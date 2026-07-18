#include "amr/amr_solver_base.hpp"
#include "pyfluid_amrex/amr_indicator.hpp"

namespace pyfluid_amrex {

amrex::Long AmrSolverBase::cell_count() const
{
   amrex::Long cells = 0;
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      if (fluid_state_.state()[lev]) {
         cells += boxArray(lev).numPts();
      }
   }
   return cells;
}

amrex::Long AmrSolverBase::active_cell_count_level(int lev) const
{
   if (!fluid_state_.state()[lev] || !solid_active(lev)) {
      return fluid_state_.state()[lev] ? boxArray(lev).numPts() : 0;
   }
   amrex::Real local_active = 0.0;
   for (amrex::MFIter mfi(*fluid_state_.state()[lev]); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const int kind = cached_mask_kind(&solid_valid_kind_[lev], mfi, *solid_mask_[lev], bx);
      if (kind == MaskKindZero) {
         local_active += static_cast<amrex::Real>(bx.numPts());
         continue;
      }
      if (kind == MaskKindNonzero) {
         continue;
      }
      const auto solid = solid_mask_[lev]->const_array(mfi);
      const auto lo = bx.smallEnd();
      const auto hi = bx.bigEnd();
#if (AMREX_SPACEDIM == 1)
      constexpr int jlo = 0;
      constexpr int jhi = 0;
      constexpr int klo = 0;
      constexpr int khi = 0;
#elif (AMREX_SPACEDIM == 2)
      const int jlo = lo[1];
      const int jhi = hi[1];
      constexpr int klo = 0;
      constexpr int khi = 0;
#else
      const int jlo = lo[1];
      const int jhi = hi[1];
      const int klo = lo[2];
      const int khi = hi[2];
#endif
      for (int k = klo; k <= khi; ++k) {
         for (int j = jlo; j <= jhi; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
               if (solid(i, j, k, 0) == 0) {
                  local_active += 1.0;
               }
            }
         }
      }
   }
   amrex::ParallelDescriptor::ReduceRealSum(local_active);
   return static_cast<amrex::Long>(std::llround(local_active));
}

amrex::Long AmrSolverBase::active_cell_count() const
{
   if (!solid_geometry_enabled(cfg_)) {
      return cell_count();
   }
   amrex::Long cells = 0;
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      cells += active_cell_count_level(lev);
   }
   return cells;
}

std::vector<amrex::Long> AmrSolverBase::level_cell_counts() const
{
   std::vector<amrex::Long> cells(maxLevel() + 1, 0);
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      if (fluid_state_.state()[lev]) {
         cells[lev] = boxArray(lev).numPts();
      }
   }
   return cells;
}

std::array<amrex::Real, AMREX_SPACEDIM> AmrSolverBase::level_cell_widths(int lev) const
{
   const auto dx = Geom(lev).CellSizeArray();
   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   const auto problo = Geom(lev).ProbLoArray();
   const amrex::Box domain = Geom(lev).Domain();
   const amrex::IntVect lo = domain.smallEnd();
   const amrex::IntVect hi = domain.bigEnd();
   std::array<amrex::Real, AMREX_SPACEDIM> widths{};
   for (int si = 0; si < 2; ++si) {
      const int i = si == 0 ? lo[0] : hi[0];
#if (AMREX_SPACEDIM >= 2)
      for (int sj = 0; sj < 2; ++sj) {
         const int j = sj == 0 ? lo[1] : hi[1];
#else
      {
         constexpr int j = 0;
#endif
#if (AMREX_SPACEDIM == 3)
         for (int sk = 0; sk < 2; ++sk) {
            const int k = sk == 0 ? lo[2] : hi[2];
#else
         {
            constexpr int k = 0;
#endif
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
               widths[dir] = amrex::max(
                  widths[dir],
                  mesh_face_gradient_spacing(coordinate, axes, problo, dx, dir, i, j, k)
               );
            }
         }
      }
   }
   return widths;
}

amrex::Real AmrSolverBase::level_cell_width(int lev) const
{
   const auto widths = level_cell_widths(lev);
   amrex::Real width = 0.0;
   for (amrex::Real directional_width : widths) {
      width = amrex::max(width, directional_width);
   }
   return width;
}

amrex::Real AmrSolverBase::active_finest_cell_width() const
{
   return level_cell_width(finestLevel());
}

amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> AmrSolverBase::min_feature_lengths() const
{
   auto lengths = configured_min_feature_lengths(cfg_);
   if (cfg_.tag_metric != "lohner" || cfg_.amr_max_level <= 1) {
      return lengths;
   }
   const auto base_widths = level_cell_widths(0);
   const amrex::Real nyquist_length_scale = amr_default_auto_min_feature_scale(cfg_.amr_max_level);
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const bool user_set_length = cfg_.min_feature_length_set;
      const bool user_set_direction =
         (dir == 0 && cfg_.min_feature_length_1_set) ||
         (dir == 1 && cfg_.min_feature_length_2_set) ||
         (dir == 2 && cfg_.min_feature_length_3_set);
      if (!user_set_length && !user_set_direction && lengths[dir] <= 0.0) {
         lengths[dir] = nyquist_length_scale * base_widths[dir];
      }
   }
   return lengths;
}

amrex::Real AmrSolverBase::min_feature_length() const
{
   const auto lengths = min_feature_lengths();
   amrex::Real summary = -1.0;
   for (amrex::Real length : lengths) {
      summary = amrex::max(summary, length);
   }
   return summary;
}

bool AmrSolverBase::auto_refinement_resolved_at_level(int lev) const
{
   const auto widths = level_cell_widths(lev);
   const auto lengths = min_feature_lengths();
   bool any_cutoff_enabled = false;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (lengths[dir] <= 0.0) {
         continue;
      }
      any_cutoff_enabled = true;
      if (widths[dir] > lengths[dir]) {
         return false;
      }
   }
   return any_cutoff_enabled;
}

AmrLevelDiagnostics AmrSolverBase::current_amr_level_diagnostics() const
{
   return summarize_amr_levels(
      level_cell_counts(),
      maxLevel(),
      cfg_.total_cell_budget,
      cfg_.level_cell_budget
   );
}

void AmrSolverBase::validate_timestep(amrex::Real dt, int step, amrex::Real time) const
{
   if (std::isfinite(dt) && dt > 0.0) {
      return;
   }
   std::ostringstream message;
   message << "Invalid timestep before step " << step
           << ": time=" << time
           << " dt=" << dt
           << ". The state likely became nonphysical in the previous step.";
   const std::string text = message.str();
   amrex::Abort(text.c_str());
}

void AmrSolverBase::validate_scalar_health(ScalarDiagnostics const& d) const
{
   if (
      std::isfinite(d.mass)
      && std::isfinite(d.energy)
      && std::isfinite(d.min_rho)
      && std::isfinite(d.max_rho)
      && std::isfinite(d.max_rho_rel_change)
      && std::isfinite(d.min_pressure)
      && d.min_rho > 0.0
      && d.max_rho >= d.min_rho
      && d.min_pressure > 0.0
   ) {
      return;
   }
   std::ostringstream message;
   message << "Nonphysical scalar diagnostics at step " << d.step
           << ": min_rho=" << d.min_rho
           << " max_rho=" << d.max_rho
           << " min_pressure=" << d.min_pressure
           << " mass=" << d.mass
           << " energy=" << d.energy
           << ".";
   const std::string text = message.str();
   amrex::Abort(text.c_str());
}

void AmrSolverBase::enforce_level_cell_budget(int lev, amrex::Long cells, char const* context) const
{
   if (cfg_.level_cell_budget <= 0 || cells <= cfg_.level_cell_budget) {
      return;
   }
   std::ostringstream message;
   message << "AMR cell budget exceeded while " << context
           << ": level " << lev
           << " has " << cells
           << " cells, but amr.level_cell_budget = "
           << cfg_.level_cell_budget
           << ". Increase the budget or reduce AMR tagging/refinement.";
   const std::string text = message.str();
   amrex::Abort(text.c_str());
}

amrex::Long AmrSolverBase::projected_cell_count(int replacement_lev, amrex::Long replacement_cells) const
{
   amrex::Long cells = replacement_cells;
   for (int lev = 0; lev < static_cast<int>(fluid_state_.state().size()); ++lev) {
      if (lev != replacement_lev && fluid_state_.state()[lev]) {
         cells += boxArray(lev).numPts();
      }
   }
   return cells;
}

void AmrSolverBase::enforce_total_cell_budget(amrex::Long cells, char const* context) const
{
   if (cfg_.total_cell_budget <= 0 || cells <= cfg_.total_cell_budget) {
      return;
   }
   std::ostringstream message;
   message << "AMR total cell budget exceeded while " << context
           << ": hierarchy has " << cells
           << " cells, but amr.total_cell_budget = "
           << cfg_.total_cell_budget
           << ". Increase the budget or reduce AMR tagging/refinement.";
   const std::string text = message.str();
   amrex::Abort(text.c_str());
}

void AmrSolverBase::enforce_projected_cell_budget(int lev, amrex::Long cells, char const* context) const
{
   enforce_level_cell_budget(lev, cells, context);
   enforce_total_cell_budget(projected_cell_count(lev, cells), context);
}

void AmrSolverBase::enforce_cell_budget(char const* context) const
{
   if (cfg_.level_cell_budget <= 0 && cfg_.total_cell_budget <= 0) {
      return;
   }
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      if (fluid_state_.state()[lev]) {
         enforce_level_cell_budget(lev, boxArray(lev).numPts(), context);
      }
   }
   enforce_total_cell_budget(cell_count(), context);
}

void AmrSolverBase::append_current_amr_level_row(std::ofstream& out, int step, amrex::Real time) const
{
   const auto finest_dx_by_dir = level_cell_widths(finestLevel());
   const auto min_feature_lengths_by_dir = min_feature_lengths();
   amrex::Real finest_dx = 0.0;
   for (amrex::Real directional_width : finest_dx_by_dir) {
      finest_dx = amrex::max(finest_dx, directional_width);
   }
   const SolidGeometryEvaluator solid_params = solid_geometry_evaluator(cfg_, time);
   const amrex::Real solid_curvature_error = solid_geometry_enabled(cfg_)
      ? solid_params.curved_boundary_error(time, finest_dx)
      : 0.0;
   const amrex::Real solid_curvature_tolerance = solid_geometry_enabled(cfg_)
      ? cfg_.solid_curvature_error
      : -1.0;
   std::vector<int> level_box_counts(maxLevel() + 1, 0);
   std::vector<amrex::Real> level_max_box_aspects(maxLevel() + 1, 0.0);
   std::array<std::vector<int>, AMREX_SPACEDIM> level_ref_ratios;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      level_ref_ratios[dir].assign(maxLevel() + 1, 1);
   }
   for (int lev = 0; lev < maxLevel(); ++lev) {
      const amrex::IntVect ratio = refRatio(lev);
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         level_ref_ratios[dir][lev] = ratio[dir];
      }
   }
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      if (!fluid_state_.state()[lev]) {
         continue;
      }
      const amrex::BoxArray& boxes = boxArray(lev);
      level_box_counts[lev] = static_cast<int>(boxes.size());
      amrex::Real max_aspect = 0.0;
      for (int ibox = 0; ibox < static_cast<int>(boxes.size()); ++ibox) {
         const amrex::Box& box = boxes[ibox];
         int min_len = box.length(0);
         int max_len = box.length(0);
#if (AMREX_SPACEDIM >= 2)
         min_len = std::min(min_len, box.length(1));
         max_len = std::max(max_len, box.length(1));
#endif
#if (AMREX_SPACEDIM == 3)
         min_len = std::min(min_len, box.length(2));
         max_len = std::max(max_len, box.length(2));
#endif
         if (min_len > 0) {
            max_aspect = std::max(max_aspect, static_cast<amrex::Real>(max_len) / static_cast<amrex::Real>(min_len));
         }
      }
      level_max_box_aspects[lev] = max_aspect;
   }
   std::vector<amrex::Long> auto_tagged_cells(maxLevel() + 1, 0);
   std::array<std::vector<amrex::Long>, AMREX_SPACEDIM> auto_direction_tagged_cells;
   std::array<std::vector<amrex::Real>, AMREX_SPACEDIM> auto_direction_strength;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      auto_direction_tagged_cells[dir].assign(maxLevel() + 1, 0);
      auto_direction_strength[dir].assign(maxLevel() + 1, 0.0);
   }
   std::vector<std::string> refinement_decisions(maxLevel() + 1, "inactive");
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      if (!fluid_state_.state()[lev]) {
         continue;
      }
      if (lev < static_cast<int>(latest_auto_tagged_cells_.size())) {
         auto_tagged_cells[lev] = latest_auto_tagged_cells_[lev];
      }
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         if (lev < static_cast<int>(latest_auto_direction_tagged_cells_[dir].size())) {
            auto_direction_tagged_cells[dir][lev] = latest_auto_direction_tagged_cells_[dir][lev];
         }
         if (lev < static_cast<int>(latest_auto_direction_strength_[dir].size())) {
            auto_direction_strength[dir][lev] = latest_auto_direction_strength_[dir][lev];
         }
      }

      if (cfg_.tag_metric == "none") {
         refinement_decisions[lev] = "tagging_off";
      } else if (auto_refinement_resolved_at_level(lev)) {
         refinement_decisions[lev] = "physical_length_guard";
      } else if (auto_tagged_cells[lev] == 0) {
         refinement_decisions[lev] = "smooth";
      } else if (lev == maxLevel()) {
         refinement_decisions[lev] = "max_level_guard";
      } else if (lev < finestLevel() && fluid_state_.state()[lev + 1]) {
         refinement_decisions[lev] = "refined";
      } else {
         refinement_decisions[lev] = "tagged_no_fine_level";
      }
   }
   append_amr_level_row(
      out,
      step,
      time,
      level_cell_counts(),
      level_steps_,
      maxLevel(),
      cfg_.total_cell_budget,
      cfg_.level_cell_budget,
      finest_dx,
      finest_dx_by_dir,
      min_feature_length(),
      min_feature_lengths_by_dir,
      cfg_.min_refine_radius,
      solid_curvature_error,
      solid_curvature_tolerance,
      level_box_counts,
      level_max_box_aspects,
      level_ref_ratios,
      auto_tagged_cells,
      auto_direction_tagged_cells,
      auto_direction_strength,
      refinement_decisions
   );
}

void AmrSolverBase::ErrorEst(
   int lev,
   amrex::TagBoxArray& tags,
   amrex::Real time,
   int /*ngrow*/
)
{
   if (lev < static_cast<int>(latest_auto_tagged_cells_.size())) {
      latest_auto_tagged_cells_[lev] = 0;
      for (auto& dir_counts : latest_auto_direction_tagged_cells_) {
         if (lev < static_cast<int>(dir_counts.size())) {
            dir_counts[lev] = 0;
         }
      }
      for (auto& dir_strength : latest_auto_direction_strength_) {
         if (lev < static_cast<int>(dir_strength.size())) {
            dir_strength[lev] = 0.0;
         }
      }
   }
   tags.setVal(amrex::TagBox::CLEAR);
   if (!fluid_state_.state()[lev]) {
      return;
   }

   const bool initial_refine_time = cfg_.initial_refine != "none"
      && cfg_.initial_refine_levels > 0
      && time <= 64.0 * std::numeric_limits<amrex::Real>::epsilon();
   const bool fixed_initial_refine = cfg_.initial_refine == "box"
      && cfg_.initial_refine_fixed_levels > 0;
   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   const bool auto_refinement_resolved = auto_refinement_resolved_at_level(lev);
   const bool solid_phi_tagging_enabled = solid_geometry_enabled(cfg_)
      && !auto_refinement_resolved
      && cfg_.solid_refine_band_cells > 0.0;
   const auto problo = Geom(lev).ProbLoArray();
   const auto dx = Geom(lev).CellSizeArray();
   const SolidGeometryEvaluator solid_params = solid_geometry_evaluator(cfg_, time);
   amrex::Real solid_refine_dx = dx[0];
#if (AMREX_SPACEDIM >= 2)
   solid_refine_dx = amrex::max(solid_refine_dx, dx[1]);
#endif
#if (AMREX_SPACEDIM == 3)
   solid_refine_dx = amrex::max(solid_refine_dx, dx[2]);
#endif
   const amrex::Real solid_refine_band = cfg_.solid_refine_band_cells * solid_refine_dx;
   const amrex::Real solid_curvature_error = solid_params.curved_boundary_error(time, solid_refine_dx);
   const bool tag_solid_phi_band = solid_phi_tagging_enabled
      && (cfg_.solid_curvature_error <= 0.0 || solid_curvature_error > cfg_.solid_curvature_error);
   const bool solid_tagging = solid_active(lev) && tag_solid_phi_band;

   if (
      cfg_.tag_metric == "none"
      && !initial_refine_time
      && !fixed_initial_refine
      && !solid_tagging
      && !tag_solid_phi_band
   ) {
      return;
   }
   // Error indicators use neighboring cells, including across coarse-fine
   // patch boundaries. Fill through the normal hierarchy contract first.
   fill_level_boundaries<false>(fluid_state_.state(), lev, time);

   const amrex::Real lohner_threshold = cfg_.tag_threshold;
   const auto amr_min_feature_lengths = min_feature_lengths();
   const amrex::Real min_refine_radius = cfg_.min_refine_radius;
   const EOSParams eos = eos_params(cfg_);
   const int eos_method = eos_method_code(cfg_);
   const PhysicalSystem physics_system = physics_system_;
   fill_level_primitive_cache(lev, time);
   const char tagval = amrex::TagBox::SET;
   const int initial_refine_levels = std::min(maxLevel(), cfg_.initial_refine_levels);
   const int fixed_initial_refine_levels = std::min(maxLevel(), cfg_.initial_refine_fixed_levels);
   const bool initial_box_refine = cfg_.initial_refine == "box"
      && (
         (initial_refine_time && lev < initial_refine_levels)
         || lev < fixed_initial_refine_levels
      );
   const bool initial_refine_only = cfg_.tag_metric == "none";
   const bool fixed_initial_hierarchy_only = initial_refine_time
      && cfg_.initial_refine_fixed_levels > 0;
   const int initial_box_count = cfg_.initial_refine_box_count;
   const auto initial_box_lo = cfg_.initial_refine_box_lo;
   const auto initial_box_hi = cfg_.initial_refine_box_hi;

   if (solid_tagging) {
      tag_solid_fluid_boundaries(tags, active_solid_mask_ptr(lev));
   }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (
      amrex::MFIter mfi(*fluid_state_.state()[lev], amrex::TilingIfNotGPU());
      mfi.isValid();
      ++mfi
   ) {
      const amrex::Box& bx = mfi.tilebox();
      const auto primitive = primitive_[lev]->const_array(mfi);
      const auto tag = tags.array(mfi);
      const bool stored_spacetime = spacetime_state_.active();
      const AmrSpacetimeMetricView spacetime{
         stored_spacetime
            ? spacetime_coupling_.cell_spacetime()[lev]->const_array(mfi)
            : amrex::Array4<amrex::Real const>{},
         stored_spacetime,
      };
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (initial_box_refine) {
            const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
            const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real z = problo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
#else
            const amrex::Real z = 0.0;
#endif
            for (int box = 0; box < initial_box_count; ++box) {
               bool inside = x >= initial_box_lo[box * AMREX_SPACEDIM]
                  && x <= initial_box_hi[box * AMREX_SPACEDIM];
#if (AMREX_SPACEDIM == 3)
               inside = inside
                  && y >= initial_box_lo[box * AMREX_SPACEDIM + 1]
                  && y <= initial_box_hi[box * AMREX_SPACEDIM + 1]
                  && z >= initial_box_lo[box * AMREX_SPACEDIM + 2]
                  && z <= initial_box_hi[box * AMREX_SPACEDIM + 2];
#elif (AMREX_SPACEDIM == 2)
               inside = inside
                  && y >= initial_box_lo[box * AMREX_SPACEDIM + 1]
                  && y <= initial_box_hi[box * AMREX_SPACEDIM + 1];
#endif
               if (inside) {
                  tag(i, j, k) = tagval;
                  break;
               }
            }
         }
         if (tag_solid_phi_band) {
            const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
            const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
#else
            const amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real z = problo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
#else
            const amrex::Real z = 0.0;
#endif
            if (amrex::Math::abs(solid_params.phi(time, x, y, z)) <= solid_refine_band) {
               tag(i, j, k) = tagval;
            }
         }
         if (initial_refine_only || fixed_initial_hierarchy_only || auto_refinement_resolved) {
            return;
         }
         const amrex::Real indicator = auto_amr_indicator_from_primitive(
            primitive,
            i,
            j,
            k,
            eos,
            eos_method,
            physics_system,
            coordinate,
            axes,
            problo,
            dx,
            amr_min_feature_lengths,
            min_refine_radius,
            spacetime
         );
         if (indicator > lohner_threshold) {
            tag(i, j, k) = tagval;
         }
      });
   }
   }

   if (
      !initial_refine_only
      && !fixed_initial_hierarchy_only
      && !auto_refinement_resolved
      && lev < static_cast<int>(latest_auto_tagged_cells_.size())
   ) {
      const DirectionalTagStatistics statistics =
         measure_directional_tag_statistics(lev, time, false);
      latest_auto_tagged_cells_[lev] = statistics.tagged_cells;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         latest_auto_direction_tagged_cells_[dir][lev] = statistics.direction_tags[dir];
         latest_auto_direction_strength_[dir][lev] = statistics.direction_strength[dir];
      }
   }
}

} // namespace pyfluid_amrex
