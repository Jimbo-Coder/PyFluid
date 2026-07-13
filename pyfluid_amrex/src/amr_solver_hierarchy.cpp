#include "pyfluid_amrex/amr_solver.hpp"

#include <AMReX_FillPatchUtil.H>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace pyfluid_amrex {

namespace {

bool refinement_ratio_is_anisotropic(amrex::IntVect const& ratio)
{
   const int first = ratio[0];
   for (int dir = 1; dir < AMREX_SPACEDIM; ++dir) {
      if (ratio[dir] != first) {
         return true;
      }
   }
   return false;
}

} // namespace

AmrSolver::DirectionalTagStatistics AmrSolver::measure_directional_tag_statistics(
   int lev,
   amrex::Real time,
   bool refresh_boundaries
)
{
   DirectionalTagStatistics statistics;
   if (!state_[lev]) {
      return statistics;
   }

   if (refresh_boundaries) {
      fill_level_boundaries<false>(state_, lev, time);
   }

   const bool mhd = is_mhd(cfg_);
   const amrex::Real threshold = cfg_.tag_threshold;
   const auto amr_min_feature_lengths = min_feature_lengths();
   const EOSParams eos = eos_params(cfg_);
   const int eos_method = eos_method_code(cfg_);
   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   const auto problo = Geom(lev).ProbLoArray();
   const auto dx = Geom(lev).CellSizeArray();
   const amrex::Real min_refine_radius = cfg_.min_refine_radius;

   amrex::Long local_tagged_cells = 0;
   std::array<amrex::Long, AMREX_SPACEDIM> local_direction_tags{};
   std::array<amrex::Real, AMREX_SPACEDIM> local_direction_strength{};
   for (amrex::MFIter mfi(*state_[lev]); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const auto s = state_[lev]->const_array(mfi);
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
               const auto directional_indicators =
                  auto_amr_directional_indicators_from_state(
                     s,
                     i,
                     j,
                     k,
                     eos,
                     eos_method,
                     mhd,
                     coordinate,
                     axes,
                     problo,
                     dx,
                     amr_min_feature_lengths,
                     min_refine_radius
                  );
               bool tagged = false;
               for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                  if (directional_indicators[dir] > threshold) {
                     local_direction_tags[dir] += 1;
                     local_direction_strength[dir] += directional_indicators[dir] / threshold;
                     tagged = true;
                  }
               }
               if (tagged) {
                  local_tagged_cells += 1;
               }
            }
         }
      }
   }

   amrex::ParallelDescriptor::ReduceLongSum(local_tagged_cells);
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::ParallelDescriptor::ReduceLongSum(local_direction_tags[dir]);
      amrex::ParallelDescriptor::ReduceRealSum(local_direction_strength[dir]);
   }

   statistics.tagged_cells = local_tagged_cells;
   statistics.direction_tags = local_direction_tags;
   statistics.direction_strength = local_direction_strength;
   return statistics;
}

void AmrSolver::validate_amr_refinement_ratios() const
{
   bool has_anisotropic_ratio = false;
   for (int lev = 0; lev < maxLevel(); ++lev) {
      const amrex::IntVect ratio = refRatio(lev);
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         if (ratio[dir] < 1) {
            amrex::Abort("amr.ref_ratio values must be positive");
         }
      }
      if (cfg_.state_interp == "quartic-conservative") {
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (ratio[dir] != 2) {
               amrex::Abort("amr.state_interp = quartic-conservative requires amr.ref_ratio = 2 in every direction");
            }
         }
      }
      has_anisotropic_ratio = has_anisotropic_ratio || refinement_ratio_is_anisotropic(ratio);
   }

#if (AMREX_SPACEDIM == 1)
   if (cfg_.state_interp == "protected") {
      amrex::Abort("amr.state_interp = protected is supported only in 2D and 3D");
   }
#endif

   if (!has_anisotropic_ratio) {
      return;
   }

   if (cfg_.mesh_patch_system != "single") {
      amrex::Abort("anisotropic amr.ref_ratio is currently supported only for mesh.patch_system = single");
   }
   if (cfg_.physics != "hydro") {
      amrex::Abort("anisotropic amr.ref_ratio is currently supported only for pyfluid.physics = hydro");
   }
   if (cfg_.ct != "off") {
      amrex::Abort("anisotropic amr.ref_ratio is currently supported only with pyfluid.ct = off");
   }
   if (cfg_.diffusion != "none") {
      amrex::Abort("anisotropic amr.ref_ratio is currently supported only with pyfluid.diffusion = none");
   }
   if (cfg_.gravity != "none") {
      amrex::Abort("anisotropic amr.ref_ratio is currently supported only with pyfluid.gravity = none");
   }
   if (cfg_.solid_geometry != "off") {
      amrex::Abort("anisotropic amr.ref_ratio is currently supported only with pyfluid.solid_geometry = off");
   }
}

void AmrSolver::validate_state_interpolation_nesting() const
{
   for (int fine_lev = 2; fine_lev <= maxLevel(); ++fine_lev) {
      const amrex::IntVect ratio = refRatio(fine_lev - 1);
      const amrex::IntVect blocking_factor = blockingFactor(fine_lev);
      if (amrex::ProperlyNested(
             ratio,
             blocking_factor,
             cfg_.ng,
             amrex::IndexType::TheCellType(),
             selected_state_interpolater()
          )) {
         continue;
      }

      std::ostringstream message;
      message
         << "AMR hierarchy is not properly nested for amr.state_interp=" << cfg_.state_interp
         << ", pyfluid.ng=" << cfg_.ng
         << ", amr.ref_ratio=" << ratio
         << ", and amr.blocking_factor=" << blocking_factor
         << " at level " << fine_lev
         << ". Increase amr.blocking_factor; an invalid interpolation footprint can corrupt "
            "coarse-fine ghost data on deep hierarchies.";
      amrex::Abort(message.str());
   }
}

void AmrSolver::validate_active_hierarchy_for_evolution() const
{
   if (patch_union_domain_enabled() && cfg_.subcycling == "on" && local_cfl_subcycling()) {
      amrex::Abort(
         "mesh.patch_box_domain = union currently supports pyfluid.subcycling_steps = spatial only; "
         "local_cfl subcycling is disabled because regridding a non-rectangular patch-union domain "
         "is not yet conservatively validated for non-refinement-ratio temporal steps"
      );
   }

   if (gravity_enabled(cfg_)) {
      if (self_gravity_enabled(cfg_) && solid_geometry_enabled(cfg_)) {
         amrex::Abort(
            "pyfluid.gravity = poisson is currently disabled with solid geometry; "
            "the fluid-only Poisson solve needs a documented internal-wall gravity boundary condition first"
         );
      }
   }

   const auto is_periodic = Geom(0).isPeriodicArray();
   const MeshAxes axes = active_mesh_axes();
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const bool lo_periodic = side_boundary_kind(cfg_, dir, false) == BoundaryKind::Periodic;
      const bool hi_periodic = side_boundary_kind(cfg_, dir, true) == BoundaryKind::Periodic;
      if (is_periodic[dir] && (!lo_periodic || !hi_periodic)) {
         amrex::Abort("geometry.is_periodic = 1 requires periodic pyfluid.bc on both sides of that direction");
      }
      if (!is_periodic[dir] && (lo_periodic || hi_periodic)) {
         amrex::Abort("geometry.is_periodic = 0 requires non-periodic pyfluid.bc on both sides of that direction");
      }
      if (mesh_axis(axes, dir) == MeshAxisPhi && !is_periodic[dir]) {
         amrex::Abort("active phi coordinate requires geometry.is_periodic = 1 and periodic pyfluid.bc on both sides");
      }
   }
}

int AmrSolver::temporal_ref_ratio(int coarse_lev) const
{
   const amrex::IntVect ratio = refRatio(coarse_lev);
   int temporal_ratio = 1;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (ratio[dir] < 1) {
         amrex::Abort("AMR temporal refinement ratio must be positive");
      }
      temporal_ratio = std::max(temporal_ratio, ratio[dir]);
   }
   return temporal_ratio;
}

int AmrSolver::level_time_factor(int lev) const
{
   int factor = 1;
   for (int coarse_lev = 0; coarse_lev < lev; ++coarse_lev) {
      factor *= temporal_ref_ratio(coarse_lev);
   }
   return factor;
}

bool AmrSolver::local_cfl_subcycling() const
{
   return cfg_.subcycling_steps == "local_cfl";
}

int AmrSolver::synced_level_step(int lev, int base_step) const
{
   if (cfg_.subcycling == "on" && cfg_.subcycling_steps == "spatial") {
      return base_step * level_time_factor(lev);
   }
   return base_step;
}

bool AmrSolver::needs_rk4_subcycling_stage_data() const
{
   return cfg_.subcycling == "on" && cfg_.timestep == "rk4" && maxLevel() > 0;
}

void AmrSolver::set_level_time_state(int lev, amrex::Real time, amrex::Real dt)
{
   level_times_[lev] = time;
   level_old_times_[lev] = time;
   level_dts_[lev] = dt;
}

void AmrSolver::copy_level_state_to_old(int lev)
{
   if (!old_state_[lev] || !state_[lev]) {
      return;
   }
   amrex::MultiFab::Copy(
      *old_state_[lev],
      *state_[lev],
      0,
      0,
      num_state_components(),
      state_[lev]->nGrowVect()
   );
}

bool AmrSolver::uses_subcycled_self_gravity_source_state() const
{
   return self_gravity_enabled(cfg_) && cfg_.subcycling == "on" && maxLevel() > 0;
}

void AmrSolver::copy_source_state_level(int lev, amrex::MultiFab const& src)
{
   if (!source_state_[lev]) {
      amrex::Abort("subcycled self-gravity source-state storage is missing on an active AMR level");
   }
   amrex::MultiFab::Copy(*source_state_[lev], src, 0, 0, num_state_components(), 0);
}

void AmrSolver::interpolate_source_state_level(int lev, amrex::Real alpha)
{
   if (!source_state_[lev] || !old_state_[lev] || !state_[lev]) {
      amrex::Abort("subcycled self-gravity source-state interpolation is missing level storage");
   }
   alpha = amrex::min(amrex::max(alpha, 0.0), 1.0);
   const amrex::Real old_weight = 1.0 - alpha;
   const int ncomp = num_state_components();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(*source_state_[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto out = source_state_[lev]->array(mfi);
      const auto old = old_state_[lev]->const_array(mfi);
      const auto current = state_[lev]->const_array(mfi);
      amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
         out(i, j, k, n) = old_weight * old(i, j, k, n) + alpha * current(i, j, k, n);
      });
   }
   }
}

void AmrSolver::fill_source_state_level_from_level_time(int lev, amrex::Real time)
{
   if (old_state_[lev] && state_[lev] && level_times_[lev] > level_old_times_[lev]) {
      const amrex::Real scale = amrex::max(
         1.0,
         amrex::max(amrex::Math::abs(level_old_times_[lev]), amrex::Math::abs(level_times_[lev]))
      );
      const amrex::Real eps = 64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
      if (time >= level_old_times_[lev] - eps && time <= level_times_[lev] + eps) {
         const amrex::Real alpha = (time - level_old_times_[lev]) / (level_times_[lev] - level_old_times_[lev]);
         interpolate_source_state_level(lev, alpha);
         return;
      }
   }
   if (old_state_[lev] && nearly_equal_time(time, level_old_times_[lev])) {
      copy_source_state_level(lev, *old_state_[lev]);
      return;
   }
   copy_source_state_level(lev, *state_[lev]);
}

void AmrSolver::fill_source_state_for_rhs(AmrSolver::LevelData& input, int lev_begin, int lev_end, amrex::Real rhs_time)
{
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      if (lev >= lev_begin && lev < lev_end) {
         copy_source_state_level(lev, *input[lev]);
      } else {
         fill_source_state_level_from_level_time(lev, rhs_time);
      }
   }
}

bool AmrSolver::nearly_equal_time(amrex::Real a, amrex::Real b)
{
   const amrex::Real scale = amrex::max(1.0, amrex::max(amrex::Math::abs(a), amrex::Math::abs(b)));
   const amrex::Real eps = 64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
   return amrex::Math::abs(a - b) <= eps;
}

void AmrSolver::invalidate_rk4_stage_cache(int lev)
{
   if (lev >= 0 && lev < static_cast<int>(rk4_stage_cache_stage_.size())) {
      rk4_stage_cache_stage_[lev] = -1;
   }
}

void AmrSolver::invalidate_rk4_stage_caches()
{
   for (int lev = 0; lev < static_cast<int>(rk4_stage_cache_stage_.size()); ++lev) {
      invalidate_rk4_stage_cache(lev);
   }
}

} // namespace pyfluid_amrex
