#include "amr/amr_solver_base.hpp"

#include "pyfluid_amrex/amr_indicator.hpp"
#include "pyfluid_amrex/primitive_cache_factory.hpp"

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

void AmrSolverBase::fill_level_primitive_cache(int lev, amrex::Real time)
{
   if (spacetime_state_.active()) {
      prepare_spacetime_coupling_range(
         spacetime_state_.state(), lev, lev + 1, time, -1.0, 0
      );
   }
   const bool relativistic = physical_system_is_relativistic(physics_system_);
   fill_primitive_cache(PrimitiveCacheLaunch{
      *primitive_[lev],
      *fluid_state_.state()[lev],
      eos_params(cfg_),
      physics_system_,
      fluid_state_.state()[lev]->nComp(),
      relativistic && spacetime_state_.active()
         ? spacetime_coupling_.cell_spacetime()[lev].get()
         : nullptr,
      nullptr,
      nullptr,
      &Geom(lev),
      fixed_spacetime_parameters(cfg_),
      time,
   });
}

AmrSolverBase::DirectionalTagStatistics AmrSolverBase::measure_directional_tag_statistics(
   int lev,
   amrex::Real time,
   bool refresh_boundaries
)
{
   DirectionalTagStatistics statistics;
   if (!fluid_state_.state()[lev]) {
      return statistics;
   }

   if (refresh_boundaries) {
      fill_level_boundaries<false>(fluid_state_.state(), lev, time);
   }

   const amrex::Real threshold = cfg_.tag_threshold;
   const auto amr_min_feature_lengths = min_feature_lengths();
   const EOSParams eos = eos_params(cfg_);
   const int eos_method = eos_method_code(cfg_);
   const int tag_policy = amr_tag_policy_code(cfg_);
   const amrex::Real density_floor = cfg_.rho_floor;
   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   const auto problo = Geom(lev).ProbLoArray();
   const auto dx = Geom(lev).CellSizeArray();
   const amrex::Real min_refine_radius = cfg_.min_refine_radius;
   if (amr_tag_policy_uses_fluid(tag_policy)) {
      fill_level_primitive_cache(lev, time);
   } else if (spacetime_state_.active()) {
      prepare_spacetime_coupling_range(
         spacetime_state_.state(), lev, lev + 1, time, -1.0, 0
      );
   }

   amrex::Long local_tagged_cells = 0;
   std::array<amrex::Long, AMREX_SPACEDIM> local_direction_tags{};
   std::array<amrex::Real, AMREX_SPACEDIM> local_direction_strength{};
   for (amrex::MFIter mfi(*fluid_state_.state()[lev]); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const auto primitive = primitive_[lev]->const_array(mfi);
      const bool stored_spacetime = spacetime_state_.active();
      const AmrSpacetimeMetricView spacetime{
         stored_spacetime
            ? spacetime_coupling_.cell_spacetime()[lev]->const_array(mfi)
            : amrex::Array4<amrex::Real const>{},
         stored_spacetime,
      };
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
                  auto_amr_directional_indicators_from_primitive(
                     primitive,
                     i,
                     j,
                     k,
                     eos,
                     eos_method,
                     physics_system_,
                     tag_policy,
                     density_floor,
                     coordinate,
                     axes,
                     problo,
                     dx,
                     amr_min_feature_lengths,
                     min_refine_radius,
                     spacetime
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

void AmrSolverBase::validate_amr_refinement_ratios() const
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
      if (cfg_.ct != "off") {
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (ratio[dir] != 2 && ratio[dir] != 4) {
               amrex::Abort(
                  "pyfluid.ct = on requires amr.ref_ratio = 2 or 4 in "
                  "every direction"
               );
            }
         }
      }
      has_anisotropic_ratio = has_anisotropic_ratio || refinement_ratio_is_anisotropic(ratio);
   }

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

void AmrSolverBase::validate_state_interpolation_nesting() const
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

void AmrSolverBase::validate_active_hierarchy_for_evolution() const
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

int AmrSolverBase::temporal_ref_ratio(int coarse_lev) const
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

int AmrSolverBase::level_time_factor(int lev) const
{
   int factor = 1;
   for (int coarse_lev = 0; coarse_lev < lev; ++coarse_lev) {
      factor *= temporal_ref_ratio(coarse_lev);
   }
   return factor;
}

bool AmrSolverBase::local_cfl_subcycling() const
{
   return cfg_.subcycling_steps == "local_cfl";
}

int AmrSolverBase::synced_level_step(int lev, int base_step) const
{
   if (cfg_.subcycling == "on" && cfg_.subcycling_steps == "spatial") {
      return base_step * level_time_factor(lev);
   }
   return base_step;
}

bool AmrSolverBase::needs_rk4_subcycling_stage_data() const
{
   return cfg_.subcycling == "on" && cfg_.timestep == "rk4" && maxLevel() > 0;
}

void AmrSolverBase::set_level_time_state(int lev, amrex::Real time, amrex::Real dt)
{
   level_times_[lev] = time;
   level_old_times_[lev] = time;
   level_dts_[lev] = dt;
}

void AmrSolverBase::copy_level_state_to_old(int lev)
{
   fluid_state_.copy_state_to_old(lev);
   spacetime_state_.copy_state_to_old(lev);
}

bool AmrSolverBase::uses_subcycled_self_gravity_source_state() const
{
   return self_gravity_enabled(cfg_) && cfg_.subcycling == "on" && maxLevel() > 0;
}

void AmrSolverBase::copy_source_state_level(int lev, amrex::MultiFab const& src)
{
   if (!fluid_state_.source_state()[lev]) {
      amrex::Abort("subcycled self-gravity source-state storage is missing on an active AMR level");
   }
   amrex::MultiFab::Copy(*fluid_state_.source_state()[lev], src, 0, 0, num_state_components(), 0);
}

void AmrSolverBase::interpolate_source_state_level(int lev, amrex::Real alpha)
{
   if (!fluid_state_.source_state()[lev] || !fluid_state_.old_state()[lev] || !fluid_state_.state()[lev]) {
      amrex::Abort("subcycled self-gravity source-state interpolation is missing level storage");
   }
   alpha = amrex::min(amrex::max(alpha, 0.0), 1.0);
   const amrex::Real old_weight = 1.0 - alpha;
   const int ncomp = num_state_components();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(*fluid_state_.source_state()[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto out = fluid_state_.source_state()[lev]->array(mfi);
      const auto old = fluid_state_.old_state()[lev]->const_array(mfi);
      const auto current = fluid_state_.state()[lev]->const_array(mfi);
      amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
         out(i, j, k, n) = old_weight * old(i, j, k, n) + alpha * current(i, j, k, n);
      });
   }
   }
}

void AmrSolverBase::fill_source_state_level_from_level_time(int lev, amrex::Real time)
{
   if (fluid_state_.old_state()[lev] && fluid_state_.state()[lev] && level_times_[lev] > level_old_times_[lev]) {
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
   if (fluid_state_.old_state()[lev] && nearly_equal_time(time, level_old_times_[lev])) {
      copy_source_state_level(lev, *fluid_state_.old_state()[lev]);
      return;
   }
   copy_source_state_level(lev, *fluid_state_.state()[lev]);
}

void AmrSolverBase::fill_source_state_for_rhs(AmrSolverBase::LevelData& input, int lev_begin, int lev_end, amrex::Real rhs_time)
{
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      if (lev >= lev_begin && lev < lev_end) {
         copy_source_state_level(lev, *input[lev]);
      } else {
         fill_source_state_level_from_level_time(lev, rhs_time);
      }
   }
}

bool AmrSolverBase::nearly_equal_time(amrex::Real a, amrex::Real b)
{
   const amrex::Real scale = amrex::max(1.0, amrex::max(amrex::Math::abs(a), amrex::Math::abs(b)));
   const amrex::Real eps = 64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
   return amrex::Math::abs(a - b) <= eps;
}

void AmrSolverBase::invalidate_rk4_stage_cache(int lev)
{
   fluid_state_.invalidate_rk4_stage_cache(lev);
   spacetime_state_.invalidate_rk4_stage_cache(lev);
}

void AmrSolverBase::invalidate_rk4_stage_caches()
{
   for (int lev = 0; lev <= maxLevel(); ++lev) {
      invalidate_rk4_stage_cache(lev);
   }
}

} // namespace pyfluid_amrex
