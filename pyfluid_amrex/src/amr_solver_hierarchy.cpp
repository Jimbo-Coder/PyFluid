#include "pyfluid_amrex/amr_solver.hpp"

#include <algorithm>
#include <cmath>

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

bool refinement_ratio_equal(amrex::IntVect const& a, amrex::IntVect const& b)
{
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (a[dir] != b[dir]) {
         return false;
      }
   }
   return true;
}

int refinement_ratio_product(amrex::IntVect const& ratio)
{
   int product = 1;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      product *= ratio[dir];
   }
   return product;
}

} // namespace

bool AmrSolver::automatic_refinement_tensor() const
{
   return cfg_.refinement_tensor == "auto";
}

AmrSolver::DirectionalRefinementDemand AmrSolver::measure_directional_refinement_demand(
   int lev,
   amrex::Real time,
   bool refresh_boundaries
)
{
   DirectionalRefinementDemand demand;
   if (!state_[lev]) {
      return demand;
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

   demand.tagged_cells = local_tagged_cells;
   demand.direction_tags = local_direction_tags;
   demand.direction_strength = local_direction_strength;
   return demand;
}

amrex::IntVect AmrSolver::refinement_ratio_from_demand(DirectionalRefinementDemand const& demand) const
{
   amrex::IntVect ratio(1);
   if (demand.tagged_cells == 0) {
      return ratio;
   }

   const int max_ratio = cfg_.max_refinement_ratio;
   constexpr int target_volume_ratio = 1 << AMREX_SPACEDIM;
   std::array<int, AMREX_SPACEDIM> requested_ratio{};
   std::array<amrex::Real, AMREX_SPACEDIM> mean_over_threshold{};
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (demand.direction_tags[dir] <= 0 || demand.direction_strength[dir] <= 0.0) {
         requested_ratio[dir] = 1;
         continue;
      }
      mean_over_threshold[dir] =
         demand.direction_strength[dir] / static_cast<amrex::Real>(demand.direction_tags[dir]);
      requested_ratio[dir] = std::min(
         max_ratio,
         std::max(2, static_cast<int>(std::ceil(mean_over_threshold[dir])))
      );
   }

   int product = 1;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (requested_ratio[dir] > 1) {
         ratio[dir] = 2;
         product *= 2;
      }
   }

   while (product < target_volume_ratio) {
      int best_dir = -1;
      amrex::Real best_demand = 0.0;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         if (ratio[dir] >= requested_ratio[dir]) {
            continue;
         }
         const int next_product = (product / ratio[dir]) * (ratio[dir] + 1);
         if (next_product > target_volume_ratio) {
            continue;
         }
         if (best_dir < 0 || mean_over_threshold[dir] > best_demand) {
            best_dir = dir;
            best_demand = mean_over_threshold[dir];
         }
      }
      if (best_dir < 0) {
         break;
      }
      product = (product / ratio[best_dir]) * (ratio[best_dir] + 1);
      ratio[best_dir] += 1;
   }
   if (refinement_ratio_product(ratio) > target_volume_ratio) {
      amrex::Abort("automatic AMR selected an excessive refinement-ratio tensor");
   }
   return ratio;
}

amrex::IntVect AmrSolver::desired_refinement_ratio(int coarse_lev, amrex::Real time)
{
   if (!automatic_refinement_tensor() || coarse_lev >= maxLevel() || !state_[coarse_lev]) {
      return refRatio(coarse_lev);
   }
   if (cfg_.tag_metric != "auto" || auto_refinement_resolved_at_level(coarse_lev)) {
      return refRatio(coarse_lev);
   }

   const DirectionalRefinementDemand demand =
      measure_directional_refinement_demand(coarse_lev, time, true);
   if (demand.tagged_cells == 0) {
      return refRatio(coarse_lev);
   }
   return refinement_ratio_from_demand(demand);
}

bool AmrSolver::set_refinement_ratio(int coarse_lev, amrex::IntVect ratio)
{
   if (coarse_lev < 0 || coarse_lev >= maxLevel()) {
      return false;
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (ratio[dir] < 1) {
         amrex::Abort("automatic AMR selected an invalid refinement ratio");
      }
   }
   if (refinement_ratio_equal(ref_ratio[coarse_lev], ratio)) {
      return false;
   }
   ref_ratio[coarse_lev] = ratio;
   return true;
}

void AmrSolver::rebuild_geometries_from_level(int coarse_lev)
{
   const int first_fine = std::max(1, coarse_lev + 1);
   for (int lev = first_fine; lev <= maxLevel(); ++lev) {
      SetGeometry(lev, amrex::refine(Geom(lev - 1), refRatio(lev - 1)));
   }
}

void AmrSolver::initialize_hierarchy_from_scratch(amrex::Real time)
{
   if (!automatic_refinement_tensor()) {
      InitFromScratch(time);
      return;
   }

   amrex::BoxArray base_grids = MakeBaseGrids();
   const amrex::DistributionMapping base_dmap = MakeDistributionMap(0, base_grids);
   MakeNewLevelFromScratch(0, time, base_grids, base_dmap);
   SetFinestLevel(0);

   while (finestLevel() < maxLevel()) {
      const int coarse_lev = finestLevel();
      set_refinement_ratio(coarse_lev, desired_refinement_ratio(coarse_lev, time));
      rebuild_geometries_from_level(coarse_lev);
      const int old_finest = finestLevel();
      regrid(coarse_lev, time, true);
      if (finestLevel() == old_finest) {
         break;
      }
   }
}

void AmrSolver::regrid_adaptive_tensor(int lbase, amrex::Real time, bool initial)
{
   if (!automatic_refinement_tensor()) {
      regrid(lbase, time, initial);
      return;
   }

   int rebuild_from = -1;
   amrex::IntVect replacement_ratio(1);
   for (int coarse_lev = lbase; coarse_lev < finestLevel(); ++coarse_lev) {
      const amrex::IntVect desired = desired_refinement_ratio(coarse_lev, time);
      if (!refinement_ratio_equal(desired, refRatio(coarse_lev))) {
         rebuild_from = coarse_lev;
         replacement_ratio = desired;
         break;
      }
   }

   if (rebuild_from >= 0) {
      average_down_state();
      for (int lev = finestLevel(); lev > rebuild_from; --lev) {
         ClearLevel(lev);
         ClearBoxArray(lev);
         ClearDistributionMap(lev);
      }
      SetFinestLevel(rebuild_from);
      set_refinement_ratio(rebuild_from, replacement_ratio);
      rebuild_geometries_from_level(rebuild_from);
   }

   const int regrid_base = rebuild_from >= 0 ? rebuild_from : lbase;
   regrid(regrid_base, time, initial);

   while (finestLevel() < maxLevel()) {
      const int coarse_lev = finestLevel();
      set_refinement_ratio(coarse_lev, desired_refinement_ratio(coarse_lev, time));
      rebuild_geometries_from_level(coarse_lev);
      const int old_finest = finestLevel();
      regrid(coarse_lev, time, initial);
      if (finestLevel() == old_finest) {
         break;
      }
   }
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
      has_anisotropic_ratio = has_anisotropic_ratio || refinement_ratio_is_anisotropic(ratio);
   }

   if (!has_anisotropic_ratio) {
      return;
   }

   if (cfg_.subcycling == "on") {
      amrex::Abort("anisotropic amr.ref_ratio currently requires pyfluid.subcycling = off");
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
   const int value = ratio[0];
   if (value < 1) {
      amrex::Abort("AMR temporal refinement ratio must be positive");
   }
   for (int dir = 1; dir < AMREX_SPACEDIM; ++dir) {
      if (ratio[dir] != value) {
         amrex::Abort("pyfluid.subcycling = on requires isotropic AMR ref_ratio");
      }
   }
   return value;
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
