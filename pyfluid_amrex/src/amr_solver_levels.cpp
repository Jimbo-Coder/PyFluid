#include "pyfluid_amrex/amr_solver.hpp"

namespace pyfluid_amrex {

void AmrSolver::define_flux_registers(int first_fine_level)
{
   first_fine_level = std::max(first_fine_level, 1);
   for (int lev = first_fine_level; lev < static_cast<int>(flux_registers_.size()); ++lev) {
      flux_registers_[lev].reset();
      gravity_energy_source_register_[lev].reset();
      coarse_reflux_volume_[lev].reset();
#if (AMREX_SPACEDIM >= 2)
      edge_flux_registers_[lev].reset();
#endif
   }
   for (int lev = first_fine_level; lev <= finestLevel(); ++lev) {
      flux_registers_[lev] = std::make_unique<amrex::YAFluxRegister>(
         boxArray(lev),
         boxArray(lev - 1),
         DistributionMap(lev),
         DistributionMap(lev - 1),
         Geom(lev),
         Geom(lev - 1),
         refRatio(lev - 1),
         lev,
         num_state_components()
      );
      prepare_weighted_flux_register(lev);
      if (gravity_energy_source_register_needed(lev)) {
         gravity_energy_source_register_[lev] = std::make_unique<amrex::MultiFab>(
            state_[lev - 1]->boxArray(),
            state_[lev - 1]->DistributionMap(),
            1,
            0
         );
         gravity_energy_source_register_[lev]->setVal(0.0);
      }
#if (AMREX_SPACEDIM >= 2)
      if (ct_enabled(cfg_)) {
         edge_flux_registers_[lev] = std::make_unique<CtEdgeFluxRegister>(
            boxArray(lev),
            boxArray(lev - 1),
            DistributionMap(lev),
            DistributionMap(lev - 1),
            Geom(lev),
            Geom(lev - 1),
            1
         );
      }
#endif
   }
}

void AmrSolver::reset_flux_registers()
{
   for (int lev = 1; lev <= finestLevel(); ++lev) {
      reset_flux_register(lev);
   }
}

void AmrSolver::reset_edge_flux_registers()
{
#if (AMREX_SPACEDIM >= 2)
   for (int lev = 1; lev <= finestLevel(); ++lev) {
      if (edge_flux_registers_[lev]) {
         edge_flux_registers_[lev]->reset();
      }
   }
#endif
}

void AmrSolver::reset_flux_register(int fine_lev)
{
   if (fine_lev > 0 && fine_lev <= finestLevel() && flux_registers_[fine_lev]) {
      flux_registers_[fine_lev]->reset();
   }
   if (
      fine_lev > 0
      && fine_lev <= finestLevel()
      && gravity_energy_source_register_[fine_lev]
   ) {
      gravity_energy_source_register_[fine_lev]->setVal(0.0);
   }
#if (AMREX_SPACEDIM >= 2)
   if (fine_lev > 0 && fine_lev <= finestLevel() && edge_flux_registers_[fine_lev]) {
      edge_flux_registers_[fine_lev]->reset();
   }
#endif
}

void AmrSolver::allocate_level(
   int lev,
   amrex::BoxArray const& ba,
   amrex::DistributionMapping const& dm,
   amrex::Real time
)
{
   invalidate_rk4_stage_cache(lev);
   enforce_projected_cell_budget(lev, ba.numPts(), "allocating AMR level");
   const int ncomp = num_state_components();
   state_[lev] = std::make_unique<amrex::MultiFab>(ba, dm, ncomp, cfg_.ng);
   old_state_[lev] = std::make_unique<amrex::MultiFab>(ba, dm, ncomp, cfg_.ng);
   if (uses_subcycled_self_gravity_source_state()) {
      source_state_[lev] = std::make_unique<amrex::MultiFab>(ba, dm, ncomp, cfg_.ng);
   }
   if (needs_rk4_subcycling_stage_data()) {
      rk4_coarse_stage_state_[lev] = std::make_unique<amrex::MultiFab>(ba, dm, ncomp, cfg_.ng);
      for (auto& stage : rk4_rhs_stage_) {
         stage[lev] = std::make_unique<amrex::MultiFab>(ba, dm, ncomp, 0);
      }
   }
   next_[lev] = std::make_unique<amrex::MultiFab>(ba, dm, ncomp, cfg_.ng);
   scratch_[lev] = std::make_unique<amrex::MultiFab>(ba, dm, ncomp, cfg_.ng);
   rhs_[lev] = std::make_unique<amrex::MultiFab>(ba, dm, ncomp, 0);
   primitive_[lev] = std::make_unique<amrex::MultiFab>(ba, dm, ncomp, cfg_.ng);
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const amrex::BoxArray flux_ba = amrex::convert(ba, amrex::IntVect::TheDimensionVector(dir));
      fluxes_[lev][dir] = std::make_unique<amrex::MultiFab>(flux_ba, dm, ncomp, 0);
   }
#if (AMREX_SPACEDIM >= 2)
   if (ct_enabled(cfg_)) {
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         flux_owner_masks_[lev][dir] = fluxes_[lev][dir]->OwnerMask(Geom(lev).periodicity());
      }
   }
#endif
   frequency_[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
   frequency_[lev]->setVal(0.0);
   pressure_[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
   if (solid_geometry_enabled(cfg_)) {
      solid_volume_fraction_[lev].reset();
      for (auto& aperture : solid_aperture_) {
         aperture[lev].reset();
      }
      solid_mask_[lev] = std::make_unique<amrex::iMultiFab>(
         ba,
         dm,
         1,
         std::max(SolidMotionSearchRadius, cfg_.ng)
      );
      fill_solid_mask(*solid_mask_[lev], Geom(lev), cfg_, time);
      refresh_solid_discrete_geometry_level(lev, time);
      solid_mask_time_[lev] = time;
      primitive_needed_mask_[lev] = std::make_unique<amrex::iMultiFab>(ba, dm, 1, cfg_.ng);
      fill_primitive_needed_mask(*primitive_needed_mask_[lev], *solid_mask_[lev], Geom(lev), cfg_.ng);
      refresh_solid_activity(lev);
      build_solid_mask_kind_tables(lev);
   } else {
      solid_active_[lev] = false;
      solid_mask_time_[lev] = std::numeric_limits<amrex::Real>::quiet_NaN();
      clear_solid_mask_kind_tables(lev);
   }
   if (needs_edge_current_density()) {
      allocate_edge_electric_field(edge_current_[lev], ba, dm);
   }
   if (gravity_enabled(cfg_)) {
      gravity_phi_[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 1);
      gravity_phi_[lev]->setVal(0.0);
      allocate_gravity_gradient(gravity_gradient_[lev], ba, dm);
      if (uses_athena_gravity_energy_correction()) {
         allocate_gravity_gradient(gravity_start_gradient_[lev], ba, dm);
         gravity_stage_energy_release_[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
         gravity_stage_energy_release_[lev]->setVal(0.0);
         for (auto& stage_flux : gravity_stage_mass_flux_) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
               const amrex::BoxArray flux_ba = amrex::convert(ba, amrex::IntVect::TheDimensionVector(dir));
               stage_flux[lev][dir] = std::make_unique<amrex::MultiFab>(flux_ba, dm, 1, 0);
               stage_flux[lev][dir]->setVal(0.0);
            }
         }
      }
      gravity_fixed_cache_valid_[lev] = false;
   }
   if (ct_enabled(cfg_)) {
      allocate_face_magnetic_field(face_b_[lev], ba, dm, cfg_.ng);
   }
   if (needs_edge_electric_field()) {
      allocate_edge_electric_field(edge_e_[lev], ba, dm);
   }
}

amrex::Long AmrSolver::active_cell_count_in_box(int lev, amrex::Box const& box, amrex::Real time) const
{
   if (!solid_geometry_enabled(cfg_)) {
      return box.numPts();
   }
   const auto problo = Geom(lev).ProbLoArray();
   const auto dx = Geom(lev).CellSizeArray();
   const SolidGeometryEvaluator params = solid_geometry_evaluator(cfg_, time);
   const auto lo = box.smallEnd();
   const auto hi = box.bigEnd();
   amrex::Long active = 0;
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
      const amrex::Real z =
#if (AMREX_SPACEDIM == 3)
         problo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
#else
         0.0;
#endif
      for (int j = jlo; j <= jhi; ++j) {
         const amrex::Real y =
#if (AMREX_SPACEDIM >= 2)
            problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
#else
            0.0;
#endif
         for (int i = lo[0]; i <= hi[0]; ++i) {
            const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
            if (!params.contains(time, x, y, z)) {
               ++active;
            }
         }
      }
   }
   return active;
}

amrex::DistributionMapping AmrSolver::obstacle_weighted_distribution_mapping(
   int lev,
   amrex::BoxArray const& ba,
   amrex::DistributionMapping const& base_dm,
   amrex::Real time
) const
{
   if (!solid_geometry_enabled(cfg_) || amrex::ParallelDescriptor::NProcs() <= 1) {
      return base_dm;
   }
   amrex::Vector<amrex::Real> costs(ba.size());
   for (int ibox = 0; ibox < static_cast<int>(ba.size()); ++ibox) {
      const amrex::Long active = active_cell_count_in_box(lev, ba[ibox], time);
      costs[ibox] = static_cast<amrex::Real>(std::max<amrex::Long>(active, 1));
   }
   return amrex::DistributionMapping::makeKnapSack(costs);
}

amrex::BoxArray AmrSolver::owned_level_box_array(int lev, amrex::BoxArray const& ba) const
{
   if (!production_multipatch_enabled()) {
      return ba;
   }
   amrex::BoxArray owned_ba = clip_to_active_multipatch_domain(lev, ba);
   if (owned_ba.empty()) {
      amrex::Abort("multipatch level construction produced no owned boxes");
   }
   return owned_ba;
}

amrex::DistributionMapping AmrSolver::owned_level_distribution_mapping(
   int lev,
   amrex::BoxArray const& original_ba,
   amrex::BoxArray const& owned_ba,
   amrex::DistributionMapping const& base_dm,
   amrex::Real time
)
{
   const amrex::DistributionMapping level_dm =
      owned_ba == original_ba ? base_dm : MakeDistributionMap(lev, owned_ba);
   return obstacle_weighted_distribution_mapping(lev, owned_ba, level_dm, time);
}

void AmrSolver::MakeNewLevelFromScratch(
   int lev,
   amrex::Real time,
   amrex::BoxArray const& ba,
   amrex::DistributionMapping const& dm
)
{
   const amrex::BoxArray level_ba = owned_level_box_array(lev, ba);
   const amrex::DistributionMapping work_dm =
      owned_level_distribution_mapping(lev, ba, level_ba, dm, time);
   SetBoxArray(lev, level_ba);
   SetDistributionMap(lev, work_dm);
   allocate_level(lev, level_ba, work_dm, time);
   set_level_time_state(lev, time, 0.0);
   initialize_state(*state_[lev], Geom(lev), cfg_);
   if (ct_enabled(cfg_) && lev > 0) {
      initialize_ct_fields_from_coarse(lev, time);
   } else {
      initialize_ct_fields(lev, time);
   }
   copy_level_state_to_old(lev);
}

void AmrSolver::MakeNewLevelFromCoarse(
   int lev,
   amrex::Real time,
   amrex::BoxArray const& ba,
   amrex::DistributionMapping const& dm
)
{
   const amrex::BoxArray level_ba = owned_level_box_array(lev, ba);
   const amrex::DistributionMapping work_dm =
      owned_level_distribution_mapping(lev, ba, level_ba, dm, time);
   SetBoxArray(lev, level_ba);
   SetDistributionMap(lev, work_dm);
   allocate_level(lev, level_ba, work_dm, time);
   set_level_time_state(lev, time, 0.0);
   if (time == 0.0) {
      initialize_state(*state_[lev], Geom(lev), cfg_);
   } else {
      fill_new_level_from_coarse_state(lev, time);
   }
   initialize_ct_fields_from_coarse(lev, time);
   copy_level_state_to_old(lev);
}

void AmrSolver::RemakeLevel(
   int lev,
   amrex::Real time,
   amrex::BoxArray const& ba,
   amrex::DistributionMapping const& dm
)
{
   std::unique_ptr<amrex::MultiFab> old_state = std::move(state_[lev]);
   FaceMagneticField old_face_b;
   if (ct_enabled(cfg_)) {
      old_face_b = std::move(face_b_[lev]);
   }
   const amrex::BoxArray level_ba = owned_level_box_array(lev, ba);
   const amrex::DistributionMapping work_dm =
      owned_level_distribution_mapping(lev, ba, level_ba, dm, time);
   SetBoxArray(lev, level_ba);
   SetDistributionMap(lev, work_dm);
   allocate_level(lev, level_ba, work_dm, time);
   set_level_time_state(lev, time, 0.0);
   fill_remade_level_state(lev, time, *old_state);
   initialize_remade_ct_fields(lev, time, old_face_b);
   copy_level_state_to_old(lev);
}

void AmrSolver::ClearLevel(int lev)
{
   state_[lev].reset();
   old_state_[lev].reset();
   source_state_[lev].reset();
   rk4_coarse_stage_state_[lev].reset();
   for (auto& stage : rk4_rhs_stage_) {
      stage[lev].reset();
   }
   next_[lev].reset();
   scratch_[lev].reset();
   rhs_[lev].reset();
   primitive_[lev].reset();
   for (auto& flux : fluxes_[lev]) {
      flux.reset();
   }
#if (AMREX_SPACEDIM >= 2)
   for (auto& mask : flux_owner_masks_[lev]) {
      mask.reset();
   }
#endif
   flux_registers_[lev].reset();
   gravity_energy_source_register_[lev].reset();
   coarse_reflux_volume_[lev].reset();
   frequency_[lev].reset();
   pressure_[lev].reset();
   solid_mask_[lev].reset();
   solid_volume_fraction_[lev].reset();
   for (auto& aperture : solid_aperture_) {
      aperture[lev].reset();
   }
   solid_mask_time_[lev] = std::numeric_limits<amrex::Real>::quiet_NaN();
   primitive_needed_mask_[lev].reset();
   clear_solid_mask_kind_tables(lev);
   solid_active_[lev] = false;
   gravity_phi_[lev].reset();
   for (auto& component : gravity_gradient_[lev]) {
      component.reset();
   }
   for (auto& component : gravity_start_gradient_[lev]) {
      component.reset();
   }
   gravity_stage_energy_release_[lev].reset();
   for (auto& stage_flux : gravity_stage_mass_flux_) {
      for (auto& component : stage_flux[lev]) {
         component.reset();
      }
   }
   gravity_fixed_cache_valid_[lev] = false;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      face_b_[lev].b[dir].reset();
   }
   for (auto& component : edge_e_[lev].e) {
      component.reset();
   }
   for (auto& component : edge_current_[lev].e) {
      component.reset();
   }
   level_steps_[lev] = 0;
   level_times_[lev] = 0.0;
   level_old_times_[lev] = 0.0;
   level_dts_[lev] = 0.0;
   invalidate_rk4_stage_cache(lev);
}

} // namespace pyfluid_amrex
