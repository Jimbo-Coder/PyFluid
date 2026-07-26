#include "amr/amr_solver_base.hpp"
#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/initial_data.hpp"

#include <limits>
#include <utility>

namespace pyfluid_amrex {

AmrSolverBase::AmrSolverBase(RunConfig cfg)
   : cfg_(std::move(cfg)),
     initial_data_provider_(cfg_),
     recovery_atmosphere_(recovery_atmosphere_state(
        eos_method_code(cfg_), eos_params(cfg_)
     ))
{
   validate_config(cfg_);
   validate_amr_refinement_ratios();
   physics_system_ = physical_system(cfg_);
   mesh_layout_ = mesh_runtime_layout(cfg_);
   const int num_levels = maxLevel() + 1;
   fluid_state_.configure(
      fluid_state_grid_fields(physics_system_, cfg_.passive_scalars),
      num_levels,
      cfg_.ng
   );
   if (spacetime_stores_state(cfg_)) {
      GridFieldSetInfo const& spacetime_fields = z4c_state_grid_fields();
      const int spacetime_stencil_width = grid_field_ghost_width(spacetime_fields);
      spacetime_state_.configure(
         spacetime_fields,
         num_levels,
         std::max(spacetime_stencil_width, cfg_.ng + 2),
         spacetime_evolves(cfg_)
      );
      spacetime_coupling_.configure(
         num_levels,
         std::max(spacetime_stencil_width, cfg_.ng)
      );
      spacetime_fillpatch_bcs_ = z4c_boundary_records(cfg_, Geom(0));
      spacetime_fillpatch_func_ = SpacetimeFillPatchFunc(Z4cFillPatchBoundary{});
   }
   analytic_fixed_spacetime_kernels_ =
      !spacetime_state_.evolves()
      && cfg_.spacetime_initial_data == "minkowski"
      && cfg_.initial_data_input_file.empty();
   state_interpolater_.configure(
      selected_state_interpolater(),
      mesh_layout_.coordinate,
      mesh_layout_.axes,
      !mhd_enabled()
   );
   validate_state_interpolation_nesting();
   recon_method_ = recon_method_code(cfg_);
   recon_fine_method_ = cfg_.recon_fine.empty()
      ? recon_method_
      : recon_method_code_from_name(cfg_.recon_fine, "pyfluid.recon_fine");
   riemann_method_ = riemann_method_code(cfg_);
   timestep_method_ = timestep_method_code(cfg_);
   diffusion_method_ = diffusion_method_code(cfg_);
   eos_method_ = eos_method_code(cfg_);
   profile_.configure(cfg_.profile, cfg_.profile_detail);
   moving_wall_work_enabled_ = cfg_.solid_motion == "translate"
      || cfg_.solid_motion == "pulse";
   boundary_data_ = make_boundary_data(cfg_);
   state_fillpatch_bcs_ = make_state_fillpatch_bcs();
   face_fillpatch_bcs_ = make_face_fillpatch_bcs();
   state_fillpatch_func_ = StateFillPatchFunc(StateFillPatchBoundary{boundary_data_});
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      face_fillpatch_func_[direction] = FaceFillPatchFunc(
         FaceBFillPatchBoundary{boundary_data_, direction}
      );
   }
   post_fillpatch_state_boundary_needed_ = compute_post_fillpatch_state_boundary_needed();
   primitive_.resize(num_levels);
   fluxes_.resize(num_levels);
#if (AMREX_SPACEDIM >= 2)
   flux_owner_masks_.resize(num_levels);
#endif
   flux_registers_.resize(num_levels);
   gravity_energy_source_register_.resize(num_levels);
   coarse_reflux_volume_.resize(num_levels);
   frequency_.resize(num_levels);
   pressure_.resize(num_levels);
   solid_mask_.resize(num_levels);
   solid_volume_fraction_.resize(num_levels);
   for (auto& aperture : solid_aperture_) {
      aperture.resize(num_levels);
   }
   primitive_needed_mask_.resize(num_levels);
   solid_valid_kind_.resize(num_levels);
   primitive_needed_kind_.resize(num_levels);
   solid_flux_kind_.resize(num_levels);
   solid_active_.resize(num_levels, false);
   solid_mask_time_.resize(
      num_levels,
      std::numeric_limits<amrex::Real>::quiet_NaN()
   );
   gravity_phi_.resize(num_levels);
   gravity_gradient_.resize(num_levels);
   gravity_start_gradient_.resize(num_levels);
   gravity_stage_energy_release_.resize(num_levels);
   for (auto& stage_flux : gravity_stage_mass_flux_) {
      stage_flux.resize(num_levels);
   }
   gravity_fixed_cache_valid_.resize(num_levels, false);
   face_b_.resize(num_levels);
   edge_e_.resize(num_levels);
   ct_stage_edge_e_.resize(num_levels);
   edge_current_.resize(num_levels);
#if (AMREX_SPACEDIM >= 2)
   edge_flux_registers_.resize(num_levels);
#endif
   level_steps_.resize(num_levels, 0);
   level_times_.resize(num_levels, 0.0);
   level_old_times_.resize(num_levels, 0.0);
   level_dts_.resize(num_levels, 0.0);
   latest_auto_tagged_cells_.resize(num_levels, 0);
   for (auto& direction_counts : latest_auto_direction_tagged_cells_) {
      direction_counts.resize(num_levels, 0);
   }
   for (auto& direction_strength : latest_auto_direction_strength_) {
      direction_strength.resize(num_levels, 0.0);
   }
}

void AmrSolverBase::define_flux_registers(int first_fine_level)
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
            fluid_state_.state()[lev - 1]->boxArray(),
            fluid_state_.state()[lev - 1]->DistributionMap(),
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

void AmrSolverBase::reset_flux_registers()
{
   for (int lev = 1; lev <= finestLevel(); ++lev) {
      reset_flux_register(lev);
   }
}

void AmrSolverBase::reset_edge_flux_registers()
{
#if (AMREX_SPACEDIM >= 2)
   for (int lev = 1; lev <= finestLevel(); ++lev) {
      if (edge_flux_registers_[lev]) {
         edge_flux_registers_[lev]->reset();
      }
   }
#endif
}

void AmrSolverBase::reset_flux_register(int fine_lev)
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

void AmrSolverBase::allocate_level(
   int lev,
   amrex::BoxArray const& ba,
   amrex::DistributionMapping const& dm,
   amrex::Real time
)
{
   invalidate_rk4_stage_cache(lev);
   enforce_projected_cell_budget(lev, ba.numPts(), "allocating AMR level");
   const int ncomp = num_state_components();
   fluid_state_.allocate_level(
      lev,
      ba,
      dm,
      uses_subcycled_self_gravity_source_state(),
      needs_rk4_subcycling_stage_data()
   );
   if (spacetime_state_.active()) {
      spacetime_state_.allocate_level(
         lev, ba, dm, false, needs_rk4_subcycling_stage_data()
      );
      spacetime_coupling_.allocate_level(lev, ba, dm);
   }
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
      allocate_edge_electric_field(ct_stage_edge_e_[lev], ba, dm);
   }
   if (needs_edge_electric_field()) {
      allocate_edge_electric_field(edge_e_[lev], ba, dm);
   }
}

amrex::Long AmrSolverBase::active_cell_count_in_box(int lev, amrex::Box const& box, amrex::Real time) const
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

amrex::DistributionMapping AmrSolverBase::obstacle_weighted_distribution_mapping(
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

amrex::BoxArray AmrSolverBase::owned_level_box_array(int lev, amrex::BoxArray const& ba) const
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

amrex::DistributionMapping AmrSolverBase::owned_level_distribution_mapping(
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

void AmrSolverBase::MakeNewLevelFromScratch(
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
   initialize_state(
      *fluid_state_.state()[lev], Geom(lev), cfg_, initial_data_provider_.view()
   );
   initialize_spacetime_level(lev, time);
   refresh_stored_spacetime_fluid_boundary(lev, time);
   if (time == 0.0) {
      initialize_ct_fields(lev, time);
   } else {
      initialize_ct_fields_from_coarse(lev, time);
   }
   copy_level_state_to_old(lev);
}

void AmrSolverBase::MakeNewLevelFromCoarse(
   int lev,
   amrex::Real time,
   amrex::BoxArray const& ba,
   amrex::DistributionMapping const& dm
)
{
   mark_regrid_level_modified(lev);
   const amrex::BoxArray level_ba = owned_level_box_array(lev, ba);
   const amrex::DistributionMapping work_dm =
      owned_level_distribution_mapping(lev, ba, level_ba, dm, time);
   SetBoxArray(lev, level_ba);
   SetDistributionMap(lev, work_dm);
   allocate_level(lev, level_ba, work_dm, time);
   set_level_time_state(lev, time, 0.0);
   if (time == 0.0) {
      initialize_state(
         *fluid_state_.state()[lev], Geom(lev), cfg_, initial_data_provider_.view()
      );
   } else {
      fill_new_level_from_coarse_state(lev, time);
   }
   fill_new_spacetime_level_from_coarse(lev, time);
   refresh_stored_spacetime_fluid_boundary(lev, time);
   if (time == 0.0) {
      initialize_ct_fields(lev, time);
   } else {
      initialize_ct_fields_from_coarse(lev, time);
   }
   copy_level_state_to_old(lev);
}

void AmrSolverBase::RemakeLevel(
   int lev,
   amrex::Real time,
   amrex::BoxArray const& ba,
   amrex::DistributionMapping const& dm
)
{
   mark_regrid_level_modified(lev);
   std::unique_ptr<amrex::MultiFab> old_state = std::move(fluid_state_.state()[lev]);
   std::unique_ptr<amrex::MultiFab> old_spacetime;
   if (spacetime_state_.active()) {
      old_spacetime = std::move(spacetime_state_.state()[lev]);
   }
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
   if (old_spacetime) {
      fill_remade_spacetime_level(lev, time, *old_spacetime);
   }
   refresh_stored_spacetime_fluid_boundary(lev, time);
   initialize_remade_ct_fields(lev, time, old_face_b);
   copy_level_state_to_old(lev);
}

void AmrSolverBase::ClearLevel(int lev)
{
   mark_regrid_level_modified(lev);
   fluid_state_.clear_level(lev);
   spacetime_state_.clear_level(lev);
   spacetime_coupling_.clear_level(lev);
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
   for (auto& component : ct_stage_edge_e_[lev].e) {
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

void AmrSolverBase::begin_regrid_change_tracking()
{
   tracking_regrid_changes_ = true;
   regrid_first_modified_level_ = std::numeric_limits<int>::max();
   regrid_last_modified_level_ = -1;
   ++regrid_attempt_count_;
}

void AmrSolverBase::mark_regrid_level_modified(int lev)
{
   if (!tracking_regrid_changes_) {
      return;
   }
   regrid_first_modified_level_ = std::min(regrid_first_modified_level_, lev);
   regrid_last_modified_level_ = std::max(regrid_last_modified_level_, lev);
}

bool AmrSolverBase::finish_regrid_change_tracking()
{
   tracking_regrid_changes_ = false;
   const bool modified = regrid_last_modified_level_ >= regrid_first_modified_level_;
   if (modified) {
      regrid_modified_level_count_ +=
         static_cast<amrex::Long>(regrid_last_modified_level_ - regrid_first_modified_level_ + 1);
   } else {
      ++regrid_noop_count_;
   }
   return modified;
}

} // namespace pyfluid_amrex
