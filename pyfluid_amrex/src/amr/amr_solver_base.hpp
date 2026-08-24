#pragma once

#include <AMReX_AmrCore.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_Interpolater.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Print.H>
#include <AMReX_TagBox.H>
#include <AMReX_VisMF.H>
#include <AMReX_YAFluxRegister.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <vector>

#include "pyfluid_amrex/boundary.hpp"
#include "pyfluid_amrex/build_options.hpp"
#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/ct_fields.hpp"
#include "pyfluid_amrex/evolution_sector_storage.hpp"
#include "pyfluid_amrex/ct_edge_flux_register.hpp"
#include "pyfluid_amrex/diagnostics_types.hpp"
#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/solid_geometry.hpp"
#include "pyfluid_amrex/gravity.hpp"
#include "pyfluid_amrex/initial_data/atmosphere.hpp"
#include "pyfluid_amrex/multipatch.hpp"
#include "pyfluid_amrex/patch_grid.hpp"
#include "pyfluid_amrex/patch_layout_builder.hpp"
#include "pyfluid_amrex/profiling.hpp"
#include "pyfluid_amrex/solid_fields.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/spacetime_fields.hpp"
#include "pyfluid_amrex/spacetime_coupling_storage.hpp"
#include "pyfluid_amrex/temporal_cadence.hpp"
#include "pyfluid_amrex/z4c_runtime.hpp"

namespace pyfluid_amrex {

class ApparentHorizonTracker;
class HierarchyInterpolationPlan;
class MeshBasisStateInterpolater;
class ParametricSurface;
class RelativisticInitialDataProvider;
struct MultipoleDiagnostics;
struct MatterModeDiagnostics;
struct EquilibriumTorusUpdateSummary;
struct PhysicalInitialDataMetadata;
struct RadiationDiagnostics;
struct RotatingStarUpdateSummary;

class AmrSolverBase : public amrex::AmrCore {
public:
   explicit AmrSolverBase(RunConfig cfg);
   ~AmrSolverBase() override;

protected:
   using StateFillPatchFunc = amrex::GpuBndryFuncFab<StateFillPatchBoundary>;
   using StateFillPatchPhysBC = amrex::PhysBCFunct<StateFillPatchFunc>;
   using FaceFillPatchFunc = amrex::GpuBndryFuncFab<FaceBFillPatchBoundary>;
   using FaceFillPatchPhysBC = amrex::PhysBCFunct<FaceFillPatchFunc>;
   using SpacetimeFillPatchFunc = amrex::GpuBndryFuncFab<Z4cFillPatchBoundary>;
   using SpacetimeFillPatchPhysBC = amrex::PhysBCFunct<SpacetimeFillPatchFunc>;

   using LevelData = EvolutionSectorStorage::LevelData;

   struct HydroConservedTotals {
      std::array<amrex::Real, NVAR_HYDRO> value{};
      amrex::Real volume = 0.0;
   };

   struct DensityTotal {
      amrex::Real mass = 0.0;
      amrex::Real volume = 0.0;
   };

   struct MassEnergy {
      amrex::Real mass = 0.0;
      amrex::Real energy = 0.0;
   };

   struct ConservationLedger {
      MassEnergy reference;
      MassEnergy boundary_change;
      MassEnergy source_change;
   };

   struct DirectionalTagStatistics {
      amrex::Long tagged_cells = 0;
      std::array<amrex::Long, AMREX_SPACEDIM> direction_tags{};
      std::array<amrex::Real, AMREX_SPACEDIM> direction_strength{};
   };

   PhysicalSystem physics_system() const
   {
      return physics_system_;
   }

   bool mhd_enabled() const
   {
      return physical_system_has_magnetic_field(physics_system_);
   }

   int num_state_components() const
   {
      return fluid_state_.num_components();
   }

   int recon_method_for_level(int lev) const
   {
      return lev >= cfg_.recon_fine_start_level ? recon_fine_method_ : recon_method_;
   }

   int mesh_coordinate_kind() const
   {
      return mesh_layout_.coordinate;
   }

   MeshAxes active_mesh_axes() const
   {
      return mesh_layout_.axes;
   }

   int mesh_patch_system_kind() const
   {
      return mesh_layout_.patch_system;
   }

   bool production_multipatch_enabled() const
   {
      return mesh_patch_system_is_multipatch(mesh_patch_system_kind());
   }

   bool mixed_patch_system_enabled() const
   {
      const int patch_system = mesh_patch_system_kind();
      return patch_system == MeshPatchMixedThornburg04
         || patch_system == MeshPatchMixedThornburg13
         || patch_system == MeshPatchMixedCylindrical5
         || patch_system == MeshPatchMixedCylindrical9;
   }

   bool patch_union_domain_enabled() const
   {
      return cartesian_patch_box_union_domain_enabled(cfg_);
   }
   amrex::IntVect patch_union_refinement_to_level(int lev) const;
   PatchDomainBoxes patch_union_domain_boxes(int lev) const;
   PatchGridSystem production_patch_grid_system_for_level(int lev) const;
   PatchGridSystem production_patch_grid_system() const;
   PatchLayout production_patch_layout(amrex::Box const& domain) const;
   PatchLayout production_patch_layout_for_level(int lev) const;

   bool mesh_is_curvilinear() const
   {
      return mesh_layout_.curvilinear;
   }

   // Shared AMR infrastructure implemented out-of-line in src/*.cpp.
   void validate_amr_refinement_ratios() const;
   void validate_state_interpolation_nesting() const;
   void validate_active_hierarchy_for_evolution() const;
   DirectionalTagStatistics measure_directional_tag_statistics(
      int lev,
      amrex::Real time,
      bool refresh_boundaries
   );
   int temporal_ref_ratio(int coarse_lev) const;
   int level_time_factor(int lev) const;
   bool local_cfl_subcycling() const;
   int synced_level_step(int lev, int base_step) const;
   bool needs_rk4_subcycling_stage_data() const;
   void set_level_time_state(int lev, amrex::Real time, amrex::Real dt);
   void copy_level_state_to_old(int lev);
   bool uses_subcycled_self_gravity_source_state() const;
   void copy_source_state_level(int lev, amrex::MultiFab const& src);
   void interpolate_source_state_level(int lev, amrex::Real alpha);
   void fill_source_state_level_from_level_time(int lev, amrex::Real time);
   void fill_source_state_for_rhs(LevelData& input, int lev_begin, int lev_end, amrex::Real rhs_time);
   static bool nearly_equal_time(amrex::Real a, amrex::Real b);
   void invalidate_rk4_stage_cache(int lev);
   void invalidate_rk4_stage_caches();

   // Evolution-bundle hooks used only by hierarchy changes and diagnostics.
   // Keeping these operations behind this narrow boundary prevents AMR
   // infrastructure translation units from including flux/reconstruction
   // templates. None of these calls are in a per-cell evolution loop.
   virtual void repair_after_hierarchy_change(HydroConservedTotals const& target) = 0;
   virtual DensityTotal valid_density_total(LevelData const& data) const = 0;
   virtual void prepare_gravity_fields(LevelData const& input, bool fill_gradient) = 0;
   virtual bool uses_athena_ssprk3_gravity_energy_correction() const = 0;
   virtual bool uses_athena_gravity_energy_correction() const = 0;

   void allocate_level(
      int lev,
      amrex::BoxArray const& ba,
      amrex::DistributionMapping const& dm,
      amrex::Real time = 0.0
   );
   amrex::Long active_cell_count_in_box(int lev, amrex::Box const& box, amrex::Real time) const;
   amrex::DistributionMapping obstacle_weighted_distribution_mapping(
      int lev,
      amrex::BoxArray const& ba,
      amrex::DistributionMapping const& base_dm,
      amrex::Real time
   ) const;
   amrex::BoxArray owned_level_box_array(int lev, amrex::BoxArray const& ba) const;
   amrex::DistributionMapping owned_level_distribution_mapping(
      int lev,
      amrex::BoxArray const& original_ba,
      amrex::BoxArray const& owned_ba,
      amrex::DistributionMapping const& base_dm,
      amrex::Real time
   );
   void MakeNewLevelFromScratch(
      int lev,
      amrex::Real time,
      amrex::BoxArray const& ba,
      amrex::DistributionMapping const& dm
   ) override;
   void MakeNewLevelFromCoarse(
      int lev,
      amrex::Real time,
      amrex::BoxArray const& ba,
      amrex::DistributionMapping const& dm
   ) override;
   void RemakeLevel(
      int lev,
      amrex::Real time,
      amrex::BoxArray const& ba,
      amrex::DistributionMapping const& dm
   ) override;
   void ClearLevel(int lev) override;
   void begin_regrid_change_tracking();
   void mark_regrid_level_modified(int lev);
   bool finish_regrid_change_tracking();
   void regrid(int lbase, amrex::Real time, bool initial = false) override;
   void PostProcessBaseGrids(amrex::BoxArray& box_array) const override;
   amrex::BoxArray clip_to_patch_union_domain(int lev, amrex::BoxArray const& box_array) const;
   amrex::BoxArray clip_to_mixed_patch_domain(int lev, amrex::BoxArray const& box_array) const;
   amrex::BoxArray clip_to_active_multipatch_domain(int lev, amrex::BoxArray const& box_array) const;
   void clip_multipatch_initial_hierarchy(amrex::Real time);

   amrex::Long cell_count() const;
   amrex::Long active_cell_count_level(int lev) const;
   amrex::Long active_cell_count() const;
   std::vector<amrex::Long> level_cell_counts() const;
   std::array<amrex::Real, AMREX_SPACEDIM> level_cell_widths(int lev) const;
   amrex::Real level_cell_width(int lev) const;
   amrex::Real active_finest_cell_width() const;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> min_feature_lengths() const;
   amrex::Real min_feature_length() const;
   void fill_level_primitive_cache(int lev, amrex::Real time);
   bool auto_refinement_resolved_at_level(int lev) const;
   AmrLevelDiagnostics current_amr_level_diagnostics() const;
   bool symmetry_axis_active(int dir) const;
   amrex::Real symmetry_volume_factor() const;
   amrex::Real symmetry_momentum_factor(int dir, amrex::Real volume_factor) const;
   amrex::Real symmetry_angular_momentum_factor(
      int dir,
      amrex::Real volume_factor
   ) const;
   CflLimiterLocation current_cfl_limiter_location() const;
   ScalarDiagnostics compute_current_scalars(
      int step,
      amrex::Real time,
      amrex::Real dt,
      amrex::Real mass0,
      amrex::Real energy0,
      amrex::Real gravity_total_energy0,
      amrex::Real passive_mass0,
      amrex::Real max_rho0,
      amrex::Real mcells_per_s,
      amrex::Real layout_mcells_per_s,
      amrex::Real sim_time_per_s,
      DiffusionStats diffusion,
      int eos_method,
      ScalarDiagnosticSelection const& selection
   );
   void validate_timestep(amrex::Real dt, int step, amrex::Real time) const;
   void validate_scalar_health(ScalarDiagnostics const& d) const;
   void enforce_level_cell_budget(int lev, amrex::Long cells, char const* context) const;
   amrex::Long projected_cell_count(int replacement_lev, amrex::Long replacement_cells) const;
   void enforce_total_cell_budget(amrex::Long cells, char const* context) const;
   void enforce_projected_cell_budget(int lev, amrex::Long cells, char const* context) const;
   void enforce_cell_budget(char const* context) const;
   void append_current_amr_level_row(std::ofstream& out, int step, amrex::Real time) const;

   void average_down_state();
   void average_down_state_level(int fine_lev);
   void average_down_state_level_weighted_hydro(int fine_lev);
   void project_metric_conservative_prolongation(
      int fine_lev,
      amrex::MultiFab const& coarse_state,
      amrex::MultiFab& fine_state
   ) const;
   void average_down_face_b();
   void average_down_face_b_level(int fine_lev);
   void average_down_face_b_level_metric(int fine_lev);
   void average_down_spacetime();
   void average_down_spacetime_level(int fine_lev);
   bool gravity_energy_source_register_needed(int fine_lev) const;
   amrex::iMultiFab coarse_fine_cell_mask(int lev) const;
   void fill_athena_ssprk3_gravity_face_energy_source(
      int lev,
      int dir,
      amrex::MultiFab& out
   ) const;
   void accumulate_athena_ssprk3_gravity_energy_source_register_dir(
      int fine_lev,
      int dir,
      amrex::MultiFab const* fine_average_source,
      amrex::Real dt,
      bool include_coarse,
      bool include_fine
   );
   void add_athena_ssprk3_gravity_energy_source_register(
      amrex::Real dt,
      int lev_begin,
      int lev_end
   );
   void apply_gravity_energy_source_register(
      int fine_lev,
      amrex::MultiFab& coarse_state
   );
   void apply_gravity_energy_source_register_scaled(
      int fine_lev,
      amrex::MultiFab& coarse_state,
      amrex::Real scale
   );
   void initialize_spacetime_level(int lev, amrex::Real time);
   bool materialize_constraint_solved_initial_data(bool seed_from_loaded_id = false);
   void finalize_constraint_solved_initial_data_state();
   void apply_configured_initial_data_perturbation();
   void materialize_configured_initial_data();
   void summarize_configured_equilibrium_torus();
   bool load_physical_initial_data_artifact();
   void write_physical_initial_data_artifact();
   void write_physical_initial_data_artifact(
      std::string const& path,
      amrex::Real angular_velocity,
      bool refresh_primitive,
      RotatingStarUpdateSummary const* rotating_star = nullptr,
      EquilibriumTorusUpdateSummary const* equilibrium_torus = nullptr
   );
   void initialize_physical_initial_data_sequence_output();
   bool physical_initial_data_sequence_output_enabled() const;
   std::string physical_initial_data_sequence_output_dir() const;
   void write_physical_initial_data_sequence_model(
      int continuation_step,
      int continuation_steps,
      amrex::Real angular_velocity,
      RotatingStarUpdateSummary const& rotating_star,
      EquilibriumTorusUpdateSummary const& equilibrium_torus
   );
   void project_materialized_initial_face_b();
   void fill_new_spacetime_level_from_coarse(int lev, amrex::Real time);
   void fill_remade_spacetime_level(
      int lev,
      amrex::Real time,
      amrex::MultiFab& old_state
   );
   void fill_spacetime_level_boundaries(
      LevelData& data,
      int lev,
      amrex::Real time,
      bool synchronous_levels = false,
      amrex::Real fine_step_start = -1.0,
      int rk_stage = 0
   );
   void fill_all_spacetime_boundaries(LevelData& data, amrex::Real time);
   void project_spacetime_state(amrex::MultiFab& state);
   void project_spacetime_state_range(
      LevelData& state,
      int lev_begin,
      int lev_end
   );
   void prepare_spacetime_coupling_range(
      LevelData& state,
      int lev_begin,
      int lev_end,
      amrex::Real stage_time,
      amrex::Real fine_step_start,
      int rk_stage
   );
   void apply_stored_spacetime_fluid_boundaries_range(
      LevelData& state,
      int lev_begin,
      int lev_end
   );
   void refresh_stored_spacetime_fluid_boundary(int lev, amrex::Real time);
   bool uses_analytic_fixed_spacetime_kernels() const noexcept;
   void fill_spacetime_matter_range(int lev_begin, int lev_end);
   void add_spacetime_ko_dissipation(amrex::MultiFab const& state, int lev);
   template <Z4cLapseGauge LapseGauge, Z4cShiftGauge ShiftGauge>
   void compute_spacetime_rhs_level(
      LevelData& input,
      int lev,
      amrex::Real stage_time
   );
   template <Z4cLapseGauge LapseGauge, Z4cShiftGauge ShiftGauge>
   void compute_spacetime_rhs_gauge_range(
      LevelData& input,
      int lev_begin,
      int lev_end,
      amrex::Real stage_time
   );
   void compute_spacetime_rhs_range(
      LevelData& input,
      int lev_begin,
      int lev_end,
      amrex::Real stage_time
   );
   template <Z4cLapseGauge LapseGauge>
   void dispatch_spacetime_rhs_shift_gauge(
      LevelData& input,
      int lev_begin,
      int lev_end,
      amrex::Real stage_time
   );
   SpacetimeConstraintDiagnostics compute_spacetime_constraint_diagnostics(
      amrex::Real time
   );
   void update_apparent_horizon(int step, amrex::Real time);
   amrex::iMultiFab const* apparent_horizon_mask_ptr(int lev);
   amrex::iMultiFab const* excision_mask_ptr(int lev);
   SpacetimeGlobalDiagnostics compute_spacetime_global_diagnostics(
      amrex::Real time,
      bool surface_requested,
      bool volume_requested
   );
   ParametricSurface diagnostic_extraction_surface() const;
   HierarchyInterpolationPlan diagnostic_surface_interpolation_plan(
      ParametricSurface const& surface,
      amrex::IndexType centering = amrex::IndexType::TheNodeType()
   ) const;
   RadiationDiagnostics compute_spacetime_radiation_diagnostics(amrex::Real time);
   MultipoleDiagnostics compute_azimuthal_surface_modes(amrex::Real time);
   MatterModeDiagnostics compute_matter_volume_modes(amrex::Real time);
   void apply_all_face_b_boundaries();
   void sync_all_cell_b_from_faces(LevelData& data, bool preserve_pressure);

   void define_flux_registers(int first_fine_level = 1);
   void reset_flux_registers();
   void reset_flux_register(int fine_lev);
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> flux_ptrs(int lev);
#if (AMREX_SPACEDIM >= 2)
   std::array<amrex::iMultiFab const*, AMREX_SPACEDIM> flux_owner_mask_ptrs(
      int lev
   ) const;
#endif
   std::array<amrex::FArrayBox const*, AMREX_SPACEDIM> flux_fabs(
      int lev,
      amrex::MFIter const& mfi
   ) const;
   amrex::Real cell_volume(int lev) const;
   amrex::Real face_area(int lev, int dir) const;
   bool cutcell_flux_register_needed(int fine_lev) const;
   bool curvilinear_flux_register_needed(int fine_lev) const;
   bool patch_union_flux_register_needed(int fine_lev) const;
   bool mapped_patch_flux_register_needed(int fine_lev) const;
   bool weighted_flux_register_needed(int fine_lev) const;
   void fill_coarse_reflux_volume(int fine_lev);
   bool prepare_weighted_flux_register(int fine_lev);
   void fill_area_weighted_flux_fabs(
      int lev,
      amrex::MFIter const& mfi,
      std::array<amrex::FArrayBox, AMREX_SPACEDIM>& weighted
   ) const;
   std::array<amrex::FArrayBox const*, AMREX_SPACEDIM> area_weighted_flux_fabs(
      int lev,
      amrex::MFIter const& mfi,
      std::array<amrex::FArrayBox, AMREX_SPACEDIM>& weighted
   ) const;
   void add_fluxes_to_registers(int lev, amrex::Real weight);
#if (AMREX_SPACEDIM == 3)
   amrex::Array<amrex::FArrayBox const*, 3> edge_electric_fabs(
      int lev,
      amrex::MFIter const& mfi
   ) const;
#endif
   void fill_edge_line_integral_fab(
      int lev,
      int component,
      amrex::FArrayBox const& edge_electric,
      amrex::FArrayBox& edge_line_integral
   ) const;
#if (AMREX_SPACEDIM == 3)
   amrex::Array<amrex::FArrayBox const*, 3> edge_line_integral_fabs(
      int lev,
      amrex::MFIter const& mfi,
      amrex::Array<amrex::FArrayBox, 3>& edge_line
   ) const;
#endif
   void add_edge_electric_to_registers(int lev, amrex::Real dt);
   void reflux_edge_flux_registers_to_face_b();
   void reflux_edge_flux_register_to_face_b(int fine_lev);
   void reflux_next_levels();
   void reflux_level_to_destination(
      int fine_lev,
      amrex::MultiFab& destination
   );
   void reflux_level_to_next(int fine_lev);
   void reflux_level_to_state(int fine_lev);

   void clear_solid_mask_kind_tables(int lev);
   void build_solid_mask_kind_tables(int lev);
   void average_down_solid_volume_fraction_from_fine(int fine_lev);
   void average_down_solid_aperture_from_fine(int fine_lev, int direction);
   void enforce_solid_mask_nesting_from_fine();
   void refresh_solid_activity(int lev);
   amrex::Real solid_geometry_max_speed() const;
   amrex::Real solid_geometry_motion_dt(int lev) const;
   bool solid_mask_refresh_needed(int lev, amrex::Real time) const;
   void allocate_solid_apertures_level(int lev);
   void fill_solid_apertures_level(int lev, amrex::Real time);
   void refresh_solid_discrete_geometry_level(int lev, amrex::Real time);
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> solid_aperture_ptrs(
      int lev
   ) const;
   void refresh_solid_geometry_level(
      int lev,
      amrex::Real time,
      bool remap_moving_solid
   );
   void refresh_solid_geometry_all_levels(
      amrex::Real time,
      bool remap_moving_solid
   );
   bool refresh_solid_geometry_synced(
      amrex::Real time,
      bool remap_moving_solid,
      bool preserve_hydro_totals
   );
   void refresh_pulse_solid_geometry_all_levels(
      amrex::Real time,
      bool remap_moving_solid
   );
   bool solid_active(int lev) const;
   amrex::iMultiFab const* active_solid_mask_ptr(int lev) const;

   amrex::iMultiFab valid_cell_mask(int lev) const;
   HydroConservedTotals hydro_valid_conserved_totals() const;
   bool external_conservation_budget_supported() const;
   MassEnergy local_rhs_mass_energy_integral(int lev) const;
   void accumulate_physical_boundary_flux(
      int lev,
      std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
      amrex::Real weight
   );
   void accumulate_source_change(
      MassEnergy const& before,
      MassEnergy const& after,
      amrex::Real weight
   );

   void write_current_state_plotfile(int step, amrex::Real time) const;
   void write_debug_stage_state(
      LevelData const& state,
      std::string const& stage_name,
      amrex::Real time,
      int lev_begin,
      int lev_end
   ) const;
   void write_current_aux_gfs(int step, amrex::Real time);
   std::string checkpoint_dir() const;
   void load_checkpoint(std::string const& checkpoint_name, amrex::Real& time, amrex::Real& dt);
   void apply_boundary(amrex::MultiFab& state, amrex::Geometry const& geom, amrex::Real time) const;
   void initialize_ct_fields(int lev, amrex::Real time);
   FaceMagneticField const* face_b_ptr(int lev) const;
   std::vector<FaceMagneticField> const* checkpoint_face_b() const;

   EdgeElectricField const* edge_current_density_ptr(int lev) const;
   EdgeElectricField const* edge_electric_field_ptr(int lev) const;
   amrex::MultiFab const* gravity_phi_ptr(int lev) const;
   bool mhd_edge_aux_requested() const;
   bool cached_edge_current_density_requested() const;
   bool cached_edge_electric_field_requested() const;
   bool needs_edge_current_density() const;
   bool needs_edge_electric_field() const;
   bool should_refresh_mhd_aux_fields(int step, bool write_aux_gfs) const;
   bool should_refresh_initial_mhd_aux_fields(bool write_aux_gfs) const;
   void refresh_mhd_aux_fields_for_output(int lev);
   void refresh_all_mhd_aux_fields_for_output();

   // Shared AMR infrastructure used by every physical-system evolution bundle.
   void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;
#include "amr/solver_sections/fillpatch.inc.hpp"
#include "amr/solver_sections/debug_contracts.inc.hpp"

   RunConfig cfg_;
   std::unique_ptr<RelativisticInitialDataProvider> initial_data_provider_;
   std::unique_ptr<ApparentHorizonTracker> apparent_horizon_tracker_;
   amrex::Vector<
      std::array<amrex::Vector<amrex::Real>, AMREX_SPACEDIM>
   > initial_data_coordinate_faces_;
   EOSThermodynamicState recovery_atmosphere_{};
   PhysicalSystem physics_system_ = PhysicalSystem::HD;
   MeshRuntimeLayout mesh_layout_{};
   int recon_method_ = ReconPWC;
   int recon_fine_method_ = ReconPWC;
   int riemann_method_ = RiemannLLF;
   int timestep_method_ = TimeEuler;
   int diffusion_method_ = DiffusionNone;
   int eos_method_ = EOSIdealGas;
   ProfileCounters profile_;
   bool analytic_fixed_spacetime_kernels_ = false;
   bool moving_wall_work_enabled_ = false;
   bool suppress_athena_gravity_energy_source_ = false;
   BoundaryData boundary_data_{};
   mutable std::unique_ptr<MeshBasisStateInterpolater> state_interpolater_;
   amrex::Vector<amrex::BCRec> state_fillpatch_bcs_;
   amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> face_fillpatch_bcs_;
   StateFillPatchFunc state_fillpatch_func_;
   amrex::Array<FaceFillPatchFunc, AMREX_SPACEDIM> face_fillpatch_func_;
   amrex::Vector<amrex::BCRec> spacetime_fillpatch_bcs_;
   SpacetimeFillPatchFunc spacetime_fillpatch_func_;
   bool post_fillpatch_state_boundary_needed_ = false;
   amrex::Real checkpoint_reference_mass_ = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real checkpoint_reference_energy_ = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real checkpoint_reference_max_rho_ = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real wall_work_reference_energy_ = 0.0;
   amrex::Real cumulative_wall_work_ = 0.0;
   amrex::Real latest_wall_power_ = 0.0;
   ConservationLedger conservation_ledger_;
   EvolutionSectorStorage fluid_state_;
   EvolutionSectorStorage spacetime_state_;
   SpacetimeCouplingStorage spacetime_coupling_;
   std::vector<std::unique_ptr<amrex::MultiFab>> primitive_;
   std::vector<std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>> fluxes_;
#if (AMREX_SPACEDIM >= 2)
   std::vector<std::array<std::unique_ptr<amrex::iMultiFab>, AMREX_SPACEDIM>> flux_owner_masks_;
#endif
   std::vector<std::unique_ptr<amrex::YAFluxRegister>> flux_registers_;
   std::vector<std::unique_ptr<amrex::MultiFab>> gravity_energy_source_register_;
   std::vector<std::unique_ptr<amrex::MultiFab>> coarse_reflux_volume_;
   std::vector<std::unique_ptr<amrex::MultiFab>> frequency_;
   std::vector<std::unique_ptr<amrex::MultiFab>> pressure_;
   std::vector<std::unique_ptr<amrex::iMultiFab>> solid_mask_;
   std::vector<std::unique_ptr<amrex::MultiFab>> solid_volume_fraction_;
   std::array<std::vector<std::unique_ptr<amrex::MultiFab>>, AMREX_SPACEDIM> solid_aperture_;
   std::vector<std::unique_ptr<amrex::iMultiFab>> primitive_needed_mask_;
   std::vector<amrex::Vector<int>> solid_valid_kind_;
   std::vector<amrex::Vector<int>> primitive_needed_kind_;
   std::vector<amrex::Array<amrex::Vector<int>, AMREX_SPACEDIM>> solid_flux_kind_;
   std::vector<bool> solid_active_;
   amrex::Vector<amrex::Real> solid_mask_time_;
   std::vector<std::unique_ptr<amrex::MultiFab>> gravity_phi_;
   std::vector<GravityGradient> gravity_gradient_;
   std::array<amrex::Real, AMREX_SPACEDIM> gravity_acceleration_offset_{};
   std::array<amrex::Real, AMREX_SPACEDIM> gravity_start_acceleration_offset_{};
   std::vector<GravityGradient> gravity_start_gradient_;
   std::array<std::vector<std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>>, 3> gravity_stage_mass_flux_;
   std::vector<std::unique_ptr<amrex::MultiFab>> gravity_stage_energy_release_;
   std::vector<bool> gravity_fixed_cache_valid_;
   std::vector<FaceMagneticField> face_b_;
   std::vector<EdgeElectricField> edge_e_;
   std::vector<EdgeElectricField> ct_stage_edge_e_;
   std::vector<EdgeElectricField> edge_current_;
#if (AMREX_SPACEDIM >= 2)
   std::vector<std::unique_ptr<CtEdgeFluxRegister>> edge_flux_registers_;
#endif
   amrex::Vector<int> level_steps_;
   amrex::Long aligned_regrid_ticks_ = 0;
   amrex::Long aligned_base_regrid_tick_ = 0;
   bool aligned_regrid_due_ = false;
   bool tracking_regrid_changes_ = false;
   int regrid_first_modified_level_ = std::numeric_limits<int>::max();
   int regrid_last_modified_level_ = -1;
   amrex::Long regrid_attempt_count_ = 0;
   amrex::Long regrid_noop_count_ = 0;
   amrex::Long regrid_modified_level_count_ = 0;
   amrex::Gpu::DeviceScalar<amrex::Long>
      step_relativistic_c2p_fallback_count_{0};
   amrex::Gpu::DeviceScalar<amrex::Long>
      step_relativistic_atmosphere_reset_count_{0};
   amrex::Gpu::DeviceScalar<amrex::Long>
      step_rmhd_pwl_correction_count_{0};
   amrex::Gpu::DeviceScalar<amrex::Long>
      step_rmhd_pwc_correction_count_{0};
   amrex::Gpu::DeviceScalar<amrex::Long>
      step_rmhd_riemann_hll_fallback_count_{0};
   amrex::Long relativistic_c2p_fallback_count_ = 0;
   amrex::Long relativistic_atmosphere_reset_count_ = 0;
   amrex::Long rmhd_pwl_correction_count_ = 0;
   amrex::Long rmhd_pwc_correction_count_ = 0;
   amrex::Long rmhd_riemann_hll_fallback_count_ = 0;
   amrex::Vector<amrex::Real> level_times_;
   amrex::Vector<amrex::Real> level_old_times_;
   amrex::Vector<amrex::Real> level_dts_;
   std::vector<amrex::Long> latest_auto_tagged_cells_;
   std::array<std::vector<amrex::Long>, AMREX_SPACEDIM> latest_auto_direction_tagged_cells_;
   std::array<std::vector<amrex::Real>, AMREX_SPACEDIM> latest_auto_direction_strength_;
   std::unique_ptr<RotatingStarUpdateSummary>
      initial_data_rotating_star_summary_;
   std::unique_ptr<EquilibriumTorusUpdateSummary>
      initial_data_equilibrium_torus_summary_;
   std::unique_ptr<PhysicalInitialDataMetadata> loaded_physical_id_metadata_;
};

} // namespace pyfluid_amrex
