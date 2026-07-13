#pragma once

#include <AMReX_AmrCore.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_GpuLaunch.H>
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
#include <sstream>
#include <vector>

#include "pyfluid_amrex/boundary.hpp"
#include "pyfluid_amrex/build_options.hpp"
#include "pyfluid_amrex/checkpoint.hpp"
#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/ct_edge_flux_register.hpp"
#include "pyfluid_amrex/amr_indicator.hpp"
#include "pyfluid_amrex/diagnostics.hpp"
#include "pyfluid_amrex/diffusion.hpp"
#include "pyfluid_amrex/implicit_resistivity.hpp"
#include "pyfluid_amrex/solid_geometry.hpp"
#include "pyfluid_amrex/gravity.hpp"
#include "pyfluid_amrex/initial_data.hpp"
#include "pyfluid_amrex/mapped_patch_fill.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/multipatch.hpp"
#include "pyfluid_amrex/output.hpp"
#include "pyfluid_amrex/patch_grid.hpp"
#include "pyfluid_amrex/patch_layout_builder.hpp"
#include "pyfluid_amrex/profiling.hpp"
#include "pyfluid_amrex/rhs.hpp"
#include "pyfluid_amrex/solid_conservation.hpp"
#include "pyfluid_amrex/solid_fields.hpp"
#include "pyfluid_amrex/sources.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/state_interpolation.hpp"

namespace pyfluid_amrex {

class AmrSolver : public amrex::AmrCore {
public:
   explicit AmrSolver(RunConfig cfg)
      : cfg_(std::move(cfg))
   {
      validate_config(cfg_);
      validate_amr_refinement_ratios();
      physics_system_ = physics_system_kind(cfg_);
      state_component_count_ = state_component_count(cfg_);
      mesh_layout_ = mesh_runtime_layout(cfg_);
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
      moving_wall_work_enabled_ = cfg_.solid_motion == "translate" || cfg_.solid_motion == "pulse";
      boundary_data_ = make_boundary_data(cfg_);
      state_fillpatch_bcs_ = make_state_fillpatch_bcs();
      face_fillpatch_bcs_ = make_face_fillpatch_bcs();
      state_fillpatch_func_ = StateFillPatchFunc(StateFillPatchBoundary{boundary_data_});
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         face_fillpatch_func_[dir] = FaceFillPatchFunc(FaceBFillPatchBoundary{boundary_data_, dir});
      }
      post_fillpatch_state_boundary_needed_ = compute_post_fillpatch_state_boundary_needed();
      const int nlev = maxLevel() + 1;
      state_.resize(nlev);
      old_state_.resize(nlev);
      source_state_.resize(nlev);
      rk4_coarse_stage_state_.resize(nlev);
      next_.resize(nlev);
      scratch_.resize(nlev);
      rhs_.resize(nlev);
      primitive_.resize(nlev);
      fluxes_.resize(nlev);
#if (AMREX_SPACEDIM >= 2)
      flux_owner_masks_.resize(nlev);
#endif
      flux_registers_.resize(nlev);
      gravity_energy_source_register_.resize(nlev);
      coarse_reflux_volume_.resize(nlev);
      frequency_.resize(nlev);
      pressure_.resize(nlev);
      solid_mask_.resize(nlev);
      solid_volume_fraction_.resize(nlev);
      for (auto& aperture : solid_aperture_) {
         aperture.resize(nlev);
      }
      primitive_needed_mask_.resize(nlev);
      solid_valid_kind_.resize(nlev);
      primitive_needed_kind_.resize(nlev);
      solid_flux_kind_.resize(nlev);
      solid_active_.resize(nlev, false);
      solid_mask_time_.resize(nlev, std::numeric_limits<amrex::Real>::quiet_NaN());
      gravity_phi_.resize(nlev);
      gravity_gradient_.resize(nlev);
      gravity_start_gradient_.resize(nlev);
      gravity_stage_energy_release_.resize(nlev);
      for (auto& stage_flux : gravity_stage_mass_flux_) {
         stage_flux.resize(nlev);
      }
      gravity_fixed_cache_valid_.resize(nlev, false);
      face_b_.resize(nlev);
      edge_e_.resize(nlev);
      edge_current_.resize(nlev);
#if (AMREX_SPACEDIM >= 2)
      edge_flux_registers_.resize(nlev);
#endif
      for (auto& stage : rk4_rhs_stage_) {
         stage.resize(nlev);
      }
      level_steps_.resize(nlev, 0);
      level_times_.resize(nlev, 0.0);
      level_old_times_.resize(nlev, 0.0);
      level_dts_.resize(nlev, 0.0);
      latest_auto_tagged_cells_.resize(nlev, 0);
      for (auto& dir_counts : latest_auto_direction_tagged_cells_) {
         dir_counts.resize(nlev, 0);
      }
      for (auto& dir_strength : latest_auto_direction_strength_) {
         dir_strength.resize(nlev, 0.0);
      }
      rk4_stage_cache_stage_.resize(nlev, -1);
      rk4_stage_cache_fine_start_.resize(nlev, 0.0);
      rk4_stage_cache_time_.resize(nlev, 0.0);
      rk4_stage_cache_dt_.resize(nlev, 0.0);
   }

   void evolve()
   {
      dispatch_evolve();
   }

private:
   using StateFillPatchFunc = amrex::GpuBndryFuncFab<StateFillPatchBoundary>;
   using StateFillPatchPhysBC = amrex::PhysBCFunct<StateFillPatchFunc>;
   using FaceFillPatchFunc = amrex::GpuBndryFuncFab<FaceBFillPatchBoundary>;
   using FaceFillPatchPhysBC = amrex::PhysBCFunct<FaceFillPatchFunc>;

   using LevelData = std::vector<std::unique_ptr<amrex::MultiFab>>;

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

   PhysicsSystemKind physics_system() const
   {
      return physics_system_;
   }

   bool mhd_enabled() const
   {
      return physics_system_has_magnetic_field(physics_system_);
   }

   int num_state_components() const
   {
      return state_component_count_;
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

   // Private AmrSolver interface implemented out-of-line in src/*.cpp.
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
   bool auto_refinement_resolved_at_level(int lev) const;
   AmrLevelDiagnostics current_amr_level_diagnostics() const;
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
   void apply_all_face_b_boundaries();
   void sync_all_cell_b_from_faces(LevelData& data, bool preserve_pressure);

   void define_flux_registers(int first_fine_level = 1);
   void reset_flux_registers();
   void reset_edge_flux_registers();
   void reset_flux_register(int fine_lev);

   amrex::iMultiFab valid_cell_mask(int lev) const;
   HydroConservedTotals hydro_valid_conserved_totals() const;

   void write_current_state_plotfile(int step, amrex::Real time) const;
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
   bool should_refresh_initial_mhd_aux_fields() const;
   void refresh_mhd_aux_fields_for_output(int lev);
   void refresh_all_mhd_aux_fields_for_output();

   static DiffusionStats combine_diffusion_stats(DiffusionStats a, DiffusionStats b)
   {
      a.residual = amrex::max(a.residual, b.residual);
      a.iterations += b.iterations;
      return a;
   }

   bool profile_enabled() const
   {
      return profile_.enabled();
   }

   bool flux_profile_detail_enabled() const
   {
      return profile_.flux_detail_enabled();
   }

   amrex::Real profile_begin() const
   {
      return profile_.begin();
   }

   void profile_end(ProfileStage stage, amrex::Real start)
   {
      profile_.end(stage, start);
   }

   void profile_detail_end(ProfileDetailStage stage, amrex::Real start)
   {
      profile_.detail_end(stage, start);
   }

   void write_profile_file(amrex::Real elapsed_wall)
   {
      profile_.write(std::filesystem::path(cfg_.output_dir), elapsed_wall);
   }

   void write_profile_detail_file(amrex::Real elapsed_wall)
   {
      profile_.write_detail(std::filesystem::path(cfg_.output_dir), elapsed_wall);
   }

   // Private AmrSolver inline/template implementation sections.
#include "pyfluid_amrex/solver_sections/evolve_core.inc.hpp"
#include "pyfluid_amrex/solver_sections/amr_tagging.inc.hpp"
#include "pyfluid_amrex/solver_sections/fillpatch.inc.hpp"
#include "pyfluid_amrex/solver_sections/ct_evolution.inc.hpp"
#include "pyfluid_amrex/solver_sections/gravity_energy_sync.inc.hpp"
#include "pyfluid_amrex/solver_sections/flux_registers.inc.hpp"
#include "pyfluid_amrex/solver_sections/solid_geometry.inc.hpp"
#include "pyfluid_amrex/solver_sections/gravity_sources.inc.hpp"
#include "pyfluid_amrex/solver_sections/solid_conservation.inc.hpp"
#include "pyfluid_amrex/solver_sections/state_repair.inc.hpp"
#include "pyfluid_amrex/solver_sections/debug_contracts.inc.hpp"
#include "pyfluid_amrex/solver_sections/external_conservation.inc.hpp"
#include "pyfluid_amrex/solver_sections/diagnostics.inc.hpp"
#include "pyfluid_amrex/solver_sections/timestep_control.inc.hpp"
#include "pyfluid_amrex/solver_sections/rhs_assembly.inc.hpp"
#include "pyfluid_amrex/solver_sections/rk_stage_algebra.inc.hpp"
#include "pyfluid_amrex/solver_sections/time_integrators.inc.hpp"
#include "pyfluid_amrex/solver_sections/barotropic_energy.inc.hpp"
#include "pyfluid_amrex/solver_sections/subcycling.inc.hpp"
#include "pyfluid_amrex/solver_sections/diffusion.inc.hpp"

   RunConfig cfg_;
   PhysicsSystemKind physics_system_ = PhysicsSystemKind::Hydro;
   int state_component_count_ = NVAR_HYDRO;
   MeshRuntimeLayout mesh_layout_{};
   int recon_method_ = ReconPWC;
   int recon_fine_method_ = ReconPWC;
   int riemann_method_ = RiemannLLF;
   int timestep_method_ = TimeEuler;
   int diffusion_method_ = DiffusionNone;
   int eos_method_ = EOSIdealGas;
   ProfileCounters profile_;
   bool moving_wall_work_enabled_ = false;
   bool suppress_athena_gravity_energy_source_ = false;
   BoundaryData boundary_data_{};
   mutable MeshBasisStateInterpolater state_interpolater_{};
   amrex::Vector<amrex::BCRec> state_fillpatch_bcs_;
   amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> face_fillpatch_bcs_;
   StateFillPatchFunc state_fillpatch_func_;
   amrex::Array<FaceFillPatchFunc, AMREX_SPACEDIM> face_fillpatch_func_;
   bool post_fillpatch_state_boundary_needed_ = false;
   amrex::Real checkpoint_reference_mass_ = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real checkpoint_reference_energy_ = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real wall_work_reference_energy_ = 0.0;
   amrex::Real cumulative_wall_work_ = 0.0;
   amrex::Real latest_wall_power_ = 0.0;
   ConservationLedger conservation_ledger_;
   std::vector<std::unique_ptr<amrex::MultiFab>> state_;
   std::vector<std::unique_ptr<amrex::MultiFab>> old_state_;
   std::vector<std::unique_ptr<amrex::MultiFab>> source_state_;
   std::vector<std::unique_ptr<amrex::MultiFab>> rk4_coarse_stage_state_;
   std::vector<std::unique_ptr<amrex::MultiFab>> next_;
   std::vector<std::unique_ptr<amrex::MultiFab>> scratch_;
   std::vector<std::unique_ptr<amrex::MultiFab>> rhs_;
   std::vector<std::unique_ptr<amrex::MultiFab>> primitive_;
   std::array<std::vector<std::unique_ptr<amrex::MultiFab>>, 4> rk4_rhs_stage_;
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
   amrex::Vector<amrex::Real> level_times_;
   amrex::Vector<amrex::Real> level_old_times_;
   amrex::Vector<amrex::Real> level_dts_;
   std::vector<amrex::Long> latest_auto_tagged_cells_;
   std::array<std::vector<amrex::Long>, AMREX_SPACEDIM> latest_auto_direction_tagged_cells_;
   std::array<std::vector<amrex::Real>, AMREX_SPACEDIM> latest_auto_direction_strength_;
   amrex::Vector<int> rk4_stage_cache_stage_;
   amrex::Vector<amrex::Real> rk4_stage_cache_fine_start_;
   amrex::Vector<amrex::Real> rk4_stage_cache_time_;
   amrex::Vector<amrex::Real> rk4_stage_cache_dt_;
};

inline void run_amr_solver(RunConfig const& cfg)
{
   AmrSolver solver(cfg);
   solver.evolve();
}

} // namespace pyfluid_amrex
