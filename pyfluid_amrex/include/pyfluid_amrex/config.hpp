#pragma once

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <array>
#include <limits>
#include <string>
#include <vector>

#include "pyfluid_amrex/build_options.hpp"
#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/fixed_spacetime_types.hpp"
#include "pyfluid_amrex/mesh_limits.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/physical_system.hpp"

namespace pyfluid_amrex {

struct RunConfig {
   RunConfig();

   int ng = 1;
   int max_step = 32;
   amrex::Real final_time = -1.0;
   amrex::Real cfl = 0.9;
   amrex::Real gamma = 5.0 / 3.0;
   std::string eos;
   std::string eos_preset;
   amrex::Real polytropic_k = 1.0;
   amrex::Real polytropic_gamma = 5.0 / 3.0;
   std::vector<amrex::Real> polytropic_density_breaks;
   std::vector<amrex::Real> polytropic_segment_gammas;
   amrex::Real thermal_gamma = 5.0 / 3.0;
   amrex::Real rho_floor = 1.0e-12;
   amrex::Real pressure_floor = 1.0e-12;

   std::string run_name = "amrex_hydro_sine";
   std::string output_dir = "runs/current/amrex_hydro_sine";
   std::string validation_policy = "";
   int scalar_int = 1;
   int plot_int = -1;
   std::string plotfile_format = "hdf5";
   std::string plotfile_compression = "best";
   std::string checkpoint_compression = "best";
   bool cfl_limiter_diagnostics = false;
   std::string scalar_output_mode = "stream";
   int aux_gf_compute_int = -1;
   int aux_gf_int = -1;
   std::string aux_gfs = "none";
   int checkpoint_int = -1;
   std::string checkpoint_dir = "";
   std::string restart_file = "";
   std::string profile = "on";
   std::string profile_detail = "basic";
   int physical_check_int = 0;
   bool debug_poison = false;
   bool debug_check_finite = false;
   bool debug_check_physical = false;

   std::string mesh_coordinate = "cartesian";
   std::string mesh_axes = "";
   std::string mesh_patch_system = "single";
   std::array<int, 3> mesh_patch_counts{{0, 0, 0}};
   int mesh_patch_overlap_cells = 0;
   int mesh_patch_box_count = 0;
   std::array<int, MaxMeshPatchBoxValues> mesh_patch_box_small{};
   std::array<int, MaxMeshPatchBoxValues> mesh_patch_box_big{};
   std::string mesh_patch_box_ownership = "exact";
   std::string mesh_patch_box_domain = "rectangular";
   amrex::Real mesh_patch_inner_radius = 1.0;
   amrex::Real mesh_patch_medium_radius = 2.5;
   amrex::Real mesh_patch_outer_radius = 4.0;
   amrex::Real mesh_patch_origin1 = 0.0;
   amrex::Real mesh_patch_origin2 = 0.0;
   amrex::Real mesh_patch_origin3 = 0.0;
   std::array<int, 3> mesh_patch_central_cells{{16, 16, 16}};
   int mesh_patch_angular_cells = 16;
   int mesh_patch_radial_cells = 16;
   int mesh_patch_outer_radial_cells = 16;
   int mesh_patch_axial_cells = 16;
   amrex::Real mesh_patch_axial_lo = -1.0;
   amrex::Real mesh_patch_axial_hi = 1.0;
   int mesh_patch_pack_direction = 0;
   int mesh_patch_gap_cells = 0;
   bool mesh_allow_experimental_mixed_shell = false;
   std::string mesh_metric_divergence = "auto";

   std::string physics;
   std::string spacetime_evolution = "none";
   std::string spacetime_matter_coupling = "coupled";
   std::string spacetime_fixed_metric = "minkowski";
   std::string spacetime_initial_data = "minkowski";
   std::string initial_data_magnetic = "id";
   std::string initial_data_spacetime = "id";
   std::string initial_data_lapse = "id";
   std::string initial_data_shift = "id";
   std::string spacetime_lapse_gauge = "frozen";
   std::string spacetime_shift_gauge = "frozen";
   amrex::Real spacetime_amplitude = 1.0e-8;
   amrex::Real spacetime_wave_number = 1.0;
   amrex::Real spacetime_kappa1 = 0.02;
   amrex::Real spacetime_kappa2 = 0.0;
   amrex::Real spacetime_shift_eta = 2.0;
   amrex::Real spacetime_slow_start_time = 10.0;
   amrex::Real spacetime_ko_dissipation = 0.1;
   amrex::Real spacetime_acceleration1 = 0.0;
   amrex::Real spacetime_acceleration2 = 0.0;
   amrex::Real spacetime_acceleration3 = 0.0;
   amrex::Real spacetime_origin1 = 0.0;
   amrex::Real spacetime_origin2 = 0.0;
   amrex::Real spacetime_origin3 = 0.0;
   amrex::Real spacetime_mass = 1.0;
   amrex::Real tov_central_density = 1.28e-3;
   amrex::Real tov_radial_step = 1.0e-3;
   int tov_radial_points = 32768;
   std::string ct;
   std::string ic = "sin_wave";
   int passive_scalars = 0;
   std::string passive_profile = "none";
   amrex::Real passive_value = 1.0;
   amrex::Real passive_background = 0.0;
   amrex::Real passive_amplitude = 1.0;
   amrex::Real passive_left = 1.0;
   amrex::Real passive_right = 0.0;
   amrex::Real passive_sigma = -1.0;
   amrex::Real passive_x0 = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real passive_y0 = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real passive_z0 = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real inflow_passive_value = 1.0;
   amrex::Real inflow_background_passive_value = 0.0;
   amrex::Real rho0 = 1.0;
   amrex::Real pressure0 = 1.0;
   amrex::Real amplitude = 1.0e-3;
   amrex::Real bump_sigma = 0.08;
   amrex::Real acoustic_transverse_sigma = 0.2;
   amrex::Real bump_x0 = 0.5;
   amrex::Real bump_y0 = 0.5;
   amrex::Real bump_z0 = 0.5;
   amrex::Real beam_length = 0.25;
   amrex::Real beam_radius = 0.08;
   amrex::Real shock_x0 = 0.5;
   amrex::Real shock_y0 = 0.5;
   amrex::Real shock_z0 = 0.5;
   amrex::Real shock_nx = 1.0;
   amrex::Real shock_ny = 0.0;
   amrex::Real shock_nz = 0.0;
   amrex::Real shock_speed = 0.0;
   amrex::Real rho_left = 1.0;
   amrex::Real pressure_left = 1.0;
   amrex::Real v1_left = 0.0;
   amrex::Real v2_left = 0.0;
   amrex::Real v3_left = 0.0;
   amrex::Real rho_right = 0.125;
   amrex::Real pressure_right = 0.1;
   amrex::Real v1_right = 0.0;
   amrex::Real v2_right = 0.0;
   amrex::Real v3_right = 0.0;
   amrex::Real b1_left = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real b2_left = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real b3_left = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real b1_right = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real b2_right = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real b3_right = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real v1_0 = 0.2;
   amrex::Real v2_0 = 0.0;
   amrex::Real v3_0 = 0.0;
   amrex::Real b1_0 = 0.0;
   amrex::Real b2_0 = 0.0;
   amrex::Real b3_0 = 0.0;
   amrex::Real alfven_amplitude = 1.0e-3;
   amrex::Real wave_number = 1.0;
   amrex::Real wave_mode_1 = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real wave_mode_2 = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real wave_mode_3 = std::numeric_limits<amrex::Real>::quiet_NaN();
   std::string wave_family = "acoustic_plus";
   amrex::Real kh_secondary_amplitude = 0.0;
   amrex::Real kh_secondary_mode = 3.0;
   amrex::Real kh_z_mode = 1.0;
   amrex::Real kh_phase = 0.37;
   amrex::Real kh_envelope_sigma = 0.0;
   amrex::Real kh_density_amplitude = 0.0;
   std::string kh_perturbation = "normal";
   amrex::Real rotor_radius = 0.10;
   amrex::Real rotor_transition_radius = 0.115;
   amrex::Real rotor_density = 10.0;
   amrex::Real rotor_omega = 20.0;

   std::string recon;
   std::string recon_fine = "";
   int recon_fine_start_level = -1;
   std::string riemann;
   std::string timestep;
   std::string bc = "periodic";
   std::string bc_xlo = "";
   std::string bc_xhi = "";
   std::string bc_ylo = "";
   std::string bc_yhi = "";
   std::string bc_zlo = "";
   std::string bc_zhi = "";
   std::string boundary_profile = "none";
   std::string inflow_profile = "uniform";
   amrex::Real inflow_rho = 1.0;
   amrex::Real inflow_pressure = 1.0;
   amrex::Real inflow_v1 = 0.0;
   amrex::Real inflow_v2 = 0.0;
   amrex::Real inflow_v3 = 0.0;
   amrex::Real inflow_b1 = 0.0;
   amrex::Real inflow_b2 = 0.0;
   amrex::Real inflow_b3 = 0.0;
   amrex::Real inflow_background_rho = 1.0;
   amrex::Real inflow_background_pressure = 1.0;
   amrex::Real inflow_background_v1 = 0.0;
   amrex::Real inflow_background_v2 = 0.0;
   amrex::Real inflow_background_v3 = 0.0;
   amrex::Real inflow_background_b1 = 0.0;
   amrex::Real inflow_background_b2 = 0.0;
   amrex::Real inflow_background_b3 = 0.0;
   amrex::Real inflow_center_y = 0.5;
   amrex::Real inflow_center_z = 0.5;
   amrex::Real inflow_radius = 0.08;
   std::string diffusion;
   amrex::Real viscosity = 0.0;
   amrex::Real resistivity = 0.0;
   amrex::Real diffusion_cfl = 0.9;
   amrex::Real diffusion_tolerance = 1.0e-10;
   int diffusion_max_iterations = 200;
   amrex::Real diffusion_sor_omega = 1.5;
   std::string source = "none";
   amrex::Real source_a1 = 0.0;
   amrex::Real source_a2 = 0.0;
   amrex::Real source_a3 = 0.0;
   amrex::Real source_power = 0.0;
   amrex::Real source_radius = 0.5;
   amrex::Real source_sigma = 0.05;
   amrex::Real source_omega = 0.0;
   amrex::Real source_ramp_time = 0.0;
   amrex::Real source_phase = 0.0;
   int source_azimuthal_mode = 2;
   std::string gravity = "none";
   std::string gravity_coupling = "athena";
   std::string gravity_fixed_potential = "uniform_acceleration";
   amrex::Real gravity_constant = 1.0;
   amrex::Real gravity_a1 = 0.0;
   amrex::Real gravity_a2 = 0.0;
   amrex::Real gravity_a3 = 0.0;
   amrex::Real gravity_omega = 0.0;
   amrex::Real gravity_center_x = 0.5;
   amrex::Real gravity_center_y = 0.5;
   amrex::Real gravity_center_z = 0.5;
   amrex::Real gravity_tolerance = 1.0e-10;
   int gravity_max_iterations = 200;
   int gravity_max_fmg_iterations = 20;
   int gravity_max_order = 2;
   int gravity_verbose = 0;
   std::string subcycling;
   std::string subcycling_steps;
   std::string solid_geometry = "off";
   std::string solid_motion = "static";
   std::string solid_wall_condition = "slip";
   int solid_volume_fraction_samples_per_dim = 1;
   int solid_box_count = 0;
   std::array<amrex::Real, MaxSolidBoxValues> solid_box_lo{};
   std::array<amrex::Real, MaxSolidBoxValues> solid_box_hi{};
   amrex::Real solid_radius = 0.1;
   amrex::Real solid_radius_amplitude = 0.0;
   amrex::Real solid_radius_frequency = 0.0;
   amrex::Real solid_radius_phase = 0.0;
   amrex::Real solid_radius_ramp_time = 0.0;
   amrex::Real solid_swept_reference_time = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real solid_center_x = 0.5;
   amrex::Real solid_center_y = 0.5;
   amrex::Real solid_center_z = 0.5;
   amrex::Real solid_velocity_1 = 0.0;
   amrex::Real solid_velocity_2 = 0.0;
   amrex::Real solid_velocity_3 = 0.0;
   amrex::Real solid_angular_velocity_1 = 0.0;
   amrex::Real solid_angular_velocity_2 = 0.0;
   amrex::Real solid_angular_velocity_3 = 0.0;
   amrex::Real solid_wedge_x0 = 0.35;
   amrex::Real solid_wedge_y0 = 0.15;
   amrex::Real solid_wedge_angle_deg = 10.0;
   std::string tag_metric;
   std::string state_interp;
   amrex::Real tag_threshold;
   bool tag_threshold_set = false;
   amrex::Real min_feature_length;
   bool min_feature_length_set = false;
   amrex::Real min_feature_length_1;
   bool min_feature_length_1_set = false;
   amrex::Real min_feature_length_2;
   bool min_feature_length_2_set = false;
   amrex::Real min_feature_length_3;
   bool min_feature_length_3_set = false;
   amrex::Real min_refine_radius;
   amrex::Real solid_refine_band_cells;
   amrex::Real solid_curvature_error;
   std::string initial_refine;
   int initial_refine_levels;
   int initial_refine_fixed_levels;
   int initial_refine_box_count = 0;
   std::array<amrex::Real, MaxInitialRefineBoxValues> initial_refine_box_lo{};
   std::array<amrex::Real, MaxInitialRefineBoxValues> initial_refine_box_hi{};
   int amr_max_level;
   int regrid_int;
   std::string regrid_mode;
   amrex::Long level_cell_budget;
   int level_cell_budget_side;
   amrex::Long total_cell_budget;
};

struct DiffusionStats {
   amrex::Real residual = 0.0;
   int iterations = 0;
};

void require_implicit_diffusion_converged(
   DiffusionStats const& stats,
   RunConfig const& cfg,
   char const* context
);

int riemann_method_code(RunConfig const& cfg);
int recon_method_code_from_name(std::string const& name, char const* key);
int recon_method_code(RunConfig const& cfg);
int timestep_method_code(RunConfig const& cfg);
int diffusion_method_code(RunConfig const& cfg);
int eos_method_code(RunConfig const& cfg);
bool valid_plotfile_compression(std::string const& value);
std::string resolved_plotfile_compression(std::string const& value);

RunConfig read_config();
void write_run_parameter_file(std::string const& path);


std::string trim_copy(std::string value);
std::vector<std::string> aux_gf_names(RunConfig const& cfg);
bool aux_gfs_enabled(RunConfig const& cfg);
bool aux_gfs_contains(RunConfig const& cfg, AuxGridField target);
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> configured_min_feature_lengths(
   RunConfig const& cfg
);
amrex::Real amr_default_lohner_tag_threshold();
amrex::Real amr_default_auto_min_feature_scale(int max_level);

enum class BoundaryKind {
   Periodic,
   Outflow,
   Reflecting,
   Symmetry,
   Inflow,
   Diode,
   Origin,
   Analytic
};

std::string side_bc_name(RunConfig const& cfg, int direction, bool high_side);
BoundaryKind boundary_kind_from_name(std::string const& value);
BoundaryKind side_boundary_kind(
   RunConfig const& cfg,
   int direction,
   bool high_side
);
bool reflective_boundary(BoundaryKind kind);
bool fixed_inflow_boundary(BoundaryKind kind);
bool diode_boundary(BoundaryKind kind);

bool curvilinear_mesh_coordinate(int coordinate);
bool curvilinear_mesh_enabled(RunConfig const& cfg);
bool cartesian_patch_box_union_domain_enabled(RunConfig const& cfg);
MeshAxes mesh_axes_layout(RunConfig const& cfg);
MeshRuntimeLayout mesh_runtime_layout(RunConfig const& cfg);
PhysicalSystem physical_system(RunConfig const& cfg);
SpacetimeEvolution spacetime_evolution_mode(RunConfig const& cfg);
bool spacetime_evolves(RunConfig const& cfg) noexcept;
bool spacetime_stores_state(RunConfig const& cfg) noexcept;
bool spacetime_includes_matter(RunConfig const& cfg) noexcept;
FixedSpacetimeParameters fixed_spacetime_parameters(RunConfig const& cfg);

void validate_config(RunConfig const& cfg);


bool is_mhd(RunConfig const& cfg);
bool viscosity_enabled(RunConfig const& cfg);
bool resistivity_enabled(RunConfig const& cfg);
bool gravity_enabled(RunConfig const& cfg);
bool athena_gravity_coupling_enabled(RunConfig const& cfg);
bool self_gravity_enabled(RunConfig const& cfg);
bool fixed_gravity_enabled(RunConfig const& cfg);
int state_component_count(RunConfig const& cfg);

} // namespace pyfluid_amrex
