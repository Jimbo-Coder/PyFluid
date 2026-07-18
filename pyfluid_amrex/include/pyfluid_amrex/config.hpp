#pragma once

#include <AMReX_Array.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/build_options.hpp"
#include "pyfluid_amrex/config_registry.hpp"
#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/mesh_limits.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

struct RunConfig {
   RunConfig();

   int ng = 1;
   int max_step = 32;
   amrex::Real final_time = -1.0;
   amrex::Real cfl = 0.9;
   amrex::Real gamma = 5.0 / 3.0;
   std::string eos;
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

struct NamedOptionCode {
   std::string_view name;
   int code;
};

static constexpr std::array<NamedOptionCode, 5> RiemannOptions{{
   {"llf", RiemannLLF},
   {"hll", RiemannHLL},
   {"hlle", RiemannHLLE},
   {"hllc", RiemannHLLC},
   {"hlld", RiemannHLLD},
}};

static constexpr std::array<NamedOptionCode, 8> ReconOptions{{
   {"pwc", ReconPWC},
   {"pwl", ReconPWL},
   {"wenoz", ReconWENOZ},
   {"wenoz-char", ReconWENOZChar},
   {"mp5", ReconMP5},
   {"mp5-char", ReconMP5Char},
   {"teno5", ReconTENO5},
   {"teno5-char", ReconTENO5Char},
}};

static constexpr std::array<NamedOptionCode, 5> TimestepOptions{{
   {"euler", TimeEuler},
   {"corrector", TimeCorrector},
   {"midpoint", TimeMidpoint},
   {"ssprk3", TimeSSPRK3},
   {"rk4", TimeRK4},
}};

static constexpr std::array<NamedOptionCode, 3> DiffusionOptions{{
   {"none", DiffusionNone},
   {"explicit", DiffusionExplicit},
   {"implicit", DiffusionImplicit},
}};

static constexpr std::array<NamedOptionCode, 3> EOSOptions{{
   {"ideal_gas", EOSIdealGas},
   {"polytrope", EOSPolytrope},
   {"hybrid_polytrope", EOSHybridPolytrope},
}};

static constexpr std::array<std::string_view, 4> PhysicsOptions{{"hydro", "mhd", "rhd", "rmhd"}};
static constexpr std::array<std::string_view, 3> SpacetimeEvolutionOptions{{
   "none", "fixed", "evolved"
}};
static constexpr std::array<std::string_view, 2> SpacetimeMatterCouplingOptions{{"coupled", "vacuum"}};
static constexpr std::array<std::string_view, 6> SpacetimeInitialDataOptions{{
   "minkowski", "gauge_wave", "linear_wave", "robust_stability", "ctt", "xcts"
}};
static constexpr std::array<std::string_view, 3> MagneticInitialDataOptions{{
   "id", "zero", "none"
}};
static constexpr std::array<std::string_view, 3> SpacetimeGroupInitialDataOptions{{
   "id", "flat", "none"
}};
static constexpr std::array<std::string_view, 3> LapseInitialDataOptions{{
   "id", "one", "none"
}};
static constexpr std::array<std::string_view, 3> ShiftInitialDataOptions{{
   "id", "zero", "none"
}};
static constexpr std::array<std::string_view, 4> Z4cLapseGaugeOptions{{
   "frozen", "harmonic", "one_plus_log", "slow_start"
}};
static constexpr std::array<std::string_view, 2> Z4cShiftGaugeOptions{{
   "frozen", "gamma_driver"
}};
static constexpr std::array<std::string_view, 3> FixedSpacetimeMetricOptions{{
   "minkowski", "rindler", "schwarzschild_isotropic",
}};
static constexpr std::array<std::string_view, 2> CTOptions{{"off", "on"}};
static constexpr std::array<std::string_view, 2> PlotfileFormatOptions{{"amrex", "hdf5"}};
static constexpr std::array<std::string_view, 5> PassiveProfileOptions{{"none", "uniform", "gaussian", "sin_wave", "step_x"}};
static constexpr std::array<std::string_view, 2> ProfileOptions{{"on", "off"}};
static constexpr std::array<std::string_view, 2> ProfileDetailOptions{{"basic", "flux"}};
static constexpr std::array<std::string_view, 8> BoundaryOptions{{
   "periodic",
   "outflow",
   "reflecting",
   "symmetry",
   "inflow",
   "diode",
   "origin",
   "analytic",
}};
static constexpr std::array<std::string_view, 2> BoundaryProfileOptions{{"none", "double_mach_reflection"}};
static constexpr std::array<std::string_view, 3> InflowProfileOptions{{"uniform", "gaussian_beam", "uniform_beam"}};
static constexpr std::array<std::string_view, 2> ScalarOutputModeOptions{{"stream", "final"}};
static constexpr std::array<std::string_view, 9> ValidationPolicyOptions{{
   "closed",
   "open",
   "open_mhd",
   "driven",
   "wall",
   "diffusive",
   "diffusive_wall",
   "smoke",
   "elliptic",
}};
static constexpr std::array<std::string_view, 3> SourceOptions{{"none", "uniform_acceleration", "rotating_compact_heating"}};
static constexpr std::array<std::string_view, 3> GravityOptions{{"none", "poisson", "fixed"}};
static constexpr std::array<std::string_view, 2> GravityCouplingOptions{{"athena", "source"}};
static constexpr std::array<std::string_view, 2> GravityFixedPotentialOptions{{"uniform_acceleration", "harmonic"}};
static constexpr std::array<std::string_view, 2> SubcyclingOptions{{"off", "on"}};
static constexpr std::array<std::string_view, 2> SubcyclingStepOptions{{"spatial", "local_cfl"}};
static constexpr std::array<std::string_view, 4> SolidGeometryOptions{{"off", "boxes", "wedge", "circle"}};
static constexpr std::array<std::string_view, 4> SolidMotionOptions{{"static", "translate", "rotate", "pulse"}};
static constexpr std::array<std::string_view, 2> SolidWallConditionOptions{{"slip", "no_slip"}};
static constexpr std::array<std::string_view, 3> MeshPatchBoxOwnershipOptions{{"exact", "owner_first", "owner_last"}};

inline amrex::Real amr_default_lohner_tag_threshold();
static constexpr std::array<std::string_view, 2> MeshPatchBoxDomainOptions{{"rectangular", "union"}};
static constexpr std::array<std::string_view, 2> AMRTagMetricOptions{{"none", "lohner"}};
static constexpr std::array<std::string_view, 2> AMRRegridModeOptions{{"synchronized", "aligned"}};
static constexpr std::array<std::string_view, 5> AMRStateInterpOptions{{
   "conservative",
   "protected",
   "quadratic",
   "quartic",
   "quartic-conservative",
}};
static constexpr std::array<std::string_view, 2> MetricRegridOptions{{"auto", "always"}};
static constexpr std::array<std::string_view, 2> InitialRefineOptions{{"none", "box"}};
static constexpr std::array<std::string_view, 3> KHPerturbationOptions{{"normal", "stream", "vortex"}};
static constexpr std::array<std::string_view, 5> HydroWaveFamilyOptions{{
   "acoustic_plus",
   "acoustic_minus",
   "entropy",
   "shear_y",
   "shear_z",
}};
static constexpr std::array<std::string_view, 7> MHDWaveFamilyOptions{{
   "fast_plus",
   "fast_minus",
   "alfven_plus",
   "alfven_minus",
   "slow_plus",
   "slow_minus",
   "entropy",
}};
using PhysicalSystemSet = std::uint8_t;

constexpr PhysicalSystemSet physical_system_set(PhysicalSystem system) noexcept
{
   return PhysicalSystemSet{1U} << static_cast<std::uint8_t>(system);
}

constexpr PhysicalSystemSet operator|(PhysicalSystem lhs, PhysicalSystem rhs) noexcept
{
   return physical_system_set(lhs) | physical_system_set(rhs);
}

constexpr PhysicalSystemSet operator|(
   PhysicalSystemSet systems,
   PhysicalSystem system
) noexcept
{
   return systems | physical_system_set(system);
}

constexpr bool contains_physical_system(
   PhysicalSystemSet systems,
   PhysicalSystem system
) noexcept
{
   return (systems & physical_system_set(system)) != 0;
}

struct InitialDataOption {
   std::string_view name;
   PhysicalSystemSet physical_systems;
};

static constexpr PhysicalSystemSet HydroSystems = PhysicalSystem::HD | PhysicalSystem::RHD;
static constexpr PhysicalSystemSet MagneticSystems = PhysicalSystem::MHD | PhysicalSystem::RMHD;
static constexpr PhysicalSystemSet AllFluidSystems = HydroSystems | PhysicalSystem::MHD | PhysicalSystem::RMHD;

// This is the single source of truth for IC/system compatibility. A Newtonian
// profile admitted for RHD/RMHD is converted through primitive variables; that
// does not make its Newtonian analytic evolution an exact relativistic test.
static constexpr std::array<InitialDataOption, 32> InitialDataOptions{{
   {"sin_wave", AllFluidSystems},
   {"advection_medley", HydroSystems},
   {"characteristic_medley", HydroSystems},
   {"mhd_characteristic_medley", PhysicalSystem::MHD | PhysicalSystem::RMHD},
   {"uniform", AllFluidSystems},
   {"cartesian_advection", AllFluidSystems},
   {"linear_wave_x", AllFluidSystems},
   {"acoustic_wave_x", HydroSystems},
   {"acoustic_packet_x", HydroSystems},
   {"jeans_wave_x", physical_system_set(PhysicalSystem::HD)},
   {"shear_wave_x", HydroSystems},
   {"pressure_bump", AllFluidSystems},
   {"vortex_blast", AllFluidSystems},
   {"density_cloud", AllFluidSystems},
   {"hse_polytrope_x", physical_system_set(PhysicalSystem::HD)},
   {"fixed_metric_hse", physical_system_set(PhysicalSystem::RHD)},
   {"tov", physical_system_set(PhysicalSystem::RHD)},
   {"hydro_beam", HydroSystems},
   {"sod_x", AllFluidSystems},
   {"sod_plane", AllFluidSystems},
   {"sod_periodic_plane", AllFluidSystems},
   {"double_mach_reflection", physical_system_set(PhysicalSystem::HD)},
   {"circular_alfven_x", MagneticSystems},
   {"resistive_wave_x", physical_system_set(PhysicalSystem::MHD)},
   {"magnetic_tube_z", MagneticSystems},
   {"mhd_rotor", MagneticSystems},
   {"kelvin_helmholtz", HydroSystems},
   {"kelvin_helmholtz_mhd", MagneticSystems},
   {"orszag_tang", MagneticSystems},
   {"magnetic_blast_noise", MagneticSystems},
   {"resistive_current_sheet", physical_system_set(PhysicalSystem::MHD)},
   {"reconnecting_shear_layer", physical_system_set(PhysicalSystem::MHD)},
}};

template <std::size_t N>
inline bool option_supported(std::string const& value, std::array<std::string_view, N> const& options)
{
   for (std::string_view option : options) {
      if (value == option) {
         return true;
      }
   }
   return false;
}

template <std::size_t N>
inline bool option_supported(std::string const& value, std::array<NamedOptionCode, N> const& options)
{
   for (NamedOptionCode const& option : options) {
      if (value == option.name) {
         return true;
      }
   }
   return false;
}

template <std::size_t N>
inline bool option_supported(
   std::string const& value,
   std::array<InitialDataOption, N> const& options
)
{
   for (InitialDataOption const& option : options) {
      if (value == option.name) {
         return true;
      }
   }
   return false;
}

template <std::size_t N>
inline std::string supported_option_names(std::array<std::string_view, N> const& options)
{
   std::string names;
   for (std::size_t n = 0; n < N; ++n) {
      if (n > 0) {
         names += ", ";
      }
      names += options[n];
   }
   return names;
}

template <std::size_t N>
inline std::string supported_option_names(std::array<NamedOptionCode, N> const& options)
{
   std::string names;
   for (std::size_t n = 0; n < N; ++n) {
      if (n > 0) {
         names += ", ";
      }
      names += options[n].name;
   }
   return names;
}

template <std::size_t N>
inline std::string supported_option_names(
   std::array<InitialDataOption, N> const& options
)
{
   std::string names;
   for (std::size_t n = 0; n < N; ++n) {
      if (n > 0) {
         names += ", ";
      }
      names += options[n].name;
   }
   return names;
}

template <std::size_t N>
inline int option_code(std::string const& value, std::array<NamedOptionCode, N> const& options, char const* key)
{
   for (NamedOptionCode const& option : options) {
      if (value == option.name) {
         return option.code;
      }
   }
   amrex::Abort(std::string(key) + " supports " + supported_option_names(options));
   return -1;
}

template <std::size_t N>
inline void require_supported_option(std::string const& value, std::array<std::string_view, N> const& options, char const* key)
{
   if (!option_supported(value, options)) {
      amrex::Abort(std::string(key) + " supports " + supported_option_names(options));
   }
}

template <std::size_t N>
inline void require_supported_option(std::string const& value, std::array<NamedOptionCode, N> const& options, char const* key)
{
   if (!option_supported(value, options)) {
      amrex::Abort(std::string(key) + " supports " + supported_option_names(options));
   }
}

template <std::size_t N>
inline void require_supported_option(
   std::string const& value,
   std::array<InitialDataOption, N> const& options,
   char const* key
)
{
   if (!option_supported(value, options)) {
      amrex::Abort(std::string(key) + " supports " + supported_option_names(options));
   }
}

inline InitialDataOption const& initial_data_option(std::string const& name)
{
   for (InitialDataOption const& option : InitialDataOptions) {
      if (name == option.name) {
         return option;
      }
   }
   amrex::Abort("pyfluid.ic supports " + supported_option_names(InitialDataOptions));
   return InitialDataOptions.front();
}

inline void require_implicit_diffusion_converged(DiffusionStats const& stats, RunConfig const& cfg, char const* context)
{
   if (stats.residual <= cfg.diffusion_tolerance) {
      return;
   }
   amrex::Abort(
      std::string(context)
      + " failed to converge: residual=" + std::to_string(static_cast<double>(stats.residual))
      + " tolerance=" + std::to_string(static_cast<double>(cfg.diffusion_tolerance))
      + " iterations=" + std::to_string(stats.iterations)
      + " max_iterations=" + std::to_string(cfg.diffusion_max_iterations)
   );
}

inline int riemann_method_code(RunConfig const& cfg)
{
   return option_code(cfg.riemann, RiemannOptions, "pyfluid.riemann");
}

inline bool valid_recon_name(std::string const& name)
{
   return option_supported(name, ReconOptions);
}

inline int recon_method_code_from_name(std::string const& name, char const* key)
{
   return option_code(name, ReconOptions, key);
}

inline int recon_method_code(RunConfig const& cfg)
{
   return recon_method_code_from_name(cfg.recon, "pyfluid.recon");
}

inline int timestep_method_code(RunConfig const& cfg)
{
   return option_code(cfg.timestep, TimestepOptions, "pyfluid.timestep");
}

inline int diffusion_method_code(RunConfig const& cfg)
{
   return option_code(cfg.diffusion, DiffusionOptions, "pyfluid.diffusion");
}

inline int eos_method_code(RunConfig const& cfg)
{
   return option_code(cfg.eos, EOSOptions, "pyfluid.eos");
}

inline EOSParams eos_params(RunConfig const& cfg)
{
   EOSParams eos{};
   eos.gamma = cfg.gamma;
   eos.thermal_gamma = cfg.thermal_gamma;
   eos.rho_floor = cfg.rho_floor;
   eos.pressure_floor = cfg.pressure_floor;
   eos.polytrope_segments = 1 + static_cast<int>(cfg.polytropic_density_breaks.size());
   eos.segment_k[0] = cfg.polytropic_k;
   eos.segment_gamma[0] = cfg.polytropic_gamma;
   for (int segment = 1; segment < eos.polytrope_segments; ++segment) {
      const amrex::Real density_break = cfg.polytropic_density_breaks[segment - 1];
      const amrex::Real previous_gamma = eos.segment_gamma[segment - 1];
      const amrex::Real gamma = cfg.polytropic_segment_gammas[segment - 1];
      const amrex::Real previous_k = eos.segment_k[segment - 1];
      const amrex::Real k = previous_k * std::pow(
         density_break, previous_gamma - gamma
      );
      eos.segment_density_break[segment - 1] = density_break;
      eos.segment_gamma[segment] = gamma;
      eos.segment_k[segment] = k;
      eos.segment_energy_offset[segment] =
         eos.segment_energy_offset[segment - 1]
         + previous_k * std::pow(density_break, previous_gamma - 1.0)
            / (previous_gamma - 1.0)
         - k * std::pow(density_break, gamma - 1.0) / (gamma - 1.0);
   }
   return eos;
}

inline bool valid_plotfile_format(std::string const& value)
{
   return option_supported(value, PlotfileFormatOptions);
}

inline bool valid_plotfile_compression(std::string const& value)
{
   if (value == "none" || value == "best" || value == "zlib") {
      return true;
   }
   if (value.empty()) {
      return false;
   }
   if (value.size() == 5 && value.rfind("zlib", 0) == 0 && std::isdigit(static_cast<unsigned char>(value[4])) != 0) {
      return true;
   }
   if (value.rfind("zlib@", 0) == 0) {
      const std::string level = value.substr(5);
      if (level.empty()) {
         return false;
      }
      for (char digit : level) {
         if (!std::isdigit(static_cast<unsigned char>(digit))) {
            return false;
         }
      }
      return true;
   }
   return false;
}

inline std::string resolved_plotfile_compression(std::string const& value)
{
   if (value == "none") {
      return "None@0";
   }
   if (value == "best") {
      return "ZLIB@9";
   }
   if (value == "zlib") {
      return "ZLIB@6";
   }
   if (value.size() == 5 && value.rfind("zlib", 0) == 0 && std::isdigit(static_cast<unsigned char>(value[4])) != 0) {
      return "ZLIB@" + value.substr(4);
   }
   if (value.rfind("zlib@", 0) == 0 && value.size() > 5) {
      const std::string level = value.substr(5);
      bool all_digits = !level.empty();
      for (char digit : level) {
         if (!std::isdigit(static_cast<unsigned char>(digit))) {
            all_digits = false;
            break;
         }
      }
      if (all_digits) {
         return "ZLIB@" + level;
      }
   }
   return value;
}

RunConfig read_config();


inline std::string trim_copy(std::string value)
{
   while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
      value.erase(value.begin());
   }
   while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
      value.pop_back();
   }
   return value;
}

inline std::vector<std::string> aux_gf_names(RunConfig const& cfg)
{
   std::vector<std::string> names;
   std::stringstream stream(cfg.aux_gfs);
   std::string token;
   while (std::getline(stream, token, ',')) {
      token = trim_copy(token);
      if (token.empty() || token == "none") {
         continue;
      }
      names.push_back(token);
   }
   return names;
}

inline bool aux_gfs_enabled(RunConfig const& cfg)
{
   return !aux_gf_names(cfg).empty();
}

inline bool aux_gfs_contains(RunConfig const& cfg, AuxGridField target)
{
   for (std::string const& name : aux_gf_names(cfg)) {
      if (aux_grid_field_from_name(name) == target) {
         return true;
      }
   }
   return false;
}

inline amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> configured_min_feature_lengths(RunConfig const& cfg)
{
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> values{};
   values[0] = std::isfinite(cfg.min_feature_length_1) ? cfg.min_feature_length_1 : cfg.min_feature_length;
#if (AMREX_SPACEDIM >= 2)
   values[1] = std::isfinite(cfg.min_feature_length_2) ? cfg.min_feature_length_2 : cfg.min_feature_length;
#endif
#if (AMREX_SPACEDIM == 3)
   values[2] = std::isfinite(cfg.min_feature_length_3) ? cfg.min_feature_length_3 : cfg.min_feature_length;
#endif
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      values[dir] = values[dir] > 0.0 ? values[dir] : -1.0;
   }
   return values;
}

inline amrex::Real amr_default_lohner_tag_threshold()
{
   return 0.08;
}

inline amrex::Real amr_default_auto_min_feature_scale(int max_level)
{
   // Keep about two points per feature on the finest level for a curvature-style trigger:
   // dx_fine = dx_base / 2^max_level.
   constexpr amrex::Real points_per_feature = 2.0;
   const amrex::Real finest_ratio = std::ldexp(1.0, -max_level);
   return points_per_feature * finest_ratio;
}

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

inline std::string side_bc_name(RunConfig const& cfg, int dir, bool high_side)
{
   std::string value;
   if (dir == 0) {
      value = high_side ? cfg.bc_xhi : cfg.bc_xlo;
   }
#if (AMREX_SPACEDIM >= 2)
   if (dir == 1) {
      value = high_side ? cfg.bc_yhi : cfg.bc_ylo;
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (dir == 2) {
      value = high_side ? cfg.bc_zhi : cfg.bc_zlo;
   }
#endif
   value = trim_copy(value);
   return value.empty() ? cfg.bc : value;
}

inline BoundaryKind boundary_kind_from_name(std::string const& value)
{
   if (value == "periodic") {
      return BoundaryKind::Periodic;
   }
   if (value == "outflow") {
      return BoundaryKind::Outflow;
   }
   if (value == "reflecting") {
      return BoundaryKind::Reflecting;
   }
   if (value == "symmetry") {
      return BoundaryKind::Symmetry;
   }
   if (value == "inflow") {
      return BoundaryKind::Inflow;
   }
   if (value == "diode") {
      return BoundaryKind::Diode;
   }
   if (value == "origin") {
      return BoundaryKind::Origin;
   }
   if (value == "analytic") {
      return BoundaryKind::Analytic;
   }
   amrex::Abort("unsupported boundary condition: " + value);
   return BoundaryKind::Outflow;
}

inline BoundaryKind side_boundary_kind(RunConfig const& cfg, int dir, bool high_side)
{
   return boundary_kind_from_name(side_bc_name(cfg, dir, high_side));
}

inline bool reflective_boundary(BoundaryKind kind)
{
   return kind == BoundaryKind::Reflecting || kind == BoundaryKind::Symmetry || kind == BoundaryKind::Origin;
}

inline bool fixed_inflow_boundary(BoundaryKind kind)
{
   return kind == BoundaryKind::Inflow;
}

inline bool diode_boundary(BoundaryKind kind)
{
   return kind == BoundaryKind::Diode;
}

inline bool valid_bc_name(std::string const& value)
{
   return option_supported(value, BoundaryOptions);
}

inline bool valid_hydro_wave_family(std::string const& value)
{
   return option_supported(value, HydroWaveFamilyOptions);
}

inline bool valid_mhd_wave_family(std::string const& value)
{
   return option_supported(value, MHDWaveFamilyOptions);
}

inline bool curvilinear_mesh_coordinate(int coordinate)
{
   return coordinate == MeshCylindrical || coordinate == MeshSpherical;
}

inline bool curvilinear_mesh_enabled(RunConfig const& cfg)
{
   return curvilinear_mesh_coordinate(mesh_coordinate_code(cfg.mesh_coordinate));
}

inline bool cartesian_patch_box_union_domain_enabled(RunConfig const& cfg)
{
   return cfg.mesh_patch_system == "multi_homogeneous_cartesian"
      && cfg.mesh_patch_box_domain == "union"
      && cfg.mesh_patch_box_count > 0;
}

inline MeshAxes mesh_axes_layout(RunConfig const& cfg)
{
   return mesh_axes_layout(mesh_coordinate_code(cfg.mesh_coordinate), cfg.mesh_axes);
}

inline MeshRuntimeLayout mesh_runtime_layout(RunConfig const& cfg)
{
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   return {
      coordinate,
      mesh_axes_layout(coordinate, cfg.mesh_axes),
      mesh_patch_system_code(cfg.mesh_patch_system),
      curvilinear_mesh_coordinate(coordinate),
   };
}

inline PhysicalSystem physical_system(RunConfig const& cfg)
{
   if (cfg.physics == "mhd") {
      return PhysicalSystem::MHD;
   }
   if (cfg.physics == "rhd") {
      return PhysicalSystem::RHD;
   }
   if (cfg.physics == "rmhd") {
      return PhysicalSystem::RMHD;
   }
   return PhysicalSystem::HD;
}

inline SpacetimeEvolution spacetime_evolution_mode(RunConfig const& cfg)
{
   if (cfg.spacetime_evolution == "fixed") {
      return SpacetimeEvolution::Fixed;
   }
   if (cfg.spacetime_evolution == "evolved") {
      return SpacetimeEvolution::Evolved;
   }
   return SpacetimeEvolution::None;
}

inline bool spacetime_evolves(RunConfig const& cfg) noexcept
{
   return spacetime_evolves(spacetime_evolution_mode(cfg));
}

inline bool spacetime_stores_state(RunConfig const& cfg) noexcept
{
   return cfg.spacetime_evolution == "fixed" || spacetime_evolves(cfg);
}

inline bool spacetime_includes_matter(RunConfig const& cfg) noexcept
{
   return spacetime_stores_state(cfg)
      && cfg.spacetime_matter_coupling == "coupled";
}

inline FixedSpacetimeParameters fixed_spacetime_parameters(RunConfig const& cfg)
{
   return {
      cfg.spacetime_fixed_metric == "rindler"
         ? FixedSpacetimeMetric::Rindler
         : (cfg.spacetime_fixed_metric == "schwarzschild_isotropic"
            ? FixedSpacetimeMetric::SchwarzschildIsotropic
            : FixedSpacetimeMetric::Minkowski),
      cfg.spacetime_acceleration1,
      cfg.spacetime_acceleration2,
      cfg.spacetime_acceleration3,
      cfg.spacetime_origin1,
      cfg.spacetime_origin2,
      cfg.spacetime_origin3,
      cfg.spacetime_mass,
   };
}

void validate_config(RunConfig const& cfg);


inline bool is_mhd(RunConfig const& cfg)
{
   return physical_system_has_magnetic_field(physical_system(cfg));
}

inline bool viscosity_enabled(RunConfig const& cfg)
{
   return cfg.viscosity > 0.0;
}

inline bool resistivity_enabled(RunConfig const& cfg)
{
   return is_mhd(cfg) && cfg.resistivity > 0.0;
}

inline bool gravity_enabled(RunConfig const& cfg)
{
   return cfg.gravity != "none";
}

inline bool athena_gravity_coupling_enabled(RunConfig const& cfg)
{
   return gravity_enabled(cfg) && cfg.gravity_coupling == "athena";
}

inline bool self_gravity_enabled(RunConfig const& cfg)
{
   return cfg.gravity == "poisson";
}

inline bool fixed_gravity_enabled(RunConfig const& cfg)
{
   return cfg.gravity == "fixed";
}

inline int state_component_count(RunConfig const& cfg)
{
   return base_component_count(physical_system(cfg)) + cfg.passive_scalars;
}

} // namespace pyfluid_amrex
