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
static constexpr std::array<std::string_view, 3> SpacetimeEvolutionOptions{{"none", "fixed", "evolved"}};
static constexpr std::array<std::string_view, 2> SpacetimeMatterCouplingOptions{{"coupled", "vacuum"}};
static constexpr std::array<std::string_view, 4> SpacetimeInitialDataOptions{{
   "minkowski", "gauge_wave", "linear_wave", "robust_stability"
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
static constexpr std::array<std::string_view, 32> InitialDataOptions{{
   "sin_wave",
   "advection_medley",
   "characteristic_medley",
   "mhd_characteristic_medley",
   "uniform",
   "cartesian_advection",
   "linear_wave_x",
   "acoustic_wave_x",
   "acoustic_packet_x",
   "jeans_wave_x",
   "shear_wave_x",
   "pressure_bump",
   "vortex_blast",
   "density_cloud",
   "hse_polytrope_x",
   "fixed_metric_hse",
   "tov",
   "hydro_beam",
   "sod_x",
   "sod_plane",
   "sod_periodic_plane",
   "double_mach_reflection",
   "circular_alfven_x",
   "resistive_wave_x",
   "magnetic_tube_z",
   "mhd_rotor",
   "kelvin_helmholtz",
   "kelvin_helmholtz_mhd",
   "orszag_tang",
   "magnetic_blast_noise",
   "resistive_current_sheet",
   "reconnecting_shear_layer",
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

inline RunConfig read_config()
{
   RunConfig cfg;
   amrex::ParmParse pp("pyfluid");
   query_registered_method_parameters(pp, cfg);
   pp.query("ng", cfg.ng);
   pp.query("max_step", cfg.max_step);
   pp.query("final_time", cfg.final_time);
   pp.query("cfl", cfg.cfl);
   pp.query("gamma", cfg.gamma);
   pp.query("polytropic_k", cfg.polytropic_k);
   pp.query("polytropic_gamma", cfg.polytropic_gamma);
   pp.queryarr("polytropic_density_breaks", cfg.polytropic_density_breaks);
   pp.queryarr("polytropic_segment_gammas", cfg.polytropic_segment_gammas);
   pp.query("thermal_gamma", cfg.thermal_gamma);
   pp.query("rho_floor", cfg.rho_floor);
   pp.query("pressure_floor", cfg.pressure_floor);
   pp.query("run_name", cfg.run_name);
   pp.query("output_dir", cfg.output_dir);
   pp.query("validation_policy", cfg.validation_policy);
   pp.query("scalar_int", cfg.scalar_int);
   pp.query("scalar_output_mode", cfg.scalar_output_mode);
   pp.query("plot_int", cfg.plot_int);
   pp.query("plotfile_format", cfg.plotfile_format);
   pp.query("plotfile_compression", cfg.plotfile_compression);
   pp.query("checkpoint_compression", cfg.checkpoint_compression);
   pp.query("cfl_limiter_diagnostics", cfg.cfl_limiter_diagnostics);
   pp.query("aux_gf_compute_int", cfg.aux_gf_compute_int);
   pp.query("aux_gf_int", cfg.aux_gf_int);
   pp.query("aux_gfs", cfg.aux_gfs);
   pp.query("checkpoint_int", cfg.checkpoint_int);
   pp.query("checkpoint_dir", cfg.checkpoint_dir);
   pp.query("restart_file", cfg.restart_file);
   pp.query("profile", cfg.profile);
   pp.query("profile_detail", cfg.profile_detail);
   pp.query("physical_check_int", cfg.physical_check_int);
   pp.query("debug_poison", cfg.debug_poison);
   pp.query("debug_check_finite", cfg.debug_check_finite);
   pp.query("debug_check_physical", cfg.debug_check_physical);
   amrex::ParmParse spacetime_pp("spacetime");
   spacetime_pp.queryAdd("evolution", cfg.spacetime_evolution);
   spacetime_pp.queryAdd("matter_coupling", cfg.spacetime_matter_coupling);
   spacetime_pp.queryAdd("fixed_metric", cfg.spacetime_fixed_metric);
   spacetime_pp.queryAdd("initial_data", cfg.spacetime_initial_data);
   amrex::ParmParse initial_data_pp("initial_data");
   initial_data_pp.queryAdd("magnetic", cfg.initial_data_magnetic);
   initial_data_pp.queryAdd("spacetime", cfg.initial_data_spacetime);
   initial_data_pp.queryAdd("lapse", cfg.initial_data_lapse);
   initial_data_pp.queryAdd("shift", cfg.initial_data_shift);
   spacetime_pp.queryAdd("lapse_gauge", cfg.spacetime_lapse_gauge);
   spacetime_pp.queryAdd("shift_gauge", cfg.spacetime_shift_gauge);
   spacetime_pp.queryAdd("amplitude", cfg.spacetime_amplitude);
   spacetime_pp.queryAdd("wave_number", cfg.spacetime_wave_number);
   spacetime_pp.queryAdd("kappa1", cfg.spacetime_kappa1);
   spacetime_pp.queryAdd("kappa2", cfg.spacetime_kappa2);
   spacetime_pp.queryAdd("shift_eta", cfg.spacetime_shift_eta);
   spacetime_pp.queryAdd("slow_start_time", cfg.spacetime_slow_start_time);
   spacetime_pp.queryAdd("ko_dissipation", cfg.spacetime_ko_dissipation);
   spacetime_pp.queryAdd("acceleration1", cfg.spacetime_acceleration1);
   spacetime_pp.queryAdd("acceleration2", cfg.spacetime_acceleration2);
   spacetime_pp.queryAdd("acceleration3", cfg.spacetime_acceleration3);
   spacetime_pp.queryAdd("origin1", cfg.spacetime_origin1);
   spacetime_pp.queryAdd("origin2", cfg.spacetime_origin2);
   spacetime_pp.queryAdd("origin3", cfg.spacetime_origin3);
   spacetime_pp.queryAdd("mass", cfg.spacetime_mass);
   amrex::ParmParse mesh_pp("mesh");
   mesh_pp.query("coordinate", cfg.mesh_coordinate);
   std::vector<std::string> mesh_axis_values;
   mesh_pp.queryarr("axes", mesh_axis_values);
   if (!mesh_axis_values.empty()) {
      std::ostringstream axes;
      for (int n = 0; n < static_cast<int>(mesh_axis_values.size()); ++n) {
         if (n > 0) {
            axes << ' ';
         }
         axes << mesh_axis_values[n];
      }
      cfg.mesh_axes = axes.str();
   }
   mesh_pp.query("patch_system", cfg.mesh_patch_system);
   std::vector<int> mesh_patch_counts;
   mesh_pp.queryarr("patch_counts", mesh_patch_counts);
   if (!mesh_patch_counts.empty()) {
      if (mesh_patch_counts.size() != AMREX_SPACEDIM) {
         amrex::Abort("mesh.patch_counts must list exactly AMREX_SPACEDIM values");
      }
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         cfg.mesh_patch_counts[dir] = mesh_patch_counts[dir];
      }
   }
   mesh_pp.query("patch_overlap_cells", cfg.mesh_patch_overlap_cells);
   std::vector<int> mesh_patch_box_small;
   std::vector<int> mesh_patch_box_big;
   mesh_pp.queryarr("patch_box_small", mesh_patch_box_small);
   mesh_pp.queryarr("patch_box_big", mesh_patch_box_big);
   mesh_pp.query("patch_box_ownership", cfg.mesh_patch_box_ownership);
   mesh_pp.query("patch_box_domain", cfg.mesh_patch_box_domain);
   mesh_pp.query("patch_inner_radius", cfg.mesh_patch_inner_radius);
   mesh_pp.query("patch_medium_radius", cfg.mesh_patch_medium_radius);
   mesh_pp.query("patch_outer_radius", cfg.mesh_patch_outer_radius);
   mesh_pp.query("patch_origin1", cfg.mesh_patch_origin1);
   mesh_pp.query("patch_origin2", cfg.mesh_patch_origin2);
   mesh_pp.query("patch_origin3", cfg.mesh_patch_origin3);
   std::vector<int> mesh_patch_central_cells;
   mesh_pp.queryarr("patch_central_cells", mesh_patch_central_cells);
   if (!mesh_patch_central_cells.empty()) {
      if (mesh_patch_central_cells.size() != AMREX_SPACEDIM) {
         amrex::Abort("mesh.patch_central_cells must list exactly AMREX_SPACEDIM values");
      }
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         cfg.mesh_patch_central_cells[dir] = mesh_patch_central_cells[dir];
      }
   }
   mesh_pp.query("patch_angular_cells", cfg.mesh_patch_angular_cells);
   mesh_pp.query("patch_radial_cells", cfg.mesh_patch_radial_cells);
   mesh_pp.query("patch_outer_radial_cells", cfg.mesh_patch_outer_radial_cells);
   mesh_pp.query("patch_axial_cells", cfg.mesh_patch_axial_cells);
   mesh_pp.query("patch_axial_lo", cfg.mesh_patch_axial_lo);
   mesh_pp.query("patch_axial_hi", cfg.mesh_patch_axial_hi);
   mesh_pp.query("patch_pack_direction", cfg.mesh_patch_pack_direction);
   mesh_pp.query("patch_gap_cells", cfg.mesh_patch_gap_cells);
   mesh_pp.query("allow_experimental_mixed_shell", cfg.mesh_allow_experimental_mixed_shell);
   if (!mesh_patch_box_small.empty() || !mesh_patch_box_big.empty()) {
      if (mesh_patch_box_small.size() != mesh_patch_box_big.size()) {
         amrex::Abort("mesh.patch_box_small and mesh.patch_box_big must have the same number of values");
      }
      if (mesh_patch_box_small.size() % AMREX_SPACEDIM != 0) {
         amrex::Abort("mesh.patch_box_small/big values must contain nboxes * AMREX_SPACEDIM entries");
      }
      const int box_count = static_cast<int>(mesh_patch_box_small.size()) / AMREX_SPACEDIM;
      if (box_count > MaxMeshPatchBoxes) {
         amrex::Abort("too many mesh patch boxes; increase MaxMeshPatchBoxes if needed");
      }
      cfg.mesh_patch_box_count = box_count;
      for (int n = 0; n < static_cast<int>(mesh_patch_box_small.size()); ++n) {
         cfg.mesh_patch_box_small[n] = mesh_patch_box_small[n];
         cfg.mesh_patch_box_big[n] = mesh_patch_box_big[n];
      }
   }
   mesh_pp.query("metric_divergence", cfg.mesh_metric_divergence);
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(mesh_coordinate, cfg.mesh_axes);
   const int mesh_patch_system = mesh_patch_system_code(cfg.mesh_patch_system);
   amrex::ParmParse geometry_pp("geometry");
   int geometry_coord_sys = amrex_coord_sys_for_mesh_coordinate(mesh_coordinate, mesh_axes);
   const bool coord_sys_was_set = geometry_pp.query("coord_sys", geometry_coord_sys) != 0;
   if (coord_sys_was_set) {
      validate_amrex_geometry_coord_sys(mesh_coordinate, mesh_axes, geometry_coord_sys);
   } else if (mesh_coordinate_uses_amrex_builtin_metric(mesh_coordinate, mesh_axes)) {
      amrex::Abort(
         std::string("mesh.coordinate=") + mesh_coordinate_name(mesh_coordinate)
         + " requires explicit geometry.coord_sys=" + std::to_string(amrex_coord_sys_for_mesh_coordinate(mesh_coordinate, mesh_axes))
      );
   }
   validate_mesh_geometry_for_current_solver(mesh_coordinate, mesh_patch_system);
   pp.query("ic", cfg.ic);
   pp.query("passive_scalars", cfg.passive_scalars);
   pp.query("passive_profile", cfg.passive_profile);
   pp.query("passive_value", cfg.passive_value);
   pp.query("passive_background", cfg.passive_background);
   pp.query("passive_amplitude", cfg.passive_amplitude);
   pp.query("passive_left", cfg.passive_left);
   pp.query("passive_right", cfg.passive_right);
   pp.query("passive_sigma", cfg.passive_sigma);
   pp.query("passive_x0", cfg.passive_x0);
   pp.query("passive_y0", cfg.passive_y0);
   pp.query("passive_z0", cfg.passive_z0);
   pp.query("inflow_passive_value", cfg.inflow_passive_value);
   pp.query("inflow_background_passive_value", cfg.inflow_background_passive_value);
   pp.query("rho0", cfg.rho0);
   pp.query("pressure0", cfg.pressure0);
   pp.query("tov_central_density", cfg.tov_central_density);
   pp.query("tov_radial_step", cfg.tov_radial_step);
   pp.query("tov_radial_points", cfg.tov_radial_points);
   pp.query("amplitude", cfg.amplitude);
   pp.query("bump_sigma", cfg.bump_sigma);
   pp.query("acoustic_transverse_sigma", cfg.acoustic_transverse_sigma);
   pp.query("bump_x0", cfg.bump_x0);
   pp.query("bump_y0", cfg.bump_y0);
   pp.query("bump_z0", cfg.bump_z0);
   pp.query("beam_length", cfg.beam_length);
   pp.query("beam_radius", cfg.beam_radius);
   pp.query("shock_x0", cfg.shock_x0);
   pp.query("shock_y0", cfg.shock_y0);
   pp.query("shock_z0", cfg.shock_z0);
   pp.query("shock_nx", cfg.shock_nx);
   pp.query("shock_ny", cfg.shock_ny);
   pp.query("shock_nz", cfg.shock_nz);
   pp.query("shock_speed", cfg.shock_speed);
   pp.query("rho_left", cfg.rho_left);
   pp.query("pressure_left", cfg.pressure_left);
   pp.query("v1_left", cfg.v1_left);
   pp.query("v2_left", cfg.v2_left);
   pp.query("v3_left", cfg.v3_left);
   pp.query("rho_right", cfg.rho_right);
   pp.query("pressure_right", cfg.pressure_right);
   pp.query("v1_right", cfg.v1_right);
   pp.query("v2_right", cfg.v2_right);
   pp.query("v3_right", cfg.v3_right);
   pp.query("b1_left", cfg.b1_left);
   pp.query("b2_left", cfg.b2_left);
   pp.query("b3_left", cfg.b3_left);
   pp.query("b1_right", cfg.b1_right);
   pp.query("b2_right", cfg.b2_right);
   pp.query("b3_right", cfg.b3_right);
   pp.query("v1_0", cfg.v1_0);
   pp.query("v2_0", cfg.v2_0);
   pp.query("v3_0", cfg.v3_0);
   pp.query("b1_0", cfg.b1_0);
   pp.query("b2_0", cfg.b2_0);
   pp.query("b3_0", cfg.b3_0);
   pp.query("alfven_amplitude", cfg.alfven_amplitude);
   pp.query("wave_number", cfg.wave_number);
   pp.query("wave_mode_1", cfg.wave_mode_1);
   pp.query("wave_mode_2", cfg.wave_mode_2);
   pp.query("wave_mode_3", cfg.wave_mode_3);
   pp.query("wave_family", cfg.wave_family);
   pp.query("kh_secondary_amplitude", cfg.kh_secondary_amplitude);
   pp.query("kh_secondary_mode", cfg.kh_secondary_mode);
   pp.query("kh_z_mode", cfg.kh_z_mode);
   pp.query("kh_phase", cfg.kh_phase);
   pp.query("kh_envelope_sigma", cfg.kh_envelope_sigma);
   pp.query("kh_density_amplitude", cfg.kh_density_amplitude);
   pp.query("kh_perturbation", cfg.kh_perturbation);
   pp.query("rotor_radius", cfg.rotor_radius);
   pp.query("rotor_transition_radius", cfg.rotor_transition_radius);
   pp.query("rotor_density", cfg.rotor_density);
   pp.query("rotor_omega", cfg.rotor_omega);
   pp.query("recon_fine", cfg.recon_fine);
   pp.query("recon_fine_start_level", cfg.recon_fine_start_level);
   pp.query("bc", cfg.bc);
   pp.query("bc_xlo", cfg.bc_xlo);
   pp.query("bc_xhi", cfg.bc_xhi);
   pp.query("bc_ylo", cfg.bc_ylo);
   pp.query("bc_yhi", cfg.bc_yhi);
   pp.query("bc_zlo", cfg.bc_zlo);
   pp.query("bc_zhi", cfg.bc_zhi);
   pp.query("boundary_profile", cfg.boundary_profile);
   pp.query("inflow_profile", cfg.inflow_profile);
   pp.query("inflow_rho", cfg.inflow_rho);
   pp.query("inflow_pressure", cfg.inflow_pressure);
   pp.query("inflow_v1", cfg.inflow_v1);
   pp.query("inflow_v2", cfg.inflow_v2);
   pp.query("inflow_v3", cfg.inflow_v3);
   pp.query("inflow_b1", cfg.inflow_b1);
   pp.query("inflow_b2", cfg.inflow_b2);
   pp.query("inflow_b3", cfg.inflow_b3);
   pp.query("inflow_background_rho", cfg.inflow_background_rho);
   pp.query("inflow_background_pressure", cfg.inflow_background_pressure);
   pp.query("inflow_background_v1", cfg.inflow_background_v1);
   pp.query("inflow_background_v2", cfg.inflow_background_v2);
   pp.query("inflow_background_v3", cfg.inflow_background_v3);
   pp.query("inflow_background_b1", cfg.inflow_background_b1);
   pp.query("inflow_background_b2", cfg.inflow_background_b2);
   pp.query("inflow_background_b3", cfg.inflow_background_b3);
   pp.query("inflow_center_y", cfg.inflow_center_y);
   pp.query("inflow_center_z", cfg.inflow_center_z);
   pp.query("inflow_radius", cfg.inflow_radius);
   pp.query("viscosity", cfg.viscosity);
   pp.query("resistivity", cfg.resistivity);
   pp.query("diffusion_cfl", cfg.diffusion_cfl);
   pp.query("diffusion_tolerance", cfg.diffusion_tolerance);
   pp.query("diffusion_max_iterations", cfg.diffusion_max_iterations);
   pp.query("diffusion_sor_omega", cfg.diffusion_sor_omega);
   pp.query("source", cfg.source);
   pp.query("source_a1", cfg.source_a1);
   pp.query("source_a2", cfg.source_a2);
   pp.query("source_a3", cfg.source_a3);
   pp.query("source_power", cfg.source_power);
   pp.query("source_radius", cfg.source_radius);
   pp.query("source_sigma", cfg.source_sigma);
   pp.query("source_omega", cfg.source_omega);
   pp.query("source_ramp_time", cfg.source_ramp_time);
   pp.query("source_phase", cfg.source_phase);
   pp.query("source_azimuthal_mode", cfg.source_azimuthal_mode);
   pp.query("gravity", cfg.gravity);
   pp.query("gravity_coupling", cfg.gravity_coupling);
   pp.query("gravity_fixed_potential", cfg.gravity_fixed_potential);
   pp.query("gravity_constant", cfg.gravity_constant);
   pp.query("gravity_a1", cfg.gravity_a1);
   pp.query("gravity_a2", cfg.gravity_a2);
   pp.query("gravity_a3", cfg.gravity_a3);
   pp.query("gravity_omega", cfg.gravity_omega);
   pp.query("gravity_center_x", cfg.gravity_center_x);
   pp.query("gravity_center_y", cfg.gravity_center_y);
   pp.query("gravity_center_z", cfg.gravity_center_z);
   pp.query("gravity_tolerance", cfg.gravity_tolerance);
   pp.query("gravity_max_iterations", cfg.gravity_max_iterations);
   pp.query("gravity_max_fmg_iterations", cfg.gravity_max_fmg_iterations);
   pp.query("gravity_max_order", cfg.gravity_max_order);
   pp.query("gravity_verbose", cfg.gravity_verbose);
   pp.query("solid_geometry", cfg.solid_geometry);
   pp.query("solid_motion", cfg.solid_motion);
   pp.query("solid_wall_condition", cfg.solid_wall_condition);
   pp.query("solid_volume_fraction_samples_per_dim", cfg.solid_volume_fraction_samples_per_dim);
   pp.query("solid_radius", cfg.solid_radius);
   pp.query("solid_radius_amplitude", cfg.solid_radius_amplitude);
   pp.query("solid_radius_frequency", cfg.solid_radius_frequency);
   pp.query("solid_radius_phase", cfg.solid_radius_phase);
   pp.query("solid_radius_ramp_time", cfg.solid_radius_ramp_time);
   pp.query("solid_swept_reference_time", cfg.solid_swept_reference_time);
   pp.query("solid_center_x", cfg.solid_center_x);
   pp.query("solid_center_y", cfg.solid_center_y);
   pp.query("solid_center_z", cfg.solid_center_z);
   pp.query("solid_velocity_1", cfg.solid_velocity_1);
   pp.query("solid_velocity_2", cfg.solid_velocity_2);
   pp.query("solid_velocity_3", cfg.solid_velocity_3);
   pp.query("solid_angular_velocity_1", cfg.solid_angular_velocity_1);
   pp.query("solid_angular_velocity_2", cfg.solid_angular_velocity_2);
   pp.query("solid_angular_velocity_3", cfg.solid_angular_velocity_3);
   pp.query("solid_wedge_x0", cfg.solid_wedge_x0);
   pp.query("solid_wedge_y0", cfg.solid_wedge_y0);
   pp.query("solid_wedge_angle_deg", cfg.solid_wedge_angle_deg);
   std::vector<amrex::Real> solid_box_lo;
   std::vector<amrex::Real> solid_box_hi;
   pp.queryarr("solid_box_lo", solid_box_lo);
   pp.queryarr("solid_box_hi", solid_box_hi);
   if (!solid_box_lo.empty() || !solid_box_hi.empty()) {
      if (solid_box_lo.size() != solid_box_hi.size()) {
         amrex::Abort("pyfluid.solid_box_lo and solid_box_hi must have the same number of values");
      }
      if (solid_box_lo.size() % AMREX_SPACEDIM != 0) {
         amrex::Abort("pyfluid.solid_box_lo/hi values must contain nboxes * AMREX_SPACEDIM entries");
      }
      const int box_count = static_cast<int>(solid_box_lo.size()) / AMREX_SPACEDIM;
      if (box_count > MaxSolidBoxes) {
         amrex::Abort("too many solid boxes; increase MaxSolidBoxes if needed");
      }
      cfg.solid_box_count = box_count;
      for (int n = 0; n < static_cast<int>(solid_box_lo.size()); ++n) {
         cfg.solid_box_lo[n] = solid_box_lo[n];
         cfg.solid_box_hi[n] = solid_box_hi[n];
      }
   }
   amrex::ParmParse amr_pp("amr");
   query_registered_amr_parameters(amr_pp, cfg);
   std::vector<amrex::Real> initial_refine_box_lo;
   std::vector<amrex::Real> initial_refine_box_hi;
   amr_pp.queryarr("initial_refine_box_lo", initial_refine_box_lo);
   amr_pp.queryarr("initial_refine_box_hi", initial_refine_box_hi);
   if (!initial_refine_box_lo.empty() || !initial_refine_box_hi.empty()) {
      if (initial_refine_box_lo.size() != initial_refine_box_hi.size()) {
         amrex::Abort("amr.initial_refine_box_lo and amr.initial_refine_box_hi must have the same number of values");
      }
      if (initial_refine_box_lo.size() % AMREX_SPACEDIM != 0) {
         amrex::Abort("amr.initial_refine_box_lo/hi values must contain nboxes * AMREX_SPACEDIM entries");
      }
      const int box_count = static_cast<int>(initial_refine_box_lo.size()) / AMREX_SPACEDIM;
      if (box_count > MaxInitialRefineBoxes) {
         amrex::Abort("too many AMR initial refinement boxes; increase MaxInitialRefineBoxes if needed");
      }
      cfg.initial_refine_box_count = box_count;
      for (int n = 0; n < static_cast<int>(initial_refine_box_lo.size()); ++n) {
         cfg.initial_refine_box_lo[n] = initial_refine_box_lo[n];
         cfg.initial_refine_box_hi[n] = initial_refine_box_hi[n];
      }
   }
   if (cfg.tag_metric == "lohner" && !cfg.tag_threshold_set) {
      cfg.tag_threshold = amr_default_lohner_tag_threshold();
   }
   if (cfg.level_cell_budget_side > 0) {
      amrex::Long derived_cells = 1;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         derived_cells *= cfg.level_cell_budget_side;
      }
      if (cfg.level_cell_budget > 0 && cfg.level_cell_budget != derived_cells) {
         amrex::Abort("amr.level_cell_budget and amr.level_cell_budget_side specify inconsistent budgets");
      }
      cfg.level_cell_budget = derived_cells;
   }
   sync_registered_amr_parameters(amr_pp, cfg);
   return cfg;
}

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

inline bool spacetime_matter_is_coupled(RunConfig const& cfg) noexcept
{
   return cfg.spacetime_evolution == "evolved"
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

inline void validate_config(RunConfig const& cfg)
{
   validate_registered_parameters(cfg);
   require_supported_option(
      cfg.spacetime_evolution,
      SpacetimeEvolutionOptions,
      "spacetime.evolution"
   );
   require_supported_option(
      cfg.spacetime_matter_coupling,
      SpacetimeMatterCouplingOptions,
      "spacetime.matter_coupling"
   );
   require_supported_option(
      cfg.spacetime_fixed_metric,
      FixedSpacetimeMetricOptions,
      "spacetime.fixed_metric"
   );
   require_supported_option(
      cfg.spacetime_initial_data,
      SpacetimeInitialDataOptions,
      "spacetime.initial_data"
   );
   require_supported_option(
      cfg.initial_data_magnetic,
      MagneticInitialDataOptions,
      "initial_data.magnetic"
   );
   require_supported_option(
      cfg.initial_data_spacetime,
      SpacetimeGroupInitialDataOptions,
      "initial_data.spacetime"
   );
   require_supported_option(
      cfg.initial_data_lapse,
      LapseInitialDataOptions,
      "initial_data.lapse"
   );
   require_supported_option(
      cfg.initial_data_shift,
      ShiftInitialDataOptions,
      "initial_data.shift"
   );
   require_supported_option(
      cfg.spacetime_lapse_gauge,
      Z4cLapseGaugeOptions,
      "spacetime.lapse_gauge"
   );
   require_supported_option(
      cfg.spacetime_shift_gauge,
      Z4cShiftGaugeOptions,
      "spacetime.shift_gauge"
   );
   if (cfg.gamma <= 1.0) {
      amrex::Abort("pyfluid.gamma must be greater than 1");
   }
   if (cfg.polytropic_k <= 0.0) {
      amrex::Abort("pyfluid.polytropic_k must be positive");
   }
   if (cfg.polytropic_gamma <= 1.0) {
      amrex::Abort("pyfluid.polytropic_gamma must be greater than 1");
   }
   if (cfg.polytropic_density_breaks.size()
       != cfg.polytropic_segment_gammas.size()) {
      amrex::Abort(
         "pyfluid.polytropic_density_breaks and polytropic_segment_gammas "
         "must have equal lengths"
      );
   }
   if (cfg.polytropic_density_breaks.size()
       >= static_cast<std::size_t>(MaxPolytropeSegments)) {
      amrex::Abort("piecewise polytropes support at most 8 density segments");
   }
   for (std::size_t segment = 0; segment < cfg.polytropic_density_breaks.size(); ++segment) {
      if (!(cfg.polytropic_density_breaks[segment] > 0.0)
          || (segment > 0
              && !(cfg.polytropic_density_breaks[segment]
                   > cfg.polytropic_density_breaks[segment - 1]))) {
         amrex::Abort(
            "pyfluid.polytropic_density_breaks must be positive and strictly increasing"
         );
      }
      if (!(cfg.polytropic_segment_gammas[segment] > 1.0)) {
         amrex::Abort(
            "pyfluid.polytropic_segment_gammas entries must be greater than 1"
         );
      }
   }
   if (cfg.thermal_gamma <= 1.0) {
      amrex::Abort("pyfluid.thermal_gamma must be greater than 1");
   }
   if (cfg.rho_floor <= 0.0 || cfg.pressure_floor <= 0.0) {
      amrex::Abort("pyfluid.rho_floor and pyfluid.pressure_floor must be positive");
   }
   if (cfg.passive_scalars < 0 || cfg.passive_scalars > MaxPassiveScalars) {
      amrex::Abort("pyfluid.passive_scalars currently supports 0 or 1");
   }
   require_supported_option(cfg.passive_profile, PassiveProfileOptions, "pyfluid.passive_profile");
   if (cfg.passive_sigma == 0.0 || cfg.passive_sigma < -1.0) {
      amrex::Abort("pyfluid.passive_sigma must be positive, or -1 to reuse pyfluid.bump_sigma");
   }
   require_supported_option(cfg.profile, ProfileOptions, "pyfluid.profile");
   require_supported_option(cfg.profile_detail, ProfileDetailOptions, "pyfluid.profile_detail");
   require_supported_option(cfg.mesh_patch_box_ownership, MeshPatchBoxOwnershipOptions, "mesh.patch_box_ownership");
   require_supported_option(cfg.mesh_patch_box_domain, MeshPatchBoxDomainOptions, "mesh.patch_box_domain");
   if (cfg.physical_check_int < 0) {
      amrex::Abort("pyfluid.physical_check_int must be non-negative; 0 disables accepted-state physical checks");
   }
   const PhysicalSystem selected_system = physical_system(cfg);
   if (physical_system_has_magnetic_field(selected_system)
       && cfg.initial_data_magnetic == "none") {
      amrex::Abort(
         "initial_data.magnetic = none requires a non-magnetized physical system; "
         "use zero to initialize an allocated magnetic field to zero"
      );
   }
   if (cfg.spacetime_evolution == "evolved"
       && (cfg.initial_data_spacetime == "none"
           || cfg.initial_data_lapse == "none"
           || cfg.initial_data_shift == "none")) {
      amrex::Abort(
         "evolved spacetime requires initial_data.spacetime, lapse, and shift storage; "
         "select id or the group-specific neutral value"
      );
   }
   if (!physical_system_is_relativistic(selected_system)
       && cfg.spacetime_fixed_metric != "minkowski") {
      amrex::Abort("spacetime.fixed_metric applies only to RHD/RMHD");
   }
   if (cfg.ct == "on" && !physical_system_supports_ct(selected_system)) {
      amrex::Abort("pyfluid.ct = on requires a magnetic physical system");
   }
   if (!valid_spacetime_evolution(selected_system, spacetime_evolution_mode(cfg))) {
      amrex::Abort(
         "invalid pyfluid.physics/spacetime.evolution combination: Newtonian hydro/MHD use none; "
         "RHD/RMHD use fixed or evolved"
      );
   }
   if (physical_system_is_relativistic(selected_system)) {
      if (selected_system == PhysicalSystem::RMHD && cfg.eos != "ideal_gas") {
         amrex::Abort("RMHD runtime evolution currently requires pyfluid.eos = ideal_gas");
      }
      if (cfg.passive_scalars != 0 || cfg.diffusion != "none"
          || cfg.source != "none" || cfg.gravity != "none" || cfg.solid_geometry != "off") {
         amrex::Abort(
            "relativistic-fluid runtime evolution currently supports no passive scalars, diffusion, "
            "Newtonian sources/gravity, or solid geometry"
         );
      }
      if (cfg.mesh_coordinate != "cartesian" || cfg.mesh_patch_system != "single") {
         amrex::Abort("relativistic-fluid runtime evolution currently supports a single Cartesian patch");
      }
      if (cfg.recon == "wenoz-char" || cfg.recon == "mp5-char" || cfg.recon == "teno5-char"
          || cfg.recon_fine == "wenoz-char" || cfg.recon_fine == "mp5-char"
          || cfg.recon_fine == "teno5-char") {
         amrex::Abort("relativistic characteristic reconstruction is not implemented; select a component method");
      }
      if (cfg.riemann != "llf" && cfg.riemann != "hll" && cfg.riemann != "hlle") {
         amrex::Abort("relativistic-fluid runtime evolution supports pyfluid.riemann = llf, hll, or hlle");
      }
      if (cfg.ic == "fixed_metric_hse" && cfg.eos != "ideal_gas") {
         amrex::Abort("fixed_metric_hse currently requires pyfluid.eos = ideal_gas");
      }
      if (cfg.ic != "uniform" && cfg.ic != "sin_wave" && cfg.ic != "sod_x"
          && cfg.ic != "sod_plane" && cfg.ic != "sod_periodic_plane"
          && cfg.ic != "fixed_metric_hse" && cfg.ic != "tov") {
         amrex::Abort("relativistic-fluid runtime evolution supports uniform, sin_wave, Sod, fixed-metric HSE, and TOV initial data");
      }
      if (cfg.ic == "fixed_metric_hse" && cfg.spacetime_fixed_metric == "minkowski") {
         amrex::Abort("pyfluid.ic = fixed_metric_hse requires a non-Minkowski fixed metric");
      }
      if (cfg.ic == "tov") {
         if (selected_system != PhysicalSystem::RHD
             || cfg.spacetime_evolution != "evolved"
             || (cfg.eos != "polytrope" && cfg.eos != "hybrid_polytrope")) {
            amrex::Abort(
               "pyfluid.ic = tov requires RHD, evolved spacetime, and a polytrope or hybrid_polytrope EOS"
            );
         }
         if (!(cfg.tov_central_density > cfg.rho_floor)
             || !(cfg.tov_radial_step > 0.0) || cfg.tov_radial_points < 16) {
            amrex::Abort(
               "TOV initial data requires central_density > rho_floor, radial_step > 0, and radial_points >= 16"
            );
         }
      }
      if (cfg.spacetime_evolution == "evolved") {
         if (cfg.ng < 2) {
            amrex::Abort(
               "spacetime.evolution = evolved requires pyfluid.ng >= 2 for fourth-order cell/vertex coupling"
            );
         }
         if (cfg.spacetime_fixed_metric != "minkowski") {
            amrex::Abort("spacetime.fixed_metric must be minkowski when spacetime.evolution = evolved");
         }
         if (cfg.spacetime_amplitude < 0.0 || cfg.spacetime_wave_number <= 0.0) {
            amrex::Abort("evolved spacetime amplitude must be non-negative and wave_number positive");
         }
         if (cfg.spacetime_slow_start_time <= 0.0) {
            amrex::Abort("spacetime.slow_start_time must be positive");
         }
         if (cfg.spacetime_ko_dissipation < 0.0) {
            amrex::Abort("spacetime.ko_dissipation must be nonnegative");
         }
      } else if (cfg.spacetime_matter_coupling != "coupled") {
         amrex::Abort(
            "spacetime.matter_coupling = vacuum is only meaningful for evolved spacetime"
         );
      }
   }
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(mesh_coordinate, cfg.mesh_axes);
   const int mesh_patch_system = mesh_patch_system_code(cfg.mesh_patch_system);
   bool any_patch_count = false;
   bool all_patch_counts_positive = true;
   int patch_count_product = 1;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const int count = cfg.mesh_patch_counts[dir];
      any_patch_count = any_patch_count || count != 0;
      all_patch_counts_positive = all_patch_counts_positive && count > 0;
      patch_count_product *= std::max(count, 1);
   }
   const bool has_patch_boxes = cfg.mesh_patch_box_count > 0;
   const bool mixed_patch_system = mesh_patch_system == MeshPatchMixedThornburg04
      || mesh_patch_system == MeshPatchMixedThornburg13
      || mesh_patch_system == MeshPatchMixedCylindrical5
      || mesh_patch_system == MeshPatchMixedCylindrical9;
   if (mesh_patch_system == MeshPatchSingle && (any_patch_count || has_patch_boxes)) {
      amrex::Abort("mesh.patch_counts/patch_box_* require mesh.patch_system = multi_homogeneous_cartesian");
   }
   if (cfg.mesh_patch_overlap_cells < 0) {
      amrex::Abort("mesh.patch_overlap_cells must be non-negative");
   }
   if (mesh_patch_system == MeshPatchSingle && cfg.mesh_patch_overlap_cells != 0) {
      amrex::Abort("mesh.patch_overlap_cells requires mesh.patch_system = multi_homogeneous_cartesian");
   }
   if (!has_patch_boxes && cfg.mesh_patch_box_ownership != "exact") {
      amrex::Abort("mesh.patch_box_ownership requires explicit mesh.patch_box_* values");
   }
   if (!has_patch_boxes && cfg.mesh_patch_box_domain != "rectangular") {
      amrex::Abort("mesh.patch_box_domain requires explicit mesh.patch_box_* values");
   }
   if (mesh_patch_system == MeshPatchHomogeneousCartesian) {
      if (mesh_coordinate != MeshCartesian) {
         amrex::Abort("mesh.patch_system = multi_homogeneous_cartesian requires mesh.coordinate = cartesian");
      }
#if (AMREX_SPACEDIM == 1)
      amrex::Abort("Cartesian multipatch production runs are supported for 2D and 3D executables");
#endif
      if (any_patch_count && !all_patch_counts_positive) {
         amrex::Abort("mesh.patch_counts must be either omitted or positive in every active direction");
      }
      if (any_patch_count && patch_count_product <= 1) {
         amrex::Abort("mesh.patch_counts must describe at least two Cartesian patches");
      }
      if (any_patch_count && has_patch_boxes) {
         amrex::Abort("mesh.patch_counts and mesh.patch_box_* are mutually exclusive");
      }
      if (any_patch_count && cfg.mesh_patch_box_ownership != "exact") {
         amrex::Abort("mesh.patch_box_ownership only applies to explicit mesh.patch_box_* layouts");
      }
      if (has_patch_boxes && cfg.mesh_patch_overlap_cells != 0) {
         amrex::Abort("mesh.patch_overlap_cells applies to mesh.patch_counts/default tiled layouts, not explicit mesh.patch_box_* layouts");
      }
      if (has_patch_boxes && cfg.mesh_patch_box_count <= 1) {
         amrex::Abort("mesh.patch_box_* must describe at least two Cartesian patches");
      }
      if (cfg.mesh_patch_box_domain == "union") {
         if (!has_patch_boxes) {
            amrex::Abort("mesh.patch_box_domain = union requires explicit mesh.patch_box_* values");
         }
         if (cfg.mesh_patch_box_ownership != "exact") {
            amrex::Abort("mesh.patch_box_domain = union currently requires mesh.patch_box_ownership = exact");
         }
         if (cfg.physics != "hydro") {
            amrex::Abort("mesh.patch_box_domain = union currently supports pure HD only");
         }
      }
      for (int box = 0; box < cfg.mesh_patch_box_count; ++box) {
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const int idx = box * AMREX_SPACEDIM + dir;
            if (cfg.mesh_patch_box_small[idx] > cfg.mesh_patch_box_big[idx]) {
               amrex::Abort("mesh.patch_box_small entries must be <= mesh.patch_box_big entries");
            }
         }
      }
   }
   if (mixed_patch_system) {
      if (!cfg.mesh_allow_experimental_mixed_shell) {
         amrex::Abort(
            "mixed_thornburg*/mixed_cylindrical* currently select true radial curvilinear shell maps, "
            "but production mixed Cartesian-curvilinear evolution still needs active-cell ownership and "
            "conservative overset/seam reconciliation. Use mesh.allow_experimental_mixed_shell = 1 only "
            "for map/prototype tests."
         );
      }
      if (mesh_coordinate != MeshCartesian) {
         amrex::Abort("mixed patch systems use mesh.coordinate = cartesian; patch maps provide the local curved coordinates");
      }
      if (cfg.physics != "hydro") {
         amrex::Abort("mixed patch production evolution is currently enabled for pure HD only");
      }
      if (cfg.diffusion != "none") {
         amrex::Abort("mixed patch production evolution does not yet support diffusion");
      }
      if (cfg.gravity != "none") {
         amrex::Abort("mixed patch production evolution does not yet support gravity sources");
      }
      if (cfg.solid_geometry != "off") {
         amrex::Abort("mixed patch production evolution does not yet support solid geometry");
      }
      if (any_patch_count || has_patch_boxes || cfg.mesh_patch_overlap_cells != 0) {
         amrex::Abort("mixed patch systems use mesh.patch_* grid parameters, not Cartesian patch_counts/patch_box_*");
      }
      if (cfg.mesh_patch_inner_radius <= 0.0 || cfg.mesh_patch_outer_radius <= cfg.mesh_patch_inner_radius) {
         amrex::Abort("mesh.patch_inner_radius and mesh.patch_outer_radius must satisfy 0 < inner < outer");
      }
      if (
         (mesh_patch_system == MeshPatchMixedThornburg13 || mesh_patch_system == MeshPatchMixedCylindrical9)
         && (
            cfg.mesh_patch_medium_radius <= cfg.mesh_patch_inner_radius
            || cfg.mesh_patch_outer_radius <= cfg.mesh_patch_medium_radius
         )
      ) {
         amrex::Abort("13/9-patch systems require inner < medium < outer radii");
      }
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         if (cfg.mesh_patch_central_cells[dir] <= 0) {
            amrex::Abort("mesh.patch_central_cells must be positive in every active direction");
         }
      }
      if (
         cfg.mesh_patch_angular_cells <= 0
         || cfg.mesh_patch_radial_cells <= 0
         || cfg.mesh_patch_outer_radial_cells <= 0
         || cfg.mesh_patch_axial_cells <= 0
      ) {
         amrex::Abort("mesh.patch_*_cells values must be positive");
      }
      if (cfg.mesh_patch_axial_hi <= cfg.mesh_patch_axial_lo) {
         amrex::Abort("mesh.patch_axial_hi must be larger than mesh.patch_axial_lo");
      }
      if (cfg.mesh_patch_pack_direction < 0 || cfg.mesh_patch_pack_direction >= AMREX_SPACEDIM) {
         amrex::Abort("mesh.patch_pack_direction is outside the executable dimension");
      }
      if (cfg.mesh_patch_gap_cells != 0) {
         amrex::Abort("mixed patch production evolution currently requires mesh.patch_gap_cells = 0");
      }
   }
   if (curvilinear_mesh_coordinate(mesh_coordinate)) {
      if (mesh_patch_system != MeshPatchSingle) {
         amrex::Abort("curvilinear cylindrical/spherical coordinates currently require mesh.patch_system = single");
      }
      if (cfg.gravity != "none") {
         amrex::Abort("curvilinear cylindrical/spherical gravity is not implemented yet");
      }
      if (cfg.solid_geometry != "off") {
         amrex::Abort("curvilinear cylindrical/spherical solid geometry is not implemented yet");
      }
      if (
         mesh_has_axis(mesh_axes, MeshAxisR)
         && (
            side_bc_name(cfg, mesh_direction_for_axis(mesh_axes, MeshAxisR), false) == "periodic"
            || side_bc_name(cfg, mesh_direction_for_axis(mesh_axes, MeshAxisR), true) == "periodic"
         )
      ) {
         amrex::Abort("curvilinear cylindrical/spherical hydro cannot use periodic radial boundary conditions");
      }
      if (
         mesh_coordinate == MeshSpherical
         && mesh_has_axis(mesh_axes, MeshAxisTheta)
         && (
            side_bc_name(cfg, mesh_direction_for_axis(mesh_axes, MeshAxisTheta), false) == "periodic"
            || side_bc_name(cfg, mesh_direction_for_axis(mesh_axes, MeshAxisTheta), true) == "periodic"
         )
      ) {
         amrex::Abort("spherical hydro cannot use periodic theta boundary conditions");
      }
   } else if (mesh_coordinate != MeshCartesian) {
      amrex::Abort("only Cartesian and curvilinear cylindrical/spherical mesh coordinates are currently supported");
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const bool lo_origin = side_bc_name(cfg, dir, false) == "origin";
      const bool hi_origin = side_bc_name(cfg, dir, true) == "origin";
      if (!lo_origin && !hi_origin) {
         continue;
      }
      if (!curvilinear_mesh_coordinate(mesh_coordinate) || !mesh_has_axis(mesh_axes, MeshAxisR)) {
         amrex::Abort("pyfluid.bc_* = origin requires cylindrical/spherical coordinates with an active radial axis");
      }
      const int radial_dir = mesh_direction_for_axis(mesh_axes, MeshAxisR);
      if (dir != radial_dir || hi_origin) {
         amrex::Abort("pyfluid.bc_* = origin is only valid on the lower radial boundary");
      }
   }
   require_supported_option(cfg.ic, InitialDataOptions, "pyfluid.ic");
   if (cfg.ic == "advection_medley") {
      if (cfg.physics != "hydro" || mesh_coordinate != MeshCartesian
          || AMREX_SPACEDIM != 1 || mesh_axis(mesh_axes, 0) != MeshAxisX) {
         amrex::Abort("advection_medley requires 1D Cartesian x hydro");
      }
      if (cfg.rho0 <= 0.0 || cfg.pressure0 <= 0.0 || cfg.amplitude <= 0.0) {
         amrex::Abort("advection_medley requires positive rho0, pressure0, and amplitude");
      }
   }
   if (cfg.ic == "characteristic_medley") {
      if (cfg.physics != "hydro" || cfg.eos != "ideal_gas" || mesh_coordinate != MeshCartesian
          || AMREX_SPACEDIM != 1 || mesh_axis(mesh_axes, 0) != MeshAxisX) {
         amrex::Abort("characteristic_medley requires 1D Cartesian x ideal-gas hydro");
      }
      if (cfg.rho0 <= 0.0 || cfg.pressure0 <= 0.0 || cfg.amplitude <= 0.0 || cfg.v1_0 != 0.0) {
         amrex::Abort("characteristic_medley requires positive rho0, pressure0, and amplitude with v1_0 = 0");
      }
   }
   if (cfg.ic == "mhd_characteristic_medley") {
      if (cfg.physics != "mhd" || cfg.eos != "ideal_gas" || mesh_coordinate != MeshCartesian
          || AMREX_SPACEDIM != 1 || mesh_axis(mesh_axes, 0) != MeshAxisX) {
         amrex::Abort("mhd_characteristic_medley requires 1D Cartesian x ideal-gas MHD");
      }
      const amrex::Real transverse_b2 = cfg.b2_0 * cfg.b2_0 + cfg.b3_0 * cfg.b3_0;
      if (cfg.rho0 <= 0.0 || cfg.pressure0 <= 0.0 || cfg.amplitude <= 0.0
          || cfg.v1_0 != 0.0 || cfg.v2_0 != 0.0 || cfg.v3_0 != 0.0
          || cfg.b1_0 == 0.0 || transverse_b2 == 0.0) {
         amrex::Abort(
            "mhd_characteristic_medley requires positive rho0/pressure0/amplitude, zero background velocity, "
            "and nonzero normal and transverse magnetic fields"
         );
      }
   }
   if (cfg.ic == "sod_plane" || cfg.ic == "sod_periodic_plane") {
      amrex::Real shock_normal2 = cfg.shock_nx * cfg.shock_nx;
#if (AMREX_SPACEDIM >= 2)
      shock_normal2 += cfg.shock_ny * cfg.shock_ny;
#endif
#if (AMREX_SPACEDIM == 3)
      shock_normal2 += cfg.shock_nz * cfg.shock_nz;
#endif
      if (shock_normal2 <= 0.0) {
         amrex::Abort("sod_plane requires a nonzero active shock normal");
      }
   }
   if (cfg.physics == "mhd" && cfg.ic == "acoustic_wave_x") {
      amrex::Abort("acoustic_wave_x is hydro-only; use a magnetosonic-wave IC for MHD EOS validation");
   }
   if (cfg.ic == "acoustic_packet_x") {
      if (cfg.physics != "hydro") {
         amrex::Abort("acoustic_packet_x is hydro-only");
      }
      if (cfg.bump_sigma <= 0.0) {
         amrex::Abort("acoustic_packet_x requires positive pyfluid.bump_sigma");
      }
      if (cfg.acoustic_transverse_sigma < 0.0) {
         amrex::Abort("acoustic_packet_x requires non-negative pyfluid.acoustic_transverse_sigma; zero selects a planar packet");
      }
      if (cfg.v1_0 != 0.0 || cfg.v2_0 != 0.0 || cfg.v3_0 != 0.0) {
         amrex::Abort("acoustic_packet_x currently requires zero background velocity");
      }
   }
   if (cfg.ic == "linear_wave_x") {
      if (cfg.physics == "hydro" && !valid_hydro_wave_family(cfg.wave_family)) {
         amrex::Abort("hydro linear_wave_x supports wave_family " + supported_option_names(HydroWaveFamilyOptions));
      }
      if (cfg.physics == "mhd" && !valid_mhd_wave_family(cfg.wave_family)) {
         amrex::Abort("MHD linear_wave_x supports wave_family " + supported_option_names(MHDWaveFamilyOptions));
      }
   }
   if (cfg.physics == "mhd" && cfg.ic == "jeans_wave_x") {
      amrex::Abort("jeans_wave_x is hydro-only");
   }
   if (cfg.ic == "jeans_wave_x" && cfg.gravity != "poisson") {
      amrex::Abort("jeans_wave_x requires pyfluid.gravity = poisson");
   }
   if (cfg.ic == "hse_polytrope_x"
      && (cfg.gravity != "fixed" || cfg.gravity_fixed_potential != "uniform_acceleration" || cfg.gravity_a1 == 0.0)) {
      amrex::Abort("hse_polytrope_x requires pyfluid.gravity=fixed uniform_acceleration with nonzero gravity_a1");
   }
   if (cfg.physics == "mhd" && cfg.ic == "shear_wave_x") {
      amrex::Abort("shear_wave_x is hydro-only");
   }
   if (cfg.physics == "mhd" && cfg.ic == "hydro_beam") {
      amrex::Abort("hydro_beam is hydro-only");
   }
   if (cfg.physics == "hydro" && cfg.ic == "circular_alfven_x") {
      amrex::Abort("circular_alfven_x requires pyfluid.physics = mhd");
   }
   if (cfg.physics == "hydro" && cfg.ic == "resistive_wave_x") {
      amrex::Abort("resistive_wave_x requires pyfluid.physics = mhd");
   }
   if (cfg.physics == "hydro" && cfg.ic == "magnetic_tube_z") {
      amrex::Abort("magnetic_tube_z requires pyfluid.physics = mhd");
   }
   if (cfg.physics == "hydro" && cfg.ic == "mhd_rotor") {
      amrex::Abort("mhd_rotor requires pyfluid.physics = mhd");
   }
   if (cfg.physics == "hydro" && cfg.ic == "kelvin_helmholtz_mhd") {
      amrex::Abort("kelvin_helmholtz_mhd requires pyfluid.physics = mhd");
   }
   if (cfg.physics == "hydro" && cfg.ic == "orszag_tang") {
      amrex::Abort("orszag_tang requires pyfluid.physics = mhd");
   }
   if (cfg.physics == "mhd" && cfg.ic == "kelvin_helmholtz") {
      amrex::Abort("kelvin_helmholtz is hydro-only; use kelvin_helmholtz_mhd for magnetized KH");
   }
   if (cfg.physics == "hydro" && cfg.ic == "magnetic_blast_noise") {
      amrex::Abort("magnetic_blast_noise requires pyfluid.physics = mhd");
   }
   if (cfg.ic == "resistive_current_sheet" || cfg.ic == "reconnecting_shear_layer") {
      if (cfg.physics != "mhd" || cfg.ct != "on") {
         amrex::Abort(cfg.ic + " requires pyfluid.physics = mhd and pyfluid.ct = on");
      }
      if (mesh_coordinate_code(cfg.mesh_coordinate) != MeshCartesian) {
         amrex::Abort(cfg.ic + " requires Cartesian coordinates");
      }
      const MeshAxes axes = mesh_axes_layout(cfg);
      if (mesh_axis(axes, 0) != MeshAxisX || mesh_axis(axes, 1) != MeshAxisY) {
         amrex::Abort(cfg.ic + " requires active axes x y or x y z");
      }
      if (cfg.bump_sigma <= 0.0 || cfg.wave_number <= 0.0 || cfg.alfven_amplitude < 0.0) {
         amrex::Abort(
            cfg.ic + " requires positive bump_sigma/wave_number and non-negative alfven_amplitude"
         );
      }
      if (cfg.b1_0 == 0.0) {
         amrex::Abort(cfg.ic + " requires nonzero pyfluid.b1_0");
      }
      if (cfg.ic == "resistive_current_sheet") {
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (side_bc_name(cfg, dir, false) != "periodic" || side_bc_name(cfg, dir, true) != "periodic") {
               amrex::Abort("resistive_current_sheet requires periodic boundaries in every active direction");
            }
         }
      } else {
         if (cfg.kh_secondary_mode <= 0.0) {
            amrex::Abort("reconnecting_shear_layer requires pyfluid.kh_secondary_mode > 0");
         }
         if (side_bc_name(cfg, 0, false) != "periodic" || side_bc_name(cfg, 0, true) != "periodic") {
            amrex::Abort("reconnecting_shear_layer requires periodic boundaries along x");
         }
         if (side_bc_name(cfg, 1, false) == "periodic" || side_bc_name(cfg, 1, true) == "periodic") {
            amrex::Abort("reconnecting_shear_layer requires non-periodic boundaries normal to the sheet");
         }
      }
   }
   if (cfg.ic == "orszag_tang") {
      if (mesh_coordinate_code(cfg.mesh_coordinate) != MeshCartesian) {
         amrex::Abort("orszag_tang requires Cartesian coordinates");
      }
      const MeshAxes axes = mesh_axes_layout(cfg);
      if (mesh_axis(axes, 0) != MeshAxisX || mesh_axis(axes, 1) != MeshAxisY) {
         amrex::Abort("orszag_tang requires active axes x y or x y z");
      }
      if (cfg.wave_number <= 0.0 || cfg.amplitude <= 0.0 || cfg.b1_0 == 0.0) {
         amrex::Abort("orszag_tang requires positive wave_number/amplitude and nonzero b1_0");
      }
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         if (side_bc_name(cfg, dir, false) != "periodic" || side_bc_name(cfg, dir, true) != "periodic") {
            amrex::Abort("orszag_tang requires periodic boundaries in every active direction");
         }
      }
   }
#if (AMREX_SPACEDIM == 1)
   if (cfg.ic == "kelvin_helmholtz" || cfg.ic == "kelvin_helmholtz_mhd") {
      amrex::Abort("Kelvin-Helmholtz initial data requires the 2D or 3D AMReX executable");
   }
   if (cfg.ic == "resistive_current_sheet" || cfg.ic == "reconnecting_shear_layer") {
      amrex::Abort(cfg.ic + " requires the 2D or 3D AMReX executable");
   }
   if (cfg.ic == "orszag_tang") {
      amrex::Abort("orszag_tang requires the 2D or 3D AMReX executable");
   }
   if (cfg.ic == "hydro_beam") {
      amrex::Abort("hydro_beam requires the 2D or 3D AMReX executable");
   }
#endif
#if (AMREX_SPACEDIM != 3)
   if (cfg.ic == "magnetic_blast_noise") {
      amrex::Abort("magnetic_blast_noise requires the 3D AMReX executable");
   }
#endif
   if (!cfg.recon_fine.empty() && !valid_recon_name(cfg.recon_fine)) {
      amrex::Abort("pyfluid.recon_fine supports " + supported_option_names(ReconOptions));
   }
   if (!cfg.recon_fine.empty() && cfg.recon_fine_start_level < 0) {
      amrex::Abort("pyfluid.recon_fine_start_level must be nonnegative when pyfluid.recon_fine is set");
   }
   if (cfg.physics == "mhd" && cfg.riemann == "hllc") {
      amrex::Abort("pyfluid.riemann = hllc is hydro-only; use llf, hll, hlle, or hlld for MHD");
   }
   if (cfg.physics == "hydro" && cfg.riemann == "hlld") {
      amrex::Abort("pyfluid.riemann = hlld is MHD-only; use llf, hll, hlle, or hllc for hydro");
   }
   if (!valid_bc_name(cfg.bc)) {
      amrex::Abort("pyfluid.bc supports " + supported_option_names(BoundaryOptions));
   }
   for (std::string const& side_bc : {cfg.bc_xlo, cfg.bc_xhi, cfg.bc_ylo, cfg.bc_yhi, cfg.bc_zlo, cfg.bc_zhi}) {
      const std::string name = trim_copy(side_bc);
      if (!name.empty() && !valid_bc_name(name)) {
         amrex::Abort("side-specific pyfluid.bc_* supports " + supported_option_names(BoundaryOptions));
      }
   }
   require_supported_option(cfg.boundary_profile, BoundaryProfileOptions, "pyfluid.boundary_profile");
   bool has_analytic_boundary = false;
   for (std::string const& side_bc : {cfg.bc_xlo, cfg.bc_xhi, cfg.bc_ylo, cfg.bc_yhi, cfg.bc_zlo, cfg.bc_zhi}) {
      has_analytic_boundary = has_analytic_boundary || trim_copy(side_bc) == "analytic";
   }
   has_analytic_boundary = has_analytic_boundary || cfg.bc == "analytic";
   if (has_analytic_boundary && cfg.boundary_profile == "none") {
      amrex::Abort("analytic boundary conditions require pyfluid.boundary_profile");
   }
   if (!has_analytic_boundary && cfg.boundary_profile != "none") {
      amrex::Abort("pyfluid.boundary_profile requires at least one analytic boundary");
   }
   if (cfg.boundary_profile == "double_mach_reflection") {
#if (AMREX_SPACEDIM != 2)
      amrex::Abort("double_mach_reflection requires the 2D AMReX executable");
#endif
      if (cfg.mesh_coordinate != "cartesian" || cfg.physics != "hydro") {
         amrex::Abort("double_mach_reflection requires Cartesian hydro");
      }
      if (cfg.ic != "double_mach_reflection") {
         amrex::Abort("double_mach_reflection boundary profile requires the matching initial condition");
      }
      if (
         side_boundary_kind(cfg, 0, false) != BoundaryKind::Analytic
         || side_boundary_kind(cfg, 0, true) != BoundaryKind::Outflow
         || side_boundary_kind(cfg, 1, false) != BoundaryKind::Analytic
         || side_boundary_kind(cfg, 1, true) != BoundaryKind::Analytic
      ) {
         amrex::Abort(
            "double_mach_reflection requires analytic xlo/ylo/yhi boundaries and outflow xhi"
         );
      }
      if (cfg.shock_speed <= 0.0) {
         amrex::Abort("double_mach_reflection requires positive pyfluid.shock_speed");
      }
   } else if (cfg.ic == "double_mach_reflection") {
      amrex::Abort("double_mach_reflection initial data requires its analytic boundary profile");
   }
   if (cfg.inflow_rho <= 0.0 || cfg.inflow_pressure <= 0.0) {
      amrex::Abort("pyfluid.inflow_rho and pyfluid.inflow_pressure must be positive");
   }
   if (cfg.inflow_background_rho <= 0.0 || cfg.inflow_background_pressure <= 0.0) {
      amrex::Abort("pyfluid.inflow_background_rho and pyfluid.inflow_background_pressure must be positive");
   }
   require_supported_option(cfg.inflow_profile, InflowProfileOptions, "pyfluid.inflow_profile");
   if (cfg.inflow_radius <= 0.0) {
      amrex::Abort("pyfluid.inflow_radius must be positive");
   }
   if (cfg.viscosity < 0.0) {
      amrex::Abort("pyfluid.viscosity must be non-negative");
   }
   if (cfg.resistivity < 0.0) {
      amrex::Abort("pyfluid.resistivity must be non-negative");
   }
   if (cfg.diffusion != "none" && cfg.physics == "hydro" && cfg.resistivity > 0.0) {
      amrex::Abort("pyfluid.resistivity requires pyfluid.physics = mhd");
   }
   if (cfg.diffusion != "none" && cfg.physics == "mhd" && cfg.resistivity > 0.0 && cfg.ct != "on") {
      amrex::Abort("resistive MHD requires pyfluid.ct = on");
   }
   if (cfg.diffusion == "implicit" && cfg.physics == "mhd" && cfg.resistivity > 0.0) {
      if (mesh_coordinate_code(cfg.mesh_coordinate) != MeshCartesian) {
         amrex::Abort("implicit resistive MHD requires Cartesian coordinates because AMReX MLCurlCurl is Cartesian-only");
      }
      if (cfg.amr_max_level != 0) {
         amrex::Abort("implicit resistive MHD requires amr.max_level = 0 because AMReX MLCurlCurl is single-level");
      }
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         if (side_bc_name(cfg, dir, false) != "periodic" || side_bc_name(cfg, dir, true) != "periodic") {
            amrex::Abort("implicit resistive MHD currently requires periodic boundaries in every active direction");
         }
      }
   }
   if (cfg.diffusion != "none" && cfg.viscosity == 0.0 && cfg.resistivity == 0.0) {
      amrex::Abort("pyfluid.diffusion requires positive viscosity and/or resistivity");
   }
   if (cfg.diffusion_cfl <= 0.0) {
      amrex::Abort("pyfluid.diffusion_cfl must be positive");
   }
   if (cfg.diffusion == "implicit") {
      if (cfg.diffusion_tolerance <= 0.0) {
         amrex::Abort("pyfluid.diffusion_tolerance must be positive");
      }
      if (cfg.diffusion_max_iterations < 1) {
         amrex::Abort("pyfluid.diffusion_max_iterations must be positive");
      }
      if (cfg.diffusion_sor_omega <= 0.0 || cfg.diffusion_sor_omega >= 2.0) {
         amrex::Abort("pyfluid.diffusion_sor_omega must satisfy 0 < omega < 2");
      }
   }
   require_supported_option(cfg.source, SourceOptions, "pyfluid.source");
   if (cfg.source == "rotating_compact_heating") {
#if (AMREX_SPACEDIM < 2)
      amrex::Abort("pyfluid.source = rotating_compact_heating requires a 2D or 3D executable");
#endif
      const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
      const MeshAxes axes = mesh_axes_layout(cfg);
      const bool cartesian_plane = coordinate == MeshCartesian
         && mesh_has_axis(axes, MeshAxisX)
         && mesh_has_axis(axes, MeshAxisY);
      const bool angular_plane = (coordinate == MeshCylindrical || coordinate == MeshSpherical)
         && mesh_has_axis(axes, MeshAxisR)
         && mesh_has_axis(axes, MeshAxisPhi);
      if (!cartesian_plane && !angular_plane) {
         amrex::Abort("pyfluid.source = rotating_compact_heating requires an x-y or r-phi plane");
      }
      if (cfg.source_power <= 0.0) {
         amrex::Abort("pyfluid.source = rotating_compact_heating requires positive pyfluid.source_power");
      }
      if (cfg.source_radius <= 0.0) {
         amrex::Abort("pyfluid.source_radius must be positive");
      }
      if (cfg.source_sigma <= 0.0) {
         amrex::Abort("pyfluid.source_sigma must be positive");
      }
      if (cfg.source_ramp_time < 0.0) {
         amrex::Abort("pyfluid.source_ramp_time must be non-negative");
      }
      if (cfg.source_azimuthal_mode < 1) {
         amrex::Abort("pyfluid.source_azimuthal_mode must be positive");
      }
      if (cfg.source_azimuthal_mode > 16) {
         amrex::Abort("pyfluid.source_azimuthal_mode must be <= 16");
      }
   }
   require_supported_option(cfg.gravity, GravityOptions, "pyfluid.gravity");
   require_supported_option(cfg.gravity_coupling, GravityCouplingOptions, "pyfluid.gravity_coupling");
   if (cfg.gravity_constant < 0.0) {
      amrex::Abort("pyfluid.gravity_constant must be non-negative");
   }
   if (cfg.gravity == "fixed") {
      require_supported_option(cfg.gravity_fixed_potential, GravityFixedPotentialOptions, "pyfluid.gravity_fixed_potential");
      if (cfg.gravity_fixed_potential == "uniform_acceleration"
         && cfg.gravity_a1 == 0.0
         && cfg.gravity_a2 == 0.0
         && cfg.gravity_a3 == 0.0) {
         amrex::Abort("pyfluid.gravity=fixed uniform_acceleration requires a nonzero gravity_a* component");
      }
      if (cfg.gravity_fixed_potential == "harmonic" && cfg.gravity_omega <= 0.0) {
         amrex::Abort("pyfluid.gravity=fixed harmonic requires positive pyfluid.gravity_omega");
      }
   } else if (cfg.gravity == "poisson") {
      if (cfg.gravity_constant <= 0.0) {
         amrex::Abort("pyfluid.gravity = poisson requires positive pyfluid.gravity_constant");
      }
      if (cfg.gravity_tolerance <= 0.0) {
         amrex::Abort("pyfluid.gravity_tolerance must be positive");
      }
      if (cfg.gravity_max_iterations < 1) {
         amrex::Abort("pyfluid.gravity_max_iterations must be positive");
      }
      if (cfg.gravity_max_fmg_iterations < 0) {
         amrex::Abort("pyfluid.gravity_max_fmg_iterations must be non-negative");
      }
      if (cfg.gravity_max_order < 1 || cfg.gravity_max_order > 4) {
         amrex::Abort("pyfluid.gravity_max_order must be in [1, 4]");
      }
      if (cfg.gravity_verbose < 0) {
         amrex::Abort("pyfluid.gravity_verbose must be non-negative");
      }
   }
   require_supported_option(cfg.solid_geometry, SolidGeometryOptions, "pyfluid.solid_geometry");
   require_supported_option(cfg.solid_motion, SolidMotionOptions, "pyfluid.solid_motion");
   require_supported_option(cfg.solid_wall_condition, SolidWallConditionOptions, "pyfluid.solid_wall_condition");
   if (cfg.solid_volume_fraction_samples_per_dim < 1 || cfg.solid_volume_fraction_samples_per_dim > 8) {
      amrex::Abort("pyfluid.solid_volume_fraction_samples_per_dim must be in [1, 8]");
   }
   if (cfg.solid_geometry == "boxes" && cfg.solid_box_count == 0) {
      amrex::Abort("pyfluid.solid_geometry = boxes requires pyfluid.solid_box_lo and pyfluid.solid_box_hi");
   }
   if (cfg.solid_geometry != "boxes" && cfg.solid_box_count > 0) {
      amrex::Abort("solid boxes were provided but pyfluid.solid_geometry is not boxes");
   }
   if (cfg.solid_geometry == "wedge") {
#if (AMREX_SPACEDIM < 2)
      amrex::Abort("pyfluid.solid_geometry = wedge requires a 2D or 3D executable");
#endif
      if (cfg.solid_wedge_angle_deg <= 0.0 || cfg.solid_wedge_angle_deg >= 45.0) {
         amrex::Abort("pyfluid.solid_wedge_angle_deg must satisfy 0 < angle < 45 degrees");
      }
   }
   if (cfg.solid_geometry == "circle") {
#if (AMREX_SPACEDIM < 2)
      amrex::Abort("pyfluid.solid_geometry = circle requires a 2D or 3D executable");
#endif
      if (cfg.solid_radius <= 0.0) {
         amrex::Abort("pyfluid.solid_radius must be positive");
      }
      if (cfg.solid_radius_frequency < 0.0) {
         amrex::Abort("pyfluid.solid_radius_frequency must be non-negative");
      }
      if (cfg.solid_radius_ramp_time < 0.0) {
         amrex::Abort("pyfluid.solid_radius_ramp_time must be non-negative");
      }
      if (cfg.solid_motion == "pulse" && cfg.solid_radius <= std::abs(cfg.solid_radius_amplitude)) {
         amrex::Abort("pyfluid.solid_motion = pulse requires solid_radius > abs(solid_radius_amplitude)");
      }
   }
   const bool solid_enabled = cfg.solid_geometry == "boxes" || cfg.solid_geometry == "wedge" || cfg.solid_geometry == "circle";
   if (cfg.solid_motion == "translate") {
      int nonzero_velocity_components = 0;
      if (std::abs(cfg.solid_velocity_1) > 0.0) {
         ++nonzero_velocity_components;
      }
#if (AMREX_SPACEDIM >= 2)
      if (std::abs(cfg.solid_velocity_2) > 0.0) {
         ++nonzero_velocity_components;
      }
#endif
#if (AMREX_SPACEDIM == 3)
      if (std::abs(cfg.solid_velocity_3) > 0.0) {
         ++nonzero_velocity_components;
      }
#endif
      if (cfg.solid_geometry != "boxes") {
         amrex::Abort("pyfluid.solid_motion = translate currently supports pyfluid.solid_geometry = boxes only");
      }
      if (cfg.solid_wall_condition != "slip") {
         amrex::Abort("pyfluid.solid_motion = translate currently supports pyfluid.solid_wall_condition = slip only");
      }
      if (nonzero_velocity_components != 1) {
         amrex::Abort("pyfluid.solid_motion = translate requires exactly one nonzero solid_velocity component");
      }
   }
   if (cfg.solid_motion == "rotate") {
      if (cfg.solid_geometry != "circle") {
         amrex::Abort(
            "pyfluid.solid_motion = rotate currently requires pyfluid.solid_geometry = circle. "
            "Rotating non-circular solids change the active cell volume and need swept-volume/cut-cell support."
         );
      }
      if (cfg.solid_wall_condition != "no_slip") {
         amrex::Abort("pyfluid.solid_motion = rotate requires pyfluid.solid_wall_condition = no_slip");
      }
      if (cfg.diffusion != "explicit" || cfg.viscosity <= 0.0) {
         amrex::Abort("pyfluid.solid_motion = rotate requires explicit viscosity so the moving wall can transfer tangential momentum");
      }
   }
   if (cfg.solid_motion == "pulse" && cfg.solid_geometry != "circle") {
      amrex::Abort("pyfluid.solid_motion = pulse currently requires pyfluid.solid_geometry = circle");
   }
   if (cfg.solid_wall_condition == "no_slip" && (cfg.diffusion != "explicit" || cfg.viscosity <= 0.0)) {
      amrex::Abort("pyfluid.solid_wall_condition = no_slip requires pyfluid.diffusion = explicit and positive pyfluid.viscosity");
   }
   if (solid_enabled && cfg.physics != "hydro") {
      amrex::Abort("solid obstacles currently support pyfluid.physics = hydro; MHD internal walls need CT-aware magnetic wall treatment");
   }
   if (solid_enabled && cfg.diffusion == "implicit") {
      amrex::Abort("solid obstacles currently support pyfluid.diffusion = none or explicit; implicit wall diffusion is not mask-aware yet");
   }
   if (solid_enabled && cfg.diffusion == "explicit" && cfg.resistivity > 0.0) {
      amrex::Abort("solid obstacles currently support explicit viscosity only; resistive MHD internal walls need CT-aware magnetic wall treatment");
   }
   for (int box = 0; box < cfg.solid_box_count; ++box) {
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         const int idx = box * AMREX_SPACEDIM + dir;
         if (cfg.solid_box_lo[idx] >= cfg.solid_box_hi[idx]) {
            amrex::Abort("solid-box lower bounds must be smaller than upper bounds");
         }
      }
   }
   require_supported_option(cfg.mesh_metric_divergence, MetricRegridOptions, "mesh.metric_divergence");
   if (cfg.tag_threshold <= 0.0) {
      amrex::Abort("amr.tag_threshold must be positive");
   }
   if (cfg.min_feature_length < 0.0) {
      amrex::Abort("amr.min_feature_length must be non-negative; 0 disables the physical-length cutoff");
   }
   if (std::isfinite(cfg.min_feature_length_1) && cfg.min_feature_length_1 < 0.0) {
      amrex::Abort("amr.min_feature_length_1 must be non-negative");
   }
   if (std::isfinite(cfg.min_feature_length_2) && cfg.min_feature_length_2 < 0.0) {
      amrex::Abort("amr.min_feature_length_2 must be non-negative");
   }
   if (std::isfinite(cfg.min_feature_length_3) && cfg.min_feature_length_3 < 0.0) {
      amrex::Abort("amr.min_feature_length_3 must be non-negative");
   }
   if (cfg.min_refine_radius < 0.0) {
      amrex::Abort("amr.min_refine_radius must be non-negative; 0 disables the radial-origin AMR veto");
   }
   if (cfg.solid_refine_band_cells < 0.0) {
      amrex::Abort("amr.solid_refine_band_cells must be non-negative; 0 disables analytic solid-boundary tagging");
   }
   if (cfg.solid_curvature_error < 0.0) {
      amrex::Abort("amr.solid_curvature_error must be non-negative; 0 disables the curvature cutoff");
   }
   if (cfg.initial_refine_levels < 0) {
      amrex::Abort("amr.initial_refine_levels must be non-negative; 0 disables initial AMR box seeding");
   }
   if (cfg.initial_refine_fixed_levels < 0) {
      amrex::Abort("amr.initial_refine_fixed_levels must be non-negative; 0 lets AMR fully regrid initial boxes");
   }
   if (cfg.initial_refine_fixed_levels > cfg.initial_refine_levels) {
      amrex::Abort("amr.initial_refine_fixed_levels cannot exceed amr.initial_refine_levels");
   }
   if (cfg.initial_refine_fixed_levels > 0 && cfg.initial_refine != "box") {
      amrex::Abort("amr.initial_refine_fixed_levels requires amr.initial_refine = box");
   }
   if (cfg.initial_refine == "box" && cfg.initial_refine_levels > 0 && cfg.initial_refine_box_count == 0) {
      amrex::Abort("amr.initial_refine = box requires amr.initial_refine_box_lo and amr.initial_refine_box_hi");
   }
   if (cfg.initial_refine == "none" && cfg.initial_refine_box_count > 0) {
      amrex::Abort("amr.initial_refine boxes were provided but amr.initial_refine = none");
   }
   for (int box = 0; box < cfg.initial_refine_box_count; ++box) {
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         const int idx = box * AMREX_SPACEDIM + dir;
         if (cfg.initial_refine_box_lo[idx] >= cfg.initial_refine_box_hi[idx]) {
            amrex::Abort("AMR initial refinement box lower bounds must be smaller than upper bounds");
         }
      }
   }
   if (
      cfg.kh_secondary_amplitude < 0.0
      || cfg.kh_secondary_mode < 0.0
      || cfg.kh_z_mode < 0.0
      || cfg.kh_envelope_sigma < 0.0
   ) {
      amrex::Abort("KH perturbation amplitudes, modes, and envelope width must be non-negative");
   }
   if (std::abs(cfg.kh_density_amplitude) >= 0.95) {
      amrex::Abort("pyfluid.kh_density_amplitude must satisfy |value| < 0.95");
   }
   require_supported_option(cfg.kh_perturbation, KHPerturbationOptions, "pyfluid.kh_perturbation");
   if (cfg.kh_perturbation == "stream" && cfg.wave_number <= 0.0) {
      amrex::Abort("pyfluid.kh_perturbation = stream requires pyfluid.wave_number > 0");
   }
   if (cfg.rotor_radius <= 0.0 || cfg.rotor_transition_radius <= cfg.rotor_radius) {
      amrex::Abort("mhd_rotor requires 0 < pyfluid.rotor_radius < pyfluid.rotor_transition_radius");
   }
   if (cfg.rotor_density <= 0.0) {
      amrex::Abort("pyfluid.rotor_density must be positive");
   }
   if (cfg.beam_length <= 0.0 || cfg.beam_radius <= 0.0) {
      amrex::Abort("pyfluid.beam_length and pyfluid.beam_radius must be positive");
   }
   if (cfg.regrid_int < -1) {
      amrex::Abort("amr.regrid_int must be -1 or a positive integer");
   }
   if (cfg.level_cell_budget < 0) {
      amrex::Abort("amr.level_cell_budget must be non-negative; 0 disables the per-level AMR cell budget");
   }
   if (cfg.level_cell_budget_side < 0) {
      amrex::Abort("amr.level_cell_budget_side must be non-negative; 0 disables the side-length cell budget");
   }
   if (cfg.total_cell_budget < 0) {
      amrex::Abort("amr.total_cell_budget must be non-negative; 0 disables the total AMR cell budget");
   }
   if (cfg.ng < 1) {
      amrex::Abort("pyfluid.ng must be at least 1 for PWC/LLF");
   }
   const bool uses_pwl = cfg.recon == "pwl" || cfg.recon_fine == "pwl";
   const bool uses_five_point =
      is_five_point_reconstruction(recon_method_code(cfg))
      || (!cfg.recon_fine.empty()
          && is_five_point_reconstruction(
             recon_method_code_from_name(cfg.recon_fine, "pyfluid.recon_fine")));
   if (uses_pwl && cfg.ng < 2) {
      amrex::Abort("pyfluid.ng must be at least 2 for PWL reconstruction");
   }
   if (uses_five_point && cfg.ng < 3) {
      amrex::Abort("pyfluid.ng must be at least 3 for fifth-order reconstruction");
   }
   if (cfg.max_step < 0) {
      amrex::Abort("pyfluid.max_step must be non-negative");
   }
   if (cfg.final_time == 0.0 || cfg.final_time < -1.0) {
      amrex::Abort("pyfluid.final_time must be positive or -1 to disable time targeting");
   }
   if (cfg.checkpoint_int < -1) {
      amrex::Abort("pyfluid.checkpoint_int must be -1, 0, or a positive integer");
   }
   if (cfg.plot_int < -1) {
      amrex::Abort("pyfluid.plot_int must be -1, 0, or a positive integer");
   }
   require_supported_option(cfg.plotfile_format, PlotfileFormatOptions, "pyfluid.plotfile_format");
   if (!valid_plotfile_compression(cfg.plotfile_compression)) {
      amrex::Abort(
         "pyfluid.plotfile_compression must be one of: none, best, zlib, zlibN, or zlib@N with a non-negative integer N"
      );
   }
   if (!valid_plotfile_compression(cfg.checkpoint_compression)) {
      amrex::Abort(
         "pyfluid.checkpoint_compression must be one of: none, best, zlib, zlibN, or zlib@N with a non-negative integer N"
      );
   }
#ifndef AMREX_USE_HDF5
   if (cfg.plotfile_format == "hdf5") {
      amrex::Abort("pyfluid.plotfile_format = hdf5 requires AMReX built with HDF5 support");
   }
#endif
   if (cfg.scalar_int < -1) {
      amrex::Abort("pyfluid.scalar_int must be -1, 0, or a positive integer");
   }
   require_supported_option(cfg.scalar_output_mode, ScalarOutputModeOptions, "pyfluid.scalar_output_mode");
   if (!cfg.validation_policy.empty()) {
      require_supported_option(cfg.validation_policy, ValidationPolicyOptions, "pyfluid.validation_policy");
   }
   if (cfg.aux_gf_int < -1) {
      amrex::Abort("pyfluid.aux_gf_int must be -1, 0, or a positive integer");
   }
   if (cfg.aux_gf_compute_int < -1) {
      amrex::Abort("pyfluid.aux_gf_compute_int must be -1, 0, or a positive integer");
   }
   const auto aux_names = aux_gf_names(cfg);
   if (aux_names.empty() && (cfg.aux_gf_int >= 0 || cfg.aux_gf_compute_int >= 0)) {
      amrex::Abort("pyfluid.aux_gf_int and aux_gf_compute_int require pyfluid.aux_gfs to name at least one auxiliary grid function");
   }
   if (!aux_names.empty() && cfg.aux_gf_int < 0 && cfg.aux_gf_compute_int < 0) {
      amrex::Abort("pyfluid.aux_gfs requires pyfluid.aux_gf_int >= 0 and/or pyfluid.aux_gf_compute_int >= 0");
   }
   for (std::string const& name : aux_names) {
      const AuxGridField field = aux_grid_field_from_name(name);
      const AuxGridFieldInfo field_info = aux_grid_field_info(field);
      if (field == AuxGridField::Unknown) {
         amrex::Abort("pyfluid.aux_gfs currently supports " + supported_aux_grid_field_names());
      }
      if (AMREX_SPACEDIM < field_info.min_dimension) {
         amrex::Abort("pyfluid.aux_gfs = " + name + " requires a higher-dimensional AMReX build");
      }
      if (field_info.requires_mhd
          && !physical_system_has_magnetic_field(physical_system(cfg))) {
         amrex::Abort("pyfluid.aux_gfs = " + name + " requires a magnetic physical system");
      }
      if (field_info.requires_solid_geometry && cfg.solid_geometry == "off") {
         amrex::Abort("solid geometry auxiliary grid functions require pyfluid.solid_geometry != off");
      }
      if (name == "solid_swept_volume" && !std::isfinite(cfg.solid_swept_reference_time)) {
         amrex::Abort("pyfluid.aux_gfs = solid_swept_volume requires pyfluid.solid_swept_reference_time");
      }
      if (field_info.requires_gravity && cfg.gravity == "none") {
         amrex::Abort("pyfluid.aux_gfs = " + name + " requires pyfluid.gravity != none");
      }
      if (cfg.aux_gf_int < 0 && field_info.role != FieldRole::CachedDerivative) {
         amrex::Abort(
            "pyfluid.aux_gfs = " + name
            + " is not a cached derivative; use pyfluid.aux_gf_int >= 0 to output it"
         );
      }
   }
}

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
