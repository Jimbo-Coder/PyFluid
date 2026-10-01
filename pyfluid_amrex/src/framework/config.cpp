#include "pyfluid_amrex/config.hpp"

#include <AMReX_ParmParse.H>
#include "framework/config_options.hpp"

#include <AMReX.H>

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/generated/method_catalog.hpp"
#include "pyfluid_amrex/generated/initial_data_method_catalog.hpp"
#include "pyfluid_amrex/generated/spatial_order_catalog.hpp"
#include "pyfluid_amrex/spacetime_fields.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;
using namespace config_options;
namespace {

enum class RestartPolicy {
   Mutable,
   Immutable,
};

enum class StringOptionSet {
   EOS,
   EOSPreset,
   Physics,
   RelativisticExterior,
   CT,
   Reconstruction,
   Riemann,
   Timestep,
   Diffusion,
   Subcycling,
   SubcyclingSteps,
   AMRTagMetric,
   AMRRegridMode,
   AMRStateInterp,
   AMRInitialRefine,
};

struct RegisteredStringParameter {
   std::string_view key;
   std::string RunConfig::* member;
   std::string_view default_value;
   StringOptionSet option_set;
   RestartPolicy restart_policy;
   std::string_view description;
};

template <typename T>
struct RegisteredScalarParameter {
   std::string_view key;
   T RunConfig::* member;
   T default_value;
   RestartPolicy restart_policy;
   std::string_view description;
   bool RunConfig::* explicit_flag = nullptr;
};

constexpr std::array<RegisteredStringParameter, 11> MethodParameters{{
   {"eos", &RunConfig::eos, "ideal_gas", StringOptionSet::EOS, RestartPolicy::Immutable,
    "Equation of state closure."},
   {"eos_preset", &RunConfig::eos_preset, "custom", StringOptionSet::EOSPreset,
    RestartPolicy::Immutable, "Named piecewise-polytrope parameters in G=c=M_sun=1 units."},
   {"physics", &RunConfig::physics, "hydro", StringOptionSet::Physics, RestartPolicy::Immutable,
    "Evolved hydrodynamic or magnetohydrodynamic state."},
   {"relativistic_exterior", &RunConfig::relativistic_exterior, "atmosphere",
    StringOptionSet::RelativisticExterior, RestartPolicy::Immutable,
    "Physical exterior matter policy for relativistic hydrodynamics."},
   {"ct", &RunConfig::ct, "off", StringOptionSet::CT, RestartPolicy::Immutable,
    "Face-centered constrained transport for MHD."},
   {"recon", &RunConfig::recon, "pwc", StringOptionSet::Reconstruction, RestartPolicy::Mutable,
    "Finite-volume reconstruction method."},
   {"riemann", &RunConfig::riemann, "llf", StringOptionSet::Riemann, RestartPolicy::Mutable,
    "Riemann solver used for face fluxes."},
   {"timestep", &RunConfig::timestep, "euler", StringOptionSet::Timestep, RestartPolicy::Mutable,
    "Explicit time integration method."},
   {"diffusion", &RunConfig::diffusion, "none", StringOptionSet::Diffusion, RestartPolicy::Mutable,
    "Diffusion integration method."},
   {"subcycling", &RunConfig::subcycling, "on", StringOptionSet::Subcycling, RestartPolicy::Mutable,
    "AMR level time subcycling."},
   {"subcycling_steps", &RunConfig::subcycling_steps, "local_cfl", StringOptionSet::SubcyclingSteps,
    RestartPolicy::Mutable, "Rule used to choose fine-level substeps."},
}};

constexpr std::array<RegisteredStringParameter, 4> AMRStringParameters{{
   {"tag_metric", &RunConfig::tag_metric, "none", StringOptionSet::AMRTagMetric, RestartPolicy::Mutable,
    "State-based AMR tagging metric."},
   {"regrid_mode", &RunConfig::regrid_mode, "aligned", StringOptionSet::AMRRegridMode,
    RestartPolicy::Mutable, "Synchronization cadence for rebuilding AMR descendants."},
   {"state_interp", &RunConfig::state_interp, "coupled-conservative", StringOptionSet::AMRStateInterp,
    RestartPolicy::Mutable, "Conservative state interpolation used for new fine data."},
   {"initial_refine", &RunConfig::initial_refine, "none", StringOptionSet::AMRInitialRefine,
    RestartPolicy::Mutable, "Optional initial refinement region."},
}};

constexpr std::array<RegisteredScalarParameter<amrex::Real>, 8> AMRRealParameters{{
   {"tag_threshold", &RunConfig::tag_threshold, 0.35_rt, RestartPolicy::Mutable,
    "Dimensionless selected AMR-indicator threshold.", &RunConfig::tag_threshold_set},
   {"min_feature_length", &RunConfig::min_feature_length, std::numeric_limits<amrex::Real>::quiet_NaN(),
    RestartPolicy::Mutable,
    "Global physical resolution guard.", &RunConfig::min_feature_length_set},
   {"min_feature_length_1", &RunConfig::min_feature_length_1, std::numeric_limits<amrex::Real>::quiet_NaN(),
    RestartPolicy::Mutable, "Direction-1 physical resolution guard.", &RunConfig::min_feature_length_1_set},
   {"min_feature_length_2", &RunConfig::min_feature_length_2, std::numeric_limits<amrex::Real>::quiet_NaN(),
    RestartPolicy::Mutable, "Direction-2 physical resolution guard.", &RunConfig::min_feature_length_2_set},
   {"min_feature_length_3", &RunConfig::min_feature_length_3, std::numeric_limits<amrex::Real>::quiet_NaN(),
    RestartPolicy::Mutable, "Direction-3 physical resolution guard.", &RunConfig::min_feature_length_3_set},
   {"min_refine_radius", &RunConfig::min_refine_radius, 0.0_rt, RestartPolicy::Mutable,
    "Curvilinear radial refinement guard."},
   {"solid_refine_band_cells", &RunConfig::solid_refine_band_cells, 1.5_rt, RestartPolicy::Mutable,
    "Solid-interface refinement-band width in cells."},
   {"solid_curvature_error", &RunConfig::solid_curvature_error, 0.0_rt, RestartPolicy::Mutable,
    "Solid-geometry curvature tolerance."},
}};

constexpr std::array<RegisteredScalarParameter<int>, 5> AMRIntParameters{{
   {"initial_refine_levels", &RunConfig::initial_refine_levels, 0, RestartPolicy::Mutable,
    "Maximum initial refinement depth."},
   {"initial_refine_fixed_levels", &RunConfig::initial_refine_fixed_levels, 0, RestartPolicy::Mutable,
    "Initial levels retained independently of state tagging."},
   {"max_level", &RunConfig::amr_max_level, 0, RestartPolicy::Mutable,
    "Maximum allowed AMR level."},
   {"regrid_int", &RunConfig::regrid_int, -1, RestartPolicy::Mutable,
    "Coarse-level steps between regrids."},
   {"level_cell_budget_side", &RunConfig::level_cell_budget_side, 0, RestartPolicy::Mutable,
    "Convenience side length for a per-level cell budget."},
}};

constexpr std::array<RegisteredScalarParameter<amrex::Long>, 2> AMRLongParameters{{
   {"level_cell_budget", &RunConfig::level_cell_budget, 0, RestartPolicy::Mutable,
    "Maximum cells allowed on one level."},
   {"total_cell_budget", &RunConfig::total_cell_budget, 0, RestartPolicy::Mutable,
    "Maximum cells allowed in the hierarchy."},
}};

template <typename Options>
void require_registered_option(
   RegisteredStringParameter const& parameter,
   std::string const& value,
   Options const& options,
   std::string_view group
)
{
   const std::string qualified_key = std::string(group) + "." + std::string(parameter.key);
   require_supported_option(value, options, qualified_key.c_str());
}

void validate_registered_option(
   RegisteredStringParameter const& parameter,
   std::string const& value,
   std::string_view group
)
{
   switch (parameter.option_set) {
   case StringOptionSet::EOS:
      require_registered_option(parameter, value, EOSOptions, group);
      return;
   case StringOptionSet::EOSPreset:
      require_registered_option(parameter, value, EOSPresetOptions, group);
      return;
   case StringOptionSet::Physics:
      require_registered_option(parameter, value, PhysicsOptions, group);
      return;
   case StringOptionSet::RelativisticExterior:
      require_registered_option(
         parameter, value, RelativisticExteriorOptions, group
      );
      return;
   case StringOptionSet::CT:
      require_registered_option(parameter, value, CTOptions, group);
      return;
   case StringOptionSet::Reconstruction:
      require_registered_option(parameter, value, ReconOptions, group);
      return;
   case StringOptionSet::Riemann:
      require_registered_option(parameter, value, RiemannOptions, group);
      return;
   case StringOptionSet::Timestep:
      require_registered_option(parameter, value, TimestepOptions, group);
      return;
   case StringOptionSet::Diffusion:
      require_registered_option(parameter, value, DiffusionOptions, group);
      return;
   case StringOptionSet::Subcycling:
      require_registered_option(parameter, value, SubcyclingOptions, group);
      return;
   case StringOptionSet::SubcyclingSteps:
      require_registered_option(parameter, value, SubcyclingStepOptions, group);
      return;
   case StringOptionSet::AMRTagMetric:
      require_registered_option(parameter, value, AMRTagMetricOptions, group);
      return;
   case StringOptionSet::AMRRegridMode:
      require_registered_option(parameter, value, AMRRegridModeOptions, group);
      return;
   case StringOptionSet::AMRStateInterp:
      require_registered_option(parameter, value, AMRStateInterpOptions, group);
      return;
   case StringOptionSet::AMRInitialRefine:
      require_registered_option(parameter, value, InitialRefineOptions, group);
      return;
   }
   amrex::Abort("unhandled registered string option set");
}

template <typename T, std::size_t N>
void initialize_scalars(RunConfig& cfg, std::array<RegisteredScalarParameter<T>, N> const& parameters)
{
   for (auto const& parameter : parameters) {
      cfg.*(parameter.member) = parameter.default_value;
      if (parameter.explicit_flag != nullptr) {
         cfg.*(parameter.explicit_flag) = false;
      }
   }
}

template <typename T>
bool serializable_parameter_value(T const&)
{
   return true;
}

template <>
bool serializable_parameter_value(amrex::Real const& value)
{
   return std::isfinite(value);
}

template <typename T, std::size_t N>
void query_scalars(
   amrex::ParmParse& pp,
   RunConfig& cfg,
   std::array<RegisteredScalarParameter<T>, N> const& parameters
)
{
   for (auto const& parameter : parameters) {
      T& value = cfg.*(parameter.member);
      const bool existed = pp.query(parameter.key, value) != 0;
      if (parameter.explicit_flag != nullptr) {
         cfg.*(parameter.explicit_flag) = existed;
      }
      if (!existed && serializable_parameter_value(value)) {
         pp.add(parameter.key, value);
      }
   }
}

template <typename T, std::size_t N>
void sync_scalars(
   amrex::ParmParse& pp,
   RunConfig const& cfg,
   std::array<RegisteredScalarParameter<T>, N> const& parameters
)
{
   for (auto const& parameter : parameters) {
      T const& value = cfg.*(parameter.member);
      if (serializable_parameter_value(value)) {
         pp.add(parameter.key, value);
      }
   }
}

} // namespace

static void initialize_registered_parameters(RunConfig& cfg)
{
   for (RegisteredStringParameter const& parameter : MethodParameters) {
      cfg.*(parameter.member) = parameter.default_value;
   }
   for (RegisteredStringParameter const& parameter : AMRStringParameters) {
      cfg.*(parameter.member) = parameter.default_value;
   }
   initialize_scalars(cfg, AMRRealParameters);
   initialize_scalars(cfg, AMRIntParameters);
   initialize_scalars(cfg, AMRLongParameters);
}

RunConfig::RunConfig()
{
   initialize_registered_parameters(*this);
}

static void query_registered_method_parameters(amrex::ParmParse& pp, RunConfig& cfg)
{
   for (RegisteredStringParameter const& parameter : MethodParameters) {
      pp.queryAdd(parameter.key, cfg.*(parameter.member));
   }
}

static void query_registered_amr_parameters(amrex::ParmParse& pp, RunConfig& cfg)
{
   for (RegisteredStringParameter const& parameter : AMRStringParameters) {
      pp.queryAdd(parameter.key, cfg.*(parameter.member));
   }
   query_scalars(pp, cfg, AMRRealParameters);
   query_scalars(pp, cfg, AMRIntParameters);
   query_scalars(pp, cfg, AMRLongParameters);
}

static void sync_registered_amr_parameters(amrex::ParmParse& pp, RunConfig const& cfg)
{
   for (RegisteredStringParameter const& parameter : AMRStringParameters) {
      pp.add(parameter.key, cfg.*(parameter.member));
   }
   sync_scalars(pp, cfg, AMRRealParameters);
   sync_scalars(pp, cfg, AMRIntParameters);
   sync_scalars(pp, cfg, AMRLongParameters);
}

static void validate_registered_parameters(RunConfig const& cfg)
{
   for (RegisteredStringParameter const& parameter : MethodParameters) {
      validate_registered_option(parameter, cfg.*(parameter.member), "pyfluid");
   }
   for (RegisteredStringParameter const& parameter : AMRStringParameters) {
      validate_registered_option(parameter, cfg.*(parameter.member), "amr");
   }
}

void write_run_parameter_file(std::string const& path)
{
   std::ostringstream raw_table;
   amrex::ParmParse::prettyPrintTable(raw_table);
   std::istringstream table_lines(raw_table.str());

   std::map<std::string, std::string> canonical_entries;
   for (std::string line; std::getline(table_lines, line);) {
      const std::size_t equal = line.find('=');
      if (equal == std::string::npos) {
         continue;
      }
      std::string key = line.substr(0, equal);
      const std::size_t first = key.find_first_not_of(" \t");
      const std::size_t last = key.find_last_not_of(" \t");
      if (first == std::string::npos) {
         continue;
      }
      key = key.substr(first, last - first + 1);
      canonical_entries[key] = std::move(line);
   }

   std::ofstream out(path, std::ios::out | std::ios::trunc);
   if (!out.good()) {
      amrex::FileOpenFailed(path);
   }
   for (auto const& [key, line] : canonical_entries) {
      static_cast<void>(key);
      out << line << '\n';
   }
}

RunConfig read_config()
{
   RunConfig cfg;
   amrex::ParmParse pp("pyfluid");
   query_registered_method_parameters(pp, cfg);
   pp.query("ng", cfg.ng);
   pp.query("max_step", cfg.max_step);
   pp.query("final_time", cfg.final_time);
   pp.query("cfl", cfg.cfl);
   pp.query("gamma", cfg.gamma);
   pp.query("eos_table", cfg.eos_table);
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
   pp.query("scalar_diagnostics", cfg.scalar_diagnostics);
   pp.query("scalar_output_mode", cfg.scalar_output_mode);
   pp.query("plot_int", cfg.plot_int);
   pp.query("plotfile_format", cfg.plotfile_format);
   pp.query("plotfile_compression", cfg.plotfile_compression);
   pp.query("checkpoint_compression", cfg.checkpoint_compression);
   pp.query("cfl_limiter_diagnostics", cfg.cfl_limiter_diagnostics);
   amrex::ParmParse diagnostics_pp("diagnostics");
   diagnostics_pp.queryAdd("constraint_region_extent", cfg.diagnostic_constraint_region_extent);
   diagnostics_pp.queryAdd("constraint_exclusion_radius", cfg.diagnostic_constraint_exclusion_radius);
   diagnostics_pp.queryAdd("constraint_split_radius", cfg.diagnostic_constraint_split_radius);
   diagnostics_pp.queryAdd("surface_radius", cfg.diagnostic_surface_radius);
   diagnostics_pp.queryAdd("surface_center1", cfg.diagnostic_surface_center1);
   diagnostics_pp.queryAdd("surface_center2", cfg.diagnostic_surface_center2);
   diagnostics_pp.queryAdd("surface_center3", cfg.diagnostic_surface_center3);
   diagnostics_pp.queryAdd("surface_polar_points", cfg.diagnostic_surface_polar_points);
   diagnostics_pp.queryAdd("multipole_l_max", cfg.diagnostic_multipole_l_max);
   diagnostics_pp.queryAdd("azimuthal_mode_fields", cfg.diagnostic_azimuthal_mode_fields);
   diagnostics_pp.queryAdd("azimuthal_mode_m_max", cfg.diagnostic_azimuthal_mode_m_max);
   diagnostics_pp.queryAdd("matter_mode_m_max", cfg.diagnostic_matter_mode_m_max);
   diagnostics_pp.queryAdd("matter_mode_origin1", cfg.diagnostic_matter_mode_origin1);
   diagnostics_pp.queryAdd("matter_mode_origin2", cfg.diagnostic_matter_mode_origin2);
   diagnostics_pp.queryAdd("matter_mode_origin3", cfg.diagnostic_matter_mode_origin3);
   diagnostics_pp.queryAdd("matter_mode_axis1", cfg.diagnostic_matter_mode_axis1);
   diagnostics_pp.queryAdd("matter_mode_axis2", cfg.diagnostic_matter_mode_axis2);
   diagnostics_pp.queryAdd("matter_mode_axis3", cfg.diagnostic_matter_mode_axis3);
   pp.query("aux_gfs", cfg.aux_gfs);
   const bool aux_compute_interval_set =
      pp.query("aux_gf_compute_int", cfg.aux_gf_compute_int);
   const bool aux_output_interval_set = pp.query("aux_gf_int", cfg.aux_gf_int);
   if (!aux_gf_names(cfg).empty()) {
      if (!aux_output_interval_set) {
         cfg.aux_gf_int = cfg.plot_int;
      }
      if (!aux_compute_interval_set) {
         cfg.aux_gf_compute_int = cfg.aux_gf_int;
      }
   }
   pp.query("checkpoint_int", cfg.checkpoint_int);
   pp.query("checkpoint_dir", cfg.checkpoint_dir);
   pp.query("restart_file", cfg.restart_file);
   pp.query("profile", cfg.profile);
   pp.query("profile_detail", cfg.profile_detail);
   pp.query("physical_check_int", cfg.physical_check_int);
   pp.query("debug_poison", cfg.debug_poison);
   pp.query("debug_check_finite", cfg.debug_check_finite);
   pp.query("debug_check_physical", cfg.debug_check_physical);
   pp.query("debug_stage_output", cfg.debug_stage_output);
   pp.query("rmhd_candidate_guard", cfg.rmhd_candidate_guard);
   pp.query("rmhd_candidate_riemann", cfg.rmhd_candidate_riemann);
   pp.query("rmhd_candidate_dmp", cfg.rmhd_candidate_dmp);
   pp.query("rmhd_candidate_dmp_factor", cfg.rmhd_candidate_dmp_factor);
   amrex::ParmParse spacetime_pp("spacetime");
   spacetime_pp.queryAdd("evolution", cfg.spacetime_evolution);
   spacetime_pp.queryAdd("formulation", cfg.spacetime_formulation);
   spacetime_pp.queryAdd("matter_coupling", cfg.spacetime_matter_coupling);
   spacetime_pp.queryAdd("fixed_metric", cfg.spacetime_fixed_metric);
   spacetime_pp.queryAdd("initial_data", cfg.spacetime_initial_data);
   spacetime_pp.queryAdd(
      "black_hole_interior", cfg.spacetime_black_hole_interior
   );
   spacetime_pp.queryAdd(
      "black_hole_fill_radius", cfg.spacetime_black_hole_fill_radius
   );
   spacetime_pp.queryAdd(
      "black_hole_matter", cfg.spacetime_black_hole_matter
   );
   spacetime_pp.queryAdd(
      "black_hole_max_lorentz_factor",
      cfg.spacetime_black_hole_max_lorentz_factor
   );
   spacetime_pp.queryAdd(
      "black_hole_max_specific_internal_energy",
      cfg.spacetime_black_hole_max_specific_internal_energy
   );
   spacetime_pp.queryAdd(
      "black_hole_max_magnetization",
      cfg.spacetime_black_hole_max_magnetization
   );
   spacetime_pp.queryAdd(
      "black_hole_max_density", cfg.spacetime_black_hole_max_density
   );
   amrex::ParmParse initial_data_pp("initial_data");
   initial_data_pp.queryAdd("magnetic", cfg.initial_data_magnetic);
   initial_data_pp.queryAdd(
      "magnetic_seed", cfg.initial_data_magnetic_seed
   );
   initial_data_pp.queryAdd(
      "magnetic_seed_pressure_cutoff",
      cfg.initial_data_magnetic_seed_pressure_cutoff
   );
   initial_data_pp.queryAdd(
      "magnetic_seed_exponent", cfg.initial_data_magnetic_seed_exponent
   );
   initial_data_pp.queryAdd(
      "magnetic_seed_maximum_comoving_field",
      cfg.initial_data_magnetic_seed_maximum_comoving_field
   );
   initial_data_pp.queryAdd(
      "magnetic_equilibrium", cfg.initial_data_magnetic_equilibrium
   );
   initial_data_pp.queryAdd(
      "magnetic_equilibrium_poloidal_magnetization",
      cfg.initial_data_magnetic_equilibrium_poloidal_magnetization
   );
   initial_data_pp.queryAdd(
      "magnetic_equilibrium_nonlinear_poloidal",
      cfg.initial_data_magnetic_equilibrium_nonlinear_poloidal
   );
   initial_data_pp.queryAdd(
      "magnetic_equilibrium_relaxation",
      cfg.initial_data_magnetic_equilibrium_relaxation
   );
   initial_data_pp.queryAdd("spacetime", cfg.initial_data_spacetime);
   initial_data_pp.queryAdd("lapse", cfg.initial_data_lapse);
   initial_data_pp.queryAdd("shift", cfg.initial_data_shift);
   initial_data_pp.queryAdd("input_file", cfg.initial_data_input_file);
   initial_data_pp.queryAdd("output_file", cfg.initial_data_output_file);
   initial_data_pp.queryAdd("solve_only", cfg.initial_data_solve_only);
   initial_data_pp.queryAdd("save_sequence", cfg.initial_data_save_sequence);
   initial_data_pp.queryAdd(
      "sequence_output_dir", cfg.initial_data_sequence_output_dir
   );
   initial_data_pp.queryAdd("radial_grid", cfg.initial_data_radial_grid);
   initial_data_pp.queryAdd(
      "adaptive_source_radius_factor",
      cfg.initial_data_adaptive_source_radius_factor
   );
   initial_data_pp.queryAdd(
      "adaptive_exterior_fraction",
      cfg.initial_data_adaptive_exterior_fraction
   );
   initial_data_pp.queryAdd(
      "adaptive_shell_width_ratio",
      cfg.initial_data_adaptive_shell_width_ratio
   );
   initial_data_pp.queryAdd("atmosphere", cfg.initial_data_atmosphere);
   initial_data_pp.queryAdd("atmosphere_density", cfg.initial_data_atmosphere_density);
   initial_data_pp.queryAdd("atmosphere_pressure", cfg.initial_data_atmosphere_pressure);
   initial_data_pp.queryAdd(
      "atmosphere_transition_radius", cfg.initial_data_atmosphere_transition_radius
   );
   initial_data_pp.queryAdd("atmosphere_power", cfg.initial_data_atmosphere_power);
   initial_data_pp.queryAdd("disk", cfg.initial_data_disk);
   initial_data_pp.queryAdd("disk_gravity", cfg.initial_data_disk_gravity);
   initial_data_pp.queryAdd(
      "disk_rotation_law", cfg.initial_data_disk_rotation_law
   );
   initial_data_pp.queryAdd(
      "disk_specific_angular_momentum",
      cfg.initial_data_disk_specific_angular_momentum
   );
   initial_data_pp.queryAdd(
      "disk_angular_velocity", cfg.initial_data_disk_angular_velocity
   );
   initial_data_pp.queryAdd(
      "disk_differential_rotation_length",
      cfg.initial_data_disk_differential_rotation_length
   );
   initial_data_pp.queryAdd(
      "disk_rotation_power_q", cfg.initial_data_disk_rotation_power_q
   );
   initial_data_pp.queryAdd(
      "disk_inner_radius", cfg.initial_data_disk_inner_radius
   );
   initial_data_pp.queryAdd("disk_axis1", cfg.initial_data_disk_axis1);
   initial_data_pp.queryAdd("disk_axis2", cfg.initial_data_disk_axis2);
   initial_data_pp.queryAdd("disk_axis3", cfg.initial_data_disk_axis3);
   initial_data_pp.queryAdd("rotation_law", cfg.initial_data_rotation_law);
   initial_data_pp.queryAdd("angular_velocity", cfg.initial_data_angular_velocity);
   initial_data_pp.queryAdd(
      "differential_rotation_length",
      cfg.initial_data_differential_rotation_length
   );
   initial_data_pp.queryAdd(
      "differential_rotation_length_b",
      cfg.initial_data_differential_rotation_length_b
   );
   initial_data_pp.queryAdd(
      "rotation_power_q", cfg.initial_data_rotation_power_q
   );
   initial_data_pp.queryAdd("rotation_axis1", cfg.initial_data_rotation_axis1);
   initial_data_pp.queryAdd("rotation_axis2", cfg.initial_data_rotation_axis2);
   initial_data_pp.queryAdd("rotation_axis3", cfg.initial_data_rotation_axis3);
   initial_data_pp.queryAdd("rotation_center1", cfg.initial_data_rotation_center1);
   initial_data_pp.queryAdd("rotation_center2", cfg.initial_data_rotation_center2);
   initial_data_pp.queryAdd("rotation_center3", cfg.initial_data_rotation_center3);
   initial_data_pp.queryAdd("matter_relaxation", cfg.initial_data_matter_relaxation);
   initial_data_pp.queryAdd(
      "matter_relative_tolerance", cfg.initial_data_matter_relative_tolerance
   );
   initial_data_pp.queryAdd(
      "matter_max_iterations", cfg.initial_data_matter_max_iterations
   );
   initial_data_pp.queryAdd(
      "rotation_continuation_steps",
      cfg.initial_data_rotation_continuation_steps
   );
   initial_data_pp.queryAdd(
      "perturbation", cfg.initial_data_perturbation
   );
   initial_data_pp.queryAdd(
      "perturbation_pressure_scale",
      cfg.initial_data_perturbation_pressure_scale
   );
   initial_data_pp.queryAdd(
      "perturbation_amplitude", cfg.initial_data_perturbation_amplitude
   );
   initial_data_pp.queryAdd(
      "perturbation_mode", cfg.initial_data_perturbation_mode
   );
   initial_data_pp.queryAdd(
      "perturbation_phase", cfg.initial_data_perturbation_phase
   );
   initial_data_pp.queryAdd(
      "perturbation_radius", cfg.initial_data_perturbation_radius
   );
   initial_data_pp.queryarr(
      "rotation_continuation_targets",
      cfg.initial_data_rotation_continuation_targets
   );
   spacetime_pp.queryAdd("lapse_gauge", cfg.spacetime_lapse_gauge);
   spacetime_pp.queryAdd("shift_gauge", cfg.spacetime_shift_gauge);
   spacetime_pp.queryAdd("outer_boundary", cfg.spacetime_outer_boundary);
   spacetime_pp.queryAdd("sommerfeld_power", cfg.spacetime_sommerfeld_power);
   spacetime_pp.queryAdd("amplitude", cfg.spacetime_amplitude);
   spacetime_pp.queryAdd("wave_number", cfg.spacetime_wave_number);
   spacetime_pp.queryAdd("kappa1", cfg.spacetime_kappa1);
   spacetime_pp.queryAdd("kappa2", cfg.spacetime_kappa2);
   spacetime_pp.queryAdd("shift_eta", cfg.spacetime_shift_eta);
   spacetime_pp.queryAdd(
      "slow_start_strength", cfg.spacetime_slow_start_strength
   );
   spacetime_pp.queryAdd("slow_start_time", cfg.spacetime_slow_start_time);
   spacetime_pp.queryAdd("derivative_order", cfg.spacetime_derivative_order);
   spacetime_pp.queryAdd("prolongation_order", cfg.spacetime_prolongation_order);
   spacetime_pp.queryAdd("ko_dissipation", cfg.spacetime_ko_dissipation);
   spacetime_pp.queryAdd("chi_floor", cfg.spacetime_chi_floor);
   spacetime_pp.queryAdd("lapse_floor", cfg.spacetime_lapse_floor);
   spacetime_pp.queryAdd("acceleration1", cfg.spacetime_acceleration1);
   spacetime_pp.queryAdd("acceleration2", cfg.spacetime_acceleration2);
   spacetime_pp.queryAdd("acceleration3", cfg.spacetime_acceleration3);
   spacetime_pp.queryAdd("origin1", cfg.spacetime_origin1);
   spacetime_pp.queryAdd("origin2", cfg.spacetime_origin2);
   spacetime_pp.queryAdd("origin3", cfg.spacetime_origin3);
   spacetime_pp.queryAdd("mass", cfg.spacetime_mass);
   spacetime_pp.queryAdd(
      "dimensionless_spin1", cfg.spacetime_dimensionless_spin1
   );
   spacetime_pp.queryAdd(
      "dimensionless_spin2", cfg.spacetime_dimensionless_spin2
   );
   spacetime_pp.queryAdd(
      "dimensionless_spin3", cfg.spacetime_dimensionless_spin3
   );
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
   pp.query("tov_absolute_tolerance", cfg.tov_absolute_tolerance);
   pp.query("tov_relative_tolerance", cfg.tov_relative_tolerance);
   pp.query("tov_max_steps", cfg.tov_max_steps);
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
   if (cfg.tag_metric != "none" && !cfg.tag_threshold_set) {
      cfg.tag_threshold = amr_tag_policy_flags(
         amr_tag_policy_code(cfg)
      ).default_threshold;
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
   prepare_eos(cfg);
   return cfg;
}

void validate_config(RunConfig const& cfg)
{
   if (!is_supported_spatial_order(cfg.spacetime_derivative_order)) {
      amrex::Abort("spacetime.derivative_order must be a catalogued spatial order");
   }
   validate_registered_parameters(cfg);
   require_supported_option(cfg.spacetime_formulation, SpacetimeFormulationOptions, "spacetime.formulation");
   if ((cfg.tag_metric == "spacetime" || cfg.tag_metric == "compact_object")
       && !spacetime_stores_state(cfg)) {
      amrex::Abort(
         "amr.tag_metric = spacetime or compact_object requires stored fixed/evolved spacetime"
      );
   }
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
   const amrex::Real black_hole_spin_squared =
      cfg.spacetime_dimensionless_spin1
         * cfg.spacetime_dimensionless_spin1
      + cfg.spacetime_dimensionless_spin2
         * cfg.spacetime_dimensionless_spin2
      + cfg.spacetime_dimensionless_spin3
         * cfg.spacetime_dimensionless_spin3;
   if (!std::isfinite(black_hole_spin_squared)) {
      amrex::Abort("spacetime.dimensionless_spin components must be finite");
   }
   const bool kerr_schild_initial_data =
      cfg.spacetime_initial_data == "kerr_schild";
   const bool kerr_schild_fixed_metric =
      cfg.spacetime_fixed_metric == "kerr_schild";
   const bool kerr_schild_requested =
      kerr_schild_initial_data || kerr_schild_fixed_metric;
   const bool static_trumpet_initial_data =
      cfg.spacetime_initial_data == "static_trumpet";
   const bool static_trumpet_fixed_metric =
      cfg.spacetime_fixed_metric == "static_trumpet";
   const bool static_trumpet_requested =
      static_trumpet_initial_data || static_trumpet_fixed_metric;
   const int initial_data_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const std::vector<std::string> initial_data_axis_tokens = mesh_axis_tokens(
      initial_data_coordinate, cfg.mesh_axes
   );
   std::string initial_data_axes;
   for (std::size_t index = 0; index < initial_data_axis_tokens.size(); ++index) {
      if (index > 0) {
         initial_data_axes += ' ';
      }
      initial_data_axes += initial_data_axis_tokens[index];
   }
   const InitialDataOwnerInfo& spherical_tov_owner = initial_data_owner_info(
      InitialDataOwner::SphericalTov
   );
   const InitialDataOwnerInfo& spherical_axisymmetric_xcts_owner =
      initial_data_owner_info(InitialDataOwner::SphericalAxisymmetricXcts);
   const InitialDataOwnerInfo& spherical_axisymmetric_xcts_3d_owner =
      initial_data_owner_info(InitialDataOwner::SphericalAxisymmetricXcts3d);
   const InitialDataOwnerInfo& cartesian_conformal_owner =
      initial_data_owner_info(InitialDataOwner::CartesianConformalConstraints);
   const InitialDataOwnerInfo& spherical_black_hole_xcts_owner =
      initial_data_owner_info(
         InitialDataOwner::SphericalAxisymmetricBlackHoleXcts
      );
   const InitialDataOwnerInfo& spherical_black_hole_xcts_3d_owner =
      initial_data_owner_info(
         InitialDataOwner::SphericalAxisymmetricBlackHoleXcts3d
      );
   const InitialDataOwnerInfo& cartesian_kerr_owner = initial_data_owner_info(
      InitialDataOwner::CartesianKerrSchild
   );
   const InitialDataOwnerInfo& cartesian_trumpet_owner =
      initial_data_owner_info(InitialDataOwner::CartesianStaticTrumpet);
   const bool spherical_tov_layout = initial_data_owner_layout_matches(
      spherical_tov_owner,
      AMREX_SPACEDIM,
      cfg.mesh_coordinate,
      initial_data_axes
   );
   const bool spherical_axisymmetric_xcts_2d_layout =
      initial_data_owner_layout_matches(
         spherical_axisymmetric_xcts_owner,
         AMREX_SPACEDIM,
         cfg.mesh_coordinate,
         initial_data_axes
      );
   const bool spherical_axisymmetric_xcts_3d_layout =
      initial_data_owner_layout_matches(
         spherical_axisymmetric_xcts_3d_owner,
         AMREX_SPACEDIM,
         cfg.mesh_coordinate,
         initial_data_axes
      );
   const bool spherical_axisymmetric_xcts_layout =
      spherical_axisymmetric_xcts_2d_layout
      || spherical_axisymmetric_xcts_3d_layout;
#if (AMREX_SPACEDIM == 1)
   const bool spherical_reference_metric_rhd_layout =
      initial_data_coordinate == MeshSpherical && initial_data_axes == "r";
#elif (AMREX_SPACEDIM == 2)
   const bool spherical_reference_metric_rhd_layout =
      initial_data_coordinate == MeshSpherical && initial_data_axes == "r theta";
#else
   const bool spherical_reference_metric_rhd_layout = false;
#endif
   const bool cartesian_conformal_layout = initial_data_owner_layout_matches(
      cartesian_conformal_owner,
      AMREX_SPACEDIM,
      cfg.mesh_coordinate,
      initial_data_axes
   );
   const bool spherical_black_hole_xcts_layout =
      initial_data_owner_layout_matches(
         spherical_black_hole_xcts_owner,
         AMREX_SPACEDIM,
         cfg.mesh_coordinate,
         initial_data_axes
      );
   const bool spherical_black_hole_xcts_3d_layout =
      initial_data_owner_layout_matches(
         spherical_black_hole_xcts_3d_owner,
         AMREX_SPACEDIM,
         cfg.mesh_coordinate,
         initial_data_axes
      );
   const bool cartesian_kerr_layout = initial_data_owner_layout_matches(
      cartesian_kerr_owner,
      AMREX_SPACEDIM,
      cfg.mesh_coordinate,
      initial_data_axes
   );
   const bool cartesian_trumpet_layout = initial_data_owner_layout_matches(
      cartesian_trumpet_owner,
      AMREX_SPACEDIM,
      cfg.mesh_coordinate,
      initial_data_axes
   );
   const bool spherical_black_hole_xcts_request =
      cfg.ic == "black_hole"
      && cfg.spacetime_initial_data == "xcts"
      && (spherical_black_hole_xcts_layout
          || spherical_black_hole_xcts_3d_layout);
   const bool native_spherical_black_hole_solve =
      spherical_black_hole_xcts_request && cfg.initial_data_solve_only;
   const bool loaded_cartesian_black_hole_artifact =
      cfg.ic == "black_hole"
      && cfg.spacetime_initial_data == "xcts"
      && AMREX_SPACEDIM == 3
      && cfg.mesh_coordinate == "cartesian"
      && initial_data_axes == "x y z"
      && !cfg.initial_data_input_file.empty()
      && cfg.spacetime_black_hole_interior == "filled";
   if (kerr_schild_requested) {
      if (AMREX_SPACEDIM != 3) {
         amrex::Abort(
            "Kerr-Schild materialization/evolution currently requires the "
            "3D Cartesian executable; reduced-dimensional ID support belongs "
            "to the elliptic solve/materialization interface"
         );
      }
      if (!(cfg.spacetime_mass > 0.0_rt)
          || !std::isfinite(cfg.spacetime_mass)
          || !(black_hole_spin_squared < 1.0_rt)) {
         amrex::Abort(
            "Kerr-Schild spacetime requires positive finite mass and "
            "dimensionless spin magnitude below one"
         );
      }
   } else if (static_trumpet_requested) {
      if (!cartesian_trumpet_layout
          || !(cfg.spacetime_mass > 0.0_rt)
          || !std::isfinite(cfg.spacetime_mass)
          || black_hole_spin_squared != 0.0_rt) {
         amrex::Abort(
            "static-trumpet spacetime requires a positive finite mass, zero "
            "spin, and the 3D Cartesian owner"
         );
      }
      if (cfg.spacetime_black_hole_interior != "puncture"
          || cfg.spacetime_black_hole_fill_radius != -1.0_rt) {
         amrex::Abort(
            "static-trumpet spacetime requires black_hole_interior = "
            "puncture and no fill radius"
         );
      }
   } else if (black_hole_spin_squared != 0.0_rt
              && !spherical_black_hole_xcts_request
              && !loaded_cartesian_black_hole_artifact) {
      amrex::Abort(
         "spacetime.dimensionless_spin requires Kerr-Schild data or a "
         "native/loaded constraint-solved black-hole ID"
      );
   }
   if (spherical_black_hole_xcts_request
       && (cfg.spacetime_dimensionless_spin1 != 0.0_rt
           || cfg.spacetime_dimensionless_spin2 != 0.0_rt
           || !(black_hole_spin_squared < 1.0_rt))) {
      amrex::Abort(
         "the axisymmetric spherical black-hole XCTS owner requires spin "
         "aligned with its third axis and magnitude below one"
      );
   }
   if (cfg.ic == "black_hole"
       && ((!native_spherical_black_hole_solve
            && !loaded_cartesian_black_hole_artifact)
           || !(cfg.spacetime_mass > 0.0_rt)
           || !std::isfinite(cfg.spacetime_mass)
           || cfg.spacetime_origin1 != 0.0_rt
           || cfg.spacetime_origin2 != 0.0_rt
           || cfg.spacetime_origin3 != 0.0_rt)) {
      amrex::Abort(
         "black_hole initial data require a positive-mass, origin-centered "
         "native spherical XCTS shell solve or a Cartesian filled-artifact "
         "materialization/evolution"
      );
   }
   if (loaded_cartesian_black_hole_artifact) {
      const amrex::Real expected_fill_radius = 0.5_rt * cfg.spacetime_mass;
      const amrex::Real fill_tolerance = 64.0_rt
         * std::numeric_limits<amrex::Real>::epsilon()
         * std::max(amrex::Real(1.0_rt), expected_fill_radius);
      if (std::abs(
             cfg.spacetime_black_hole_fill_radius - expected_fill_radius
          ) > fill_tolerance) {
         amrex::Abort(
            "a Cartesian fill of the inversion-symmetric XCTS shell must "
            "end at its r=M/2 apparent-horizon throat"
         );
      }
   }
   require_supported_option(
      cfg.spacetime_initial_data,
      SpacetimeInitialDataOptions,
      "spacetime.initial_data"
   );
   require_supported_option(
      cfg.spacetime_black_hole_interior,
      BlackHoleInteriorOptions,
      "spacetime.black_hole_interior"
   );
   require_supported_option(
      cfg.spacetime_black_hole_matter,
      BlackHoleMatterOptions,
      "spacetime.black_hole_matter"
   );
   const bool black_hole_matter_special =
      cfg.spacetime_black_hole_matter != "evolve";
   if (black_hole_matter_special
       && !physical_system_is_relativistic(physical_system(cfg))) {
      amrex::Abort(
         "spacetime.black_hole_matter requires RHD or RMHD"
      );
   }
   if (black_hole_matter_special) {
      amrex::ParmParse horizon_pp("horizon");
      bool horizon_enabled = false;
      horizon_pp.queryAdd("enabled", horizon_enabled);
      if (!horizon_enabled) {
         amrex::Abort(
            "bounded or excised black-hole matter requires horizon.enabled = 1"
         );
      }
   }
   if (cfg.spacetime_black_hole_matter == "bounded") {
      if (cfg.spacetime_black_hole_interior != "filled"
          && cfg.spacetime_black_hole_interior != "puncture") {
         amrex::Abort(
            "bounded black-hole matter requires a filled or puncture/trumpet "
            "complete-grid interior"
         );
      }
      if (!(cfg.spacetime_black_hole_max_lorentz_factor > 1.0_rt)
          || !std::isfinite(cfg.spacetime_black_hole_max_lorentz_factor)
          || !(cfg.spacetime_black_hole_max_specific_internal_energy > 0.0_rt)
          || !std::isfinite(
             cfg.spacetime_black_hole_max_specific_internal_energy
          )
          || !(cfg.spacetime_black_hole_max_magnetization > 0.0_rt)
          || !std::isfinite(cfg.spacetime_black_hole_max_magnetization)
          || (cfg.spacetime_black_hole_max_density != -1.0_rt
              && (!(cfg.spacetime_black_hole_max_density > cfg.rho_floor)
                  || !std::isfinite(
                     cfg.spacetime_black_hole_max_density
                  )))) {
         amrex::Abort(
            "invalid bounded black-hole matter ceilings"
         );
      }
   }
   if (cfg.spacetime_black_hole_matter == "excision"
       && cfg.spacetime_black_hole_interior != "excision") {
      amrex::Abort(
         "excised black-hole matter requires black_hole_interior = excision"
      );
   }
   if (cfg.spacetime_black_hole_interior == "excision"
       && cfg.spacetime_black_hole_matter != "excision") {
      amrex::Abort(
         "black_hole_interior = excision requires excised matter ownership"
      );
   }
   // Evolved excision uses a fixed coordinate cut at black_hole_fill_radius.
   // Horizon containment and physical/gauge outflow are checked at runtime.
   if (!kerr_schild_requested && !static_trumpet_requested
       && !loaded_cartesian_black_hole_artifact
       && (cfg.spacetime_black_hole_interior != "exterior"
           || cfg.spacetime_black_hole_fill_radius != -1.0_rt)) {
      amrex::Abort(
         "black-hole interior options require Kerr-Schild or static-trumpet "
         "initial data"
      );
   }
   if (kerr_schild_requested
       && (cfg.spacetime_black_hole_interior == "filled"
           || cfg.spacetime_black_hole_interior == "excision")) {
      const amrex::Real outer_horizon_radius = cfg.spacetime_mass * (
         1.0_rt + std::sqrt(1.0_rt - black_hole_spin_squared)
      );
      const amrex::Real spin_length = cfg.spacetime_mass
         * std::sqrt(black_hole_spin_squared);
      const amrex::Real safe_inner_radius = spin_length
         + 256.0_rt * std::numeric_limits<amrex::Real>::epsilon()
            * cfg.spacetime_mass;
      if (!(cfg.spacetime_black_hole_fill_radius > 0.0_rt)
          || !std::isfinite(cfg.spacetime_black_hole_fill_radius)
          || !(cfg.spacetime_black_hole_fill_radius > safe_inner_radius)
          || !(cfg.spacetime_black_hole_fill_radius
               < outer_horizon_radius)) {
         amrex::Abort(
            "spacetime.black_hole_fill_radius must lie safely outside |a| "
            "and strictly inside the Kerr outer horizon"
         );
      }
   } else if (kerr_schild_requested
              && cfg.spacetime_black_hole_fill_radius != -1.0_rt) {
      amrex::Abort(
         "spacetime.black_hole_fill_radius applies only to filled or "
         "excised black-hole interiors"
      );
   }
   if (cfg.spacetime_evolution == "fixed"
       && (kerr_schild_initial_data != kerr_schild_fixed_metric)) {
      amrex::Abort(
         "fixed Kerr-Schild runs must select kerr_schild consistently for "
         "both spacetime.initial_data and spacetime.fixed_metric"
      );
   }
   if (cfg.spacetime_evolution == "fixed"
       && (static_trumpet_initial_data != static_trumpet_fixed_metric)) {
      amrex::Abort(
         "fixed static-trumpet runs must select static_trumpet consistently "
         "for both spacetime.initial_data and spacetime.fixed_metric"
      );
   }
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
      cfg.initial_data_atmosphere,
      InitialAtmosphereOptions,
      "initial_data.atmosphere"
   );
   require_supported_option(
      cfg.initial_data_disk,
      InitialDiskOptions,
      "initial_data.disk"
   );
   require_supported_option(
      cfg.initial_data_disk_gravity,
      InitialDiskGravityOptions,
      "initial_data.disk_gravity"
   );
   bool disk_rotation_law_supported = false;
   for (DiskRotationLawInfo const& law : DiskRotationLawCatalog) {
      disk_rotation_law_supported = disk_rotation_law_supported
         || law.name == cfg.initial_data_disk_rotation_law;
   }
   if (!disk_rotation_law_supported) {
      amrex::Abort(
         "initial_data.disk_rotation_law is not present in "
         "schema/methods.toml"
      );
   }
   const bool stationary_torus_enabled =
      cfg.initial_data_disk == "equilibrium_torus";
   const bool tilted_torus_enabled =
      cfg.initial_data_disk == "tilted_torus";
   const bool equilibrium_torus_enabled = stationary_torus_enabled
      || tilted_torus_enabled;
   if (!equilibrium_torus_enabled
       && cfg.initial_data_disk_gravity != "none") {
      amrex::Abort(
         "initial_data.disk_gravity must be none when no disk is selected"
      );
   }
   if (equilibrium_torus_enabled
       && cfg.initial_data_disk_gravity == "none") {
      amrex::Abort(
         "a selected disk requires initial_data.disk_gravity = test_fluid "
         "or coupled"
      );
   }
   const amrex::Real disk_axis_norm = std::sqrt(
      cfg.initial_data_disk_axis1 * cfg.initial_data_disk_axis1
      + cfg.initial_data_disk_axis2 * cfg.initial_data_disk_axis2
      + cfg.initial_data_disk_axis3 * cfg.initial_data_disk_axis3
   );
   if (equilibrium_torus_enabled
       && (!(disk_axis_norm > 0.0_rt) || !std::isfinite(disk_axis_norm))) {
      amrex::Abort("initial_data.disk_axis must be a finite nonzero vector");
   }
   const amrex::Real inverse_disk_axis_norm = equilibrium_torus_enabled
      ? 1.0_rt / disk_axis_norm : 1.0_rt;
   const amrex::Real disk_axis1 = cfg.initial_data_disk_axis1
      * inverse_disk_axis_norm;
   const amrex::Real disk_axis2 = cfg.initial_data_disk_axis2
      * inverse_disk_axis_norm;
   const amrex::Real disk_axis3 = cfg.initial_data_disk_axis3
      * inverse_disk_axis_norm;
   const amrex::Real disk_axis_tolerance = 64.0_rt
      * std::numeric_limits<amrex::Real>::epsilon();
   const bool analytic_black_hole_disk_request =
      ((cartesian_kerr_layout && kerr_schild_requested)
       || (cartesian_trumpet_layout && static_trumpet_requested))
      && cfg.ic == "equilibrium_torus"
      && stationary_torus_enabled
      && initial_data_owner_list_contains(
         static_trumpet_requested
            ? cartesian_trumpet_owner.disk_kinds
            : cartesian_kerr_owner.disk_kinds,
         cfg.initial_data_disk
      )
      && initial_data_owner_list_contains(
         static_trumpet_requested
            ? cartesian_trumpet_owner.disk_gravity_modes
            : cartesian_kerr_owner.disk_gravity_modes,
         cfg.initial_data_disk_gravity
      );
   const bool cartesian_stellar_torus_request =
      cartesian_conformal_layout
      && initial_data_owner_list_contains(
         cartesian_conformal_owner.central_objects, cfg.ic
      )
      && cfg.spacetime_initial_data == "xcts"
      && initial_data_owner_list_contains(
         cartesian_conformal_owner.disk_kinds, cfg.initial_data_disk
      )
      && initial_data_owner_list_contains(
         cartesian_conformal_owner.disk_gravity_modes,
         cfg.initial_data_disk_gravity
      );
   const InitialDataOwnerInfo& selected_spherical_stellar_owner =
      spherical_axisymmetric_xcts_3d_layout
         ? spherical_axisymmetric_xcts_3d_owner
         : spherical_axisymmetric_xcts_owner;
   const bool spherical_stellar_torus_request =
      spherical_axisymmetric_xcts_layout
      && initial_data_owner_list_contains(
         selected_spherical_stellar_owner.central_objects,
         cfg.ic
      )
      && cfg.spacetime_initial_data == "xcts"
      && initial_data_owner_list_contains(
         selected_spherical_stellar_owner.disk_kinds, cfg.initial_data_disk
      )
      && initial_data_owner_list_contains(
         selected_spherical_stellar_owner.disk_gravity_modes,
         cfg.initial_data_disk_gravity
      );
   const InitialDataOwnerInfo& selected_spherical_black_hole_owner =
      spherical_black_hole_xcts_3d_layout
         ? spherical_black_hole_xcts_3d_owner
         : spherical_black_hole_xcts_owner;
   const bool spherical_black_hole_torus_request =
      spherical_black_hole_xcts_request
      && initial_data_owner_list_contains(
         selected_spherical_black_hole_owner.disk_kinds,
         cfg.initial_data_disk
      )
      && initial_data_owner_list_contains(
         selected_spherical_black_hole_owner.disk_gravity_modes,
         cfg.initial_data_disk_gravity
      );
   const bool loaded_3d_materialization_request =
      AMREX_SPACEDIM == 3 && cfg.initial_data_solve_only
      && !cfg.initial_data_input_file.empty();
   if (equilibrium_torus_enabled
       && !analytic_black_hole_disk_request
       && !cartesian_stellar_torus_request
       && !spherical_stellar_torus_request
       && !spherical_black_hole_torus_request
       && !loaded_cartesian_black_hole_artifact
       && !loaded_3d_materialization_request) {
      amrex::Abort(
         "the selected central object, solve owner, and disk_gravity mode "
         "do not form a supported disk initial-data composition"
      );
   }
   if (equilibrium_torus_enabled
       && (spherical_stellar_torus_request
           || spherical_black_hole_torus_request)
       && (std::abs(disk_axis1) > disk_axis_tolerance
           || std::abs(disk_axis2) > disk_axis_tolerance
           || std::abs(disk_axis3 - 1.0_rt) > disk_axis_tolerance)) {
      amrex::Abort(
         "the reduced r-theta equilibrium-torus owner requires disk_axis = "
         "(0,0,1); arbitrary axes require the 3D Cartesian XCTS owner"
      );
   }
   if (tilted_torus_enabled
       && (!cartesian_conformal_owner.relative_tilt
           || !cartesian_stellar_torus_request)
       && !loaded_3d_materialization_request) {
      amrex::Abort(
         "initial_data.disk = tilted_torus requires a 3D Cartesian TOV or "
         "rotating-star XCTS solve"
      );
   }
   if (stationary_torus_enabled
       && cartesian_stellar_torus_request
       && cfg.ic == "rotating_star"
       && cfg.initial_data_angular_velocity != 0.0_rt) {
      const amrex::Real star_axis_norm = std::sqrt(
         cfg.initial_data_rotation_axis1 * cfg.initial_data_rotation_axis1
         + cfg.initial_data_rotation_axis2 * cfg.initial_data_rotation_axis2
         + cfg.initial_data_rotation_axis3 * cfg.initial_data_rotation_axis3
      );
      if (star_axis_norm > 0.0_rt && std::isfinite(star_axis_norm)) {
         const amrex::Real star_axis1 =
            cfg.initial_data_rotation_axis1 / star_axis_norm;
         const amrex::Real star_axis2 =
            cfg.initial_data_rotation_axis2 / star_axis_norm;
         const amrex::Real star_axis3 =
            cfg.initial_data_rotation_axis3 / star_axis_norm;
         const amrex::Real cross1 = disk_axis2 * star_axis3
            - disk_axis3 * star_axis2;
         const amrex::Real cross2 = disk_axis3 * star_axis1
            - disk_axis1 * star_axis3;
         const amrex::Real cross3 = disk_axis1 * star_axis2
            - disk_axis2 * star_axis1;
         if (std::sqrt(
                cross1 * cross1 + cross2 * cross2 + cross3 * cross3
             ) > disk_axis_tolerance) {
            amrex::Abort(
               "stationary equilibrium_torus data require aligned stellar "
               "and disk axes; use initial_data.disk = tilted_torus for an "
               "explicit constraint-consistent, nonstationary 3D seed"
            );
         }
      }
   }
   if (equilibrium_torus_enabled && analytic_black_hole_disk_request
       && kerr_schild_requested
       && black_hole_spin_squared > 0.0_rt) {
      const amrex::Real inverse_spin_norm = 1.0_rt / std::sqrt(
         black_hole_spin_squared
      );
      const amrex::Real spin1 = cfg.spacetime_dimensionless_spin1
         * inverse_spin_norm;
      const amrex::Real spin2 = cfg.spacetime_dimensionless_spin2
         * inverse_spin_norm;
      const amrex::Real spin3 = cfg.spacetime_dimensionless_spin3
         * inverse_spin_norm;
      const amrex::Real cross1 = disk_axis2 * spin3 - disk_axis3 * spin2;
      const amrex::Real cross2 = disk_axis3 * spin1 - disk_axis1 * spin3;
      const amrex::Real cross3 = disk_axis1 * spin2 - disk_axis2 * spin1;
      if (std::sqrt(
             cross1 * cross1 + cross2 * cross2 + cross3 * cross3
          ) > disk_axis_tolerance) {
         amrex::Abort(
            "stationary Kerr equilibrium-torus data require disk_axis "
            "parallel or antiparallel to the black-hole spin; tilted Kerr "
            "disks require the explicit nonequilibrium 3D owner"
         );
      }
   }
   if (equilibrium_torus_enabled
       && (!(cfg.initial_data_disk_inner_radius > 0.0_rt)
           || !std::isfinite(cfg.initial_data_disk_inner_radius))) {
      amrex::Abort(
         "equilibrium-torus initial data require a positive finite "
         "disk_inner_radius"
      );
   }
   if (equilibrium_torus_enabled
       && cfg.initial_data_disk_rotation_law == "constant_l"
       && (!std::isfinite(cfg.initial_data_disk_specific_angular_momentum)
           || cfg.initial_data_disk_specific_angular_momentum == 0.0_rt)) {
      amrex::Abort(
         "constant-l equilibrium-torus initial data require a nonzero finite "
         "disk_specific_angular_momentum"
      );
   }
   if (equilibrium_torus_enabled
       && cfg.initial_data_disk_rotation_law != "constant_l"
       && (!std::isfinite(cfg.initial_data_disk_angular_velocity)
           || cfg.initial_data_disk_angular_velocity == 0.0_rt)) {
      amrex::Abort(
         "uniform and differential equilibrium-torus laws require a nonzero "
         "finite disk_angular_velocity"
      );
   }
   if (equilibrium_torus_enabled
       && (cfg.initial_data_disk_rotation_law == "j_constant"
           || cfg.initial_data_disk_rotation_law == "q_power")
       && (!(cfg.initial_data_disk_differential_rotation_length > 0.0_rt)
           || !std::isfinite(
                cfg.initial_data_disk_differential_rotation_length
              ))) {
      amrex::Abort(
         "differential equilibrium-torus laws require a positive finite "
         "disk_differential_rotation_length"
      );
   }
   if (equilibrium_torus_enabled
       && cfg.initial_data_disk_rotation_law == "q_power"
       && (!(cfg.initial_data_disk_rotation_power_q >= 1.0_rt)
           || !std::isfinite(cfg.initial_data_disk_rotation_power_q))) {
      amrex::Abort(
         "q-power equilibrium-torus initial data require finite "
         "disk_rotation_power_q >= 1"
      );
   }
   bool rotation_law_supported = false;
   for (RotationLawInfo const& law : RotationLawCatalog) {
      rotation_law_supported = rotation_law_supported
         || law.name == cfg.initial_data_rotation_law;
   }
   if (!rotation_law_supported) {
      amrex::Abort(
         "initial_data.rotation_law is not present in schema/methods.toml"
      );
   }
   if (cfg.ic == "rotating_star"
       && cfg.initial_data_rotation_law != "uniform"
       && (!(cfg.initial_data_differential_rotation_length > 0.0_rt)
           || !std::isfinite(cfg.initial_data_differential_rotation_length))) {
      amrex::Abort(
         "differential rotating-star initial data requires a positive finite "
         "differential_rotation_length"
      );
   }
   const bool uryu_rotation_law =
      cfg.initial_data_rotation_law == "uryu_plus_13"
      || cfg.initial_data_rotation_law == "uryu_j_envelope_p1";
   if (cfg.ic == "rotating_star" && uryu_rotation_law
       && (!(cfg.initial_data_differential_rotation_length_b > 0.0_rt)
           || !std::isfinite(
              cfg.initial_data_differential_rotation_length_b
           ))) {
      amrex::Abort(
         "Uryu rotating-star initial data requires a positive finite "
         "differential_rotation_length_b"
      );
   }
   if (cfg.ic == "rotating_star"
       && cfg.initial_data_rotation_law == "q_power"
       && (!(cfg.initial_data_rotation_power_q >= 1.0_rt)
           || !std::isfinite(cfg.initial_data_rotation_power_q))) {
      amrex::Abort(
         "q-power rotating-star initial data requires finite "
         "rotation_power_q >= 1"
      );
   }
   bool perturbation_supported = false;
   for (InitialDataPerturbationInfo const& perturbation
        : InitialDataPerturbationCatalog) {
      perturbation_supported = perturbation_supported
         || perturbation.name == cfg.initial_data_perturbation;
   }
   if (!perturbation_supported) {
      amrex::Abort(
         "initial_data.perturbation is not present in schema/methods.toml"
      );
   }
   if (cfg.initial_data_perturbation != "none") {
      const bool stellar_equilibrium =
         cfg.ic == "tov" || cfg.ic == "rotating_star";
      const bool disk_equilibrium = cfg.ic == "equilibrium_torus";
      if (!stellar_equilibrium && !disk_equilibrium) {
         amrex::Abort(
            "initial-data equilibrium perturbations require pyfluid.ic = "
            "tov, rotating_star, or equilibrium_torus"
         );
      }
      if (disk_equilibrium
          && cfg.initial_data_perturbation == "density_radial_font2002") {
         amrex::Abort(
            "density_radial_font2002 is a stellar-collapse perturbation and "
            "does not support pyfluid.ic = equilibrium_torus"
         );
      }
      if (cfg.eos != "polytrope" && cfg.eos != "hybrid_polytrope"
          && cfg.eos != "compose") {
         amrex::Abort(
            "initial-data equilibrium perturbations require a cold "
            "polytrope, hybrid-polytrope, or CompOSE EoS"
         );
      }
   }
   if (cfg.initial_data_perturbation == "cold_pressure_scale"
       && (!(cfg.initial_data_perturbation_pressure_scale > 0.0_rt)
           || !std::isfinite(
              cfg.initial_data_perturbation_pressure_scale
           )
           || std::abs(cfg.initial_data_perturbation_pressure_scale - 1.0_rt)
              > 0.1_rt)) {
      amrex::Abort(
         "cold_pressure_scale requires a finite positive pressure scale "
         "within 10 percent of unity"
      );
   }
   if (cfg.initial_data_perturbation
       == "density_azimuthal_espino2019") {
#if (AMREX_SPACEDIM != 3)
      amrex::Abort(
         "density_azimuthal_espino2019 requires the 3D executable"
      );
#endif
      if (!std::isfinite(cfg.initial_data_perturbation_amplitude)
          || std::abs(cfg.initial_data_perturbation_amplitude) > 0.1_rt
          || cfg.initial_data_perturbation_mode < 1
          || cfg.initial_data_perturbation_mode > 16
          || !std::isfinite(cfg.initial_data_perturbation_phase)
          || cfg.initial_data_perturbation_radius == 0.0_rt
          || !std::isfinite(cfg.initial_data_perturbation_radius)) {
         amrex::Abort(
            "density_azimuthal_espino2019 requires |amplitude| <= 0.1, "
            "mode in [1,16], a finite phase, and positive or automatic "
            "perturbation_radius"
         );
      }
   }
   if (cfg.initial_data_perturbation == "density_radial_font2002"
       && (!std::isfinite(cfg.initial_data_perturbation_amplitude)
           || std::abs(cfg.initial_data_perturbation_amplitude) > 0.1_rt
           || cfg.initial_data_perturbation_radius == 0.0_rt
           || !std::isfinite(cfg.initial_data_perturbation_radius))) {
      amrex::Abort(
         "density_radial_font2002 requires |amplitude| <= 0.1 and a "
         "positive or automatic perturbation_radius"
      );
   }
   if (cfg.initial_data_perturbation == "density_radial_font2002"
       && cfg.eos != "hybrid_polytrope" && cfg.eos != "compose") {
      amrex::Abort(
         "density_radial_font2002 requires a hybrid-polytrope or CompOSE "
         "EoS so pressure can remain fixed while thermodynamics are rebuilt"
      );
   }
   bool magnetic_seed_supported = false;
   for (InitialDataMagneticSeedInfo const& seed
        : InitialDataMagneticSeedCatalog) {
      magnetic_seed_supported = magnetic_seed_supported
         || seed.name == cfg.initial_data_magnetic_seed;
   }
   if (!magnetic_seed_supported) {
      amrex::Abort(
         "initial_data.magnetic_seed is not present in schema/methods.toml"
      );
   }
   const bool constructs_magnetic_seed =
      cfg.initial_data_magnetic_seed != "none";
   if (constructs_magnetic_seed) {
#if (AMREX_SPACEDIM != 3)
      amrex::Abort("magnetic initial-data seeds require the 3D executable");
#endif
      if (physical_system(cfg) != PhysicalSystem::RMHD || cfg.ct != "on"
          || cfg.initial_data_magnetic != "id"
          || cfg.mesh_coordinate != "cartesian"
          || (cfg.ic != "tov" && cfg.ic != "rotating_star")
          || cfg.initial_data_disk != "none"
          || cfg.spacetime_initial_data != "xcts"
          || cfg.spacetime_matter_coupling != "coupled"
          || !cfg.initial_data_input_file.empty()
          || !cfg.initial_data_solve_only) {
         amrex::Abort(
            "magnetic initial-data seeds currently require a fresh 3D "
            "Cartesian RMHD+CT stellar coupled-XCTS solve-only artifact "
            "without a disk"
         );
      }
      if (!(cfg.initial_data_magnetic_seed_pressure_cutoff >= 0.0_rt)
          || !(cfg.initial_data_magnetic_seed_pressure_cutoff < 1.0_rt)
          || !std::isfinite(cfg.initial_data_magnetic_seed_pressure_cutoff)
          || !(cfg.initial_data_magnetic_seed_exponent >= 1.0_rt)
          || !std::isfinite(cfg.initial_data_magnetic_seed_exponent)
          || !(cfg.initial_data_magnetic_seed_maximum_comoving_field > 0.0_rt)
          || !std::isfinite(
             cfg.initial_data_magnetic_seed_maximum_comoving_field
          )) {
         amrex::Abort(
            "pressure-poloidal magnetic seed requires 0 <= pressure_cutoff "
            "< 1, finite exponent >= 1, and positive finite maximum "
            "comoving field"
         );
      }
   }
   bool magnetic_equilibrium_supported = false;
   for (InitialDataMagneticEquilibriumInfo const& equilibrium
        : InitialDataMagneticEquilibriumCatalog) {
      magnetic_equilibrium_supported = magnetic_equilibrium_supported
         || equilibrium.name == cfg.initial_data_magnetic_equilibrium;
   }
   if (!magnetic_equilibrium_supported) {
      amrex::Abort(
         "initial_data.magnetic_equilibrium is not present in "
         "schema/methods.toml"
      );
   }
   const bool constructs_magnetic_equilibrium =
      cfg.initial_data_magnetic_equilibrium != "none";
   if (constructs_magnetic_seed && constructs_magnetic_equilibrium) {
      amrex::Abort(
         "a kinematic magnetic seed and a magnetohydrostatic magnetic "
         "equilibrium cannot be selected together"
      );
   }
   if (constructs_magnetic_equilibrium) {
#if (AMREX_SPACEDIM != 3)
      amrex::Abort(
         "magnetohydrostatic initial data require the 3D executable"
      );
#endif
      if (physical_system(cfg) != PhysicalSystem::RMHD || cfg.ct != "on"
          || cfg.initial_data_magnetic != "id"
          || cfg.mesh_coordinate != "cartesian"
          || (cfg.ic != "tov" && cfg.ic != "rotating_star")
          || cfg.initial_data_disk != "none"
          || cfg.spacetime_initial_data != "xcts"
          || cfg.spacetime_matter_coupling != "coupled"
          || !cfg.initial_data_input_file.empty()
          || !cfg.initial_data_solve_only) {
         amrex::Abort(
            "magnetohydrostatic initial data currently require a fresh 3D "
            "Cartesian RMHD+CT stellar coupled-XCTS solve-only artifact "
            "without a disk"
         );
      }
      if (!std::isfinite(
             cfg.initial_data_magnetic_equilibrium_poloidal_magnetization
          )
          || !std::isfinite(
             cfg.initial_data_magnetic_equilibrium_nonlinear_poloidal
          )
          || !(cfg.initial_data_magnetic_equilibrium_relaxation > 0.0_rt)
          || !(cfg.initial_data_magnetic_equilibrium_relaxation <= 1.0_rt)
          || !std::isfinite(
             cfg.initial_data_magnetic_equilibrium_relaxation
          )) {
         amrex::Abort(
            "poloidal_grad_shafranov requires a finite poloidal "
            "magnetization, a finite nonlinear-poloidal coefficient, and "
            "0 < magnetic-equilibrium relaxation <= 1"
         );
      }
   }
   const bool oriented_poloidal_field =
      cfg.initial_data_magnetic_seed == "pressure_poloidal"
      || (cfg.initial_data_magnetic_equilibrium == "poloidal_grad_shafranov"
          && cfg.initial_data_magnetic_equilibrium_poloidal_magnetization
             != 0.0_rt);
   if (oriented_poloidal_field) {
      const amrex::Real axis_norm = std::sqrt(
         cfg.initial_data_rotation_axis1 * cfg.initial_data_rotation_axis1
         + cfg.initial_data_rotation_axis2 * cfg.initial_data_rotation_axis2
         + cfg.initial_data_rotation_axis3 * cfg.initial_data_rotation_axis3
      );
      const std::array<amrex::Real, 3> axis{{
         cfg.initial_data_rotation_axis1,
         cfg.initial_data_rotation_axis2,
         cfg.initial_data_rotation_axis3,
      }};
      const std::array<std::array<std::string, 2>, 3> side_boundaries{{
         {{cfg.bc_xlo, cfg.bc_xhi}},
         {{cfg.bc_ylo, cfg.bc_yhi}},
         {{cfg.bc_zlo, cfg.bc_zhi}},
      }};
      const amrex::Real axis_tolerance = 64.0_rt
         * std::numeric_limits<amrex::Real>::epsilon()
         * std::max(axis_norm, amrex::Real(1.0_rt));
      for (int direction = 0; direction < 3; ++direction) {
         for (std::string const& side : side_boundaries[direction]) {
            const std::string selected = trim_copy(side).empty()
               ? trim_copy(cfg.bc) : trim_copy(side);
            if (selected == "symmetry"
                && std::abs(std::abs(axis[direction]) - axis_norm)
                   > axis_tolerance) {
               amrex::Abort(
                  "an oriented poloidal magnetic field may use a Cartesian "
                  "reflection plane only normal to its magnetic axis"
               );
            }
         }
      }
   }
   const PhysicalSystem selected_system = physical_system(cfg);
   if (cfg.relativistic_exterior == "vacuum"
       && selected_system != PhysicalSystem::RHD) {
      amrex::Abort(
         "pyfluid.relativistic_exterior = vacuum currently requires "
         "pyfluid.physics = rhd; magnetized vacuum needs an explicit "
         "electromagnetic exterior closure"
      );
   }
   if (cfg.relativistic_exterior == "vacuum"
       && (cfg.initial_data_atmosphere != "flat"
           || cfg.initial_data_atmosphere_density != -1.0_rt
           || cfg.initial_data_atmosphere_pressure != -1.0_rt)) {
      amrex::Abort(
         "pyfluid.relativistic_exterior = vacuum does not accept a "
         "nondefault initial_data atmosphere profile or state"
      );
   }
   const bool loaded_physical_id = !cfg.initial_data_input_file.empty();
   const bool rmhd_stellar_physical_id =
      selected_system == PhysicalSystem::RMHD
      && cfg.ct == "on"
      && cfg.initial_data_magnetic == "id"
      && (cfg.ic == "tov" || cfg.ic == "rotating_star")
      && (constructs_magnetic_seed || constructs_magnetic_equilibrium
          || loaded_physical_id);
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
   require_supported_option(
      cfg.spacetime_outer_boundary,
      Z4cOuterBoundaryOptions,
      "spacetime.outer_boundary"
   );
   if (cfg.spacetime_sommerfeld_power < -1) {
      amrex::Abort("spacetime.sommerfeld_power must be -1 (pure wave) or nonnegative");
   }
   if (cfg.gamma <= 1.0_rt) {
      amrex::Abort("pyfluid.gamma must be greater than 1");
   }
   if (cfg.polytropic_k <= 0.0_rt) {
      amrex::Abort("pyfluid.polytropic_k must be positive");
   }
   if (cfg.polytropic_gamma <= 1.0_rt) {
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
      if (!(cfg.polytropic_density_breaks[segment] > 0.0_rt)
          || (segment > 0
              && !(cfg.polytropic_density_breaks[segment]
                   > cfg.polytropic_density_breaks[segment - 1]))) {
         amrex::Abort(
            "pyfluid.polytropic_density_breaks must be positive and strictly increasing"
         );
      }
      if (!(cfg.polytropic_segment_gammas[segment] > 1.0_rt)) {
         amrex::Abort(
            "pyfluid.polytropic_segment_gammas entries must be greater than 1"
         );
      }
   }
   if (cfg.thermal_gamma <= 1.0_rt) {
      amrex::Abort("pyfluid.thermal_gamma must be greater than 1");
   }
   if (cfg.eos_preset != "custom"
       && cfg.eos != "polytrope"
       && cfg.eos != "hybrid_polytrope") {
      amrex::Abort(
         "pyfluid.eos_preset requires pyfluid.eos = polytrope or hybrid_polytrope"
      );
   }
   if (cfg.eos == "compose") {
      if (cfg.eos_table.empty()) {
         amrex::Abort("pyfluid.eos = compose requires pyfluid.eos_table");
      }
      if (!cfg.tabulated_eos) {
         amrex::Abort("the configured CompOSE table was not prepared");
      }
      if (selected_system != PhysicalSystem::RHD
          && selected_system != PhysicalSystem::RMHD) {
         amrex::Abort(
            "native cold CompOSE evolution requires pyfluid.physics = rhd or rmhd"
         );
      }
   } else if (!cfg.eos_table.empty()) {
      amrex::Abort("pyfluid.eos_table requires pyfluid.eos = compose");
   }
   if (cfg.rho_floor <= 0.0_rt || cfg.pressure_floor <= 0.0_rt) {
      amrex::Abort("pyfluid.rho_floor and pyfluid.pressure_floor must be positive");
   }
   if (!((cfg.initial_data_atmosphere_density > 0.0_rt)
         || cfg.initial_data_atmosphere_density == -1.0_rt)
       || !((cfg.initial_data_atmosphere_pressure > 0.0_rt)
            || cfg.initial_data_atmosphere_pressure == -1.0_rt)) {
      amrex::Abort(
         "initial_data.atmosphere_density and atmosphere_pressure must be positive, "
         "or -1 to inherit the runtime C2P floors"
      );
   }
   if ((cfg.initial_data_atmosphere_density > 0.0_rt
        && cfg.initial_data_atmosphere_density < cfg.rho_floor)
       || (cfg.initial_data_atmosphere_pressure > 0.0_rt
           && cfg.initial_data_atmosphere_pressure < cfg.pressure_floor)) {
      amrex::Abort(
         "the initial atmosphere state must not be below the runtime C2P floors"
      );
   }
   if (cfg.initial_data_atmosphere == "power_law"
       && (!(cfg.initial_data_atmosphere_power > 3.0_rt)
           || !((cfg.initial_data_atmosphere_transition_radius > 0.0_rt)
                || cfg.initial_data_atmosphere_transition_radius == -1.0_rt))) {
      amrex::Abort(
         "power-law initial atmosphere requires atmosphere_power > 3 for finite 3D mass "
         "and a positive atmosphere_transition_radius, or -1 for the TOV-derived radius"
      );
   }
   const amrex::Real rotation_axis_norm_squared =
      cfg.initial_data_rotation_axis1 * cfg.initial_data_rotation_axis1
      + cfg.initial_data_rotation_axis2 * cfg.initial_data_rotation_axis2
      + cfg.initial_data_rotation_axis3 * cfg.initial_data_rotation_axis3;
   if (!std::isfinite(cfg.initial_data_angular_velocity)
       || !(rotation_axis_norm_squared > 0.0_rt)
       || !std::isfinite(rotation_axis_norm_squared)
       || !std::isfinite(cfg.initial_data_rotation_center1)
       || !std::isfinite(cfg.initial_data_rotation_center2)
       || !std::isfinite(cfg.initial_data_rotation_center3)
       || !(cfg.initial_data_matter_relaxation > 0.0_rt)
       || !(cfg.initial_data_matter_relaxation <= 1.0_rt)
       || !(cfg.initial_data_matter_relative_tolerance > 0.0_rt)
       || cfg.initial_data_matter_max_iterations < 2
       || cfg.initial_data_rotation_continuation_steps < 1) {
      amrex::Abort(
         "rotating initial data requires finite angular velocity/center, a "
         "nonzero finite axis, 0 < matter_relaxation <= 1, positive "
         "matter_relative_tolerance, matter_max_iterations >= 2, and "
         "rotation_continuation_steps >= 1"
      );
   }
   amrex::Real previous_rotation_magnitude = 0.0_rt;
   for (amrex::Real target : cfg.initial_data_rotation_continuation_targets) {
      const amrex::Real magnitude = std::abs(target);
      if (!std::isfinite(target)
          || target * cfg.initial_data_angular_velocity <= 0.0_rt
          || !(magnitude > previous_rotation_magnitude)
          || !(magnitude < std::abs(cfg.initial_data_angular_velocity))) {
         amrex::Abort(
            "initial_data.rotation_continuation_targets must be finite, "
            "strictly increasing in magnitude with the target rotation sign, "
            "and exclude the final initial_data.angular_velocity"
         );
      }
      previous_rotation_magnitude = magnitude;
   }
   if (!cfg.initial_data_rotation_continuation_targets.empty()
       && cfg.initial_data_rotation_continuation_steps != 1) {
      amrex::Abort(
         "explicit initial_data.rotation_continuation_targets require "
         "rotation_continuation_steps = 1"
      );
   }
   if (cfg.passive_scalars < 0 || cfg.passive_scalars > MaxPassiveScalars) {
      amrex::Abort("pyfluid.passive_scalars currently supports 0 or 1");
   }
   require_supported_option(cfg.passive_profile, PassiveProfileOptions, "pyfluid.passive_profile");
   if (cfg.passive_sigma == 0.0_rt || cfg.passive_sigma < -1.0_rt) {
      amrex::Abort("pyfluid.passive_sigma must be positive, or -1 to reuse pyfluid.bump_sigma");
   }
   require_supported_option(cfg.profile, ProfileOptions, "pyfluid.profile");
   require_supported_option(cfg.profile_detail, ProfileDetailOptions, "pyfluid.profile_detail");
   require_supported_option(cfg.mesh_patch_box_ownership, MeshPatchBoxOwnershipOptions, "mesh.patch_box_ownership");
   require_supported_option(cfg.mesh_patch_box_domain, MeshPatchBoxDomainOptions, "mesh.patch_box_domain");
   if (cfg.physical_check_int < 0) {
      amrex::Abort("pyfluid.physical_check_int must be non-negative; 0 disables accepted-state physical checks");
   }
   if (cfg.debug_stage_output && cfg.timestep != "ssprk3") {
      amrex::Abort(
         "pyfluid.debug_stage_output currently requires pyfluid.timestep = ssprk3"
      );
   }
   if (cfg.debug_stage_output && cfg.amr_max_level > 0) {
      amrex::Abort(
         "pyfluid.debug_stage_output currently requires amr.max_level = 0"
      );
   }
   if (cfg.rmhd_candidate_guard != "off"
       && cfg.rmhd_candidate_guard != "pwc"
       && cfg.rmhd_candidate_guard != "pwl-pwc") {
      amrex::Abort(
         "pyfluid.rmhd_candidate_guard supports off, pwc, or pwl-pwc"
      );
   }
   if (cfg.rmhd_candidate_riemann != "same"
       && cfg.rmhd_candidate_riemann != "llf"
       && cfg.rmhd_candidate_riemann != "hll") {
      amrex::Abort(
         "pyfluid.rmhd_candidate_riemann supports same, llf, or hll"
      );
   }
   if (!(cfg.rmhd_candidate_dmp_factor >= 1.0_rt)
       || !std::isfinite(cfg.rmhd_candidate_dmp_factor)) {
      amrex::Abort(
         "pyfluid.rmhd_candidate_dmp_factor must be finite and at least 1"
      );
   }
   if (cfg.rmhd_candidate_guard == "off"
       && cfg.rmhd_candidate_riemann != "same") {
      amrex::Abort(
         "pyfluid.rmhd_candidate_riemann requires an enabled "
         "pyfluid.rmhd_candidate_guard"
      );
   }
   if (selected_system != PhysicalSystem::RMHD
       && (cfg.rmhd_candidate_guard != "off"
           || cfg.rmhd_candidate_riemann != "same")) {
      amrex::Abort(
         "RMHD candidate correction options apply only to "
         "pyfluid.physics = rmhd"
      );
   }
   const InitialDataOption& selected_initial_data = initial_data_option(cfg.ic);
   if (!contains_physical_system(
          selected_initial_data.physical_systems, selected_system
       ) && !rmhd_stellar_physical_id) {
      amrex::Abort(
         "pyfluid.ic = " + cfg.ic
         + " is incompatible with the selected pyfluid.physics system"
      );
   }
   if (physical_system_has_magnetic_field(selected_system)
       && cfg.initial_data_magnetic == "none") {
      amrex::Abort(
         "initial_data.magnetic = none requires a non-magnetized physical system; "
         "use zero to initialize an allocated magnetic field to zero"
      );
   }
   if (spacetime_stores_state(cfg)
       && (cfg.initial_data_spacetime == "none"
           || cfg.initial_data_lapse == "none"
           || cfg.initial_data_shift == "none")) {
      amrex::Abort(
         "stored fixed/evolved spacetime requires initial_data.spacetime, lapse, and shift storage; "
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
      if (cfg.passive_scalars != 0 || cfg.diffusion != "none"
          || cfg.source != "none" || cfg.gravity != "none" || cfg.solid_geometry != "off") {
         amrex::Abort(
            "relativistic-fluid runtime evolution currently supports no passive scalars, diffusion, "
            "Newtonian sources/gravity, or solid geometry"
         );
      }
      const bool reference_metric_rhd_evolution =
         !cfg.initial_data_solve_only
         && selected_system == PhysicalSystem::RHD
         && spherical_reference_metric_rhd_layout
         && cfg.spacetime_evolution == "fixed"
         && cfg.initial_data_input_file.empty()
         && analytic_fixed_spacetime_kernels_selected(
               cfg, cfg.ic, cfg.spacetime_initial_data
            );
      const bool evolved_reference_metric_rhd =
         selected_system == PhysicalSystem::RHD
         && cfg.spacetime_evolution == "evolved"
         && cfg.spacetime_formulation == "fccz4";
      if ((!cfg.initial_data_solve_only && cfg.mesh_coordinate != "cartesian"
           && !reference_metric_rhd_evolution && !evolved_reference_metric_rhd)
          || cfg.mesh_patch_system != "single") {
         amrex::Abort(
            "relativistic-fluid runtime evolution supports a single Cartesian patch, "
            "regular-chart evolved fCCZ4 RHD, or analytic fixed-spacetime RHD on the single-patch spherical "
            "r/r-theta reference-metric layouts"
         );
      }
      if (reference_metric_rhd_evolution && cfg.amr_max_level != 0) {
         amrex::Abort(
            "spherical reference-metric RHD currently requires amr.max_level = 0; "
            "coarse-fine well-balanced momentum reflux is not yet validated"
         );
      }
      if (reference_metric_rhd_evolution
          && cfg.spacetime_fixed_metric == "schwarzschild_isotropic"
          && (cfg.spacetime_origin1 != 0.0_rt
              || cfg.spacetime_origin2 != 0.0_rt
              || cfg.spacetime_origin3 != 0.0_rt)) {
         amrex::Abort(
            "spherical isotropic-Schwarzschild reference-metric evolution "
            "requires spacetime.origin1/2/3 = 0"
         );
      }
      const bool uses_characteristic_reconstruction =
         reconstruction_is_characteristic(recon_method_code(cfg))
         || (!cfg.recon_fine.empty()
             && reconstruction_is_characteristic(recon_method_code_from_name(
                cfg.recon_fine, "pyfluid.recon_fine"
             )));
      if (uses_characteristic_reconstruction) {
         amrex::Abort(
            "relativistic characteristic reconstruction is not implemented; "
            "select a component method"
         );
      }
      if (selected_system == PhysicalSystem::RHD
          && cfg.riemann != "llf" && cfg.riemann != "hll"
          && cfg.riemann != "hlle") {
         amrex::Abort(
            "RHD runtime evolution supports pyfluid.riemann = llf, hll, or hlle"
         );
      }
      if (cfg.ic == "fixed_metric_hse" && cfg.eos != "ideal_gas") {
         amrex::Abort("fixed_metric_hse currently requires pyfluid.eos = ideal_gas");
      }
      if (cfg.ic == "fixed_metric_hse" && cfg.spacetime_fixed_metric == "minkowski") {
         amrex::Abort("pyfluid.ic = fixed_metric_hse requires a non-Minkowski fixed metric");
      }
      if (cfg.ic == "tov" || cfg.ic == "rotating_star") {
         if ((selected_system != PhysicalSystem::RHD
              && !rmhd_stellar_physical_id)
             || !spacetime_stores_state(cfg)
             || (cfg.eos != "polytrope" && cfg.eos != "hybrid_polytrope"
                 && cfg.eos != "compose")) {
            amrex::Abort(
               "TOV/rotating-star initial data require RHD, fixed or evolved spacetime, and a cold-capable polytrope, hybrid_polytrope, or compose EOS"
            );
         }
         if (!(cfg.tov_central_density > cfg.rho_floor)
             || !(cfg.tov_absolute_tolerance > 0.0_rt)
             || !(cfg.tov_relative_tolerance > 0.0_rt)
             || cfg.tov_max_steps < 16) {
            amrex::Abort(
               "TOV initial data requires central_density > rho_floor, positive solve tolerances, and max_steps >= 16"
            );
         }
      }
      if (cfg.ic == "black_hole"
          && (selected_system != PhysicalSystem::RHD
              || !spacetime_stores_state(cfg))) {
         amrex::Abort(
            "black_hole XCTS initial data require RHD and stored fixed or "
            "evolved spacetime"
         );
      }
      if (cfg.ic == "rotating_star"
          && cfg.initial_data_angular_velocity != 0.0_rt
          && cfg.spacetime_initial_data != "xcts") {
         amrex::Abort(
            "nonzero-spin rotating_star currently requires "
            "spacetime.initial_data = xcts; the shared matter law can feed "
            "CTT/CTTK, but those branches do not yet close stationarity"
         );
      }
      if (cfg.ic == "rotating_star"
          && cfg.initial_data_angular_velocity != 0.0_rt
          && AMREX_SPACEDIM != 3) {
         const bool reduced_axisymmetric_xcts =
            spherical_axisymmetric_xcts_layout
            && cfg.initial_data_solve_only;
         if (!reduced_axisymmetric_xcts) {
            amrex::Abort(
               "nonzero-spin rotating_star requires either 3D Cartesian "
               "materialization or a 2D spherical r-theta solve-only XCTS run"
            );
         }
      }
      if (spacetime_evolves(cfg) && cfg.spacetime_fixed_metric != "minkowski") {
         amrex::Abort("spacetime.fixed_metric applies to fixed spacetime only");
      }
      if (spacetime_evolves(cfg)) {
         if (cfg.spacetime_amplitude < 0.0_rt || cfg.spacetime_wave_number <= 0.0_rt) {
            amrex::Abort("evolved spacetime amplitude must be non-negative and wave_number positive");
         }
         if (cfg.spacetime_slow_start_strength < 0.0_rt) {
            amrex::Abort("spacetime.slow_start_strength must be nonnegative");
         }
         if (cfg.spacetime_slow_start_time <= 0.0_rt) {
            amrex::Abort("spacetime.slow_start_time must be positive");
         }
         if (!is_supported_spatial_order(cfg.spacetime_prolongation_order)) {
            amrex::Abort("spacetime.prolongation_order must be a catalogued spatial order");
         }
         if (cfg.spacetime_ko_dissipation < 0.0_rt) {
            amrex::Abort("spacetime.ko_dissipation must be nonnegative");
         }
         if (cfg.spacetime_chi_floor <= 0.0_rt) {
            amrex::Abort("spacetime.chi_floor must be positive");
         }
         if (cfg.spacetime_lapse_floor <= 0.0_rt) {
            amrex::Abort("spacetime.lapse_floor must be positive");
         }
      } else if (cfg.spacetime_matter_coupling != "coupled") {
         amrex::Abort(
            "spacetime.matter_coupling = vacuum is only meaningful for evolved spacetime"
         );
      }
   }
   if (cfg.spacetime_formulation == "fccz4") {
      if (!spacetime_evolves(cfg)) {
         amrex::Abort("spacetime.formulation = fccz4 requires evolved spacetime");
      }
      if (cfg.spacetime_black_hole_interior == "excision") {
         amrex::Abort("fCCZ4 excision characteristics and one-sided reference stencils are not implemented");
      }
      if (cfg.mesh_coordinate != "cartesian" && cfg.spacetime_outer_boundary != "extrapolate") {
         amrex::Abort("curvilinear fCCZ4 radiative boundary transport is not implemented; "
                      "regular-chart controls currently use extrapolate");
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
      if (cfg.mesh_patch_inner_radius <= 0.0_rt || cfg.mesh_patch_outer_radius <= cfg.mesh_patch_inner_radius) {
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
   if (cfg.ic == "advection_medley") {
      if (mesh_coordinate != MeshCartesian
          || AMREX_SPACEDIM != 1 || mesh_axis(mesh_axes, 0) != MeshAxisX) {
         amrex::Abort("advection_medley requires a 1D Cartesian x layout");
      }
      if (cfg.rho0 <= 0.0_rt || cfg.pressure0 <= 0.0_rt || cfg.amplitude <= 0.0_rt) {
         amrex::Abort("advection_medley requires positive rho0, pressure0, and amplitude");
      }
   }
   if (cfg.ic == "characteristic_medley") {
      if (cfg.eos != "ideal_gas" || mesh_coordinate != MeshCartesian
          || AMREX_SPACEDIM != 1 || mesh_axis(mesh_axes, 0) != MeshAxisX) {
         amrex::Abort("characteristic_medley requires a 1D Cartesian x ideal-gas layout");
      }
      if (cfg.rho0 <= 0.0_rt || cfg.pressure0 <= 0.0_rt || cfg.amplitude <= 0.0_rt || cfg.v1_0 != 0.0_rt) {
         amrex::Abort("characteristic_medley requires positive rho0, pressure0, and amplitude with v1_0 = 0");
      }
   }
   if (cfg.ic == "mhd_characteristic_medley") {
      if (cfg.eos != "ideal_gas" || mesh_coordinate != MeshCartesian
          || AMREX_SPACEDIM != 1 || mesh_axis(mesh_axes, 0) != MeshAxisX) {
         amrex::Abort("mhd_characteristic_medley requires a 1D Cartesian x ideal-gas layout");
      }
      const amrex::Real transverse_b2 = cfg.b2_0 * cfg.b2_0 + cfg.b3_0 * cfg.b3_0;
      if (cfg.rho0 <= 0.0_rt || cfg.pressure0 <= 0.0_rt || cfg.amplitude <= 0.0_rt
          || cfg.v1_0 != 0.0_rt || cfg.v2_0 != 0.0_rt || cfg.v3_0 != 0.0_rt
          || cfg.b1_0 == 0.0_rt || transverse_b2 == 0.0_rt) {
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
      if (shock_normal2 <= 0.0_rt) {
         amrex::Abort("sod_plane requires a nonzero active shock normal");
      }
   }
   if (cfg.ic == "acoustic_packet_x") {
      if (cfg.bump_sigma <= 0.0_rt) {
         amrex::Abort("acoustic_packet_x requires positive pyfluid.bump_sigma");
      }
      if (cfg.acoustic_transverse_sigma < 0.0_rt) {
         amrex::Abort("acoustic_packet_x requires non-negative pyfluid.acoustic_transverse_sigma; zero selects a planar packet");
      }
      if (cfg.v1_0 != 0.0_rt || cfg.v2_0 != 0.0_rt || cfg.v3_0 != 0.0_rt) {
         amrex::Abort("acoustic_packet_x currently requires zero background velocity");
      }
   }
   if (cfg.ic == "linear_wave_x") {
      if (!physical_system_has_magnetic_field(selected_system)
          && !valid_hydro_wave_family(cfg.wave_family)) {
         amrex::Abort("hydro linear_wave_x supports wave_family " + supported_option_names(HydroWaveFamilyOptions));
      }
      if (physical_system_has_magnetic_field(selected_system)
          && !valid_mhd_wave_family(cfg.wave_family)) {
         amrex::Abort("MHD linear_wave_x supports wave_family " + supported_option_names(MHDWaveFamilyOptions));
      }
   }
   if (cfg.ic == "jeans_wave_x" && cfg.gravity != "poisson") {
      amrex::Abort("jeans_wave_x requires pyfluid.gravity = poisson");
   }
   if (cfg.ic == "hse_polytrope_x"
      && (cfg.gravity != "fixed" || cfg.gravity_fixed_potential != "uniform_acceleration" || cfg.gravity_a1 == 0.0_rt)) {
      amrex::Abort("hse_polytrope_x requires pyfluid.gravity=fixed uniform_acceleration with nonzero gravity_a1");
   }
   if (cfg.ic == "hse_polytrope_x" && cfg.eos == "compose") {
      amrex::Abort(
         "hse_polytrope_x is an analytic polytropic initial condition and "
         "does not support pyfluid.eos = compose"
      );
   }
   if (cfg.ic == "resistive_current_sheet" || cfg.ic == "reconnecting_shear_layer") {
      if (cfg.ct != "on") {
         amrex::Abort(cfg.ic + " requires pyfluid.ct = on");
      }
      if (mesh_coordinate_code(cfg.mesh_coordinate) != MeshCartesian) {
         amrex::Abort(cfg.ic + " requires Cartesian coordinates");
      }
      const MeshAxes axes = mesh_axes_layout(cfg);
      if (mesh_axis(axes, 0) != MeshAxisX || mesh_axis(axes, 1) != MeshAxisY) {
         amrex::Abort(cfg.ic + " requires active axes x y or x y z");
      }
      if (cfg.bump_sigma <= 0.0_rt || cfg.wave_number <= 0.0_rt || cfg.alfven_amplitude < 0.0_rt) {
         amrex::Abort(
            cfg.ic + " requires positive bump_sigma/wave_number and non-negative alfven_amplitude"
         );
      }
      if (cfg.b1_0 == 0.0_rt) {
         amrex::Abort(cfg.ic + " requires nonzero pyfluid.b1_0");
      }
      if (cfg.ic == "resistive_current_sheet") {
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (side_bc_name(cfg, dir, false) != "periodic" || side_bc_name(cfg, dir, true) != "periodic") {
               amrex::Abort("resistive_current_sheet requires periodic boundaries in every active direction");
            }
         }
      } else {
         if (cfg.kh_secondary_mode <= 0.0_rt) {
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
      if (cfg.wave_number <= 0.0_rt || cfg.amplitude <= 0.0_rt || cfg.b1_0 == 0.0_rt) {
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
      if (cfg.shock_speed <= 0.0_rt) {
         amrex::Abort("double_mach_reflection requires positive pyfluid.shock_speed");
      }
   } else if (cfg.ic == "double_mach_reflection") {
      amrex::Abort("double_mach_reflection initial data requires its analytic boundary profile");
   }
   if (cfg.inflow_rho <= 0.0_rt || cfg.inflow_pressure <= 0.0_rt) {
      amrex::Abort("pyfluid.inflow_rho and pyfluid.inflow_pressure must be positive");
   }
   if (cfg.inflow_background_rho <= 0.0_rt || cfg.inflow_background_pressure <= 0.0_rt) {
      amrex::Abort("pyfluid.inflow_background_rho and pyfluid.inflow_background_pressure must be positive");
   }
   require_supported_option(cfg.inflow_profile, InflowProfileOptions, "pyfluid.inflow_profile");
   if (cfg.inflow_radius <= 0.0_rt) {
      amrex::Abort("pyfluid.inflow_radius must be positive");
   }
   if (cfg.viscosity < 0.0_rt) {
      amrex::Abort("pyfluid.viscosity must be non-negative");
   }
   if (cfg.resistivity < 0.0_rt) {
      amrex::Abort("pyfluid.resistivity must be non-negative");
   }
   if (cfg.diffusion != "none" && cfg.physics == "hydro" && cfg.resistivity > 0.0_rt) {
      amrex::Abort("pyfluid.resistivity requires pyfluid.physics = mhd");
   }
   if (cfg.diffusion != "none" && cfg.physics == "mhd" && cfg.resistivity > 0.0_rt && cfg.ct != "on") {
      amrex::Abort("resistive MHD requires pyfluid.ct = on");
   }
   if (cfg.diffusion == "implicit" && cfg.physics == "mhd" && cfg.resistivity > 0.0_rt) {
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
   if (cfg.diffusion != "none" && cfg.viscosity == 0.0_rt && cfg.resistivity == 0.0_rt) {
      amrex::Abort("pyfluid.diffusion requires positive viscosity and/or resistivity");
   }
   if (cfg.diffusion_cfl <= 0.0_rt) {
      amrex::Abort("pyfluid.diffusion_cfl must be positive");
   }
   if (cfg.diffusion == "implicit") {
      if (cfg.diffusion_tolerance <= 0.0_rt) {
         amrex::Abort("pyfluid.diffusion_tolerance must be positive");
      }
      if (cfg.diffusion_max_iterations < 1) {
         amrex::Abort("pyfluid.diffusion_max_iterations must be positive");
      }
      if (cfg.diffusion_sor_omega <= 0.0_rt || cfg.diffusion_sor_omega >= 2.0_rt) {
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
      if (cfg.source_power <= 0.0_rt) {
         amrex::Abort("pyfluid.source = rotating_compact_heating requires positive pyfluid.source_power");
      }
      if (cfg.source_radius <= 0.0_rt) {
         amrex::Abort("pyfluid.source_radius must be positive");
      }
      if (cfg.source_sigma <= 0.0_rt) {
         amrex::Abort("pyfluid.source_sigma must be positive");
      }
      if (cfg.source_ramp_time < 0.0_rt) {
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
   if (cfg.gravity_constant < 0.0_rt) {
      amrex::Abort("pyfluid.gravity_constant must be non-negative");
   }
   if (cfg.gravity == "fixed") {
      require_supported_option(cfg.gravity_fixed_potential, GravityFixedPotentialOptions, "pyfluid.gravity_fixed_potential");
      if (cfg.gravity_fixed_potential == "uniform_acceleration"
         && cfg.gravity_a1 == 0.0_rt
         && cfg.gravity_a2 == 0.0_rt
         && cfg.gravity_a3 == 0.0_rt) {
         amrex::Abort("pyfluid.gravity=fixed uniform_acceleration requires a nonzero gravity_a* component");
      }
      if (cfg.gravity_fixed_potential == "harmonic" && cfg.gravity_omega <= 0.0_rt) {
         amrex::Abort("pyfluid.gravity=fixed harmonic requires positive pyfluid.gravity_omega");
      }
   } else if (cfg.gravity == "poisson") {
      if (cfg.gravity_constant <= 0.0_rt) {
         amrex::Abort("pyfluid.gravity = poisson requires positive pyfluid.gravity_constant");
      }
      if (cfg.gravity_tolerance <= 0.0_rt) {
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
      if (cfg.solid_wedge_angle_deg <= 0.0_rt || cfg.solid_wedge_angle_deg >= 45.0_rt) {
         amrex::Abort("pyfluid.solid_wedge_angle_deg must satisfy 0 < angle < 45 degrees");
      }
   }
   if (cfg.solid_geometry == "circle") {
#if (AMREX_SPACEDIM < 2)
      amrex::Abort("pyfluid.solid_geometry = circle requires a 2D or 3D executable");
#endif
      if (cfg.solid_radius <= 0.0_rt) {
         amrex::Abort("pyfluid.solid_radius must be positive");
      }
      if (cfg.solid_radius_frequency < 0.0_rt) {
         amrex::Abort("pyfluid.solid_radius_frequency must be non-negative");
      }
      if (cfg.solid_radius_ramp_time < 0.0_rt) {
         amrex::Abort("pyfluid.solid_radius_ramp_time must be non-negative");
      }
      if (cfg.solid_motion == "pulse" && cfg.solid_radius <= std::abs(cfg.solid_radius_amplitude)) {
         amrex::Abort("pyfluid.solid_motion = pulse requires solid_radius > abs(solid_radius_amplitude)");
      }
   }
   const bool solid_enabled = cfg.solid_geometry == "boxes" || cfg.solid_geometry == "wedge" || cfg.solid_geometry == "circle";
   if (cfg.solid_motion == "translate") {
      int nonzero_velocity_components = 0;
      if (std::abs(cfg.solid_velocity_1) > 0.0_rt) {
         ++nonzero_velocity_components;
      }
#if (AMREX_SPACEDIM >= 2)
      if (std::abs(cfg.solid_velocity_2) > 0.0_rt) {
         ++nonzero_velocity_components;
      }
#endif
#if (AMREX_SPACEDIM == 3)
      if (std::abs(cfg.solid_velocity_3) > 0.0_rt) {
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
      if (cfg.diffusion != "explicit" || cfg.viscosity <= 0.0_rt) {
         amrex::Abort("pyfluid.solid_motion = rotate requires explicit viscosity so the moving wall can transfer tangential momentum");
      }
   }
   if (cfg.solid_motion == "pulse" && cfg.solid_geometry != "circle") {
      amrex::Abort("pyfluid.solid_motion = pulse currently requires pyfluid.solid_geometry = circle");
   }
   if (cfg.solid_wall_condition == "no_slip" && (cfg.diffusion != "explicit" || cfg.viscosity <= 0.0_rt)) {
      amrex::Abort("pyfluid.solid_wall_condition = no_slip requires pyfluid.diffusion = explicit and positive pyfluid.viscosity");
   }
   if (solid_enabled && cfg.physics != "hydro") {
      amrex::Abort("solid obstacles currently support pyfluid.physics = hydro; MHD internal walls need CT-aware magnetic wall treatment");
   }
   if (solid_enabled && cfg.diffusion == "implicit") {
      amrex::Abort("solid obstacles currently support pyfluid.diffusion = none or explicit; implicit wall diffusion is not mask-aware yet");
   }
   if (solid_enabled && cfg.diffusion == "explicit" && cfg.resistivity > 0.0_rt) {
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
   if (cfg.tag_threshold <= 0.0_rt) {
      amrex::Abort("amr.tag_threshold must be positive");
   }
   if (cfg.min_feature_length < 0.0_rt) {
      amrex::Abort("amr.min_feature_length must be non-negative; 0 disables the physical-length cutoff");
   }
   if (std::isfinite(cfg.min_feature_length_1) && cfg.min_feature_length_1 < 0.0_rt) {
      amrex::Abort("amr.min_feature_length_1 must be non-negative");
   }
   if (std::isfinite(cfg.min_feature_length_2) && cfg.min_feature_length_2 < 0.0_rt) {
      amrex::Abort("amr.min_feature_length_2 must be non-negative");
   }
   if (std::isfinite(cfg.min_feature_length_3) && cfg.min_feature_length_3 < 0.0_rt) {
      amrex::Abort("amr.min_feature_length_3 must be non-negative");
   }
   if (cfg.min_refine_radius < 0.0_rt) {
      amrex::Abort("amr.min_refine_radius must be non-negative; 0 disables the radial-origin AMR veto");
   }
   if (cfg.solid_refine_band_cells < 0.0_rt) {
      amrex::Abort("amr.solid_refine_band_cells must be non-negative; 0 disables analytic solid-boundary tagging");
   }
   if (cfg.solid_curvature_error < 0.0_rt) {
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
      cfg.kh_secondary_amplitude < 0.0_rt
      || cfg.kh_secondary_mode < 0.0_rt
      || cfg.kh_z_mode < 0.0_rt
      || cfg.kh_envelope_sigma < 0.0_rt
   ) {
      amrex::Abort("KH perturbation amplitudes, modes, and envelope width must be non-negative");
   }
   if (std::abs(cfg.kh_density_amplitude) >= 0.95_rt) {
      amrex::Abort("pyfluid.kh_density_amplitude must satisfy |value| < 0.95");
   }
   require_supported_option(cfg.kh_perturbation, KHPerturbationOptions, "pyfluid.kh_perturbation");
   if (cfg.kh_perturbation == "stream" && cfg.wave_number <= 0.0_rt) {
      amrex::Abort("pyfluid.kh_perturbation = stream requires pyfluid.wave_number > 0");
   }
   if (cfg.rotor_radius <= 0.0_rt || cfg.rotor_transition_radius <= cfg.rotor_radius) {
      amrex::Abort("mhd_rotor requires 0 < pyfluid.rotor_radius < pyfluid.rotor_transition_radius");
   }
   if (cfg.rotor_density <= 0.0_rt) {
      amrex::Abort("pyfluid.rotor_density must be positive");
   }
   if (cfg.beam_length <= 0.0_rt || cfg.beam_radius <= 0.0_rt) {
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
      amrex::Abort("pyfluid.ng must be at least 3 for five-point reconstruction");
   }
   if (cfg.max_step < 0) {
      amrex::Abort("pyfluid.max_step must be non-negative");
   }
   if (cfg.final_time == 0.0_rt || cfg.final_time < -1.0_rt) {
      amrex::Abort("pyfluid.final_time must be positive or -1 to disable time targeting");
   }
   if (cfg.checkpoint_int < -1) {
      amrex::Abort("pyfluid.checkpoint_int must be -1, 0, or a positive integer");
   }
   if (!cfg.restart_file.empty()
       && (!cfg.initial_data_input_file.empty()
           || !cfg.initial_data_output_file.empty()
           || !cfg.initial_data_sequence_output_dir.empty())) {
      amrex::Abort(
         "pyfluid.restart_file is mutually exclusive with physical initial-data "
         "input or output"
      );
   }
   if (!cfg.initial_data_input_file.empty()
       && cfg.initial_data_input_file == cfg.initial_data_output_file) {
      amrex::Abort(
         "initial_data.input_file and output_file must not name the same artifact"
      );
   }
   if (cfg.initial_data_solve_only && cfg.initial_data_output_file.empty()) {
      amrex::Abort(
         "initial_data.solve_only requires initial_data.output_file"
      );
   }
   if (cfg.initial_data_solve_only && !cfg.restart_file.empty()) {
      amrex::Abort("initial_data.solve_only cannot restart evolution checkpoints");
   }
   const bool reduced_axisymmetric_xcts =
      spherical_axisymmetric_xcts_layout
      && (cfg.ic == "rotating_star"
          || (cfg.ic == "tov" && stationary_torus_enabled))
      && cfg.spacetime_initial_data == "xcts";
   const bool reduced_black_hole_xcts = spherical_black_hole_xcts_request;
   const bool reduced_radial_tov =
      spherical_tov_layout
      && cfg.ic == "tov";
   const bool mapped_radial_owner =
      (reduced_axisymmetric_xcts
       && initial_data_owner_list_contains(
          spherical_axisymmetric_xcts_3d_layout
             ? spherical_axisymmetric_xcts_3d_owner.grid_kinds
             : spherical_axisymmetric_xcts_owner.grid_kinds,
          "mapped_radial"
       ))
      || (reduced_radial_tov
          && initial_data_owner_list_contains(
             spherical_tov_owner.grid_kinds, "mapped_radial"
          ))
      || (reduced_black_hole_xcts
          && (initial_data_owner_list_contains(
                 spherical_black_hole_xcts_owner.grid_kinds,
                 "mapped_radial_shell"
              )
              || initial_data_owner_list_contains(
                 spherical_black_hole_xcts_3d_owner.grid_kinds,
                 "mapped_radial_shell"
              )));
   const bool analytic_black_hole_torus =
      ((cartesian_kerr_layout && kerr_schild_requested)
       || (cartesian_trumpet_layout && static_trumpet_requested))
      && cfg.ic == "equilibrium_torus"
      && cfg.initial_data_disk_gravity == "test_fluid";
   if (cfg.initial_data_radial_grid != "uniform"
       && cfg.initial_data_radial_grid != "adaptive") {
      amrex::Abort("initial_data.radial_grid must be uniform or adaptive");
   }
   if (!(cfg.initial_data_adaptive_source_radius_factor > 1.0_rt)
       || cfg.initial_data_adaptive_source_radius_factor > 3.0_rt) {
      amrex::Abort(
         "initial_data.adaptive_source_radius_factor must be in (1, 3]"
      );
   }
   if (!(cfg.initial_data_adaptive_exterior_fraction >= 0.125_rt
         && cfg.initial_data_adaptive_exterior_fraction <= 0.5_rt)) {
      amrex::Abort(
         "initial_data.adaptive_exterior_fraction must be in [0.125, 0.5]"
      );
   }
   if (!(cfg.initial_data_adaptive_shell_width_ratio >= 1.0_rt)
       || cfg.initial_data_adaptive_shell_width_ratio > 128.0_rt
       || !std::isfinite(cfg.initial_data_adaptive_shell_width_ratio)) {
      amrex::Abort(
         "initial_data.adaptive_shell_width_ratio must be finite and in "
         "[1, 128]"
      );
   }
   if (cfg.initial_data_radial_grid == "adaptive"
       && (!mapped_radial_owner || !cfg.initial_data_solve_only)) {
      amrex::Abort(
         "initial_data.radial_grid = adaptive currently requires a solve-only "
         "spherical-r TOV, spherical r-theta rotating-star XCTS solve, or "
         "spherical r-theta or axisymmetric r-theta-phi black-hole XCTS "
         "shell solve"
      );
   }
   if (equilibrium_torus_enabled
       && cfg.initial_data_input_file.empty()
       && (((!reduced_axisymmetric_xcts && !reduced_black_hole_xcts)
             || !cfg.initial_data_solve_only)
           && !analytic_black_hole_torus
           && !cartesian_stellar_torus_request)) {
      amrex::Abort(
         "torus construction requires a solve-only spherical stellar or "
         "black-hole XCTS owner, a 3D Cartesian stellar XCTS owner, or a 3D "
         "Cartesian test-fluid analytic black-hole central object"
      );
   }
   if (equilibrium_torus_enabled
       && cfg.ic != "tov" && cfg.ic != "rotating_star"
       && cfg.ic != "black_hole" && cfg.ic != "equilibrium_torus") {
      amrex::Abort(
         "equilibrium-torus initial data currently require a TOV, "
         "rotating_star, or black_hole central object, or pyfluid.ic = "
         "equilibrium_torus for an analytic black hole"
      );
   }
   if (cfg.ic == "equilibrium_torus"
       && !stationary_torus_enabled) {
      amrex::Abort(
         "pyfluid.ic = equilibrium_torus requires "
         "initial_data.disk = equilibrium_torus"
      );
   }
   if (cfg.ic == "equilibrium_torus" && !analytic_black_hole_torus) {
      amrex::Abort(
         "black-hole equilibrium-torus initial data currently require a 3D "
         "Cartesian Kerr-Schild or static-trumpet spacetime and "
         "disk_gravity = test_fluid"
      );
   }
   if (cfg.initial_data_solve_only && cfg.mesh_coordinate != "cartesian"
       && cfg.ic != "tov" && !reduced_axisymmetric_xcts
       && !reduced_black_hole_xcts
       && !loaded_3d_materialization_request) {
      amrex::Abort(
         "reduced curvilinear initial-data solves currently support TOV or "
         "rotating-star spherical r-theta XCTS, or black-hole spherical "
         "r-theta/axisymmetric r-theta-phi XCTS"
      );
   }
   if (!cfg.initial_data_sequence_output_dir.empty()
       && (cfg.ic != "rotating_star"
           || cfg.spacetime_initial_data != "xcts"
           || cfg.initial_data_angular_velocity == 0.0_rt)) {
      amrex::Abort(
         "initial_data.sequence_output_dir currently requires nonzero "
         "rotating_star XCTS initial data"
      );
   }
   if (!cfg.initial_data_save_sequence
       && !cfg.initial_data_sequence_output_dir.empty()) {
      amrex::Abort(
         "initial_data.sequence_output_dir requires save_sequence = 1"
      );
   }
   if (!cfg.initial_data_sequence_output_dir.empty()
       && (cfg.initial_data_sequence_output_dir == cfg.initial_data_input_file
           || cfg.initial_data_sequence_output_dir
              == cfg.initial_data_output_file)) {
      amrex::Abort(
         "initial_data.sequence_output_dir must not replace a singular input "
         "or output artifact"
      );
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
   if (!std::isfinite(cfg.diagnostic_constraint_region_extent)
       || !std::isfinite(cfg.diagnostic_constraint_exclusion_radius)
       || !std::isfinite(cfg.diagnostic_constraint_split_radius)
       || cfg.diagnostic_constraint_region_extent == 0.0_rt
       || cfg.diagnostic_constraint_exclusion_radius < 0.0_rt
       || cfg.diagnostic_constraint_split_radius < 0.0_rt) {
      amrex::Abort("diagnostics.constraint_region_extent must be finite and positive "
         "(or negative to disable); constraint_exclusion_radius and constraint_split_radius "
         "must be finite and nonnegative");
   }
   if (cfg.diagnostic_constraint_region_extent > 0.0_rt
       && !(cfg.diagnostic_constraint_exclusion_radius < cfg.diagnostic_constraint_split_radius
            && cfg.diagnostic_constraint_split_radius < cfg.diagnostic_constraint_region_extent)) {
      amrex::Abort("fixed constraint regions require 0 <= constraint_exclusion_radius "
         "< constraint_split_radius < constraint_region_extent");
   }
   if (cfg.diagnostic_surface_radius == 0.0_rt) {
      amrex::Abort("diagnostics.surface_radius must be positive or negative for an automatic inscribed sphere");
   }
   if (cfg.diagnostic_surface_polar_points < 2) {
      amrex::Abort("diagnostics.surface_polar_points must be at least 2");
   }
   if (cfg.diagnostic_multipole_l_max < 0 || cfg.diagnostic_multipole_l_max == 1
       || cfg.diagnostic_multipole_l_max > 16) {
      amrex::Abort("diagnostics.multipole_l_max must be 0 (disabled) or between 2 and 16");
   }
   if (cfg.diagnostic_azimuthal_mode_m_max < 0
       || cfg.diagnostic_azimuthal_mode_m_max > 64) {
      amrex::Abort("diagnostics.azimuthal_mode_m_max must be between 0 and 64");
   }
   if (cfg.diagnostic_matter_mode_m_max < -1
       || cfg.diagnostic_matter_mode_m_max > 64) {
      amrex::Abort(
         "diagnostics.matter_mode_m_max must be -1 (disabled) or between 0 "
         "and 64"
      );
   }
   if (cfg.diagnostic_matter_mode_m_max >= 0) {
      const amrex::Real axis_squared =
         cfg.diagnostic_matter_mode_axis1 * cfg.diagnostic_matter_mode_axis1
         + cfg.diagnostic_matter_mode_axis2 * cfg.diagnostic_matter_mode_axis2
         + cfg.diagnostic_matter_mode_axis3 * cfg.diagnostic_matter_mode_axis3;
      if (!(axis_squared > 0.0_rt) || !std::isfinite(axis_squared)
          || !std::isfinite(cfg.diagnostic_matter_mode_origin1)
          || !std::isfinite(cfg.diagnostic_matter_mode_origin2)
          || !std::isfinite(cfg.diagnostic_matter_mode_origin3)) {
         amrex::Abort(
            "matter azimuthal modes require a finite origin and nonzero "
            "finite axis"
         );
      }
   }
   for (std::string const& field : azimuthal_mode_field_names(cfg)) {
      const bool primitive = field == "rho" || field == "pressure";
      const bool fluid_scalar = scalar_grid_field_component(
         fluid_state_grid_fields(physical_system(cfg), cfg.passive_scalars), field
      ) >= 0;
      const bool spacetime_scalar = spacetime_stores_state(cfg)
         && scalar_grid_field_component(z4c_state_grid_fields(), field) >= 0;
      if (!primitive && !fluid_scalar && !spacetime_scalar) {
         amrex::Abort(
            "diagnostics.azimuthal_mode_fields contains unsupported scalar field '"
            + field + "'"
         );
      }
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

} // namespace pyfluid_amrex
