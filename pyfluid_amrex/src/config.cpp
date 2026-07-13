#include "pyfluid_amrex/config.hpp"

#include <AMReX.H>

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <string_view>

namespace pyfluid_amrex {
namespace {

enum class RestartPolicy {
   Mutable,
   Immutable,
};

enum class StringOptionSet {
   EOS,
   Physics,
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

constexpr std::array<RegisteredStringParameter, 9> MethodParameters{{
   {"eos", &RunConfig::eos, "ideal_gas", StringOptionSet::EOS, RestartPolicy::Immutable,
    "Equation of state closure."},
   {"physics", &RunConfig::physics, "hydro", StringOptionSet::Physics, RestartPolicy::Immutable,
    "Evolved hydrodynamic or magnetohydrodynamic state."},
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
   {"state_interp", &RunConfig::state_interp, "quartic-conservative", StringOptionSet::AMRStateInterp,
    RestartPolicy::Mutable, "Conservative state interpolation used for new fine data."},
   {"initial_refine", &RunConfig::initial_refine, "none", StringOptionSet::AMRInitialRefine,
    RestartPolicy::Mutable, "Optional initial refinement region."},
}};

constexpr std::array<RegisteredScalarParameter<amrex::Real>, 8> AMRRealParameters{{
   {"tag_threshold", &RunConfig::tag_threshold, 0.35, RestartPolicy::Mutable,
    "Dimensionless state-curvature threshold.", &RunConfig::tag_threshold_set},
   {"min_feature_length", &RunConfig::min_feature_length, 0.0, RestartPolicy::Mutable,
    "Global physical resolution guard.", &RunConfig::min_feature_length_set},
   {"min_feature_length_1", &RunConfig::min_feature_length_1, std::numeric_limits<amrex::Real>::quiet_NaN(),
    RestartPolicy::Mutable, "Direction-1 physical resolution guard.", &RunConfig::min_feature_length_1_set},
   {"min_feature_length_2", &RunConfig::min_feature_length_2, std::numeric_limits<amrex::Real>::quiet_NaN(),
    RestartPolicy::Mutable, "Direction-2 physical resolution guard.", &RunConfig::min_feature_length_2_set},
   {"min_feature_length_3", &RunConfig::min_feature_length_3, std::numeric_limits<amrex::Real>::quiet_NaN(),
    RestartPolicy::Mutable, "Direction-3 physical resolution guard.", &RunConfig::min_feature_length_3_set},
   {"min_refine_radius", &RunConfig::min_refine_radius, 0.0, RestartPolicy::Mutable,
    "Curvilinear radial refinement guard."},
   {"solid_refine_band_cells", &RunConfig::solid_refine_band_cells, 1.5, RestartPolicy::Mutable,
    "Solid-interface refinement-band width in cells."},
   {"solid_curvature_error", &RunConfig::solid_curvature_error, 0.0, RestartPolicy::Mutable,
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
   case StringOptionSet::Physics:
      require_registered_option(parameter, value, PhysicsOptions, group);
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

RunConfig::RunConfig()
{
   initialize_registered_parameters(*this);
}

void initialize_registered_parameters(RunConfig& cfg)
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

void query_registered_method_parameters(amrex::ParmParse& pp, RunConfig& cfg)
{
   for (RegisteredStringParameter const& parameter : MethodParameters) {
      pp.queryAdd(parameter.key, cfg.*(parameter.member));
   }
}

void query_registered_amr_parameters(amrex::ParmParse& pp, RunConfig& cfg)
{
   for (RegisteredStringParameter const& parameter : AMRStringParameters) {
      pp.queryAdd(parameter.key, cfg.*(parameter.member));
   }
   query_scalars(pp, cfg, AMRRealParameters);
   query_scalars(pp, cfg, AMRIntParameters);
   query_scalars(pp, cfg, AMRLongParameters);
}

void sync_registered_amr_parameters(amrex::ParmParse& pp, RunConfig const& cfg)
{
   for (RegisteredStringParameter const& parameter : AMRStringParameters) {
      pp.add(parameter.key, cfg.*(parameter.member));
   }
   sync_scalars(pp, cfg, AMRRealParameters);
   sync_scalars(pp, cfg, AMRIntParameters);
   sync_scalars(pp, cfg, AMRLongParameters);
}

void validate_registered_parameters(RunConfig const& cfg)
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

} // namespace pyfluid_amrex
