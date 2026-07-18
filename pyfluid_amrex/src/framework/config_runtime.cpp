#include "pyfluid_amrex/config.hpp"

#include <AMReX.H>

#include <cctype>
#include <cmath>
#include <sstream>

#include "framework/config_options.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

void require_implicit_diffusion_converged(
   DiffusionStats const& stats,
   RunConfig const& cfg,
   char const* context
)
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

int riemann_method_code(RunConfig const& cfg)
{
   return config_options::option_code(
      cfg.riemann, config_options::RiemannOptions, "pyfluid.riemann"
   );
}

int recon_method_code_from_name(std::string const& name, char const* key)
{
   return config_options::option_code(name, config_options::ReconOptions, key);
}

int recon_method_code(RunConfig const& cfg)
{
   return recon_method_code_from_name(cfg.recon, "pyfluid.recon");
}

int timestep_method_code(RunConfig const& cfg)
{
   return config_options::option_code(
      cfg.timestep, config_options::TimestepOptions, "pyfluid.timestep"
   );
}

int diffusion_method_code(RunConfig const& cfg)
{
   return config_options::option_code(
      cfg.diffusion, config_options::DiffusionOptions, "pyfluid.diffusion"
   );
}

int eos_method_code(RunConfig const& cfg)
{
   return config_options::option_code(cfg.eos, config_options::EOSOptions, "pyfluid.eos");
}

EOSParams eos_params(RunConfig const& cfg)
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

bool valid_plotfile_compression(std::string const& value)
{
   if (value == "none" || value == "best" || value == "zlib") {
      return true;
   }
   if (value.empty()) {
      return false;
   }
   if (value.size() == 5 && value.rfind("zlib", 0) == 0
       && std::isdigit(static_cast<unsigned char>(value[4])) != 0) {
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

std::string resolved_plotfile_compression(std::string const& value)
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
   if (value.size() == 5 && value.rfind("zlib", 0) == 0
       && std::isdigit(static_cast<unsigned char>(value[4])) != 0) {
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

std::string trim_copy(std::string value)
{
   while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
      value.erase(value.begin());
   }
   while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
      value.pop_back();
   }
   return value;
}

std::vector<std::string> aux_gf_names(RunConfig const& cfg)
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

bool aux_gfs_enabled(RunConfig const& cfg)
{
   return !aux_gf_names(cfg).empty();
}

bool aux_gfs_contains(RunConfig const& cfg, AuxGridField target)
{
   for (std::string const& name : aux_gf_names(cfg)) {
      if (aux_grid_field_from_name(name) == target) {
         return true;
      }
   }
   return false;
}

amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> configured_min_feature_lengths(
   RunConfig const& cfg
)
{
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> values{};
   values[0] = std::isfinite(cfg.min_feature_length_1)
      ? cfg.min_feature_length_1 : cfg.min_feature_length;
#if (AMREX_SPACEDIM >= 2)
   values[1] = std::isfinite(cfg.min_feature_length_2)
      ? cfg.min_feature_length_2 : cfg.min_feature_length;
#endif
#if (AMREX_SPACEDIM == 3)
   values[2] = std::isfinite(cfg.min_feature_length_3)
      ? cfg.min_feature_length_3 : cfg.min_feature_length;
#endif
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      values[direction] = values[direction] > 0.0 ? values[direction] : -1.0;
   }
   return values;
}

amrex::Real amr_default_lohner_tag_threshold()
{
   return 0.08;
}

amrex::Real amr_default_auto_min_feature_scale(int max_level)
{
   constexpr amrex::Real points_per_feature = 2.0;
   const amrex::Real finest_ratio = std::ldexp(1.0, -max_level);
   return points_per_feature * finest_ratio;
}

std::string side_bc_name(RunConfig const& cfg, int direction, bool high_side)
{
   std::string value;
   if (direction == 0) {
      value = high_side ? cfg.bc_xhi : cfg.bc_xlo;
   }
#if (AMREX_SPACEDIM >= 2)
   if (direction == 1) {
      value = high_side ? cfg.bc_yhi : cfg.bc_ylo;
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (direction == 2) {
      value = high_side ? cfg.bc_zhi : cfg.bc_zlo;
   }
#endif
   value = trim_copy(value);
   return value.empty() ? cfg.bc : value;
}

BoundaryKind boundary_kind_from_name(std::string const& value)
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

BoundaryKind side_boundary_kind(RunConfig const& cfg, int direction, bool high_side)
{
   return boundary_kind_from_name(side_bc_name(cfg, direction, high_side));
}

bool reflective_boundary(BoundaryKind kind)
{
   return kind == BoundaryKind::Reflecting
      || kind == BoundaryKind::Symmetry
      || kind == BoundaryKind::Origin;
}

bool fixed_inflow_boundary(BoundaryKind kind)
{
   return kind == BoundaryKind::Inflow;
}

bool diode_boundary(BoundaryKind kind)
{
   return kind == BoundaryKind::Diode;
}

bool curvilinear_mesh_coordinate(int coordinate)
{
   return coordinate == MeshCylindrical || coordinate == MeshSpherical;
}

bool curvilinear_mesh_enabled(RunConfig const& cfg)
{
   return curvilinear_mesh_coordinate(mesh_coordinate_code(cfg.mesh_coordinate));
}

bool cartesian_patch_box_union_domain_enabled(RunConfig const& cfg)
{
   return cfg.mesh_patch_system == "multi_homogeneous_cartesian"
      && cfg.mesh_patch_box_domain == "union"
      && cfg.mesh_patch_box_count > 0;
}

MeshAxes mesh_axes_layout(RunConfig const& cfg)
{
   return mesh_axes_layout(mesh_coordinate_code(cfg.mesh_coordinate), cfg.mesh_axes);
}

MeshRuntimeLayout mesh_runtime_layout(RunConfig const& cfg)
{
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   return {
      coordinate,
      mesh_axes_layout(coordinate, cfg.mesh_axes),
      mesh_patch_system_code(cfg.mesh_patch_system),
      curvilinear_mesh_coordinate(coordinate),
   };
}

PhysicalSystem physical_system(RunConfig const& cfg)
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

SpacetimeEvolution spacetime_evolution_mode(RunConfig const& cfg)
{
   if (cfg.spacetime_evolution == "fixed") {
      return SpacetimeEvolution::Fixed;
   }
   if (cfg.spacetime_evolution == "evolved") {
      return SpacetimeEvolution::Evolved;
   }
   return SpacetimeEvolution::None;
}

bool spacetime_evolves(RunConfig const& cfg) noexcept
{
   return spacetime_evolves(spacetime_evolution_mode(cfg));
}

bool spacetime_stores_state(RunConfig const& cfg) noexcept
{
   return cfg.spacetime_evolution == "fixed" || spacetime_evolves(cfg);
}

bool spacetime_includes_matter(RunConfig const& cfg) noexcept
{
   return spacetime_stores_state(cfg)
      && cfg.spacetime_matter_coupling == "coupled";
}

FixedSpacetimeParameters fixed_spacetime_parameters(RunConfig const& cfg)
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

bool is_mhd(RunConfig const& cfg)
{
   return physical_system_has_magnetic_field(physical_system(cfg));
}

bool viscosity_enabled(RunConfig const& cfg)
{
   return cfg.viscosity > 0.0;
}

bool resistivity_enabled(RunConfig const& cfg)
{
   return is_mhd(cfg) && cfg.resistivity > 0.0;
}

bool gravity_enabled(RunConfig const& cfg)
{
   return cfg.gravity != "none";
}

bool athena_gravity_coupling_enabled(RunConfig const& cfg)
{
   return gravity_enabled(cfg) && cfg.gravity_coupling == "athena";
}

bool self_gravity_enabled(RunConfig const& cfg)
{
   return cfg.gravity == "poisson";
}

bool fixed_gravity_enabled(RunConfig const& cfg)
{
   return cfg.gravity == "fixed";
}

int state_component_count(RunConfig const& cfg)
{
   return base_component_count(physical_system(cfg)) + cfg.passive_scalars;
}

} // namespace pyfluid_amrex
