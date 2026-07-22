#include "pyfluid_amrex/initial_data/physical_id.hpp"

#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/multilevel_field_io.hpp"
#include "pyfluid_amrex/output_metadata.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace pyfluid_amrex {
namespace {

constexpr int PhysicalIdSchemaVersion = 1;
constexpr char const* PhysicalIdFormat = "PyFluidPhysicalIDHDF5-v1";
constexpr char const* PhysicalFluidField = "physical_fluid";
constexpr char const* AdmGeometryField = "adm_geometry";
constexpr char const* EvolutionFluidField = "evolution_fluid_cache";
constexpr char const* EvolutionSpacetimeField = "evolution_spacetime_cache";

enum AdmComponent : int {
   AdmGamma11 = 0,
   AdmGamma12,
   AdmGamma13,
   AdmGamma22,
   AdmGamma23,
   AdmGamma33,
   AdmK11,
   AdmK12,
   AdmK13,
   AdmK22,
   AdmK23,
   AdmK33,
   AdmLapse,
   AdmShift1,
   AdmShift2,
   AdmShift3,
   AdmNumComponents,
};

std::string trim(std::string_view text)
{
   const std::size_t first = text.find_first_not_of(" \t\r\n");
   if (first == std::string_view::npos) {
      return {};
   }
   const std::size_t last = text.find_last_not_of(" \t\r\n");
   return std::string(text.substr(first, last - first + 1));
}

std::map<std::string, std::string> read_metadata(std::string const& path)
{
   const std::filesystem::path metadata =
      std::filesystem::path(path) / "Header.meta";
   std::ifstream input(metadata);
   if (!input.good()) {
      amrex::Abort("physical initial data require " + metadata.string());
   }
   std::map<std::string, std::string> entries;
   for (std::string line; std::getline(input, line);) {
      if (line.empty() || line.front() != '#') {
         continue;
      }
      const std::size_t equal = line.find('=');
      if (equal == std::string::npos) {
         continue;
      }
      const std::string key = trim(std::string_view(line).substr(1, equal - 1));
      const std::string value = trim(std::string_view(line).substr(equal + 1));
      if (!key.empty()) {
         entries[key] = value;
      }
   }
   return entries;
}

std::string domain_string(amrex::Geometry const& geometry)
{
   std::ostringstream output;
   output << geometry.Domain();
   return output.str();
}

std::string real_vector_string(amrex::Geometry const& geometry, bool upper)
{
   std::ostringstream output;
   output << std::setprecision(17);
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      if (direction != 0) {
         output << ',';
      }
      output << (upper ? geometry.ProbHi(direction) : geometry.ProbLo(direction));
   }
   return output.str();
}

std::string periodicity_string(amrex::Geometry const& geometry)
{
   std::ostringstream output;
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      if (direction != 0) {
         output << ',';
      }
      output << geometry.isPeriodic(direction);
   }
   return output.str();
}

std::string refinement_string(
   amrex::Vector<amrex::IntVect> const& refinement_ratios
)
{
   std::ostringstream output;
   for (std::size_t level = 0; level < refinement_ratios.size(); ++level) {
      if (level != 0) {
         output << ';';
      }
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         if (direction != 0) {
            output << ',';
         }
         output << refinement_ratios[level][direction];
      }
   }
   return output.str();
}

void require_metadata(
   std::map<std::string, std::string> const& metadata,
   std::string const& key,
   std::string const& expected,
   std::string const& path
)
{
   const auto found = metadata.find(key);
   if (found == metadata.end()) {
      amrex::Abort(
         "physical initial-data metadata missing '" + key + "' in " + path
      );
   }
   if (found->second != expected) {
      amrex::Abort(
         "physical initial-data metadata mismatch for '" + key
         + "': artifact='" + found->second + "' current='" + expected + "'"
      );
   }
}

std::string required_metadata_value(
   std::map<std::string, std::string> const& metadata,
   std::string const& key,
   std::string const& path
)
{
   const auto found = metadata.find(key);
   if (found == metadata.end()) {
      amrex::Abort(
         "physical initial-data metadata missing '" + key + "' in " + path
      );
   }
   return found->second;
}

void validate_level_fields(
   ConstPhysicalIdField const& fields,
   int levels,
   int components,
   char const* name
)
{
   if (static_cast<int>(fields.size()) != levels) {
      amrex::Abort(std::string("physical initial-data field count mismatch: ") + name);
   }
   for (amrex::MultiFab const* field : fields) {
      if (field == nullptr || field->nComp() != components) {
         amrex::Abort(
            std::string("physical initial-data component mismatch: ") + name
         );
      }
   }
}

void materialize_adm_geometry(
   ConstPhysicalIdField const& spacetime,
   amrex::Vector<std::unique_ptr<amrex::MultiFab>>& adm
)
{
   adm.resize(spacetime.size());
   for (std::size_t level = 0; level < spacetime.size(); ++level) {
      adm[level] = std::make_unique<amrex::MultiFab>(
         spacetime[level]->boxArray(),
         spacetime[level]->DistributionMap(),
         AdmNumComponents,
         0
      );
      for (amrex::MFIter mfi(*adm[level], amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto source = spacetime[level]->const_array(mfi);
         const auto output = adm[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const Z4cState state = load_z4c_state(source, i, j, k);
            const amrex::Real inverse_chi = 1.0 / state.chi;
            const amrex::Real trace_k = state.k_hat + 2.0 * state.theta;
            const amrex::Real trace_third = trace_k / 3.0;
            output(i, j, k, AdmGamma11) = state.gamma_tilde11 * inverse_chi;
            output(i, j, k, AdmGamma12) = state.gamma_tilde12 * inverse_chi;
            output(i, j, k, AdmGamma13) = state.gamma_tilde13 * inverse_chi;
            output(i, j, k, AdmGamma22) = state.gamma_tilde22 * inverse_chi;
            output(i, j, k, AdmGamma23) = state.gamma_tilde23 * inverse_chi;
            output(i, j, k, AdmGamma33) = state.gamma_tilde33 * inverse_chi;
            output(i, j, k, AdmK11) = inverse_chi
               * (state.a_tilde11 + state.gamma_tilde11 * trace_third);
            output(i, j, k, AdmK12) = inverse_chi
               * (state.a_tilde12 + state.gamma_tilde12 * trace_third);
            output(i, j, k, AdmK13) = inverse_chi
               * (state.a_tilde13 + state.gamma_tilde13 * trace_third);
            output(i, j, k, AdmK22) = inverse_chi
               * (state.a_tilde22 + state.gamma_tilde22 * trace_third);
            output(i, j, k, AdmK23) = inverse_chi
               * (state.a_tilde23 + state.gamma_tilde23 * trace_third);
            output(i, j, k, AdmK33) = inverse_chi
               * (state.a_tilde33 + state.gamma_tilde33 * trace_third);
            output(i, j, k, AdmLapse) = state.lapse;
            output(i, j, k, AdmShift1) = state.shift1;
            output(i, j, k, AdmShift2) = state.shift2;
            output(i, j, k, AdmShift3) = state.shift3;
         });
      }
   }
}

ConstPhysicalIdField const_view(
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> const& fields
)
{
   ConstPhysicalIdField output(fields.size());
   for (std::size_t level = 0; level < fields.size(); ++level) {
      output[level] = fields[level].get();
   }
   return output;
}

void write_field(
   std::string const& path,
   std::string const& name,
   ConstPhysicalIdField const& field,
   amrex::Vector<amrex::Geometry> const& geometry,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   std::string const& compression
)
{
   amrex::Vector<int> level_steps(field.size(), 0);
   write_multilevel_hdf5_field(
      field,
      geometry,
      level_steps,
      refinement_ratios,
      static_cast<int>(field.size()) - 1,
      0.0,
      path,
      name,
      compression,
      PhysicalIdFormat
   );
}

} // namespace

void write_physical_initial_data(
   std::string const& path,
   RunConfig const& cfg,
   PhysicalSystem system,
   amrex::Vector<amrex::Geometry> const& geometry,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   ConstPhysicalIdField const& primitive,
   ConstPhysicalIdField const& evolution_fluid,
   ConstPhysicalIdField const& evolution_spacetime,
   ConstPhysicalIdFaceFields const& face_magnetic
)
{
   const int levels = static_cast<int>(geometry.size());
   if (levels == 0 || static_cast<int>(refinement_ratios.size()) != levels - 1) {
      amrex::Abort("physical initial data require a nonempty consistent hierarchy");
   }
   const int fluid_components = grid_field_component_count(
      physical_fluid_grid_fields(system, cfg.passive_scalars)
   );
   validate_level_fields(primitive, levels, fluid_components, PhysicalFluidField);
   validate_level_fields(
      evolution_fluid, levels, evolution_fluid.front()->nComp(), EvolutionFluidField
   );
   const bool has_spacetime = !evolution_spacetime.empty();
   if (has_spacetime) {
      validate_level_fields(
         evolution_spacetime,
         levels,
         grid_field_component_count(z4c_state_grid_fields()),
         EvolutionSpacetimeField
      );
   }

   if (amrex::ParallelDescriptor::IOProcessor()) {
      if (std::filesystem::exists(path)) {
         amrex::Abort("physical initial-data output already exists: " + path);
      }
      std::filesystem::create_directories(path);
   }
   amrex::ParallelDescriptor::Barrier();
   const std::string compression = resolved_plotfile_compression(
      cfg.checkpoint_compression
   );
   write_field(
      path, PhysicalFluidField, primitive, geometry, refinement_ratios, compression
   );
   write_field(
      path,
      EvolutionFluidField,
      evolution_fluid,
      geometry,
      refinement_ratios,
      compression
   );

   if (has_spacetime) {
      amrex::Vector<std::unique_ptr<amrex::MultiFab>> adm;
      materialize_adm_geometry(evolution_spacetime, adm);
      write_field(
         path,
         AdmGeometryField,
         const_view(adm),
         geometry,
         refinement_ratios,
         compression
      );
      write_field(
         path,
         EvolutionSpacetimeField,
         evolution_spacetime,
         geometry,
         refinement_ratios,
         compression
      );
   }

   const bool has_ct = !face_magnetic[0].empty();
   if (has_ct) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         validate_level_fields(
            face_magnetic[direction], levels, 1, "face magnetic field"
         );
         write_field(
            path,
            checkpoint_face_b_field_name(direction),
            face_magnetic[direction],
            geometry,
            refinement_ratios,
            compression
         );
      }
   }

   if (amrex::ParallelDescriptor::IOProcessor()) {
      std::ofstream header(std::filesystem::path(path) / "Header");
      header << "PyFluid physical initial data v1\n"
             << "Provider-independent physical fields with exact same-grid "
                "evolution caches.\n";
      std::ofstream metadata(std::filesystem::path(path) / "Header.meta");
      metadata << "# type=physical_initial_data\n";
      metadata << "# schema_version=" << PhysicalIdSchemaVersion << '\n';
      metadata << "# format=hdf5\n";
      metadata << "# compression=" << compression << '\n';
      metadata << "# amrex_spacedim=" << AMREX_SPACEDIM << '\n';
      metadata << "# levels=" << levels << '\n';
      metadata << "# physics=" << cfg.physics << '\n';
      metadata << "# fluid_components=" << evolution_fluid.front()->nComp() << '\n';
      metadata << "# physical_fluid_components=" << fluid_components << '\n';
      metadata << "# mesh_coordinate=" << cfg.mesh_coordinate << '\n';
      metadata << "# mesh_axes="
               << mesh_axes_or_default(
                     mesh_coordinate_code(cfg.mesh_coordinate), cfg.mesh_axes
                  )
               << '\n';
      metadata << "# domain=" << domain_string(geometry.front()) << '\n';
      metadata << "# prob_lo=" << real_vector_string(geometry.front(), false) << '\n';
      metadata << "# prob_hi=" << real_vector_string(geometry.front(), true) << '\n';
      metadata << "# periodicity=" << periodicity_string(geometry.front()) << '\n';
      metadata << "# refinement_ratios=" << refinement_string(refinement_ratios) << '\n';
      metadata << "# eos=" << cfg.eos << '\n';
      metadata << "# gamma=" << output_metadata_real_string(cfg.gamma) << '\n';
      metadata << "# polytropic_k="
               << output_metadata_real_string(cfg.polytropic_k) << '\n';
      metadata << "# polytropic_gamma="
               << output_metadata_real_string(cfg.polytropic_gamma) << '\n';
      metadata << "# polytropic_density_breaks="
               << output_metadata_real_vector_string(
                     cfg.polytropic_density_breaks
                  )
               << '\n';
      metadata << "# polytropic_segment_gammas="
               << output_metadata_real_vector_string(
                     cfg.polytropic_segment_gammas
                  )
               << '\n';
      metadata << "# thermal_gamma="
               << output_metadata_real_string(cfg.thermal_gamma) << '\n';
      metadata << "# passive_scalars=" << cfg.passive_scalars << '\n';
      metadata << "# has_spacetime=" << has_spacetime << '\n';
      metadata << "# has_ct=" << has_ct << '\n';
      metadata << "# source_ic=" << cfg.ic << '\n';
      metadata << "# tov_central_density="
               << output_metadata_real_string(cfg.tov_central_density) << '\n';
      metadata << "# rotation_law=" << cfg.initial_data_rotation_law << '\n';
      metadata << "# angular_velocity="
               << output_metadata_real_string(cfg.initial_data_angular_velocity)
               << '\n';
   }
   amrex::ParallelDescriptor::Barrier();
}

PhysicalInitialDataMetadata read_physical_initial_data(
   std::string const& path,
   RunConfig const& cfg,
   PhysicalSystem system,
   amrex::Vector<amrex::Geometry> const& geometry,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   MutablePhysicalIdField const& evolution_fluid,
   MutablePhysicalIdField const& evolution_spacetime,
   MutablePhysicalIdFaceFields const& face_magnetic
)
{
   const int levels = static_cast<int>(geometry.size());
   if (levels == 0
       || static_cast<int>(refinement_ratios.size()) != levels - 1
       || static_cast<int>(evolution_fluid.size()) != levels) {
      amrex::Abort(
         "physical initial data require a nonempty consistent target hierarchy"
      );
   }
   const int fluid_components = evolution_fluid.front()->nComp();
   validate_level_fields(
      ConstPhysicalIdField(evolution_fluid.begin(), evolution_fluid.end()),
      levels,
      fluid_components,
      EvolutionFluidField
   );
   const int physical_fluid_components = grid_field_component_count(
      physical_fluid_grid_fields(system, cfg.passive_scalars)
   );
   const auto metadata = read_metadata(path);
   require_metadata(metadata, "type", "physical_initial_data", path);
   require_metadata(metadata, "schema_version", std::to_string(PhysicalIdSchemaVersion), path);
   require_metadata(metadata, "format", "hdf5", path);
   require_metadata(metadata, "amrex_spacedim", std::to_string(AMREX_SPACEDIM), path);
   require_metadata(metadata, "levels", std::to_string(levels), path);
   require_metadata(metadata, "physics", cfg.physics, path);
   require_metadata(metadata, "fluid_components", std::to_string(fluid_components), path);
   require_metadata(
      metadata,
      "physical_fluid_components",
      std::to_string(physical_fluid_components),
      path
   );
   require_metadata(metadata, "mesh_coordinate", cfg.mesh_coordinate, path);
   require_metadata(
      metadata,
      "mesh_axes",
      mesh_axes_or_default(
         mesh_coordinate_code(cfg.mesh_coordinate), cfg.mesh_axes
      ),
      path
   );
   require_metadata(metadata, "domain", domain_string(geometry.front()), path);
   require_metadata(metadata, "prob_lo", real_vector_string(geometry.front(), false), path);
   require_metadata(metadata, "prob_hi", real_vector_string(geometry.front(), true), path);
   require_metadata(metadata, "periodicity", periodicity_string(geometry.front()), path);
   require_metadata(metadata, "refinement_ratios", refinement_string(refinement_ratios), path);
   require_metadata(metadata, "eos", cfg.eos, path);
   if (cfg.eos == "ideal_gas") {
      require_metadata(
         metadata, "gamma", output_metadata_real_string(cfg.gamma), path
      );
   } else {
      require_metadata(
         metadata,
         "polytropic_k",
         output_metadata_real_string(cfg.polytropic_k),
         path
      );
      require_metadata(
         metadata,
         "polytropic_gamma",
         output_metadata_real_string(cfg.polytropic_gamma),
         path
      );
      require_metadata(
         metadata,
         "polytropic_density_breaks",
         output_metadata_real_vector_string(cfg.polytropic_density_breaks),
         path
      );
      require_metadata(
         metadata,
         "polytropic_segment_gammas",
         output_metadata_real_vector_string(cfg.polytropic_segment_gammas),
         path
      );
      if (cfg.eos == "hybrid_polytrope") {
         require_metadata(
            metadata,
            "thermal_gamma",
            output_metadata_real_string(cfg.thermal_gamma),
            path
         );
      }
   }
   require_metadata(metadata, "passive_scalars", std::to_string(cfg.passive_scalars), path);
   const bool expects_spacetime = !evolution_spacetime.empty();
   if (expects_spacetime) {
      validate_level_fields(
         ConstPhysicalIdField(
            evolution_spacetime.begin(), evolution_spacetime.end()
         ),
         levels,
         grid_field_component_count(z4c_state_grid_fields()),
         EvolutionSpacetimeField
      );
   }
   require_metadata(metadata, "has_spacetime", expects_spacetime ? "1" : "0", path);
   const bool expects_ct = !face_magnetic[0].empty();
   if (expects_ct) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         validate_level_fields(
            ConstPhysicalIdField(
               face_magnetic[direction].begin(),
               face_magnetic[direction].end()
            ),
            levels,
            1,
            "face magnetic field"
         );
      }
   }
   require_metadata(metadata, "has_ct", expects_ct ? "1" : "0", path);
   bool has_required_fields =
      std::filesystem::is_regular_file(
         std::filesystem::path(path) / "Header"
      )
      && multilevel_hdf5_field_exists(path, PhysicalFluidField)
      && multilevel_hdf5_field_exists(path, EvolutionFluidField)
      && (!expects_spacetime
          || (multilevel_hdf5_field_exists(path, AdmGeometryField)
              && multilevel_hdf5_field_exists(
                    path, EvolutionSpacetimeField
                 )));
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      has_required_fields = has_required_fields
         && (!expects_ct
             || multilevel_hdf5_field_exists(
                path, checkpoint_face_b_field_name(direction)
             ));
   }
   if (!has_required_fields) {
      amrex::Abort(
         "physical initial-data artifact is missing a required field in " + path
      );
   }

   for (std::size_t level = 0; level < evolution_fluid.size(); ++level) {
      read_multilevel_hdf5_field(
         *evolution_fluid[level], path, static_cast<int>(level), EvolutionFluidField
      );
      if (expects_spacetime) {
         read_multilevel_hdf5_field(
            *evolution_spacetime[level],
            path,
            static_cast<int>(level),
            EvolutionSpacetimeField
         );
      }
      if (expects_ct) {
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            read_multilevel_hdf5_field(
               *face_magnetic[direction][level],
               path,
               static_cast<int>(level),
               checkpoint_face_b_field_name(direction)
            );
         }
      }
   }
   PhysicalInitialDataMetadata result;
   result.source_ic = required_metadata_value(metadata, "source_ic", path);
   result.rotation_law = required_metadata_value(metadata, "rotation_law", path);
   try {
      const std::string value = required_metadata_value(
         metadata, "angular_velocity", path
      );
      std::size_t parsed = 0;
      result.angular_velocity = std::stod(value, &parsed);
      if (parsed != value.size()) {
         throw std::invalid_argument("trailing angular_velocity metadata");
      }
   } catch (std::exception const&) {
      amrex::Abort(
         "physical initial-data angular_velocity metadata is invalid in " + path
      );
   }
   if (!std::isfinite(result.angular_velocity)) {
      amrex::Abort(
         "physical initial-data angular_velocity metadata is nonfinite in "
         + path
      );
   }
   return result;
}

} // namespace pyfluid_amrex
