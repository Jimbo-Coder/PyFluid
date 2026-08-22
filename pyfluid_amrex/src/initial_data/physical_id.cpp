#include "pyfluid_amrex/initial_data/physical_id.hpp"

#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/coordinate_locations.hpp"
#include "pyfluid_amrex/initial_data/atmosphere.hpp"
#include "pyfluid_amrex/initial_data/physical_id_interpolation.hpp"
#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/hierarchy_interpolation.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/multilevel_field_io.hpp"
#include "pyfluid_amrex/output_metadata.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"
#include "pyfluid_amrex/state.hpp"

#include <AMReX_BoxIterator.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_RealBox.H>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace pyfluid_amrex {
namespace {

constexpr int PhysicalIdSchemaVersion = 3;
constexpr int LegacyPhysicalIdSchemaVersion = 1;
// The multilevel HDF5 payload layout remains v2; schema-v3 changes metadata
// semantics and the human-readable artifact header, not the field encoding.
constexpr char const* PhysicalIdFormat = "PyFluidPhysicalIDHDF5-v2";
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

std::string coordinate_face_string(amrex::Vector<amrex::Real> const& faces)
{
   std::ostringstream output;
   output << std::setprecision(17);
   for (std::size_t index = 0; index < faces.size(); ++index) {
      if (index != 0) {
         output << ',';
      }
      output << faces[index];
   }
   return output.str();
}

std::vector<amrex::Real> parse_coordinate_faces(
   std::string const& value,
   std::string const& path,
   std::string const& key
)
{
   std::vector<amrex::Real> faces;
   std::istringstream input(value);
   for (std::string item; std::getline(input, item, ',');) {
      try {
         std::size_t parsed = 0;
         const amrex::Real face = std::stod(item, &parsed);
         if (parsed != item.size() || !std::isfinite(face)) {
            throw std::invalid_argument("invalid coordinate face");
         }
         faces.push_back(face);
      } catch (std::exception const&) {
         amrex::Abort(
            "invalid physical-ID " + key + " metadata in " + path
         );
      }
   }
   for (std::size_t index = 1; index < faces.size(); ++index) {
      if (!(faces[index] > faces[index - 1])) {
         amrex::Abort(
            "physical-ID coordinate faces must be strictly increasing in "
            + path
         );
      }
   }
   return faces;
}

RuntimePhysicalIdCoordinateFaces runtime_coordinate_faces(
   std::map<std::string, std::string> const& metadata,
   int dimension,
   int levels,
   std::string const& path
)
{
   const auto layout = metadata.find("coordinate_layout");
   if (layout == metadata.end() || layout->second == "uniform") {
      return {};
   }
   if (layout->second != "explicit_faces_v1") {
      amrex::Abort("unsupported physical-ID coordinate layout in " + path);
   }
   RuntimePhysicalIdCoordinateFaces coordinates(levels);
   bool has_explicit_faces = false;
   for (int level = 0; level < levels; ++level) {
      for (int direction = 0; direction < dimension; ++direction) {
         const std::string key = "coordinate_faces_"
            + std::to_string(level) + "_" + std::to_string(direction);
         const auto found = metadata.find(key);
         if (found != metadata.end()) {
            coordinates[level][direction] = parse_coordinate_faces(
               found->second, path, key
            );
            has_explicit_faces = true;
         }
      }
   }
   if (!has_explicit_faces) {
      amrex::Abort(
         "physical-ID explicit coordinate layout has no coordinate faces in "
         + path
      );
   }
   return coordinates;
}

bool coordinate_faces_match(
   RuntimePhysicalIdCoordinateFaces const& source,
   PhysicalIdCoordinateFaces const* target,
   int dimension
)
{
   if (source.empty()) {
      return target == nullptr || target->empty();
   }
   if (target == nullptr || source.size() != target->size()) {
      return false;
   }
   for (std::size_t level = 0; level < source.size(); ++level) {
      for (int direction = 0; direction < dimension; ++direction) {
         auto const& source_faces = source[level][direction];
         auto const& target_faces = (*target)[level][direction];
         if (source_faces.size() != target_faces.size()
             || !std::equal(
                source_faces.begin(), source_faces.end(), target_faces.begin()
             )) {
            return false;
         }
      }
   }
   return true;
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

std::string reflection_string(RunConfig const& cfg, bool upper)
{
   std::ostringstream output;
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      if (direction != 0) {
         output << ',';
      }
      const BoundaryKind kind = side_boundary_kind(cfg, direction, upper);
      output << (
         kind == BoundaryKind::Symmetry || kind == BoundaryKind::Reflecting
      );
   }
   return output.str();
}

std::vector<GridFieldSymmetryKind> component_symmetries(
   GridFieldSetInfo const& fields
)
{
   std::vector<GridFieldSymmetryKind> symmetries;
   symmetries.reserve(grid_field_component_count(fields));
   for (GridFieldInfo const& field : fields.fields) {
      for (GridFieldComponentInfo const& component : field.components) {
         symmetries.push_back(component.symmetry);
      }
   }
   return symmetries;
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

amrex::Real parse_metadata_real(
   std::string const& value,
   std::string const& key,
   std::string const& path
)
{
   try {
      std::size_t parsed = 0;
      const amrex::Real result = std::stod(value, &parsed);
      if (parsed != value.size() || !std::isfinite(result)) {
         throw std::invalid_argument("invalid real metadata");
      }
      return result;
   } catch (std::exception const&) {
      amrex::Abort(
         "physical initial-data metadata '" + key
         + "' is invalid in " + path
      );
   }
   return 0.0;
}

amrex::Real required_metadata_real(
   std::map<std::string, std::string> const& metadata,
   std::string const& key,
   std::string const& path
)
{
   return parse_metadata_real(
      required_metadata_value(metadata, key, path), key, path
   );
}

amrex::Real optional_metadata_real(
   std::map<std::string, std::string> const& metadata,
   std::string const& key,
   amrex::Real fallback,
   std::string const& path
)
{
   const auto found = metadata.find(key);
   return found == metadata.end()
      ? fallback : parse_metadata_real(found->second, key, path);
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

struct StoredPhysicalHierarchy {
   int coordinate_kind = MeshCartesian;
   MeshAxes axes{};
   amrex::Vector<amrex::Geometry> geometry;
   amrex::Vector<amrex::BoxArray> grids;
   amrex::Vector<amrex::DistributionMapping> distribution;
   std::array<int, 3> reflection_lo{};
   std::array<int, 3> reflection_hi{};
};

std::array<int, 3> parse_binary_directions(
   std::string const& value,
   std::string const& path,
   std::string const& key,
   int dimension
)
{
   if (dimension < 1 || dimension > 3) {
      amrex::Abort("invalid physical-ID dimension metadata in " + path);
   }
   std::array<int, 3> flags{};
   std::istringstream input(value);
   for (int direction = 0; direction < dimension; ++direction) {
      std::string item;
      if (!std::getline(input, item, ',')) {
         amrex::Abort(
            "invalid physical-ID " + key + " metadata in " + path
         );
      }
      try {
         flags[direction] = std::stoi(item);
      } catch (std::exception const&) {
         amrex::Abort(
            "invalid physical-ID " + key + " metadata in " + path
         );
      }
      if (flags[direction] != 0 && flags[direction] != 1) {
         amrex::Abort(
            "invalid physical-ID " + key + " metadata in " + path
         );
      }
   }
   return flags;
}

StoredPhysicalHierarchy stored_physical_hierarchy(
   std::string const& path,
   std::map<std::string, std::string> const& metadata
)
{
   const MultilevelHdf5FieldLayout layout =
      read_multilevel_hdf5_field_layout(path, PhysicalFluidField);
   StoredPhysicalHierarchy hierarchy;
   hierarchy.coordinate_kind = mesh_coordinate_code(
      required_metadata_value(metadata, "mesh_coordinate", path)
   );
   const RunConfig axes_config = [&]() {
      RunConfig config;
      config.mesh_coordinate = required_metadata_value(
         metadata, "mesh_coordinate", path
      );
      config.mesh_axes = required_metadata_value(metadata, "mesh_axes", path);
      return config;
   }();
   hierarchy.axes = mesh_axes_layout(axes_config);
   hierarchy.grids = layout.boxes;
   hierarchy.geometry.reserve(layout.boxes.size());
   hierarchy.distribution.reserve(layout.boxes.size());
   const auto periodic = parse_binary_directions(
      required_metadata_value(metadata, "periodicity", path),
      path,
      "periodicity",
      AMREX_SPACEDIM
   );
   const auto reflection_lo = metadata.find("reflection_lo");
   const auto reflection_hi = metadata.find("reflection_hi");
   if ((reflection_lo == metadata.end()) != (reflection_hi == metadata.end())) {
      amrex::Abort(
         "physical-ID reflection metadata must define both sides in " + path
      );
   }
   if (reflection_lo != metadata.end()) {
      hierarchy.reflection_lo = parse_binary_directions(
         reflection_lo->second, path, "reflection_lo", AMREX_SPACEDIM
      );
      hierarchy.reflection_hi = parse_binary_directions(
         reflection_hi->second, path, "reflection_hi", AMREX_SPACEDIM
      );
   }
   for (std::size_t level = 0; level < layout.boxes.size(); ++level) {
      amrex::RealBox real_box(
         layout.prob_lo[level].data(), layout.prob_hi[level].data()
      );
      hierarchy.geometry.emplace_back(
         layout.domains[level],
         &real_box,
         hierarchy.coordinate_kind,
         periodic.data()
      );
      hierarchy.distribution.emplace_back(layout.boxes[level]);
   }
   return hierarchy;
}

std::vector<std::unique_ptr<amrex::MultiFab>> load_stored_field(
   std::string const& path,
   std::string const& name,
   StoredPhysicalHierarchy const& hierarchy,
   amrex::IndexType centering,
   int num_components
)
{
   const MultilevelHdf5FieldLayout layout =
      read_multilevel_hdf5_field_layout(path, name);
   if (layout.num_components != num_components
       || layout.boxes.size() != hierarchy.grids.size()) {
      amrex::Abort(
         "physical initial-data field layout mismatch for " + name
      );
   }
   std::vector<std::unique_ptr<amrex::MultiFab>> field(
      hierarchy.grids.size()
   );
   for (std::size_t level = 0; level < field.size(); ++level) {
      const amrex::BoxArray boxes = amrex::convert(
         hierarchy.grids[level], centering
      );
      field[level] = std::make_unique<amrex::MultiFab>(
         boxes,
         hierarchy.distribution[level],
         num_components,
         1
      );
      read_multilevel_hdf5_field(
         *field[level], path, static_cast<int>(level), name
      );
      field[level]->FillBoundary(hierarchy.geometry[level].periodicity());
   }
   return field;
}

int canonical_component_for_runtime_direction(
   int coordinate_kind,
   MeshAxes const& axes,
   int runtime_direction
)
{
   const int axis = mesh_axis(axes, runtime_direction);
   const auto canonical = mesh_canonical_vector_axes(coordinate_kind);
   for (int component = 0; component < 3; ++component) {
      if (canonical[component] == axis) {
         return component;
      }
   }
   amrex::Abort("physical-ID axes do not map to a canonical coordinate");
   return 0;
}

struct PhysicalInterpolationPoint {
   amrex::GpuArray<amrex::Real, 3> source_point{};
   PatchCoordinate target_mesh{};
   PatchCoordinate target_physical{};
   amrex::Real source_radius = 0.0;
   amrex::Real target_radius = 0.0;
   int reflection_mask = 0;
   bool exterior = false;
};

PhysicalInterpolationPoint mapped_source_point(
   amrex::GpuArray<amrex::Real, 3> target_mesh,
   StoredPhysicalHierarchy const& source,
   amrex::IndexType centering,
   amrex::GpuArray<amrex::Real, 3> const& origin
)
{
   PhysicalInterpolationPoint mapped;
   const auto original_target_mesh = target_mesh;
   const amrex::Geometry& base = source.geometry.front();
   const auto spacing = base.CellSizeArray();
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      if (base.isPeriodic(direction)) {
         continue;
      }
      const int component = canonical_component_for_runtime_direction(
         source.coordinate_kind, source.axes, direction
      );
      if (target_mesh[component] < base.ProbLo(direction)
          && source.reflection_lo[direction] != 0) {
         target_mesh[component] =
            2.0 * base.ProbLo(direction) - target_mesh[component];
         mapped.reflection_mask ^= 1 << direction;
      } else if (target_mesh[component] > base.ProbHi(direction)
                 && source.reflection_hi[direction] != 0) {
         target_mesh[component] =
            2.0 * base.ProbHi(direction) - target_mesh[component];
         mapped.reflection_mask ^= 1 << direction;
      }
      const amrex::Real offset = centering.nodeCentered(direction) ? 0.0 : 0.5;
      const amrex::Real safe_lo = base.ProbLo(direction) + offset * spacing[direction];
      const amrex::Real safe_hi = base.ProbHi(direction) - offset * spacing[direction];
      // A finer target can have cell centers inside the same physical domain
      // but outside the source cell-center interpolation hull. Treat that
      // half-cell strip with the selected physical exterior policy instead of
      // asking the generic interpolation plan for unavailable source data.
      const bool below_interpolation_hull = target_mesh[component] < safe_lo
         && source.reflection_lo[direction] == 0;
      const bool above_interpolation_hull = target_mesh[component] > safe_hi
         && source.reflection_hi[direction] == 0;
      const bool outside_interpolation_hull = below_interpolation_hull
         || above_interpolation_hull;
      mapped.exterior = mapped.exterior || outside_interpolation_hull;
      if (outside_interpolation_hull) {
         target_mesh[component] = amrex::max(
            safe_lo, amrex::min(safe_hi, target_mesh[component])
         );
      }
   }
   const auto source_cartesian = mesh_coordinate_triplet_to_cartesian(
      source.coordinate_kind, target_mesh
   );
   mapped.source_point = source_cartesian;
   amrex::Real source_radius_squared = 0.0;
   amrex::Real target_radius_squared = 0.0;
   const auto target_cartesian = mesh_coordinate_triplet_to_cartesian(
      source.coordinate_kind, original_target_mesh
   );
   mapped.target_mesh = {
      original_target_mesh[0],
      original_target_mesh[1],
      original_target_mesh[2],
   };
   mapped.target_physical = {
      target_cartesian[0], target_cartesian[1], target_cartesian[2]
   };
   for (int component = 0; component < 3; ++component) {
      const amrex::Real source_delta = source_cartesian[component] - origin[component];
      source_radius_squared += source_delta * source_delta;
      const amrex::Real target_delta = target_cartesian[component] - origin[component];
      target_radius_squared += target_delta * target_delta;
   }
   mapped.source_radius = std::sqrt(source_radius_squared);
   mapped.target_radius = std::sqrt(target_radius_squared);
   return mapped;
}

enum class PhysicalFieldExterior {
   Constant,
   InitialAtmosphere,
   AsymptoticallyFlatAdm,
   TovSchwarzschildAdm,
};

EOSThermodynamicState initial_atmosphere_state_for_method(
   int eos_method,
   InitialAtmosphereParameters const& atmosphere,
   InitialDataSamplePoint const& sample
)
{
   static_cast<void>(eos_method);
   return initial_atmosphere_state<EOSRuntime>(atmosphere, sample);
}

void apply_initial_atmosphere_exterior(
   amrex::Real* values,
   int num_components,
   PhysicalInterpolationPoint const& point,
   PhysicalSystem system,
   int eos_method,
   InitialAtmosphereParameters const& atmosphere,
   amrex::Real matter_outer_radius
)
{
   // A cell-centred radial source has no sample exactly at the regular origin.
   // Clamping a target there to the innermost source centre is interpolation,
   // not stellar exterior.  Atmosphere continuation applies only beyond the
   // saved source radius.
   const bool beyond_source = point.exterior
      && point.target_radius > point.source_radius;
   const bool beyond_matter = matter_outer_radius > 0.0
      && point.target_radius > matter_outer_radius;
   if (!beyond_source && !beyond_matter) {
      return;
   }
   const EOSThermodynamicState state = initial_atmosphere_state_for_method(
      eos_method,
      atmosphere,
      {point.target_mesh, point.target_physical}
   );
   values[RHO] = state.rho;
   values[MOM1] = 0.0;
   values[MOM2] = 0.0;
   values[MOM3] = 0.0;
   values[ENERGY] = state.pressure;
   const int base_components = physical_system_has_magnetic_field(system)
      ? NVAR_MHD
      : NVAR_HYDRO;
   for (int component = base_components;
        component < num_components;
        ++component) {
      values[component] = 0.0;
   }
}

void apply_adm_exterior(
   amrex::Real* values,
   int num_components,
   PhysicalInterpolationPoint const& point,
   bool stationary_rotating
)
{
   if (!point.exterior || !(point.target_radius > point.source_radius)
       || !(point.source_radius > 0.0)) {
      return;
   }
   for (int component = 0; component < num_components; ++component) {
      amrex::Real asymptotic = 0.0;
      amrex::Real falloff = 1.0;
      if (component == AdmGamma11 || component == AdmGamma22
          || component == AdmGamma33 || component == AdmLapse) {
         asymptotic = 1.0;
      } else if (component >= AdmK11 && component <= AdmK33) {
         falloff = stationary_rotating ? 3.0 : 2.0;
      } else if (component >= AdmShift1 && component <= AdmShift3) {
         falloff = stationary_rotating ? 2.0 : 1.0;
      }
      const amrex::Real scale = std::pow(
         point.source_radius / point.target_radius, falloff
      );
      values[component] = asymptotic
         + (values[component] - asymptotic) * scale;
   }
}

void apply_tov_schwarzschild_adm_exterior(
   amrex::Real* values,
   PhysicalInterpolationPoint const& point
)
{
   if (!point.exterior || !(point.target_radius > point.source_radius)
       || !(point.source_radius > 0.0)) {
      return;
   }
   const amrex::Real determinant =
      values[AdmGamma11]
         * (values[AdmGamma22] * values[AdmGamma33]
            - values[AdmGamma23] * values[AdmGamma23])
      - values[AdmGamma12]
         * (values[AdmGamma12] * values[AdmGamma33]
            - values[AdmGamma13] * values[AdmGamma23])
      + values[AdmGamma13]
         * (values[AdmGamma12] * values[AdmGamma23]
            - values[AdmGamma13] * values[AdmGamma22]);
   if (!(determinant > 0.0) || !std::isfinite(determinant)) {
      amrex::Abort(
         "TOV exterior continuation requires a finite positive spatial metric"
      );
   }
   const amrex::Real source_conformal_factor = std::pow(determinant, 1.0 / 12.0);
   const amrex::Real mass = 2.0 * point.source_radius
      * (source_conformal_factor - 1.0);
   const amrex::Real compact_radius = mass / (2.0 * point.target_radius);
   const amrex::Real target_conformal_factor = 1.0 + compact_radius;
   const amrex::Real spatial_diagonal = std::pow(
      target_conformal_factor, 4.0
   );
   values[AdmGamma11] = spatial_diagonal;
   values[AdmGamma12] = 0.0;
   values[AdmGamma13] = 0.0;
   values[AdmGamma22] = spatial_diagonal;
   values[AdmGamma23] = 0.0;
   values[AdmGamma33] = spatial_diagonal;
   for (int component = AdmK11; component <= AdmK33; ++component) {
      values[component] = 0.0;
   }
   values[AdmLapse] = (1.0 - compact_radius) / target_conformal_factor;
   values[AdmShift1] = 0.0;
   values[AdmShift2] = 0.0;
   values[AdmShift3] = 0.0;
}

void interpolate_stored_field(
   std::vector<std::unique_ptr<amrex::MultiFab>> const& source_field,
   StoredPhysicalHierarchy const& source,
   amrex::Vector<amrex::Geometry> const& target_geometry,
   MutablePhysicalIdField const& target_field,
   amrex::IndexType centering,
   std::vector<GridFieldSymmetryKind> const& symmetries,
   amrex::GpuArray<amrex::Real, 3> const& origin,
   PhysicalFieldExterior exterior,
   bool stationary_rotating,
   PhysicalSystem system,
   int eos_method,
   InitialAtmosphereParameters const& atmosphere
)
{
   if (target_geometry.size() != target_field.size()) {
      amrex::Abort("physical-ID target hierarchy is inconsistent");
   }
   const int num_components = target_field.front()->nComp();
   if (static_cast<int>(symmetries.size()) != num_components) {
      amrex::Abort("physical-ID interpolation symmetry layout is inconsistent");
   }
   bool has_reflection = false;
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      has_reflection = has_reflection
         || source.reflection_lo[direction] != 0
         || source.reflection_hi[direction] != 0;
   }
   // AMReX's scalar FillDomainBoundary helper is cell-only, while a portable
   // artifact also stores nodal ADM fields and face-centered magnetic fields.
   // Fill the one-layer parity extension explicitly for every centering so the
   // common interpolation plan can use a reflected stencil without inventing
   // separate field-specific materializers.
   for (std::size_t level = 0;
        has_reflection && level < source_field.size();
        ++level) {
      amrex::MultiFab& field = *source_field[level];
      const amrex::Box centered_domain = amrex::convert(
         source.geometry[level].Domain(), centering
      );
      amrex::GpuArray<int, 3> domain_lo{{0, 0, 0}};
      amrex::GpuArray<int, 3> domain_hi{{0, 0, 0}};
      amrex::GpuArray<int, 3> periodic{{0, 0, 0}};
      amrex::GpuArray<int, 3> node_centered{{0, 0, 0}};
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         domain_lo[direction] = centered_domain.smallEnd(direction);
         domain_hi[direction] = centered_domain.bigEnd(direction);
         periodic[direction] = source.geometry[level].isPeriodic(direction)
            ? 1 : 0;
         node_centered[direction] = centering.nodeCentered(direction) ? 1 : 0;
      }
      const auto reflection_lo = source.reflection_lo;
      const auto reflection_hi = source.reflection_hi;
      for (int component = 0; component < num_components; ++component) {
         amrex::GpuArray<int, 3> parity{{1, 1, 1}};
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            parity[direction] = grid_field_reflection_parity(
               symmetries[component], direction
            );
         }
         for (amrex::MFIter mfi(field, false); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.fabbox();
            const auto values = field.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
               int i, int j, int k
            ) noexcept {
               int source_index[3]{i, j, k};
               int reflection_parity = 1;
               bool reflected = false;
               bool supported = true;
               for (int direction = 0;
                    direction < AMREX_SPACEDIM;
                    ++direction) {
                  if (source_index[direction] < domain_lo[direction]) {
                     if (reflection_lo[direction] != 0) {
                        source_index[direction] =
                           2 * domain_lo[direction] - source_index[direction]
                           - (node_centered[direction] == 0 ? 1 : 0);
                        reflection_parity *= parity[direction];
                        reflected = true;
                     } else if (periodic[direction] == 0) {
                        supported = false;
                     }
                  } else if (source_index[direction] > domain_hi[direction]) {
                     if (reflection_hi[direction] != 0) {
                        source_index[direction] =
                           2 * domain_hi[direction] - source_index[direction]
                           + (node_centered[direction] == 0 ? 1 : 0);
                        reflection_parity *= parity[direction];
                        reflected = true;
                     } else if (periodic[direction] == 0) {
                        supported = false;
                     }
                  }
               }
               if (reflected && supported) {
                  values(i, j, k, component) = reflection_parity * values(
                     source_index[0],
                     source_index[1],
                     source_index[2],
                     component
                  );
               }
            });
         }
      }
   }
   for (std::size_t level = 0; level < target_field.size(); ++level) {
      amrex::MultiFab& target = *target_field[level];
      const auto target_lo = target_geometry[level].ProbLoArray();
      const auto target_dx = target_geometry[level].CellSizeArray();
      for (int box_index = 0; box_index < target.boxArray().size(); ++box_index) {
         const amrex::Box box = target.boxArray()[box_index];
         std::vector<amrex::GpuArray<amrex::Real, 3>> source_points;
         std::vector<PhysicalInterpolationPoint> mapped_points;
         source_points.reserve(box.numPts());
         mapped_points.reserve(box.numPts());
         for (amrex::BoxIterator iterator(box); iterator.ok(); ++iterator) {
            const amrex::IntVect& index = iterator();
            const auto target_mesh = mesh_coordinate_triplet_at_location(
               source.coordinate_kind,
               source.axes,
               target_lo,
               target_dx,
               index[0],
#if (AMREX_SPACEDIM >= 2)
               index[1],
#else
               0,
#endif
#if (AMREX_SPACEDIM == 3)
               index[2],
#else
               0,
#endif
               centering.nodeCentered(0),
               AMREX_SPACEDIM >= 2 && centering.nodeCentered(1),
               AMREX_SPACEDIM == 3 && centering.nodeCentered(2)
            );
            PhysicalInterpolationPoint mapped = mapped_source_point(
               target_mesh, source, centering, origin
            );
            source_points.push_back(mapped.source_point);
            mapped_points.push_back(mapped);
         }
         const HierarchyInterpolationPlan plan(
            std::move(source_points),
            source.coordinate_kind,
            source.axes,
            source.geometry,
            source.grids,
            source.distribution,
            centering,
            source.reflection_lo,
            source.reflection_hi
         );
         std::vector<amrex::Real> values = plan.interpolate(
            source_field, 0, num_components
         );
         for (std::size_t point = 0; point < mapped_points.size(); ++point) {
            for (int component = 0; component < num_components; ++component) {
               int parity = 1;
               for (int direction = 0; direction < 3; ++direction) {
                  if ((mapped_points[point].reflection_mask
                       & (1 << direction)) != 0) {
                     parity *= grid_field_reflection_parity(
                        symmetries[component], direction
                     );
                  }
               }
               values[point * num_components + component] *= parity;
            }
         }
         if (exterior == PhysicalFieldExterior::AsymptoticallyFlatAdm) {
            for (std::size_t point = 0; point < mapped_points.size(); ++point) {
               apply_adm_exterior(
                  values.data() + point * num_components,
                  num_components,
                  mapped_points[point],
                  stationary_rotating
               );
            }
         } else if (
            exterior == PhysicalFieldExterior::TovSchwarzschildAdm
         ) {
            for (std::size_t point = 0; point < mapped_points.size(); ++point) {
               apply_tov_schwarzschild_adm_exterior(
                  values.data() + point * num_components,
                  mapped_points[point]
               );
            }
         } else if (exterior == PhysicalFieldExterior::InitialAtmosphere) {
            for (std::size_t point = 0; point < mapped_points.size(); ++point) {
               apply_initial_atmosphere_exterior(
                  values.data() + point * num_components,
                  num_components,
                  mapped_points[point],
                  system,
                  eos_method,
                  atmosphere,
                  -1.0
               );
            }
         }
         if (target.DistributionMap()[box_index]
             == amrex::ParallelDescriptor::MyProc()) {
            amrex::FArrayBox& fab = target[box_index];
            std::size_t point = 0;
            for (amrex::BoxIterator iterator(box); iterator.ok(); ++iterator) {
               for (int component = 0; component < num_components; ++component) {
                  fab(iterator(), component) = values[
                     point * num_components + component
                  ];
               }
               ++point;
            }
         }
      }
      target.FillBoundary(target_geometry[level].periodicity());
   }
}

void interpolate_portable_field(
   std::string const& path,
   std::string const& name,
   PhysicalIdChart const& source_chart,
   PhysicalIdChart const& target_chart,
   amrex::Vector<amrex::Geometry> const& target_geometry,
   MutablePhysicalIdField const& target_field,
   GridFieldSetInfo const& field_set,
   amrex::IndexType centering,
   RuntimePhysicalIdCoordinateFaces const& source_coordinate_faces,
   PhysicalIdCoordinateFaces const& target_coordinate_faces,
   PhysicalIdRigidTransform const& rigid_transform,
   amrex::GpuArray<amrex::Real, 3> const& origin,
   PhysicalFieldExterior exterior,
   bool stationary_rotating,
   PhysicalSystem system,
   int eos_method,
   InitialAtmosphereParameters const& atmosphere,
   amrex::Real matter_outer_radius
)
{
   if (target_geometry.size() != target_field.size()) {
      amrex::Abort("portable physical-ID target hierarchy is inconsistent");
   }
   const int num_components = grid_field_component_count(field_set);
   if (target_field.empty() || target_field.front()->nComp() != num_components) {
      amrex::Abort("portable physical-ID target component layout is inconsistent");
   }
   PortablePhysicalIdInterpolator interpolator(
      read_runtime_dimension_hdf5_field(path, name),
      source_chart,
      field_set,
      centering == amrex::IndexType::TheNodeType(),
      source_coordinate_faces,
      rigid_transform
   );
   const int target_coordinate = target_chart.coordinate_kind;
   MeshAxes axes{};
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      axes.axis[direction] = target_chart.axes[direction];
   }

   for (std::size_t level = 0; level < target_field.size(); ++level) {
      amrex::MultiFab& target = *target_field[level];
      const auto target_lo = target_geometry[level].ProbLoArray();
      const auto target_dx = target_geometry[level].CellSizeArray();
      for (int box_index = 0; box_index < target.boxArray().size(); ++box_index) {
         if (target.DistributionMap()[box_index]
             != amrex::ParallelDescriptor::MyProc()) {
            continue;
         }
         const amrex::Box box = target.boxArray()[box_index];
         amrex::FArrayBox& fab = target[box_index];
         for (amrex::BoxIterator iterator(box); iterator.ok(); ++iterator) {
            const amrex::IntVect& index = iterator();
            const auto target_q = mesh_coordinate_triplet_at_location(
               target_coordinate,
               axes,
               target_lo,
               target_dx,
               index[0],
#if (AMREX_SPACEDIM >= 2)
               index[1],
#else
               0,
#endif
#if (AMREX_SPACEDIM == 3)
               index[2],
#else
               0,
#endif
               centering.nodeCentered(0),
               AMREX_SPACEDIM >= 2 && centering.nodeCentered(1),
               AMREX_SPACEDIM == 3 && centering.nodeCentered(2)
            );
            auto mapped_target_q = target_q;
            if (!target_coordinate_faces.empty()) {
               for (int direction = 0; direction < AMREX_SPACEDIM;
                    ++direction) {
                  auto const& faces =
                     target_coordinate_faces[level][direction];
                  if (faces.empty()) {
                     continue;
                  }
                  const int local_index = index[direction]
                     - target_geometry[level].Domain().smallEnd(direction);
                  const amrex::Real coordinate =
                     centering.nodeCentered(direction)
                     ? faces[local_index]
                     : 0.5 * (faces[local_index] + faces[local_index + 1]);
                  mapped_target_q[canonical_component_for_runtime_direction(
                     target_coordinate, axes, direction
                  )] = coordinate;
               }
            }
            const auto target_cartesian = mesh_coordinate_triplet_to_cartesian(
               target_coordinate, mapped_target_q
            );
            PortablePhysicalIdSample sample = interpolator.sample(
               {target_cartesian[0], target_cartesian[1], target_cartesian[2]},
               target_chart
            );
            PhysicalInterpolationPoint point;
            point.exterior = sample.exterior;
            point.target_mesh = {
               mapped_target_q[0], mapped_target_q[1], mapped_target_q[2]
            };
            point.target_physical = {
               target_cartesian[0], target_cartesian[1], target_cartesian[2]
            };
            amrex::Real source_radius_squared = 0.0;
            amrex::Real target_radius_squared = 0.0;
            for (int component = 0; component < 3; ++component) {
               const amrex::Real source_delta =
                  sample.source_cartesian[component] - origin[component];
               const amrex::Real target_delta =
                  target_cartesian[component] - origin[component];
               source_radius_squared += source_delta * source_delta;
               target_radius_squared += target_delta * target_delta;
            }
            point.source_radius = std::sqrt(source_radius_squared);
            point.target_radius = std::sqrt(target_radius_squared);
            if (exterior == PhysicalFieldExterior::AsymptoticallyFlatAdm) {
               apply_adm_exterior(
                  sample.values.data(),
                  num_components,
                  point,
                  stationary_rotating
               );
            } else if (
               exterior == PhysicalFieldExterior::TovSchwarzschildAdm
            ) {
               apply_tov_schwarzschild_adm_exterior(
                  sample.values.data(), point
               );
            } else if (exterior == PhysicalFieldExterior::InitialAtmosphere) {
               apply_initial_atmosphere_exterior(
                  sample.values.data(),
                  num_components,
                  point,
                  system,
                  eos_method,
                  atmosphere,
                  matter_outer_radius
               );
            }
            for (int component = 0; component < num_components; ++component) {
               fab(iterator(), component) = sample.values[component];
            }
         }
      }
      target.FillBoundary(target_geometry[level].periodicity());
   }
}

void adm_to_z4c(
   MutablePhysicalIdField const& spacetime,
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> const& adm
)
{
   for (std::size_t level = 0; level < spacetime.size(); ++level) {
      for (amrex::MFIter mfi(*spacetime[level], amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto source = adm[level]->const_array(mfi);
         const auto destination = spacetime[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real g11 = source(i, j, k, AdmGamma11);
            const amrex::Real g12 = source(i, j, k, AdmGamma12);
            const amrex::Real g13 = source(i, j, k, AdmGamma13);
            const amrex::Real g22 = source(i, j, k, AdmGamma22);
            const amrex::Real g23 = source(i, j, k, AdmGamma23);
            const amrex::Real g33 = source(i, j, k, AdmGamma33);
            const amrex::Real determinant =
               g11 * (g22 * g33 - g23 * g23)
               - g12 * (g12 * g33 - g13 * g23)
               + g13 * (g12 * g23 - g13 * g22);
            const amrex::Real chi = 1.0 / std::cbrt(determinant);
            const amrex::Real inv11 = (g22 * g33 - g23 * g23) / determinant;
            const amrex::Real inv12 = (g13 * g23 - g12 * g33) / determinant;
            const amrex::Real inv13 = (g12 * g23 - g13 * g22) / determinant;
            const amrex::Real inv22 = (g11 * g33 - g13 * g13) / determinant;
            const amrex::Real inv23 = (g12 * g13 - g11 * g23) / determinant;
            const amrex::Real inv33 = (g11 * g22 - g12 * g12) / determinant;
            const amrex::Real k11 = source(i, j, k, AdmK11);
            const amrex::Real k12 = source(i, j, k, AdmK12);
            const amrex::Real k13 = source(i, j, k, AdmK13);
            const amrex::Real k22 = source(i, j, k, AdmK22);
            const amrex::Real k23 = source(i, j, k, AdmK23);
            const amrex::Real k33 = source(i, j, k, AdmK33);
            const amrex::Real trace_k = inv11 * k11 + 2.0 * inv12 * k12
               + 2.0 * inv13 * k13 + inv22 * k22
               + 2.0 * inv23 * k23 + inv33 * k33;
            Z4cState state;
            state.chi = chi;
            state.gamma_tilde11 = chi * g11;
            state.gamma_tilde12 = chi * g12;
            state.gamma_tilde13 = chi * g13;
            state.gamma_tilde22 = chi * g22;
            state.gamma_tilde23 = chi * g23;
            state.gamma_tilde33 = chi * g33;
            state.k_hat = trace_k;
            state.a_tilde11 = chi * (k11 - g11 * trace_k / 3.0);
            state.a_tilde12 = chi * (k12 - g12 * trace_k / 3.0);
            state.a_tilde13 = chi * (k13 - g13 * trace_k / 3.0);
            state.a_tilde22 = chi * (k22 - g22 * trace_k / 3.0);
            state.a_tilde23 = chi * (k23 - g23 * trace_k / 3.0);
            state.a_tilde33 = chi * (k33 - g33 * trace_k / 3.0);
            state.lapse = source(i, j, k, AdmLapse);
            state.shift1 = source(i, j, k, AdmShift1);
            state.shift2 = source(i, j, k, AdmShift2);
            state.shift3 = source(i, j, k, AdmShift3);
            store_z4c_state(destination, i, j, k, state);
         });
      }
   }
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
   ConstPhysicalIdFaceFields const& face_magnetic,
   PhysicalIdCoordinateFaces const& coordinate_faces
)
{
   const int levels = static_cast<int>(geometry.size());
   if (levels == 0 || static_cast<int>(refinement_ratios.size()) != levels - 1) {
      amrex::Abort("physical initial data require a nonempty consistent hierarchy");
   }
   if (!coordinate_faces.empty()
       && static_cast<int>(coordinate_faces.size()) != levels) {
      amrex::Abort("physical initial-data coordinate hierarchy is inconsistent");
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
   const bool write_exact_caches = !(
      cfg.initial_data_solve_only && cfg.mesh_coordinate != "cartesian"
   );
   if (write_exact_caches) {
      write_field(
         path,
         EvolutionFluidField,
         evolution_fluid,
         geometry,
         refinement_ratios,
         compression
      );
   }

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
      if (write_exact_caches) {
         write_field(
            path,
            EvolutionSpacetimeField,
            evolution_spacetime,
            geometry,
            refinement_ratios,
            compression
         );
      }
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
      header << "PyFluid physical initial data v3\n"
             << "Provider-independent physical fields with exact same-grid "
                "evolution caches.\n";
      std::ofstream metadata(std::filesystem::path(path) / "Header.meta");
      const EOSIdentity eos_metadata = eos_identity(cfg);
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
      metadata << "# component_basis=mesh_orthonormal\n";
      metadata << "# coordinate_layout="
               << (coordinate_faces.empty()
                     ? "uniform" : "explicit_faces_v1")
               << '\n';
      for (int level = 0;
           level < static_cast<int>(coordinate_faces.size()); ++level) {
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            auto const& faces = coordinate_faces[level][direction];
            if (!faces.empty()) {
               metadata << "# coordinate_faces_" << level << '_'
                        << direction << '='
                        << coordinate_face_string(faces) << '\n';
            }
         }
      }
      metadata << "# domain=" << domain_string(geometry.front()) << '\n';
      metadata << "# prob_lo=" << real_vector_string(geometry.front(), false) << '\n';
      metadata << "# prob_hi=" << real_vector_string(geometry.front(), true) << '\n';
      metadata << "# periodicity=" << periodicity_string(geometry.front()) << '\n';
      metadata << "# reflection_lo=" << reflection_string(cfg, false) << '\n';
      metadata << "# reflection_hi=" << reflection_string(cfg, true) << '\n';
      metadata << "# refinement_ratios=" << refinement_string(refinement_ratios) << '\n';
      metadata << "# eos=" << cfg.eos << '\n';
      metadata << "# eos_backend=" << eos_metadata.backend << '\n';
      metadata << "# eos_preset=" << cfg.eos_preset << '\n';
      metadata << "# gamma=" << output_metadata_real_string(cfg.gamma) << '\n';
      if (cfg.eos == "compose") {
         metadata << "# eos_table_fingerprint="
                  << eos_metadata.fingerprint << '\n';
         metadata << "# eos_table_source="
                  << eos_metadata.source_directory << '\n';
         metadata << "# eos_table_source_points="
                  << eos_metadata.source_points << '\n';
         metadata << "# eos_table_retained_points="
                  << eos_metadata.retained_points << '\n';
         metadata << "# eos_table_truncated_for_causality="
                  << (eos_metadata.table_truncated_for_causality ? 1 : 0)
                  << '\n';
         metadata << "# eos_table_baryon_mass_mev="
                  << output_metadata_real_string(eos_metadata.baryon_mass_mev)
                  << '\n';
      } else if (cfg.eos != "ideal_gas" && cfg.eos_preset == "custom") {
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
      }
      if (cfg.eos == "hybrid_polytrope" || cfg.eos == "compose") {
         metadata << "# thermal_gamma="
                  << output_metadata_real_string(cfg.thermal_gamma) << '\n';
      }
      metadata << "# passive_scalars=" << cfg.passive_scalars << '\n';
      metadata << "# has_spacetime=" << has_spacetime << '\n';
      metadata << "# has_ct=" << has_ct << '\n';
      metadata << "# magnetic_seed="
               << cfg.initial_data_magnetic_seed << '\n';
      metadata << "# magnetic_equilibrium="
               << cfg.initial_data_magnetic_equilibrium << '\n';
      metadata << "# magnetic_force_balance="
               << (cfg.initial_data_magnetic_equilibrium != "none"
                     ? "grad_shafranov"
                     : (cfg.initial_data_magnetic_seed == "none"
                           ? "not_applicable" : "not_solved"))
               << '\n';
      metadata << "# magnetic_face_representation="
               << (physical_system(cfg) == PhysicalSystem::RMHD
                     ? "valencia_densitized" : "physical")
               << '\n';
      metadata << "# magnetic_seed_pressure_cutoff="
               << output_metadata_real_string(
                    cfg.initial_data_magnetic_seed_pressure_cutoff
                  ) << '\n';
      metadata << "# magnetic_seed_exponent="
               << output_metadata_real_string(
                    cfg.initial_data_magnetic_seed_exponent
                  ) << '\n';
      metadata << "# magnetic_seed_maximum_comoving_field="
               << output_metadata_real_string(
                    cfg.initial_data_magnetic_seed_maximum_comoving_field
                  ) << '\n';
      metadata << "# magnetic_equilibrium_poloidal_magnetization="
               << output_metadata_real_string(
                    cfg.initial_data_magnetic_equilibrium_poloidal_magnetization
                  ) << '\n';
      metadata << "# magnetic_equilibrium_nonlinear_poloidal="
               << output_metadata_real_string(
                    cfg.initial_data_magnetic_equilibrium_nonlinear_poloidal
                  ) << '\n';
      metadata << "# magnetic_equilibrium_relaxation="
               << output_metadata_real_string(
                    cfg.initial_data_magnetic_equilibrium_relaxation
                  ) << '\n';
      metadata << "# has_exact_caches=" << write_exact_caches << '\n';
      metadata << "# source_ic=" << cfg.ic << '\n';
      metadata << "# spacetime_initial_data="
               << cfg.spacetime_initial_data << '\n';
      if (cfg.spacetime_initial_data == "kerr_schild"
          || cfg.ic == "black_hole") {
         metadata << "# black_hole_mass="
                  << output_metadata_real_string(cfg.spacetime_mass) << '\n';
         metadata << "# black_hole_spin1="
                  << output_metadata_real_string(
                       cfg.spacetime_dimensionless_spin1
                     ) << '\n';
         metadata << "# black_hole_spin2="
                  << output_metadata_real_string(
                       cfg.spacetime_dimensionless_spin2
                     ) << '\n';
         metadata << "# black_hole_spin3="
                  << output_metadata_real_string(
                       cfg.spacetime_dimensionless_spin3
                     ) << '\n';
         metadata << "# black_hole_origin1="
                  << output_metadata_real_string(cfg.spacetime_origin1)
                  << '\n';
         metadata << "# black_hole_origin2="
                  << output_metadata_real_string(cfg.spacetime_origin2)
                  << '\n';
         metadata << "# black_hole_origin3="
                  << output_metadata_real_string(cfg.spacetime_origin3)
                  << '\n';
         metadata << "# black_hole_interior="
                  << cfg.spacetime_black_hole_interior << '\n';
         metadata << "# black_hole_fill_radius="
                  << output_metadata_real_string(
                       cfg.spacetime_black_hole_fill_radius
                     ) << '\n';
      }
      metadata << "# disk=" << cfg.initial_data_disk << '\n';
      metadata << "# disk_gravity=" << cfg.initial_data_disk_gravity << '\n';
      metadata << "# disk_rotation_law="
               << cfg.initial_data_disk_rotation_law << '\n';
      metadata << "# disk_specific_angular_momentum="
               << output_metadata_real_string(
                    cfg.initial_data_disk_specific_angular_momentum
                  ) << '\n';
      metadata << "# disk_angular_velocity="
               << output_metadata_real_string(
                    cfg.initial_data_disk_angular_velocity
                  ) << '\n';
      metadata << "# disk_differential_rotation_length="
               << output_metadata_real_string(
                    cfg.initial_data_disk_differential_rotation_length
                  ) << '\n';
      metadata << "# disk_rotation_power_q="
               << output_metadata_real_string(
                    cfg.initial_data_disk_rotation_power_q
                  ) << '\n';
      metadata << "# disk_inner_radius="
               << output_metadata_real_string(
                    cfg.initial_data_disk_inner_radius
                  ) << '\n';
      metadata << "# disk_axis1="
               << output_metadata_real_string(cfg.initial_data_disk_axis1)
               << '\n';
      metadata << "# disk_axis2="
               << output_metadata_real_string(cfg.initial_data_disk_axis2)
               << '\n';
      metadata << "# disk_axis3="
               << output_metadata_real_string(cfg.initial_data_disk_axis3)
               << '\n';
      metadata << "# tov_central_density="
               << output_metadata_real_string(cfg.tov_central_density) << '\n';
      metadata << "# rotation_law=" << cfg.initial_data_rotation_law << '\n';
      metadata << "# angular_velocity="
               << output_metadata_real_string(cfg.initial_data_angular_velocity)
               << '\n';
      metadata << "# differential_rotation_length="
               << output_metadata_real_string(
                    cfg.initial_data_differential_rotation_length
                  ) << '\n';
      metadata << "# differential_rotation_length_b="
               << output_metadata_real_string(
                    cfg.initial_data_differential_rotation_length_b
                  ) << '\n';
      metadata << "# rotation_power_q="
               << output_metadata_real_string(cfg.initial_data_rotation_power_q)
               << '\n';
      metadata << "# rotation_axis1="
               << output_metadata_real_string(cfg.initial_data_rotation_axis1)
               << '\n';
      metadata << "# rotation_axis2="
               << output_metadata_real_string(cfg.initial_data_rotation_axis2)
               << '\n';
      metadata << "# rotation_axis3="
               << output_metadata_real_string(cfg.initial_data_rotation_axis3)
               << '\n';
      metadata << "# rotation_center1="
               << output_metadata_real_string(cfg.initial_data_rotation_center1)
               << '\n';
      metadata << "# rotation_center2="
               << output_metadata_real_string(cfg.initial_data_rotation_center2)
               << '\n';
      metadata << "# rotation_center3="
               << output_metadata_real_string(cfg.initial_data_rotation_center3)
               << '\n';
   }
   amrex::ParallelDescriptor::Barrier();
}

PhysicalInitialDataMetadata read_physical_initial_data(
   std::string const& path,
   RunConfig const& cfg,
   PhysicalSystem system,
   InitialAtmosphereParameters const& atmosphere,
   amrex::Vector<amrex::Geometry> const& geometry,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   MutablePhysicalIdField const& primitive,
   MutablePhysicalIdField const& evolution_fluid,
   MutablePhysicalIdField const& evolution_spacetime,
   MutablePhysicalIdFaceFields const& face_magnetic,
   PhysicalIdCoordinateFaces* coordinate_faces
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
   validate_level_fields(
      ConstPhysicalIdField(primitive.begin(), primitive.end()),
      levels,
      physical_fluid_components,
      PhysicalFluidField
   );
   const auto metadata = read_metadata(path);
   require_metadata(metadata, "type", "physical_initial_data", path);
   int schema_version = 0;
   int source_dimension = 0;
   try {
      schema_version = std::stoi(
         required_metadata_value(metadata, "schema_version", path)
      );
      source_dimension = std::stoi(
         required_metadata_value(metadata, "amrex_spacedim", path)
      );
   } catch (std::exception const&) {
      amrex::Abort("physical initial-data schema metadata is invalid in " + path);
   }
   if ((schema_version != PhysicalIdSchemaVersion
        && schema_version != 2
        && schema_version != LegacyPhysicalIdSchemaVersion)
       || source_dimension < 1 || source_dimension > 3) {
      amrex::Abort("unsupported physical initial-data schema in " + path);
   }
   require_metadata(metadata, "format", "hdf5", path);
   require_metadata(metadata, "physics", cfg.physics, path);
   require_metadata(metadata, "fluid_components", std::to_string(fluid_components), path);
   require_metadata(
      metadata,
      "physical_fluid_components",
      std::to_string(physical_fluid_components),
      path
   );
   const std::string source_coordinate = required_metadata_value(
      metadata, "mesh_coordinate", path
   );
   const std::string source_axes_value = required_metadata_value(
      metadata, "mesh_axes", path
   );
   const auto source_periodic = parse_binary_directions(
      required_metadata_value(metadata, "periodicity", path),
      path,
      "periodicity",
      source_dimension
   );
   const auto reflection_lo_entry = metadata.find("reflection_lo");
   const auto reflection_hi_entry = metadata.find("reflection_hi");
   if ((reflection_lo_entry == metadata.end())
       != (reflection_hi_entry == metadata.end())) {
      amrex::Abort(
         "physical-ID reflection metadata must define both sides in " + path
      );
   }
   const std::array<int, 3> source_reflection_lo =
      reflection_lo_entry == metadata.end()
      ? std::array<int, 3>{}
      : parse_binary_directions(
           reflection_lo_entry->second,
           path,
           "reflection_lo",
           source_dimension
        );
   const std::array<int, 3> source_reflection_hi =
      reflection_hi_entry == metadata.end()
      ? std::array<int, 3>{}
      : parse_binary_directions(
           reflection_hi_entry->second,
           path,
           "reflection_hi",
           source_dimension
        );
   const PhysicalIdChart source_chart = make_physical_id_chart(
      source_dimension,
      source_coordinate,
      source_axes_value,
      source_periodic,
      source_reflection_lo,
      source_reflection_hi
   );
   const PhysicalIdChart target_chart = current_physical_id_chart(
      cfg.mesh_coordinate, cfg.mesh_axes
   );
   const RuntimePhysicalIdCoordinateFaces source_coordinate_faces =
      runtime_coordinate_faces(
         metadata,
         source_dimension,
         std::stoi(required_metadata_value(metadata, "levels", path)),
         path
      );
   bool same_chart = source_chart.dimension == target_chart.dimension
      && source_chart.coordinate_kind == target_chart.coordinate_kind;
   for (int direction = 0; same_chart && direction < AMREX_SPACEDIM;
        ++direction) {
      same_chart = source_chart.axes[direction] == target_chart.axes[direction]
         && source_chart.periodic[direction]
            == geometry.front().isPeriodic(direction);
   }
   const bool portable_basis = metadata.find("component_basis") != metadata.end()
      && metadata.at("component_basis") == "mesh_orthonormal";
   if (!same_chart && !portable_basis) {
      amrex::Abort(
         "cross-coordinate or cross-dimensional physical-ID materialization "
         "requires a portable mesh_orthonormal artifact"
      );
   }
   require_metadata(metadata, "eos", cfg.eos, path);
   const EOSIdentity eos_metadata = eos_identity(cfg);
   if (cfg.eos == "compose" || metadata.find("eos_backend") != metadata.end()) {
      require_metadata(metadata, "eos_backend", eos_metadata.backend, path);
   }
   const auto eos_preset_entry = metadata.find("eos_preset");
   if (eos_preset_entry != metadata.end()) {
      require_metadata(metadata, "eos_preset", cfg.eos_preset, path);
   } else if (cfg.eos_preset != "custom") {
      amrex::Abort(
         "legacy physical initial data do not identify their named EoS preset; "
         "regenerate the artifact before using eos_preset = " + cfg.eos_preset
      );
   }
   if (cfg.eos == "ideal_gas") {
      require_metadata(
         metadata, "gamma", output_metadata_real_string(cfg.gamma), path
      );
   } else if (cfg.eos == "compose") {
      require_metadata(
         metadata, "eos_table_fingerprint", eos_metadata.fingerprint, path
      );
      require_metadata(
         metadata,
         "eos_table_source_points",
         std::to_string(eos_metadata.source_points),
         path
      );
      require_metadata(
         metadata,
         "eos_table_retained_points",
         std::to_string(eos_metadata.retained_points),
         path
      );
      require_metadata(
         metadata,
         "eos_table_truncated_for_causality",
         eos_metadata.table_truncated_for_causality ? "1" : "0",
         path
      );
      require_metadata(
         metadata,
         "eos_table_baryon_mass_mev",
         output_metadata_real_string(eos_metadata.baryon_mass_mev),
         path
      );
   } else if (cfg.eos_preset == "custom") {
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
   }
   if (cfg.eos == "hybrid_polytrope" || cfg.eos == "compose") {
      require_metadata(
         metadata,
         "thermal_gamma",
         output_metadata_real_string(cfg.thermal_gamma),
         path
      );
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
   if (expects_ct && system == PhysicalSystem::RMHD) {
      if (schema_version < 3) {
         amrex::Abort(
            "legacy RMHD physical-ID artifacts do not declare whether CT "
            "face fields are physical or Valencia-densitized in " + path
         );
      }
      require_metadata(
         metadata,
         "magnetic_face_representation",
         "valencia_densitized",
         path
      );
   }
   if (expects_ct && !same_chart) {
      amrex::Abort(
         "cross-coordinate physical-ID materialization does not yet support "
         "face-centered constrained transport fields"
      );
   }
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
      && (!expects_spacetime
          || multilevel_hdf5_field_exists(path, AdmGeometryField));
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

   PhysicalInitialDataMetadata result;
   result.source_ic = required_metadata_value(metadata, "source_ic", path);
   const auto magnetic_seed_entry = metadata.find("magnetic_seed");
   result.magnetic_seed = magnetic_seed_entry == metadata.end()
      ? "none" : magnetic_seed_entry->second;
   result.magnetic_seed_pressure_cutoff = optional_metadata_real(
      metadata, "magnetic_seed_pressure_cutoff", 0.0, path
   );
   result.magnetic_seed_exponent = optional_metadata_real(
      metadata, "magnetic_seed_exponent", 0.0, path
   );
   result.magnetic_seed_maximum_comoving_field =
      optional_metadata_real(
         metadata,
         "magnetic_seed_maximum_comoving_field",
         0.0,
         path
      );
   const auto magnetic_equilibrium_entry = metadata.find(
      "magnetic_equilibrium"
   );
   result.magnetic_equilibrium = magnetic_equilibrium_entry == metadata.end()
      ? "none" : magnetic_equilibrium_entry->second;
   const auto magnetic_force_balance_entry = metadata.find(
      "magnetic_force_balance"
   );
   result.magnetic_force_balance = magnetic_force_balance_entry
         == metadata.end()
      ? "" : magnetic_force_balance_entry->second;
   const auto magnetic_face_representation_entry = metadata.find(
      "magnetic_face_representation"
   );
   result.magnetic_face_representation = magnetic_face_representation_entry
         == metadata.end()
      ? "" : magnetic_face_representation_entry->second;
   if (schema_version >= 3) {
      if (result.magnetic_seed != "none"
          && result.magnetic_equilibrium != "none") {
         amrex::Abort(
            "physical-ID artifact declares both a magnetic seed and magnetic "
            "equilibrium in " + path
         );
      }
      const std::string expected_force_balance =
         result.magnetic_equilibrium != "none"
            ? "grad_shafranov"
            : (result.magnetic_seed != "none"
                  ? "not_solved" : "not_applicable");
      require_metadata(
         metadata,
         "magnetic_force_balance",
         expected_force_balance,
         path
      );
   }
   result.magnetic_equilibrium_poloidal_magnetization =
      optional_metadata_real(
         metadata,
         "magnetic_equilibrium_poloidal_magnetization",
         0.0,
         path
      );
   result.magnetic_equilibrium_nonlinear_poloidal = optional_metadata_real(
      metadata, "magnetic_equilibrium_nonlinear_poloidal", 0.0, path
   );
   result.magnetic_equilibrium_relaxation = optional_metadata_real(
      metadata, "magnetic_equilibrium_relaxation", 0.0, path
   );
   const auto disk_entry = metadata.find("disk");
   result.disk = disk_entry == metadata.end() ? "none" : disk_entry->second;
   const bool adding_torus = result.disk == "none"
      && (cfg.initial_data_disk == "equilibrium_torus"
          || cfg.initial_data_disk == "tilted_torus")
      && cfg.spacetime_initial_data == "xcts";
   if (result.disk != cfg.initial_data_disk && !adding_torus) {
      amrex::Abort(
         "physical-ID disk matter identity is incompatible with the target in "
         + path
      );
   }
   if (result.disk == "equilibrium_torus"
       || result.disk == "tilted_torus") {
      result.disk_gravity = required_metadata_value(
         metadata, "disk_gravity", path
      );
      result.disk_rotation_law = required_metadata_value(
         metadata, "disk_rotation_law", path
      );
      result.disk_specific_angular_momentum = required_metadata_real(
         metadata, "disk_specific_angular_momentum", path
      );
      result.disk_angular_velocity = optional_metadata_real(
         metadata, "disk_angular_velocity", 0.0, path
      );
      result.disk_differential_rotation_length = optional_metadata_real(
         metadata, "disk_differential_rotation_length", 10.0, path
      );
      result.disk_rotation_power_q = optional_metadata_real(
         metadata, "disk_rotation_power_q", 1.0, path
      );
      result.disk_inner_radius = required_metadata_real(
         metadata, "disk_inner_radius", path
      );
      result.disk_axis = {{
         optional_metadata_real(metadata, "disk_axis1", 0.0, path),
         optional_metadata_real(metadata, "disk_axis2", 0.0, path),
         optional_metadata_real(metadata, "disk_axis3", 1.0, path),
      }};
      result.disk_maximum_density = required_metadata_real(
         metadata, "disk_maximum_density", path
      );
      result.disk_maximum_velocity = required_metadata_real(
         metadata, "disk_maximum_velocity", path
      );
      result.disk_resolved_inner_radius = required_metadata_real(
         metadata, "disk_resolved_inner_radius", path
      );
      result.disk_outer_radius = required_metadata_real(
         metadata, "disk_outer_radius", path
      );
      result.disk_density_maximum_radius = required_metadata_real(
         metadata, "disk_density_maximum_radius", path
      );
      result.disk_maximum_log_specific_enthalpy = required_metadata_real(
         metadata, "disk_maximum_log_specific_enthalpy", path
      );
      result.disk_baryonic_mass = required_metadata_real(
         metadata, "disk_baryonic_mass", path
      );
      result.disk_angular_momentum = required_metadata_real(
         metadata, "disk_angular_momentum", path
      );
      if (result.disk_gravity != cfg.initial_data_disk_gravity
          || result.disk_rotation_law
             != cfg.initial_data_disk_rotation_law
          || result.disk_specific_angular_momentum
             != cfg.initial_data_disk_specific_angular_momentum
          || (result.disk_rotation_law != "constant_l"
              && result.disk_angular_velocity
                 != cfg.initial_data_disk_angular_velocity)
          || ((result.disk_rotation_law == "j_constant"
               || result.disk_rotation_law == "q_power")
              && result.disk_differential_rotation_length
                 != cfg.initial_data_disk_differential_rotation_length)
          || (result.disk_rotation_law == "q_power"
              && result.disk_rotation_power_q
                 != cfg.initial_data_disk_rotation_power_q)
          || result.disk_inner_radius != cfg.initial_data_disk_inner_radius) {
         amrex::Abort(
            "physical-ID equilibrium-torus parameters are incompatible with "
            "the target in " + path
         );
      }
      if (!(result.disk_maximum_density > 0.0)
          || !(result.disk_maximum_velocity >= 0.0)
          || !(result.disk_maximum_velocity < 1.0)
          || !(result.disk_resolved_inner_radius > 0.0)
          || !(result.disk_outer_radius
               > result.disk_resolved_inner_radius)
          || !(result.disk_density_maximum_radius
               >= result.disk_resolved_inner_radius)
          || !(result.disk_density_maximum_radius
               <= result.disk_outer_radius)
          || !(result.disk_maximum_log_specific_enthalpy > 0.0)
          || !(result.disk_baryonic_mass > 0.0)
          || !std::isfinite(result.disk_angular_momentum)) {
         amrex::Abort(
            "physical-ID equilibrium-torus diagnostics are invalid in " + path
         );
      }
   }
   const auto source_spacetime = metadata.find("spacetime_initial_data");
   if (source_spacetime != metadata.end()) {
      result.spacetime_initial_data = source_spacetime->second;
      if (result.spacetime_initial_data != cfg.spacetime_initial_data) {
         amrex::Abort(
            "physical-ID spacetime class is incompatible with the target in "
            + path
         );
      }
   }
   if (result.spacetime_initial_data == "kerr_schild"
       || result.source_ic == "black_hole") {
      result.black_hole_mass = required_metadata_real(
         metadata, "black_hole_mass", path
      );
      result.black_hole_spin = {{
         required_metadata_real(metadata, "black_hole_spin1", path),
         required_metadata_real(metadata, "black_hole_spin2", path),
         required_metadata_real(metadata, "black_hole_spin3", path),
      }};
      result.black_hole_origin = {{
         required_metadata_real(metadata, "black_hole_origin1", path),
         required_metadata_real(metadata, "black_hole_origin2", path),
         required_metadata_real(metadata, "black_hole_origin3", path),
      }};
      result.black_hole_interior = required_metadata_value(
         metadata, "black_hole_interior", path
      );
      result.black_hole_fill_radius = required_metadata_real(
         metadata, "black_hole_fill_radius", path
      );
      if (result.black_hole_mass != cfg.spacetime_mass
          || result.black_hole_spin[0]
             != cfg.spacetime_dimensionless_spin1
          || result.black_hole_spin[1]
             != cfg.spacetime_dimensionless_spin2
          || result.black_hole_spin[2]
             != cfg.spacetime_dimensionless_spin3
          || result.black_hole_origin[0] != cfg.spacetime_origin1
          || result.black_hole_origin[1] != cfg.spacetime_origin2
          || result.black_hole_origin[2] != cfg.spacetime_origin3
          || result.black_hole_interior
             != cfg.spacetime_black_hole_interior
          || result.black_hole_fill_radius
             != cfg.spacetime_black_hole_fill_radius) {
         amrex::Abort(
            std::string("physical-ID ")
            + (result.spacetime_initial_data == "kerr_schild"
                  ? "Kerr-Schild" : "black-hole XCTS")
            + " parameters are incompatible with the target in " + path
         );
      }
   }
   result.rotation_law = required_metadata_value(metadata, "rotation_law", path);
   result.angular_velocity = required_metadata_real(
      metadata, "angular_velocity", path
   );
   result.injection_energy = optional_metadata_real(
      metadata, "injection_energy", -1.0, path
   );
   if (result.source_ic == "rotating_star") {
      result.maximum_density = required_metadata_real(
         metadata, "maximum_density", path
      );
      result.maximum_velocity = required_metadata_real(
         metadata, "maximum_velocity", path
      );
      result.equatorial_radius = required_metadata_real(
         metadata, "equatorial_radius", path
      );
      result.polar_radius = required_metadata_real(
         metadata, "polar_radius", path
      );
      result.equatorial_log_enthalpy_gradient = required_metadata_real(
         metadata, "equatorial_log_enthalpy_gradient", path
      );
      result.polar_log_enthalpy_gradient = required_metadata_real(
         metadata, "polar_log_enthalpy_gradient", path
      );
      result.mass_shedding_indicator = required_metadata_real(
         metadata, "mass_shedding_indicator", path
      );
      result.adm_mass = required_metadata_real(metadata, "adm_mass", path);
      result.baryonic_mass = required_metadata_real(
         metadata, "baryonic_mass", path
      );
      result.angular_momentum = required_metadata_real(
         metadata, "angular_momentum", path
      );
      result.dimensionless_spin = required_metadata_real(
         metadata, "dimensionless_spin", path
      );
      if (!(result.maximum_density > 0.0)
          || !(result.maximum_velocity >= 0.0
               && result.maximum_velocity < 1.0)
          || !(result.injection_energy > 0.0)
          || !(result.equatorial_radius > 0.0)
          || !(result.polar_radius > 0.0)
          || !(result.adm_mass > 0.0)
          || !(result.baryonic_mass > 0.0)) {
         amrex::Abort(
            "physical rotating-star diagnostic metadata is invalid in "
            + path
         );
      }
   }
   result.differential_rotation_length =
      cfg.initial_data_differential_rotation_length;
   result.differential_rotation_length_b =
      cfg.initial_data_differential_rotation_length_b;
   result.rotation_power_q = cfg.initial_data_rotation_power_q;
   if (result.rotation_law == "j_constant"
       || result.rotation_law == "q_power"
       || result.rotation_law == "uryu_plus_13"
       || result.rotation_law == "uryu_j_envelope_p1") {
      result.differential_rotation_length = required_metadata_real(
         metadata, "differential_rotation_length", path
      );
      if (!(result.differential_rotation_length > 0.0)) {
         amrex::Abort(
            "physical initial-data differential rotation length must be "
            "positive in " + path
         );
      }
   }
   if (result.rotation_law == "uryu_plus_13"
       || result.rotation_law == "uryu_j_envelope_p1") {
      result.differential_rotation_length_b = required_metadata_real(
         metadata, "differential_rotation_length_b", path
      );
      if (!(result.differential_rotation_length_b > 0.0)) {
         amrex::Abort(
            "physical initial-data Uryu differential rotation length B must "
            "be positive in " + path
         );
      }
   }
   if (result.rotation_law == "q_power") {
      result.rotation_power_q = required_metadata_real(
         metadata, "rotation_power_q", path
      );
      if (!(result.rotation_power_q >= 1.0)) {
         amrex::Abort(
            "physical initial-data q-power exponent must be at least one in "
            + path
         );
      }
   }
   if (result.source_ic == "rotating_star"
       || result.disk == "equilibrium_torus"
       || result.disk == "tilted_torus") {
      try {
         const auto stored_outer = metadata.find("matter_outer_radius");
         if (stored_outer != metadata.end()) {
            std::size_t parsed = 0;
            result.matter_outer_radius = std::stod(
               stored_outer->second, &parsed
            );
            if (parsed != stored_outer->second.size()) {
               throw std::invalid_argument(
                  "trailing matter_outer_radius metadata"
               );
            }
         } else {
            const amrex::Real equatorial = result.source_ic == "rotating_star"
               ? result.equatorial_radius
               : std::stod(
                    required_metadata_value(
                       metadata, "equatorial_radius", path
                    )
                 );
            const amrex::Real polar = result.source_ic == "rotating_star"
               ? result.polar_radius
               : std::stod(
                    required_metadata_value(metadata, "polar_radius", path)
                 );
            // Older artifacts store independent cylindrical and axial
            // extents.  Their Euclidean combination is a conservative sphere
            // enclosing every source matter cell.
            result.matter_outer_radius = std::hypot(equatorial, polar);
         }
      } catch (std::exception const&) {
         amrex::Abort(
            "rotating physical-ID matter-support metadata is invalid in "
            + path
         );
      }
      if (!(result.matter_outer_radius > 0.0)
          || !std::isfinite(result.matter_outer_radius)) {
         amrex::Abort(
            "rotating physical-ID matter-support radius is invalid in " + path
         );
      }
   }
   // Schema-v1 artifacts predate explicit pose metadata and were centered at
   // the coordinate origin.  The target center is a materialization choice,
   // so it must not be used as the missing source value.
   result.rotation_center = {0.0, 0.0, 0.0};
   result.rotation_axis = {{
      optional_metadata_real(metadata, "rotation_axis1", 0.0, path),
      optional_metadata_real(metadata, "rotation_axis2", 0.0, path),
      optional_metadata_real(metadata, "rotation_axis3", 1.0, path),
   }};
   for (int component = 0; component < 3; ++component) {
      const std::string key = "rotation_center" + std::to_string(component + 1);
      const auto found = metadata.find(key);
      if (found == metadata.end()) {
         continue;
      }
      try {
         std::size_t parsed = 0;
         result.rotation_center[component] = std::stod(found->second, &parsed);
         if (parsed != found->second.size()
             || !std::isfinite(result.rotation_center[component])) {
            throw std::invalid_argument("invalid rotation center");
         }
      } catch (std::exception const&) {
         amrex::Abort(
            "physical initial-data " + key + " metadata is invalid in " + path
         );
      }
   }
   const std::array<amrex::Real, 3> target_rotation_axis{
      cfg.initial_data_rotation_axis1,
      cfg.initial_data_rotation_axis2,
      cfg.initial_data_rotation_axis3,
   };
   const std::array<amrex::Real, 3> target_disk_axis{
      cfg.initial_data_disk_axis1,
      cfg.initial_data_disk_axis2,
      cfg.initial_data_disk_axis3,
   };
   const std::array<amrex::Real, 3> target_rotation_center{
      cfg.initial_data_rotation_center1,
      cfg.initial_data_rotation_center2,
      cfg.initial_data_rotation_center3,
   };
   const std::array<amrex::Real, 3> source_rotation_axis{
      result.rotation_axis[0],
      result.rotation_axis[1],
      result.rotation_axis[2],
   };
   const std::array<amrex::Real, 3> source_disk_axis{
      result.disk_axis[0], result.disk_axis[1], result.disk_axis[2]
   };
   const std::array<amrex::Real, 3> source_rotation_center{
      result.rotation_center[0],
      result.rotation_center[1],
      result.rotation_center[2],
   };
   PhysicalIdRigidTransform rigid_transform;
   if (result.spacetime_initial_data == "kerr_schild"
       || result.source_ic == "black_hole") {
      rigid_transform.source_origin = {
         result.black_hole_origin[0],
         result.black_hole_origin[1],
         result.black_hole_origin[2],
      };
      rigid_transform.target_origin = rigid_transform.source_origin;
   } else if (result.source_ic == "rotating_star") {
      rigid_transform = make_physical_id_rigid_transform(
         source_rotation_center,
         target_rotation_center,
         source_rotation_axis,
         target_rotation_axis
      );
   } else if (result.disk == "equilibrium_torus"
              || result.disk == "tilted_torus") {
      rigid_transform = make_physical_id_rigid_transform(
         source_rotation_center,
         target_rotation_center,
         source_disk_axis,
         target_disk_axis
      );
   } else if (result.source_ic == "tov") {
      rigid_transform = make_physical_id_rigid_transform(
         source_rotation_center,
         target_rotation_center,
         {0.0, 0.0, 1.0},
         {0.0, 0.0, 1.0}
      );
   }
   result.used_rigid_transform =
      !physical_id_rigid_transform_is_identity(rigid_transform);
   if (result.disk == "equilibrium_torus"
       || result.disk == "tilted_torus") {
      std::array<amrex::Real, 3> mapped_disk_axis{};
      amrex::Real mapped_norm_squared = 0.0;
      amrex::Real target_norm_squared = 0.0;
      for (int i = 0; i < 3; ++i) {
         for (int j = 0; j < 3; ++j) {
            mapped_disk_axis[i] += rigid_transform.source_to_target[i][j]
               * result.disk_axis[j];
         }
         mapped_norm_squared += mapped_disk_axis[i] * mapped_disk_axis[i];
         target_norm_squared += target_disk_axis[i] * target_disk_axis[i];
      }
      if (!(mapped_norm_squared > 0.0) || !(target_norm_squared > 0.0)) {
         amrex::Abort(
            "physical-ID disk rotation axis must be finite and nonzero"
         );
      }
      const amrex::Real mapped_norm = std::sqrt(mapped_norm_squared);
      const amrex::Real target_norm = std::sqrt(target_norm_squared);
      for (int component = 0; component < 3; ++component) {
         if (std::abs(mapped_disk_axis[component] / mapped_norm
                      - target_disk_axis[component] / target_norm)
             > 1.0e-12) {
            amrex::Abort(
               "physical-ID stellar/disk relative orientation is "
               "incompatible with the target in " + path
            );
         }
      }
   }
   bool exact_boxes = false;
   if (source_dimension == AMREX_SPACEDIM && same_chart) {
      const MultilevelHdf5FieldLayout stored_layout =
         read_multilevel_hdf5_field_layout(path, PhysicalFluidField);
      for (int level = 0;
           level < static_cast<int>(source_coordinate_faces.size()); ++level) {
         for (int direction = 0; direction < AMREX_SPACEDIM;
              ++direction) {
            auto const& faces = source_coordinate_faces[level][direction];
            if (!faces.empty()
                && static_cast<int>(faces.size())
                   != stored_layout.domains[level].length(direction) + 1) {
               amrex::Abort(
                  "physical-ID coordinate face count is inconsistent in "
                  + path
               );
            }
         }
      }
      exact_boxes = static_cast<int>(stored_layout.boxes.size()) == levels;
      for (int level = 0; exact_boxes && level < levels; ++level) {
         const amrex::BoxArray& stored_boxes = stored_layout.boxes[level];
         const amrex::BoxArray& target_boxes = evolution_fluid[level]->boxArray();
         exact_boxes = stored_boxes.size() == target_boxes.size();
         for (int target = 0; exact_boxes && target < target_boxes.size(); ++target) {
            bool found = false;
            for (int stored = 0; stored < stored_boxes.size(); ++stored) {
               found = found || stored_boxes[stored] == target_boxes[target];
            }
            exact_boxes = found;
         }
      }
   }
   const bool exact_layout = exact_boxes
      && same_chart
      && !result.used_rigid_transform
      && source_dimension == AMREX_SPACEDIM
      && coordinate_faces_match(
         source_coordinate_faces, coordinate_faces, source_dimension
      )
      && required_metadata_value(metadata, "levels", path)
         == std::to_string(levels)
      && required_metadata_value(metadata, "domain", path)
         == domain_string(geometry.front())
      && required_metadata_value(metadata, "prob_lo", path)
         == real_vector_string(geometry.front(), false)
      && required_metadata_value(metadata, "prob_hi", path)
         == real_vector_string(geometry.front(), true)
      && required_metadata_value(metadata, "refinement_ratios", path)
         == refinement_string(refinement_ratios);
   result.exact_layout = exact_layout;
   if (coordinate_faces != nullptr) {
      if (exact_layout && !source_coordinate_faces.empty()) {
         coordinate_faces->clear();
         coordinate_faces->resize(source_coordinate_faces.size());
         for (std::size_t level = 0;
              level < source_coordinate_faces.size(); ++level) {
            for (int direction = 0; direction < AMREX_SPACEDIM;
                 ++direction) {
               auto const& source_faces =
                  source_coordinate_faces[level][direction];
               (*coordinate_faces)[level][direction].assign(
                  source_faces.begin(), source_faces.end()
               );
            }
         }
      }
   }
   bool has_exact_caches = multilevel_hdf5_field_exists(
      path, EvolutionFluidField
   );
   has_exact_caches = has_exact_caches
      && (!expects_spacetime
          || multilevel_hdf5_field_exists(path, EvolutionSpacetimeField));
   if (exact_layout && has_exact_caches) {
      for (std::size_t level = 0; level < evolution_fluid.size(); ++level) {
         read_multilevel_hdf5_field(
            *evolution_fluid[level],
            path,
            static_cast<int>(level),
            EvolutionFluidField
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
      result.used_exact_cache = true;
      return result;
   }

   // A mapped solver grid is an index-space AMReX hierarchy plus explicit
   // physical coordinates.  For an exact same-layout continuation, preserve
   // those physical degrees of freedom directly; reinterpolating them through
   // the uniform logical Geometry would change the seed before the next solve.
   if (exact_layout && !source_coordinate_faces.empty() && !expects_ct) {
      for (int level = 0; level < levels; ++level) {
         read_multilevel_hdf5_field(
            *primitive[level], path, level, PhysicalFluidField
         );
      }
      if (expects_spacetime) {
         const int adm_components = grid_field_component_count(
            adm_initial_data_grid_fields()
         );
         amrex::Vector<std::unique_ptr<amrex::MultiFab>> target_adm(levels);
         for (int level = 0; level < levels; ++level) {
            target_adm[level] = std::make_unique<amrex::MultiFab>(
               evolution_spacetime[level]->boxArray(),
               evolution_spacetime[level]->DistributionMap(),
               adm_components,
               0
            );
            read_multilevel_hdf5_field(
               *target_adm[level], path, level, AdmGeometryField
            );
         }
         adm_to_z4c(evolution_spacetime, target_adm);
      }
      return result;
   }

   if (portable_basis && !expects_ct
       && (source_dimension != AMREX_SPACEDIM || !same_chart
           || !source_coordinate_faces.empty()
           || result.used_rigid_transform)) {
      const amrex::GpuArray<amrex::Real, 3> origin{{
         rigid_transform.target_origin[0],
         rigid_transform.target_origin[1],
         rigid_transform.target_origin[2],
      }};
      const GridFieldSetInfo fluid_field_set = physical_fluid_grid_fields(
         system, cfg.passive_scalars
      );
      interpolate_portable_field(
         path,
         PhysicalFluidField,
         source_chart,
         target_chart,
         geometry,
         primitive,
         fluid_field_set,
         amrex::IndexType::TheCellType(),
         source_coordinate_faces,
         coordinate_faces == nullptr
            ? PhysicalIdCoordinateFaces{} : *coordinate_faces,
         rigid_transform,
         origin,
         result.source_ic == "tov" || result.source_ic == "rotating_star"
            ? PhysicalFieldExterior::InitialAtmosphere
            : PhysicalFieldExterior::Constant,
         false,
         system,
         eos_method_code(cfg),
         atmosphere,
         result.matter_outer_radius
      );
      if (expects_spacetime) {
         const GridFieldSetInfo& adm_field_set = adm_initial_data_grid_fields();
         const int adm_components = grid_field_component_count(adm_field_set);
         amrex::Vector<std::unique_ptr<amrex::MultiFab>> target_adm(levels);
         MutablePhysicalIdField target_adm_view(levels);
         for (int level = 0; level < levels; ++level) {
            target_adm[level] = std::make_unique<amrex::MultiFab>(
               evolution_spacetime[level]->boxArray(),
               evolution_spacetime[level]->DistributionMap(),
               adm_components,
               0
            );
            target_adm_view[level] = target_adm[level].get();
         }
         interpolate_portable_field(
            path,
            AdmGeometryField,
            source_chart,
            target_chart,
            geometry,
            target_adm_view,
            adm_field_set,
            amrex::IndexType::TheNodeType(),
            source_coordinate_faces,
            coordinate_faces == nullptr
               ? PhysicalIdCoordinateFaces{} : *coordinate_faces,
            rigid_transform,
            origin,
            result.source_ic == "tov"
               ? PhysicalFieldExterior::TovSchwarzschildAdm
               : PhysicalFieldExterior::AsymptoticallyFlatAdm,
            result.source_ic == "rotating_star",
            system,
            eos_method_code(cfg),
            atmosphere,
            -1.0
         );
         adm_to_z4c(evolution_spacetime, target_adm);
      }
      return result;
   }

   const StoredPhysicalHierarchy source = stored_physical_hierarchy(
      path, metadata
   );
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      if (!source.geometry.front().isPeriodic(direction)) {
         continue;
      }
      const amrex::Real source_extent = source.geometry.front().ProbLength(
         direction
      );
      const amrex::Real target_extent = geometry.front().ProbLength(direction);
      const amrex::Real periods = target_extent / source_extent;
      const amrex::Real nearest_periods = std::round(periods);
      if (!(nearest_periods >= 1.0)
          || std::abs(periods - nearest_periods)
             > 128.0 * std::numeric_limits<amrex::Real>::epsilon()
                * amrex::max(1.0, std::abs(periods))) {
         amrex::Abort(
            "periodic physical-ID materialization requires each target extent "
            "to contain an integer number of saved periods"
         );
      }
   }
   const auto source_fluid = load_stored_field(
      path,
      PhysicalFluidField,
      source,
      amrex::IndexType::TheCellType(),
      physical_fluid_components
   );
   const amrex::GpuArray<amrex::Real, 3> origin{{
      rigid_transform.target_origin[0],
      rigid_transform.target_origin[1],
      rigid_transform.target_origin[2],
   }};
   interpolate_stored_field(
      source_fluid,
      source,
      geometry,
      primitive,
      amrex::IndexType::TheCellType(),
      component_symmetries(
         physical_fluid_grid_fields(system, cfg.passive_scalars)
      ),
      origin,
      result.source_ic == "tov" || result.source_ic == "rotating_star"
         ? PhysicalFieldExterior::InitialAtmosphere
         : PhysicalFieldExterior::Constant,
      false,
      system,
      eos_method_code(cfg),
      atmosphere
   );
   if (expects_spacetime) {
      const int adm_components = grid_field_component_count(
         adm_initial_data_grid_fields()
      );
      const auto source_adm = load_stored_field(
         path,
         AdmGeometryField,
         source,
         amrex::IndexType::TheNodeType(),
         adm_components
      );
      amrex::Vector<std::unique_ptr<amrex::MultiFab>> target_adm(levels);
      MutablePhysicalIdField target_adm_view(levels);
      for (int level = 0; level < levels; ++level) {
         target_adm[level] = std::make_unique<amrex::MultiFab>(
            evolution_spacetime[level]->boxArray(),
            evolution_spacetime[level]->DistributionMap(),
            adm_components,
            0
         );
         target_adm_view[level] = target_adm[level].get();
      }
      interpolate_stored_field(
         source_adm,
         source,
         geometry,
         target_adm_view,
         amrex::IndexType::TheNodeType(),
         component_symmetries(adm_initial_data_grid_fields()),
         origin,
         result.source_ic == "tov"
            ? PhysicalFieldExterior::TovSchwarzschildAdm
            : PhysicalFieldExterior::AsymptoticallyFlatAdm,
         result.source_ic == "rotating_star",
         system,
         eos_method_code(cfg),
         atmosphere
      );
      adm_to_z4c(evolution_spacetime, target_adm);
   }
   if (expects_ct) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         const amrex::IndexType centering(
            amrex::IntVect::TheDimensionVector(direction)
         );
         const auto source_face = load_stored_field(
            path,
            checkpoint_face_b_field_name(direction),
            source,
            centering,
            1
         );
         interpolate_stored_field(
            source_face,
            source,
            geometry,
            face_magnetic[direction],
            centering,
            {
               ct_magnetic_grid_field().components[
                  static_cast<std::size_t>(direction)
               ].symmetry
            },
            origin,
            PhysicalFieldExterior::Constant,
            false,
            system,
            eos_method_code(cfg),
            atmosphere
         );
      }
   }
   return result;
}

namespace {

template <int EOSMethod>
void materialize_physical_initial_data_fluid_method(
   RunConfig const& cfg,
   PhysicalSystem system,
   amrex::Vector<amrex::Geometry> const& geometry,
   ConstPhysicalIdField const& primitive,
   ConstPhysicalIdField const& cell_spacetime,
   MutablePhysicalIdField const& evolution_fluid
)
{
   const EOSParams eos = eos_params(cfg);
   const bool relativistic = physical_system_is_relativistic(system);
   const bool magnetized = physical_system_has_magnetic_field(system);
   for (std::size_t level = 0; level < evolution_fluid.size(); ++level) {
      const bool stored_spacetime = !cell_spacetime.empty();
      const auto lower = geometry[level].ProbLoArray();
      const auto spacing = geometry[level].CellSizeArray();
      const FixedSpacetimeParameters fixed = fixed_spacetime_parameters(cfg);
      for (amrex::MFIter mfi(
              *evolution_fluid[level], amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto physical = primitive[level]->const_array(mfi);
         const auto output = evolution_fluid[level]->array(mfi);
         const auto stored_metric = stored_spacetime
            ? cell_spacetime[level]->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
         const int num_components = evolution_fluid[level]->nComp();
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (!relativistic) {
               if (magnetized) {
                  const MHDPrimitive value{
                     physical(i, j, k, RHO),
                     physical(i, j, k, MOM1),
                     physical(i, j, k, MOM2),
                     physical(i, j, k, MOM3),
                     physical(i, j, k, ENERGY),
                     physical(i, j, k, B1),
                     physical(i, j, k, B2),
                     physical(i, j, k, B3),
                  };
                  const MHDState conserved =
                     conserved_from_primitive<EOSMethod>(value, eos);
                  output(i, j, k, RHO) = conserved.rho;
                  output(i, j, k, MOM1) = conserved.mom1;
                  output(i, j, k, MOM2) = conserved.mom2;
                  output(i, j, k, MOM3) = conserved.mom3;
                  output(i, j, k, ENERGY) = conserved.energy;
                  output(i, j, k, B1) = conserved.b1;
                  output(i, j, k, B2) = conserved.b2;
                  output(i, j, k, B3) = conserved.b3;
               } else {
                  const HydroPrimitive value{
                     physical(i, j, k, RHO),
                     physical(i, j, k, MOM1),
                     physical(i, j, k, MOM2),
                     physical(i, j, k, MOM3),
                     physical(i, j, k, ENERGY),
                  };
                  const HydroState conserved =
                     conserved_from_primitive<EOSMethod>(value, eos);
                  output(i, j, k, RHO) = conserved.rho;
                  output(i, j, k, MOM1) = conserved.mom1;
                  output(i, j, k, MOM2) = conserved.mom2;
                  output(i, j, k, MOM3) = conserved.mom3;
                  output(i, j, k, ENERGY) = conserved.energy;
               }
            } else {
               const SpacetimeMetric metric = stored_spacetime
                  ? generated::z4c_spacetime_metric(
                       load_z4c_state(stored_metric, i, j, k)
                    )
                  : fixed_spacetime_geometry_at_cell<AMREX_SPACEDIM>(
                       fixed, lower, spacing, i, j, k, 0.0
                    ).metric;
               const amrex::Real density = physical(i, j, k, RHO);
               const amrex::Real pressure = physical(i, j, k, ENERGY);
               const amrex::Real specific_internal_energy =
                  eos_internal_energy_density_from_pressure<EOSMethod>(
                     density, pressure, eos
                  ) / density;
               if (magnetized) {
                  const RMHDPrimitive value{
                     density,
                     specific_internal_energy,
                     pressure,
                     physical(i, j, k, MOM1),
                     physical(i, j, k, MOM2),
                     physical(i, j, k, MOM3),
                     physical(i, j, k, B1),
                     physical(i, j, k, B2),
                     physical(i, j, k, B3),
                  };
                  store_rmhd_conserved(
                     output,
                     i,
                     j,
                     k,
                     generated::rmhd_primitive_to_conserved(value, metric)
                  );
               } else {
                  const RHDPrimitive value{
                     density,
                     specific_internal_energy,
                     pressure,
                     physical(i, j, k, MOM1),
                     physical(i, j, k, MOM2),
                     physical(i, j, k, MOM3),
                  };
                  store_rhd_conserved(
                     output,
                     i,
                     j,
                     k,
                     generated::rhd_primitive_to_conserved(value, metric)
                  );
               }
            }
            const int base_components = magnetized ? NVAR_MHD : NVAR_HYDRO;
            for (int component = base_components;
                 component < num_components; ++component) {
               output(i, j, k, component) = output(i, j, k, RHO)
                  * physical(i, j, k, component);
            }
         });
      }
   }
}

} // namespace

void materialize_physical_initial_data_fluid(
   RunConfig const& cfg,
   PhysicalSystem system,
   amrex::Vector<amrex::Geometry> const& geometry,
   ConstPhysicalIdField const& primitive,
   ConstPhysicalIdField const& cell_spacetime,
   MutablePhysicalIdField const& evolution_fluid
)
{
   if (geometry.size() != primitive.size()
       || geometry.size() != evolution_fluid.size()
       || (!cell_spacetime.empty()
           && geometry.size() != cell_spacetime.size())) {
      amrex::Abort("physical-ID fluid materialization hierarchy is inconsistent");
   }
   materialize_physical_initial_data_fluid_method<EOSRuntime>(
      cfg, system, geometry, primitive, cell_spacetime, evolution_fluid
   );
}

} // namespace pyfluid_amrex
