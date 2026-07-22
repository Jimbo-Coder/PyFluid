#include "pyfluid_amrex/initial_data/physical_id.hpp"

#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/coordinate_locations.hpp"
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

struct StoredPhysicalHierarchy {
   int coordinate_kind = MeshCartesian;
   MeshAxes axes{};
   amrex::Vector<amrex::Geometry> geometry;
   amrex::Vector<amrex::BoxArray> grids;
   amrex::Vector<amrex::DistributionMapping> distribution;
};

std::array<int, AMREX_SPACEDIM> parse_periodicity(
   std::string const& value,
   std::string const& path
)
{
   std::array<int, AMREX_SPACEDIM> periodic{};
   std::istringstream input(value);
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      std::string item;
      if (!std::getline(input, item, ',')) {
         amrex::Abort("invalid physical-ID periodicity metadata in " + path);
      }
      try {
         periodic[direction] = std::stoi(item);
      } catch (std::exception const&) {
         amrex::Abort("invalid physical-ID periodicity metadata in " + path);
      }
      if (periodic[direction] != 0 && periodic[direction] != 1) {
         amrex::Abort("invalid physical-ID periodicity metadata in " + path);
      }
   }
   return periodic;
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
   const auto periodic = parse_periodicity(
      required_metadata_value(metadata, "periodicity", path), path
   );
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
   amrex::Real source_radius = 0.0;
   amrex::Real target_radius = 0.0;
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
      const amrex::Real offset = centering.nodeCentered(direction) ? 0.0 : 0.5;
      const amrex::Real safe_lo = base.ProbLo(direction) + offset * spacing[direction];
      const amrex::Real safe_hi = base.ProbHi(direction) - offset * spacing[direction];
      mapped.exterior = mapped.exterior
         || target_mesh[component] < base.ProbLo(direction)
         || target_mesh[component] > base.ProbHi(direction);
      target_mesh[component] = amrex::max(
         safe_lo, amrex::min(safe_hi, target_mesh[component])
      );
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
   AsymptoticallyFlatAdm,
};

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

void interpolate_stored_field(
   std::vector<std::unique_ptr<amrex::MultiFab>> const& source_field,
   StoredPhysicalHierarchy const& source,
   amrex::Vector<amrex::Geometry> const& target_geometry,
   MutablePhysicalIdField const& target_field,
   amrex::IndexType centering,
   amrex::GpuArray<amrex::Real, 3> const& origin,
   PhysicalFieldExterior exterior,
   bool stationary_rotating
)
{
   if (target_geometry.size() != target_field.size()) {
      amrex::Abort("physical-ID target hierarchy is inconsistent");
   }
   const int num_components = target_field.front()->nComp();
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
            centering
         );
         std::vector<amrex::Real> values = plan.interpolate(
            source_field, 0, num_components
         );
         if (exterior == PhysicalFieldExterior::AsymptoticallyFlatAdm) {
            for (std::size_t point = 0; point < mapped_points.size(); ++point) {
               apply_adm_exterior(
                  values.data() + point * num_components,
                  num_components,
                  mapped_points[point],
                  stationary_rotating
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
   amrex::Vector<amrex::Geometry> const& geometry,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   MutablePhysicalIdField const& primitive,
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
   validate_level_fields(
      ConstPhysicalIdField(primitive.begin(), primitive.end()),
      levels,
      physical_fluid_components,
      PhysicalFluidField
   );
   const auto metadata = read_metadata(path);
   require_metadata(metadata, "type", "physical_initial_data", path);
   require_metadata(metadata, "schema_version", std::to_string(PhysicalIdSchemaVersion), path);
   require_metadata(metadata, "format", "hdf5", path);
   require_metadata(metadata, "amrex_spacedim", std::to_string(AMREX_SPACEDIM), path);
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
   require_metadata(metadata, "periodicity", periodicity_string(geometry.front()), path);
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
   result.rotation_center = {
      cfg.initial_data_rotation_center1,
      cfg.initial_data_rotation_center2,
      cfg.initial_data_rotation_center3,
   };
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
   const MultilevelHdf5FieldLayout stored_layout =
      read_multilevel_hdf5_field_layout(path, PhysicalFluidField);
   bool exact_boxes = static_cast<int>(stored_layout.boxes.size()) == levels;
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
   const bool exact_layout = exact_boxes
      &&
      required_metadata_value(metadata, "levels", path)
         == std::to_string(levels)
      && required_metadata_value(metadata, "domain", path)
         == domain_string(geometry.front())
      && required_metadata_value(metadata, "prob_lo", path)
         == real_vector_string(geometry.front(), false)
      && required_metadata_value(metadata, "prob_hi", path)
         == real_vector_string(geometry.front(), true)
      && required_metadata_value(metadata, "refinement_ratios", path)
         == refinement_string(refinement_ratios);
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
   const amrex::GpuArray<amrex::Real, 3> origin = result.rotation_center;
   interpolate_stored_field(
      source_fluid,
      source,
      geometry,
      primitive,
      amrex::IndexType::TheCellType(),
      origin,
      PhysicalFieldExterior::Constant,
      false
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
         origin,
         PhysicalFieldExterior::AsymptoticallyFlatAdm,
         result.source_ic == "tov" || result.source_ic == "rotating_star"
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
            origin,
            PhysicalFieldExterior::Constant,
            false
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
   switch (eos_method_code(cfg)) {
   case EOSPolytrope:
      materialize_physical_initial_data_fluid_method<EOSPolytrope>(
         cfg, system, geometry, primitive, cell_spacetime, evolution_fluid
      );
      return;
   case EOSHybridPolytrope:
      materialize_physical_initial_data_fluid_method<EOSHybridPolytrope>(
         cfg, system, geometry, primitive, cell_spacetime, evolution_fluid
      );
      return;
   default:
      materialize_physical_initial_data_fluid_method<EOSIdealGas>(
         cfg, system, geometry, primitive, cell_spacetime, evolution_fluid
      );
   }
}

} // namespace pyfluid_amrex
