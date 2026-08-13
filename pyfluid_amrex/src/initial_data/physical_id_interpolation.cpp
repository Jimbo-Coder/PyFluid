#include "pyfluid_amrex/initial_data/physical_id_interpolation.hpp"

#include <AMReX.H>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/coordinate_locations.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"

namespace pyfluid_amrex {
namespace {

using Vector = std::array<amrex::Real, 3>;
using Basis = std::array<Vector, 3>;
using Tensor = std::array<std::array<amrex::Real, 3>, 3>;

amrex::GpuArray<amrex::Real, 3> gpu_vector(Vector const& value)
{
   return {value[0], value[1], value[2]};
}

Vector host_vector(amrex::GpuArray<amrex::Real, 3> const& value)
{
   return {value[0], value[1], value[2]};
}

std::vector<std::string> chart_axis_tokens(
   int coordinate_kind,
   int dimension,
   std::string axes
)
{
   std::replace(axes.begin(), axes.end(), ',', ' ');
   std::istringstream stream(axes);
   std::vector<std::string> tokens;
   for (std::string token; stream >> token;) {
      std::transform(
         token.begin(), token.end(), token.begin(), [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
         }
      );
      tokens.push_back(std::move(token));
   }
   if (tokens.empty()) {
      const auto canonical = mesh_canonical_vector_axes(coordinate_kind);
      for (int direction = 0; direction < dimension; ++direction) {
         const int axis = canonical[direction];
         if (axis == MeshAxisX) {
            tokens.emplace_back("x");
         } else if (axis == MeshAxisY) {
            tokens.emplace_back("y");
         } else if (axis == MeshAxisZ) {
            tokens.emplace_back("z");
         } else if (axis == MeshAxisR) {
            tokens.emplace_back("r");
         } else if (axis == MeshAxisTheta) {
            tokens.emplace_back("theta");
         } else if (axis == MeshAxisPhi) {
            tokens.emplace_back("phi");
         }
      }
   }
   return tokens;
}

int chart_direction_for_axis(PhysicalIdChart const& chart, int axis)
{
   for (int direction = 0; direction < chart.dimension; ++direction) {
      if (chart.axes[direction] == axis) {
         return direction;
      }
   }
   return -1;
}

int chart_component_axis(PhysicalIdChart const& chart, int component)
{
   if (component < chart.dimension) {
      return chart.axes[component];
   }
   const auto canonical = mesh_canonical_vector_axes(chart.coordinate_kind);
   int missing_component = chart.dimension;
   for (int index = 0; index < 3; ++index) {
      if (chart_direction_for_axis(chart, canonical[index]) >= 0) {
         continue;
      }
      if (missing_component == component) {
         return canonical[index];
      }
      ++missing_component;
   }
   amrex::Abort("physical-ID chart cannot map vector component");
   return MeshAxisInvalid;
}

int canonical_coordinate_index(int coordinate_kind, int axis)
{
   const auto canonical = mesh_canonical_vector_axes(coordinate_kind);
   for (int index = 0; index < 3; ++index) {
      if (canonical[index] == axis) {
         return index;
      }
   }
   amrex::Abort("physical-ID chart contains a noncanonical axis");
   return 0;
}

Basis canonical_basis(
   int coordinate_kind,
   std::array<amrex::Real, 3> const& q
)
{
   if (coordinate_kind == MeshCylindrical) {
      const amrex::Real cosine = std::cos(q[1]);
      const amrex::Real sine = std::sin(q[1]);
      return {{{cosine, sine, 0.0}, {-sine, cosine, 0.0}, {0.0, 0.0, 1.0}}};
   }
   if (coordinate_kind == MeshSpherical) {
      const amrex::Real sin_theta = std::sin(q[1]);
      const amrex::Real cos_theta = std::cos(q[1]);
      const amrex::Real sin_phi = std::sin(q[2]);
      const amrex::Real cos_phi = std::cos(q[2]);
      return {{
         {sin_theta * cos_phi, sin_theta * sin_phi, cos_theta},
         {cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta},
         {-sin_phi, cos_phi, 0.0},
      }};
   }
   return {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
}

Basis component_basis(
   PhysicalIdChart const& chart,
   std::array<amrex::Real, 3> const& q
)
{
   const Basis canonical = canonical_basis(chart.coordinate_kind, q);
   Basis result{};
   for (int component = 0; component < 3; ++component) {
      result[component] = canonical[
         canonical_coordinate_index(
            chart.coordinate_kind, chart_component_axis(chart, component)
         )
      ];
   }
   return result;
}

amrex::Real dot(Vector const& left, Vector const& right)
{
   return left[0] * right[0] + left[1] * right[1]
      + left[2] * right[2];
}

Vector basis_vector_to_cartesian(Basis const& basis, Vector const& value)
{
   Vector result{};
   for (int component = 0; component < 3; ++component) {
      for (int direction = 0; direction < 3; ++direction) {
         result[direction] += value[component] * basis[component][direction];
      }
   }
   return result;
}

Vector cartesian_vector_to_basis(Basis const& basis, Vector const& value)
{
   return {dot(basis[0], value), dot(basis[1], value), dot(basis[2], value)};
}

Tensor basis_tensor_to_cartesian(Basis const& basis, Tensor const& value)
{
   Tensor result{};
   for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
         for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
               result[i][j] += basis[a][i] * value[a][b] * basis[b][j];
            }
         }
      }
   }
   return result;
}

Tensor cartesian_tensor_to_basis(Basis const& basis, Tensor const& value)
{
   Tensor result{};
   for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
         for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
               result[a][b] += basis[a][i] * value[i][j] * basis[b][j];
            }
         }
      }
   }
   return result;
}

bool axial_vector(GridFieldInfo const& field)
{
   if (field.components.empty()) {
      return false;
   }
   const GridFieldSymmetryKind symmetry = field.components.front().symmetry;
   return symmetry == GridFieldSymmetryKind::AxialVector1
      || symmetry == GridFieldSymmetryKind::AxialVector2
      || symmetry == GridFieldSymmetryKind::AxialVector3;
}

void reflect_vector(Vector& value, int reflection_mask, bool axial)
{
   for (int direction = 0; direction < 3; ++direction) {
      if ((reflection_mask & (1 << direction)) == 0) {
         continue;
      }
      if (axial) {
         for (int component = 0; component < 3; ++component) {
            if (component != direction) {
               value[component] = -value[component];
            }
         }
      } else {
         value[direction] = -value[direction];
      }
   }
}

void reflect_tensor(Tensor& value, int reflection_mask)
{
   for (int direction = 0; direction < 3; ++direction) {
      if ((reflection_mask & (1 << direction)) == 0) {
         continue;
      }
      for (int i = 0; i < 3; ++i) {
         for (int j = 0; j < 3; ++j) {
            if ((i == direction) != (j == direction)) {
               value[i][j] = -value[i][j];
            }
         }
      }
   }
}

int physical_reflection_direction(
   PhysicalIdChart const& chart,
   RuntimeDimensionHdf5Level const& level,
   int source_direction,
   bool upper
)
{
   const int axis = chart.axes[source_direction];
   if (chart.coordinate_kind == MeshCartesian) {
      return canonical_coordinate_index(chart.coordinate_kind, axis);
   }
   if (axis == MeshAxisZ) {
      return 2;
   }
   if (chart.coordinate_kind == MeshSpherical && axis == MeshAxisTheta) {
      const amrex::Real boundary = upper
         ? level.prob_hi[source_direction]
         : level.prob_lo[source_direction];
      const amrex::Real tolerance = 256.0
         * std::numeric_limits<amrex::Real>::epsilon();
      if (std::abs(boundary - 0.5 * constants::pi) <= tolerance) {
         return 2;
      }
   }
   amrex::Abort(
      "portable physical-ID reflection is not a Cartesian-plane symmetry"
   );
   return 0;
}

std::size_t box_point_count(
   RuntimeDimensionHdf5Box const& box,
   int dimension
)
{
   std::size_t count = 1;
   for (int direction = 0; direction < dimension; ++direction) {
      count *= static_cast<std::size_t>(
         box.upper[direction] - box.lower[direction] + 1
      );
   }
   return count;
}

bool box_contains(
   RuntimeDimensionHdf5Box const& box,
   int dimension,
   std::array<int, 3> const& index
)
{
   for (int direction = 0; direction < dimension; ++direction) {
      if (index[direction] < box.lower[direction]
          || index[direction] > box.upper[direction]) {
         return false;
      }
   }
   return true;
}

std::size_t box_linear_offset(
   RuntimeDimensionHdf5Box const& box,
   int dimension,
   std::array<int, 3> const& index
)
{
   std::size_t offset = static_cast<std::size_t>(
      index[0] - box.lower[0]
   );
   std::size_t stride = static_cast<std::size_t>(
      box.upper[0] - box.lower[0] + 1
   );
   for (int direction = 1; direction < dimension; ++direction) {
      offset += stride * static_cast<std::size_t>(
         index[direction] - box.lower[direction]
      );
      stride *= static_cast<std::size_t>(
         box.upper[direction] - box.lower[direction] + 1
      );
   }
   return offset;
}

bool field_value(
   RuntimeDimensionHdf5Field const& field,
   int level,
   std::array<int, 3> const& index,
   int component,
   amrex::Real& value
)
{
   for (RuntimeDimensionHdf5Box const& box : field.levels[level].boxes) {
      if (!box_contains(box, field.dimension, index)) {
         continue;
      }
      const std::size_t offset = box_linear_offset(
         box, field.dimension, index
      );
      value = box.data[
         offset + static_cast<std::size_t>(component)
            * box_point_count(box, field.dimension)
      ];
      return true;
   }
   return false;
}

void coalesce_complete_level(
   RuntimeDimensionHdf5Field& field,
   RuntimeDimensionHdf5Level& level,
   bool node_centered
)
{
   RuntimeDimensionHdf5Box dense;
   dense.lower = level.domain_lower;
   dense.upper = level.domain_upper;
   for (int direction = 0; direction < field.dimension; ++direction) {
      dense.upper[direction] += node_centered ? 1 : 0;
   }
   const std::size_t dense_points = box_point_count(dense, field.dimension);
   dense.data.assign(
      dense_points * static_cast<std::size_t>(field.num_components), 0.0
   );
   std::vector<unsigned char> covered(dense_points, 0);
   for (RuntimeDimensionHdf5Box const& box : level.boxes) {
      const std::size_t box_points = box_point_count(box, field.dimension);
      for (std::size_t point = 0; point < box_points; ++point) {
         std::size_t remainder = point;
         std::array<int, 3> index = box.lower;
         for (int direction = 0; direction < field.dimension; ++direction) {
            const std::size_t extent = static_cast<std::size_t>(
               box.upper[direction] - box.lower[direction] + 1
            );
            index[direction] += static_cast<int>(remainder % extent);
            remainder /= extent;
         }
         if (!box_contains(dense, field.dimension, index)) {
            amrex::Abort("physical-ID source box lies outside its level domain");
         }
         const std::size_t dense_point = box_linear_offset(
            dense, field.dimension, index
         );
         covered[dense_point] = 1;
         for (int component = 0; component < field.num_components; ++component) {
            dense.data[
               dense_point + static_cast<std::size_t>(component) * dense_points
            ] = box.data[
               point + static_cast<std::size_t>(component) * box_points
            ];
         }
      }
   }
   if (std::all_of(covered.begin(), covered.end(), [](unsigned char value) {
          return value != 0;
       })) {
      level.boxes.clear();
      level.boxes.push_back(std::move(dense));
   }
}

struct MappedSourcePoint {
   std::array<amrex::Real, 3> q{};
   std::array<amrex::Real, 3> cartesian{};
   int reflection_mask = 0;
   bool exterior = false;
};

MappedSourcePoint map_source_point(
   std::array<amrex::Real, 3> const& target_cartesian,
   PhysicalIdChart const& chart,
   RuntimeDimensionHdf5Level const& level,
   bool node_centered
)
{
   MappedSourcePoint mapped;
   mapped.q = host_vector(cartesian_to_mesh_coordinate_triplet(
      chart.coordinate_kind, gpu_vector(target_cartesian)
   ));
   for (int direction = 0; direction < chart.dimension; ++direction) {
      const int coordinate_component = canonical_coordinate_index(
         chart.coordinate_kind, chart.axes[direction]
      );
      amrex::Real& coordinate = mapped.q[coordinate_component];
      const amrex::Real lower = level.prob_lo[direction];
      const amrex::Real upper = level.prob_hi[direction];
      if (chart.periodic[direction] != 0) {
         const amrex::Real length = upper - lower;
         coordinate = lower + std::fmod(
            std::fmod(coordinate - lower, length) + length, length
         );
      } else {
         for (int attempt = 0; attempt < 4; ++attempt) {
            if (coordinate < lower && chart.reflection_lo[direction] != 0) {
               coordinate = 2.0 * lower - coordinate;
               mapped.reflection_mask ^= 1 << physical_reflection_direction(
                  chart, level, direction, false
               );
            } else if (
               coordinate > upper && chart.reflection_hi[direction] != 0
            ) {
               coordinate = 2.0 * upper - coordinate;
               mapped.reflection_mask ^= 1 << physical_reflection_direction(
                  chart, level, direction, true
               );
            } else {
               break;
            }
         }
      }
      const int cell_count = level.domain_upper[direction]
         - level.domain_lower[direction] + 1;
      const auto& faces = level.coordinate_faces[direction];
      const amrex::Real spacing = (upper - lower) / cell_count;
      const amrex::Real safe_lower = faces.empty()
         ? lower + (node_centered ? 0.0 : 0.5) * spacing
         : (node_centered ? faces.front()
                          : 0.5 * (faces[0] + faces[1]));
      const amrex::Real safe_upper = faces.empty()
         ? upper - (node_centered ? 0.0 : 0.5) * spacing
         : (node_centered ? faces.back()
                          : 0.5 * (faces[faces.size() - 2] + faces.back()));
      const amrex::Real scale = std::max(
         amrex::Real(1.0), std::max(std::abs(lower), std::abs(upper))
      );
      const bool regular_radial_origin =
         chart.axes[direction] == MeshAxisR
         && std::abs(lower) <= 64.0
            * std::numeric_limits<amrex::Real>::epsilon() * scale
         && coordinate >= lower;
      const bool regular_lower_boundary = coordinate >= lower
         && (regular_radial_origin || chart.reflection_lo[direction] != 0);
      const bool regular_upper_boundary = coordinate <= upper
         && chart.reflection_hi[direction] != 0;
      // Cell-centered radial data do not contain a sample at r=0, but the
      // origin is a regular boundary of the stellar chart rather than the
      // physical exterior.  Clamp that half-cell interpolation strip to the
      // innermost value without applying an atmosphere/exterior policy.  This
      // is essential when materializing onto a finer target grid.
      const bool outside_interpolation_hull =
         (coordinate < safe_lower && !regular_lower_boundary)
         || (coordinate > safe_upper && !regular_upper_boundary);
      if (outside_interpolation_hull) {
         mapped.exterior = true;
      }
      coordinate = std::max(safe_lower, std::min(safe_upper, coordinate));
   }
   mapped.cartesian = host_vector(mesh_coordinate_triplet_to_cartesian(
      chart.coordinate_kind, gpu_vector(mapped.q)
   ));
   return mapped;
}

bool interpolate_level(
   RuntimeDimensionHdf5Field const& field,
   int level_index,
   PhysicalIdChart const& chart,
   std::array<amrex::Real, 3> const& q,
   bool node_centered,
   std::vector<amrex::Real>& output
)
{
   RuntimeDimensionHdf5Level const& level = field.levels[level_index];
   std::array<int, 3> lower_index{};
   std::array<amrex::Real, 3> fraction{};
   for (int direction = 0; direction < field.dimension; ++direction) {
      const int coordinate_component = canonical_coordinate_index(
         chart.coordinate_kind, chart.axes[direction]
      );
      const int cell_count = level.domain_upper[direction]
         - level.domain_lower[direction] + 1;
      const int valid_lower = level.domain_lower[direction];
      const int valid_upper = level.domain_upper[direction]
         + (node_centered ? 1 : 0);
      const auto& faces = level.coordinate_faces[direction];
      amrex::Real logical = 0.0;
      if (faces.empty()) {
         const amrex::Real spacing = (
            level.prob_hi[direction] - level.prob_lo[direction]
         ) / cell_count;
         const amrex::Real offset = node_centered ? 0.0 : 0.5;
         logical = level.domain_lower[direction]
            + (q[coordinate_component] - level.prob_lo[direction]) / spacing
            - offset;
      } else {
         auto coordinate_at = [&](int local_index) {
            return node_centered
               ? faces[local_index]
               : 0.5 * (faces[local_index] + faces[local_index + 1]);
         };
         const int coordinate_count = cell_count + (node_centered ? 1 : 0);
         if (q[coordinate_component] <= coordinate_at(0)) {
            logical = valid_lower;
         } else if (
            q[coordinate_component] >= coordinate_at(coordinate_count - 1)
         ) {
            logical = valid_upper;
         } else {
            int lower_coordinate = 0;
            int upper_coordinate = coordinate_count - 1;
            while (upper_coordinate - lower_coordinate > 1) {
               const int middle = lower_coordinate
                  + (upper_coordinate - lower_coordinate) / 2;
               if (coordinate_at(middle) <= q[coordinate_component]) {
                  lower_coordinate = middle;
               } else {
                  upper_coordinate = middle;
               }
            }
            const amrex::Real left = coordinate_at(lower_coordinate);
            const amrex::Real right = coordinate_at(upper_coordinate);
            logical = valid_lower + lower_coordinate
               + (q[coordinate_component] - left) / (right - left);
         }
      }
      if (valid_lower == valid_upper) {
         lower_index[direction] = valid_lower;
         fraction[direction] = 0.0;
      } else if (logical <= valid_lower) {
         lower_index[direction] = valid_lower;
         fraction[direction] = 0.0;
      } else if (logical >= valid_upper) {
         lower_index[direction] = valid_upper - 1;
         fraction[direction] = 1.0;
      } else {
         lower_index[direction] = static_cast<int>(std::floor(logical));
         fraction[direction] = logical - lower_index[direction];
      }
   }

   const int corners = 1 << field.dimension;
   std::vector<amrex::Real> candidate(
      static_cast<std::size_t>(field.num_components), 0.0
   );
   for (int corner = 0; corner < corners; ++corner) {
      std::array<int, 3> index = lower_index;
      amrex::Real weight = 1.0;
      for (int direction = 0; direction < field.dimension; ++direction) {
         if ((corner & (1 << direction)) != 0) {
            ++index[direction];
            weight *= fraction[direction];
         } else {
            weight *= 1.0 - fraction[direction];
         }
      }
      if (weight == 0.0) {
         continue;
      }
      for (int component = 0; component < field.num_components; ++component) {
         amrex::Real value = 0.0;
         if (!field_value(field, level_index, index, component, value)) {
            return false;
         }
         candidate[component] += weight * value;
      }
   }
   output = std::move(candidate);
   return true;
}

void transform_field_values(
   std::vector<amrex::Real>& values,
   GridFieldSetInfo const& field_set,
   PhysicalIdChart const& source_chart,
   std::array<amrex::Real, 3> const& source_q,
   PhysicalIdChart const& target_chart,
   std::array<amrex::Real, 3> const& target_q,
   int reflection_mask
)
{
   const Basis source_basis = component_basis(source_chart, source_q);
   const Basis target_basis = component_basis(target_chart, target_q);
   int offset = 0;
   for (GridFieldInfo const& field : field_set.fields) {
      if (field.value_kind == GridFieldValueKind::Scalar) {
         ++offset;
         continue;
      }
      if (field.value_kind == GridFieldValueKind::Vector3) {
         if (field.components.size() != 3) {
            amrex::Abort("physical-ID Vector3 field does not have three components");
         }
         const Vector source_value{
            values[offset], values[offset + 1], values[offset + 2]
         };
         Vector cartesian = basis_vector_to_cartesian(
            source_basis, source_value
         );
         reflect_vector(cartesian, reflection_mask, axial_vector(field));
         const Vector target_value = cartesian_vector_to_basis(
            target_basis, cartesian
         );
         for (int component = 0; component < 3; ++component) {
            values[offset + component] = target_value[component];
         }
         offset += 3;
         continue;
      }
      if (field.value_kind == GridFieldValueKind::SymmetricTensor3) {
         if (field.components.size() != 6) {
            amrex::Abort(
               "physical-ID SymmetricTensor3 field does not have six components"
            );
         }
         Tensor source_value{};
         source_value[0][0] = values[offset];
         source_value[0][1] = source_value[1][0] = values[offset + 1];
         source_value[0][2] = source_value[2][0] = values[offset + 2];
         source_value[1][1] = values[offset + 3];
         source_value[1][2] = source_value[2][1] = values[offset + 4];
         source_value[2][2] = values[offset + 5];
         Tensor cartesian = basis_tensor_to_cartesian(
            source_basis, source_value
         );
         reflect_tensor(cartesian, reflection_mask);
         const Tensor target_value = cartesian_tensor_to_basis(
            target_basis, cartesian
         );
         values[offset] = target_value[0][0];
         values[offset + 1] = target_value[0][1];
         values[offset + 2] = target_value[0][2];
         values[offset + 3] = target_value[1][1];
         values[offset + 4] = target_value[1][2];
         values[offset + 5] = target_value[2][2];
         offset += 6;
         continue;
      }
      amrex::Abort(
         "portable physical-ID interpolation supports scalar, Vector3, and "
         "SymmetricTensor3 fields"
      );
   }
   if (offset != static_cast<int>(values.size())) {
      amrex::Abort("physical-ID field catalog/component count is inconsistent");
   }
}

} // namespace

PhysicalIdChart make_physical_id_chart(
   int dimension,
   std::string const& coordinate,
   std::string const& axes,
   std::array<int, 3> const& periodic,
   std::array<int, 3> const& reflection_lo,
   std::array<int, 3> const& reflection_hi
)
{
   if (dimension < 1 || dimension > 3) {
      amrex::Abort("physical-ID chart dimension must be between one and three");
   }
   PhysicalIdChart chart;
   chart.dimension = dimension;
   chart.coordinate_kind = mesh_coordinate_code(coordinate);
   chart.axes.fill(MeshAxisInvalid);
   const std::vector<std::string> tokens = chart_axis_tokens(
      chart.coordinate_kind, dimension, axes
   );
   if (static_cast<int>(tokens.size()) != dimension) {
      amrex::Abort(
         "physical-ID chart axes must list exactly the source dimension"
      );
   }
   int previous_canonical = -1;
   for (int direction = 0; direction < dimension; ++direction) {
      chart.axes[direction] = mesh_axis_code(tokens[direction]);
      const int canonical = canonical_coordinate_index(
         chart.coordinate_kind, chart.axes[direction]
      );
      if (canonical <= previous_canonical) {
         amrex::Abort(
            "physical-ID chart axes must be an ordered canonical subset"
         );
      }
      previous_canonical = canonical;
      if ((periodic[direction] != 0 && periodic[direction] != 1)
          || (reflection_lo[direction] != 0 && reflection_lo[direction] != 1)
          || (reflection_hi[direction] != 0 && reflection_hi[direction] != 1)) {
         amrex::Abort("physical-ID chart flags must be zero or one");
      }
   }
   chart.periodic = periodic;
   chart.reflection_lo = reflection_lo;
   chart.reflection_hi = reflection_hi;
   return chart;
}

PhysicalIdChart current_physical_id_chart(
   std::string const& coordinate,
   std::string const& axes
)
{
   return make_physical_id_chart(AMREX_SPACEDIM, coordinate, axes);
}

PortablePhysicalIdInterpolator::PortablePhysicalIdInterpolator(
   RuntimeDimensionHdf5Field field,
   PhysicalIdChart source_chart,
   GridFieldSetInfo field_set,
   bool node_centered,
   RuntimePhysicalIdCoordinateFaces coordinate_faces
)
   : field_(std::move(field)),
     source_chart_(std::move(source_chart)),
     field_set_(std::move(field_set)),
     node_centered_(node_centered)
{
   if (field_.dimension != source_chart_.dimension
       || field_.num_components != grid_field_component_count(field_set_)) {
      amrex::Abort(
         "portable physical-ID field, chart, and catalog are inconsistent"
      );
   }
   if (!coordinate_faces.empty()
       && coordinate_faces.size() != field_.levels.size()) {
      amrex::Abort("portable physical-ID coordinate hierarchy is inconsistent");
   }
   for (std::size_t level_index = 0;
        level_index < field_.levels.size(); ++level_index) {
      RuntimeDimensionHdf5Level& level = field_.levels[level_index];
      if (!coordinate_faces.empty()) {
         level.coordinate_faces = std::move(coordinate_faces[level_index]);
      }
      for (int direction = 0; direction < field_.dimension; ++direction) {
         auto const& faces = level.coordinate_faces[direction];
         const int cells = level.domain_upper[direction]
            - level.domain_lower[direction] + 1;
         if (!faces.empty()
             && static_cast<int>(faces.size()) != cells + 1) {
            amrex::Abort(
               "portable physical-ID coordinate face count is inconsistent"
            );
         }
      }
      coalesce_complete_level(field_, level, node_centered_);
   }
}

PortablePhysicalIdSample PortablePhysicalIdInterpolator::sample(
   std::array<amrex::Real, 3> const& target_cartesian,
   PhysicalIdChart const& target_chart
) const
{
   if (field_.levels.empty()) {
      amrex::Abort("portable physical-ID field has no levels");
   }
   const MappedSourcePoint mapped = map_source_point(
      target_cartesian, source_chart_, field_.levels.front(), node_centered_
   );
   std::vector<amrex::Real> values;
   bool found = false;
   for (int level = static_cast<int>(field_.levels.size()) - 1;
        level >= 0 && !found; --level) {
      found = interpolate_level(
         field_, level, source_chart_, mapped.q, node_centered_, values
      );
   }
   if (!found) {
      amrex::Abort(
         "portable physical-ID hierarchy does not cover an interpolation stencil"
      );
   }
   const std::array<amrex::Real, 3> target_q = host_vector(
      cartesian_to_mesh_coordinate_triplet(
         target_chart.coordinate_kind, gpu_vector(target_cartesian)
      )
   );
   transform_field_values(
      values,
      field_set_,
      source_chart_,
      mapped.q,
      target_chart,
      target_q,
      mapped.reflection_mask
   );
   return {std::move(values), mapped.cartesian, mapped.exterior};
}

} // namespace pyfluid_amrex
