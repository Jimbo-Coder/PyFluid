#include <AMReX.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/initial_data/physical_id_interpolation.hpp"
#include "pyfluid_amrex/physical_system.hpp"

namespace {

using pyfluid_amrex::PhysicalIdChart;
using pyfluid_amrex::PhysicalIdRigidTransform;
using pyfluid_amrex::PortablePhysicalIdInterpolator;
using pyfluid_amrex::RuntimeDimensionHdf5Box;
using pyfluid_amrex::RuntimeDimensionHdf5Field;
using pyfluid_amrex::RuntimeDimensionHdf5Level;

void require_close(
   amrex::Real value,
   amrex::Real expected,
   amrex::Real tolerance,
   std::string const& label
)
{
   if (std::abs(value - expected) > tolerance) {
      throw std::runtime_error(
         label + " expected " + std::to_string(expected)
         + " but got " + std::to_string(value)
      );
   }
}

RuntimeDimensionHdf5Field constant_spherical_field(
   int num_components,
   std::vector<amrex::Real> const& component_values,
   bool node_centered
)
{
   RuntimeDimensionHdf5Field field;
   field.dimension = 2;
   field.num_components = num_components;
   field.levels.resize(1);
   RuntimeDimensionHdf5Level& level = field.levels.front();
   level.domain_lower = {0, 0, 0};
   level.domain_upper = {1, 1, 0};
   level.prob_lo = {1.0, 0.0, 0.0};
   level.prob_hi = {3.0, 0.5 * std::acos(-1.0), 0.0};
   level.boxes.resize(1);
   RuntimeDimensionHdf5Box& box = level.boxes.front();
   box.lower = level.domain_lower;
   box.upper = level.domain_upper;
   if (node_centered) {
      box.upper[0] += 1;
      box.upper[1] += 1;
   }
   const int point_count = node_centered ? 9 : 4;
   box.data.reserve(static_cast<std::size_t>(point_count * num_components));
   for (amrex::Real component_value : component_values) {
      for (int point = 0; point < point_count; ++point) {
         box.data.push_back(component_value);
      }
   }
   return field;
}

void check_equatorial_reflection_and_basis_rotation()
{
   const auto field_set = pyfluid_amrex::physical_fluid_grid_fields(
      pyfluid_amrex::PhysicalSystem::RHD, 0
   );
   PortablePhysicalIdInterpolator interpolator(
      constant_spherical_field(5, {5.0, 2.0, 0.0, 0.0, 7.0}, false),
      pyfluid_amrex::make_physical_id_chart(
         2,
         "spherical",
         "r theta",
         {},
         {},
         {0, 1, 0}
      ),
      field_set,
      false
   );
   const PhysicalIdChart target_chart = pyfluid_amrex::make_physical_id_chart(
      3, "cartesian", "x y z"
   );
   const amrex::Real root_two = std::sqrt(2.0);
   const auto sample = interpolator.sample(
      {root_two, 0.0, -root_two}, target_chart
   );
   if (sample.exterior) {
      throw std::runtime_error("reflected spherical point was marked exterior");
   }
   require_close(sample.values[0], 5.0, 1.0e-14, "reflected scalar");
   require_close(sample.values[1], root_two, 2.0e-14, "reflected velocity x");
   require_close(sample.values[2], 0.0, 2.0e-14, "reflected velocity y");
   require_close(sample.values[3], -root_two, 2.0e-14, "reflected velocity z");
   require_close(sample.values[4], 7.0, 1.0e-14, "reflected pressure");
   require_close(sample.source_cartesian[0], root_two, 2.0e-14, "source x");
   require_close(sample.source_cartesian[2], root_two, 2.0e-14, "source z");
}

void check_symmetric_tensor_rotation()
{
   const auto& field_set = pyfluid_amrex::adm_initial_data_grid_fields();
   std::vector<amrex::Real> values(16, 0.0);
   values[0] = 4.0;
   values[3] = 2.0;
   values[5] = 3.0;
   values[12] = 0.8;
   PortablePhysicalIdInterpolator interpolator(
      constant_spherical_field(16, values, true),
      pyfluid_amrex::make_physical_id_chart(
         2,
         "spherical",
         "r theta",
         {},
         {},
         {0, 1, 0}
      ),
      field_set,
      true
   );
   const amrex::Real root_two = std::sqrt(2.0);
   const auto sample = interpolator.sample(
      {root_two, 0.0, -root_two},
      pyfluid_amrex::make_physical_id_chart(3, "cartesian", "x y z")
   );
   for (int component : {0, 3, 5}) {
      require_close(sample.values[component], 3.0, 2.0e-14, "metric diagonal");
   }
   for (int component : {1, 4}) {
      require_close(sample.values[component], 0.0, 2.0e-14, "metric off diagonal");
   }
   require_close(sample.values[2], -1.0, 2.0e-14, "reflected metric xz");
   require_close(sample.values[12], 0.8, 1.0e-14, "lapse");
}

void check_axial_vector_reflection()
{
   const auto field_set = pyfluid_amrex::physical_fluid_grid_fields(
      pyfluid_amrex::PhysicalSystem::RMHD, 0
   );
   PortablePhysicalIdInterpolator interpolator(
      constant_spherical_field(
         8, {5.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 2.0}, false
      ),
      pyfluid_amrex::make_physical_id_chart(
         2,
         "spherical",
         "r theta",
         {},
         {},
         {0, 1, 0}
      ),
      field_set,
      false
   );
   const amrex::Real root_two = std::sqrt(2.0);
   const auto sample = interpolator.sample(
      {root_two, 0.0, -root_two},
      pyfluid_amrex::make_physical_id_chart(3, "cartesian", "x y z")
   );
   require_close(sample.values[5], 0.0, 2.0e-14, "reflected magnetic x");
   require_close(sample.values[6], -2.0, 2.0e-14, "reflected magnetic y");
   require_close(sample.values[7], 0.0, 2.0e-14, "reflected magnetic z");
}

void check_mapped_radial_origin_is_not_exterior()
{
   auto field = constant_spherical_field(
      5, {5.0, 0.0, 0.0, 0.0, 7.0}, false
   );
   field.levels.front().prob_lo[0] = 0.0;
   field.levels.front().prob_hi[0] = 4.0;
   pyfluid_amrex::RuntimePhysicalIdCoordinateFaces coordinates(1);
   coordinates.front()[0] = {0.0, 1.0, 4.0};
   coordinates.front()[1] = {0.0, 0.25 * std::acos(-1.0),
                             0.5 * std::acos(-1.0)};
   PortablePhysicalIdInterpolator interpolator(
      std::move(field),
      pyfluid_amrex::make_physical_id_chart(
         2,
         "spherical",
         "r theta",
         {},
         {},
         {0, 1, 0}
      ),
      pyfluid_amrex::physical_fluid_grid_fields(
         pyfluid_amrex::PhysicalSystem::RHD, 0
      ),
      false,
      std::move(coordinates)
   );
   const auto sample = interpolator.sample(
      {0.1, 0.0, 0.0},
      pyfluid_amrex::make_physical_id_chart(3, "cartesian", "x y z")
   );
   if (sample.exterior) {
      throw std::runtime_error(
         "regular mapped radial origin was marked exterior"
      );
   }
   require_close(sample.values[0], 5.0, 1.0e-14, "mapped-origin scalar");
}

void check_rigid_axis_and_center_materialization()
{
   const PhysicalIdRigidTransform transform =
      pyfluid_amrex::make_physical_id_rigid_transform(
         {0.0, 0.0, 0.0},
         {3.0, -2.0, 1.0},
         {0.0, 0.0, 1.0},
         {1.0, 0.0, 0.0}
      );
   if (pyfluid_amrex::physical_id_rigid_transform_is_identity(transform)) {
      throw std::runtime_error("nontrivial physical-ID transform is identity");
   }
   PortablePhysicalIdInterpolator fluid_interpolator(
      constant_spherical_field(5, {5.0, 2.0, 0.0, 0.0, 7.0}, true),
      pyfluid_amrex::make_physical_id_chart(2, "spherical", "r theta"),
      pyfluid_amrex::physical_fluid_grid_fields(
         pyfluid_amrex::PhysicalSystem::RHD, 0
      ),
      true,
      {},
      transform
   );
   const PhysicalIdChart target_chart = pyfluid_amrex::make_physical_id_chart(
      3, "cartesian", "x y z"
   );
   const auto fluid_sample = fluid_interpolator.sample(
      {5.0, -2.0, 1.0}, target_chart
   );
   require_close(fluid_sample.values[0], 5.0, 1.0e-14, "rotated density");
   require_close(fluid_sample.values[1], 2.0, 2.0e-14, "rotated velocity x");
   require_close(fluid_sample.values[2], 0.0, 2.0e-14, "rotated velocity y");
   require_close(fluid_sample.values[3], 0.0, 2.0e-14, "rotated velocity z");
   require_close(fluid_sample.source_cartesian[0], 5.0, 2.0e-14, "mapped x");
   require_close(fluid_sample.source_cartesian[1], -2.0, 2.0e-14, "mapped y");
   require_close(fluid_sample.source_cartesian[2], 1.0, 2.0e-14, "mapped z");

   std::vector<amrex::Real> adm_values(16, 0.0);
   adm_values[0] = 4.0;
   adm_values[3] = 2.0;
   adm_values[5] = 3.0;
   adm_values[12] = 0.8;
   adm_values[13] = 2.0;
   PortablePhysicalIdInterpolator adm_interpolator(
      constant_spherical_field(16, adm_values, true),
      pyfluid_amrex::make_physical_id_chart(2, "spherical", "r theta"),
      pyfluid_amrex::adm_initial_data_grid_fields(),
      true,
      {},
      transform
   );
   const auto adm_sample = adm_interpolator.sample(
      {5.0, -2.0, 1.0}, target_chart
   );
   require_close(adm_sample.values[0], 4.0, 2.0e-14, "rotated metric xx");
   require_close(adm_sample.values[3], 3.0, 2.0e-14, "rotated metric yy");
   require_close(adm_sample.values[5], 2.0, 2.0e-14, "rotated metric zz");
   require_close(adm_sample.values[13], 2.0, 2.0e-14, "rotated shift x");
   require_close(adm_sample.values[14], 0.0, 2.0e-14, "rotated shift y");
   require_close(adm_sample.values[15], 0.0, 2.0e-14, "rotated shift z");

   const PhysicalIdRigidTransform half_turn =
      pyfluid_amrex::make_physical_id_rigid_transform(
         {}, {}, {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0}
      );
   const auto& rotation = half_turn.source_to_target;
   require_close(rotation[2][2], -1.0, 2.0e-14, "half-turn axis map");
}

} // namespace

int main(int argc, char** argv)
{
   amrex::Initialize(argc, argv);
   int result = 0;
   try {
      check_equatorial_reflection_and_basis_rotation();
      check_symmetric_tensor_rotation();
      check_axial_vector_reflection();
      check_mapped_radial_origin_is_not_exterior();
      check_rigid_axis_and_center_materialization();
      std::cout << "PASS portable physical-ID interpolation check\n";
   } catch (std::exception const& error) {
      std::cerr << "FAIL portable physical-ID interpolation check: "
                << error.what() << '\n';
      result = 1;
   }
   amrex::Finalize();
   return result;
}
