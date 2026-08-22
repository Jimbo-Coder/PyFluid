#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/coordinate_locations.hpp"
#include "pyfluid_amrex/hierarchy_interpolation.hpp"
#include "pyfluid_amrex/multipoles.hpp"
#include "pyfluid_amrex/surface_geometry.hpp"

namespace {

using pyfluid_amrex::HierarchyInterpolationPlan;
using pyfluid_amrex::ParametricSurface;

void require_close(amrex::Real value, amrex::Real expected, amrex::Real tolerance)
{
   if (std::abs(value - expected) > tolerance) {
      throw std::runtime_error(
         "surface check mismatch: value=" + std::to_string(value)
         + " expected=" + std::to_string(expected)
      );
   }
}

amrex::Geometry make_geometry(
   int cells,
   amrex::GpuArray<int, 3> periodic,
   amrex::GpuArray<amrex::Real, 3> lower = {-1.0, -1.0, -1.0},
   amrex::GpuArray<amrex::Real, 3> upper = {1.0, 1.0, 1.0}
)
{
   const amrex::Box domain(
      amrex::IntVect(0, 0, 0), amrex::IntVect(cells - 1, cells - 1, cells - 1)
   );
   const amrex::RealBox physical_domain(lower.data(), upper.data());
   return amrex::Geometry(domain, &physical_domain, 0, periodic.data());
}

std::unique_ptr<amrex::MultiFab> linear_coordinate_field(
   amrex::Geometry const& geometry,
   amrex::BoxArray const& cell_grids,
   amrex::DistributionMapping const& distribution,
   amrex::IndexType centering,
   int ghost_cells = 1
)
{
   auto field = std::make_unique<amrex::MultiFab>(
      amrex::convert(cell_grids, centering), distribution, 1, ghost_cells
   );
   field->setVal(0.0);
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   const amrex::Box centered_domain = amrex::convert(geometry.Domain(), centering);
   for (amrex::MFIter mfi(*field, false); mfi.isValid(); ++mfi) {
      const auto output = field->array(mfi);
      const amrex::Box box = mfi.validbox();
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real q1 = lower[0]
            + (i - centered_domain.smallEnd(0) + (centering.nodeCentered(0) ? 0.0 : 0.5))
               * spacing[0];
         const amrex::Real q2 = lower[1]
            + (j - centered_domain.smallEnd(1) + (centering.nodeCentered(1) ? 0.0 : 0.5))
               * spacing[1];
         const amrex::Real q3 = lower[2]
            + (k - centered_domain.smallEnd(2) + (centering.nodeCentered(2) ? 0.0 : 0.5))
               * spacing[2];
         output(i, j, k, 0) = 1.0 + 2.0 * q1 - 3.0 * q2 + 0.5 * q3;
      });
   }
   field->FillBoundary(geometry.periodicity());
   return field;
}

void check_surface_geometry()
{
   const auto surface = ParametricSurface::sphere({0.0, 0.0, 0.0}, 1.7, 20);
   amrex::Real area = 0.0;
   amrex::GpuArray<amrex::Real, 3> area_vector{};
   amrex::GpuArray<amrex::Real, 3> second_moment{};
   std::vector<amrex::Real> sampled_fluxes(surface.points().size() * 6, 0.0);
   std::size_t point_index = 0;
   for (auto const& point : surface.points()) {
      const amrex::Real weight = std::sqrt(
         point.area_vector[0] * point.area_vector[0]
         + point.area_vector[1] * point.area_vector[1]
         + point.area_vector[2] * point.area_vector[2]
      );
      area += weight;
      for (int direction = 0; direction < 3; ++direction) {
         area_vector[direction] += point.area_vector[direction];
         const amrex::Real normal = point.position[direction] / 1.7;
         second_moment[direction] += weight * normal * normal;
         sampled_fluxes[6 * point_index + direction] = normal;
      }
      sampled_fluxes[6 * point_index + 3] = 1.0;
      ++point_index;
   }
   const amrex::Real expected_area = 4.0 * pyfluid_amrex::constants::pi * 1.7 * 1.7;
   require_close(area, expected_area, 2.0e-13 * expected_area);
   for (int direction = 0; direction < 3; ++direction) {
      require_close(area_vector[direction], 0.0, 2.0e-13 * expected_area);
      require_close(second_moment[direction], expected_area / 3.0, 2.0e-13 * expected_area);
   }
   const auto surface_integrals = surface.integrate_normal_fluxes(sampled_fluxes, 2);
   require_close(surface_integrals[0], expected_area, 2.0e-13 * expected_area);
   require_close(surface_integrals[1], 0.0, 2.0e-13 * expected_area);
}

void check_coordinate_round_trip()
{
   const std::vector<std::pair<int, amrex::GpuArray<amrex::Real, 3>>> samples{
      {pyfluid_amrex::MeshCartesian, {0.4, -0.3, 0.2}},
      {pyfluid_amrex::MeshCylindrical, {0.8, 1.1, -0.2}},
      {pyfluid_amrex::MeshSpherical, {0.9, 1.2, 2.1}},
   };
   for (auto const& [coordinate, mesh_point] : samples) {
      const auto physical = pyfluid_amrex::mesh_coordinate_triplet_to_cartesian(
         coordinate, mesh_point
      );
      const auto recovered = pyfluid_amrex::cartesian_to_mesh_coordinate_triplet(
         coordinate, physical
      );
      for (int component = 0; component < 3; ++component) {
         require_close(recovered[component], mesh_point[component], 2.0e-14);
      }
   }
}

void check_spin_weighted_multipoles()
{
   const auto surface = ParametricSurface::sphere({0.0, 0.0, 0.0}, 2.0, 24);
   std::vector<amrex::Real> real_values;
   std::vector<amrex::Real> imaginary_values;
   real_values.reserve(surface.points().size());
   imaginary_values.reserve(surface.points().size());
   for (auto const& point : surface.points()) {
      const auto harmonic = pyfluid_amrex::spin_weighted_spherical_harmonic(
         -2, 2, 2, point.theta, point.phi
      );
      real_values.push_back(harmonic.real());
      imaginary_values.push_back(harmonic.imag());
   }
   require_close(
      pyfluid_amrex::surface_complex_l2(surface, real_values, imaginary_values),
      1.0,
      3.0e-13
   );
   const auto modes = pyfluid_amrex::decompose_spin_weighted_field(
      surface, real_values, imaginary_values, "psi4", -2, 2, 4
   );
   for (auto const& mode : modes) {
      const amrex::Real expected = mode.ell == 2 && mode.m == 2 ? 1.0 : 0.0;
      require_close(mode.value.real(), expected, 8.0e-13);
      require_close(mode.value.imag(), 0.0, 8.0e-13);
   }
}

void check_azimuthal_modes()
{
   constexpr amrex::Real amplitude = 0.2;
   const auto surface = ParametricSurface::sphere({0.0, 0.0, 0.0}, 1.7, 24);
   std::vector<amrex::Real> values;
   values.reserve(surface.points().size());
   for (auto const& point : surface.points()) {
      values.push_back(1.0 + amplitude * std::cos(2.0 * point.phi));
   }
   const auto modes = pyfluid_amrex::decompose_azimuthal_field(
      surface, values, "rho", 4
   );
   const amrex::Real area = 4.0 * pyfluid_amrex::constants::pi * 1.7 * 1.7;
   for (auto const& mode : modes) {
      const amrex::Real expected = mode.m == 0
         ? area
         : (mode.m == 2 ? 0.5 * amplitude * area : 0.0);
      require_close(mode.value.real(), expected, 8.0e-13 * area);
      require_close(mode.value.imag(), 0.0, 8.0e-13 * area);
   }
}

void check_amr_interpolation()
{
   const amrex::GpuArray<int, 3> periodic{{0, 0, 0}};
   std::vector<amrex::Geometry> geometry{
      make_geometry(8, periodic), make_geometry(16, periodic)
   };
   amrex::BoxArray coarse_grids(geometry[0].Domain());
   coarse_grids.maxSize(4);
   amrex::BoxArray fine_grids(amrex::Box(
      amrex::IntVect(4, 4, 4), amrex::IntVect(11, 11, 11)
   ));
   fine_grids.maxSize(4);
   std::vector<amrex::BoxArray> grids{coarse_grids, fine_grids};
   std::vector<amrex::DistributionMapping> distribution{
      amrex::DistributionMapping(coarse_grids),
      amrex::DistributionMapping(fine_grids),
   };
   std::vector<std::unique_ptr<amrex::MultiFab>> hierarchy(2);
   hierarchy[0] = linear_coordinate_field(
      geometry[0], grids[0], distribution[0], amrex::IndexType::TheNodeType()
   );
   hierarchy[1] = linear_coordinate_field(
      geometry[1], grids[1], distribution[1], amrex::IndexType::TheNodeType()
   );

   const auto surface = ParametricSurface::sphere({0.0, 0.0, 0.0}, 0.4, 10);
   std::vector<amrex::GpuArray<amrex::Real, 3>> positions;
   for (auto const& point : surface.points()) {
      positions.push_back(point.position);
   }
   const HierarchyInterpolationPlan plan(
      positions,
      pyfluid_amrex::MeshCartesian,
      pyfluid_amrex::mesh_default_axes_layout(pyfluid_amrex::MeshCartesian),
      geometry,
      grids,
      distribution,
      amrex::IndexType::TheNodeType()
   );
   const auto values = plan.interpolate(hierarchy, 0, 1);
   for (std::size_t point = 0; point < positions.size(); ++point) {
      const auto& position = positions[point];
      const amrex::Real expected =
         1.0 + 2.0 * position[0] - 3.0 * position[1] + 0.5 * position[2];
      require_close(values[point], expected, 2.0e-13);
   }

   // The origin lies between cell centers and exactly on the internal FAB
   // boundaries at both levels.  Linear interpolation therefore needs values
   // exchanged into the one-cell ghost layer from neighboring FABs.
   std::vector<std::unique_ptr<amrex::MultiFab>> cell_hierarchy(2);
   for (int level = 0; level < 2; ++level) {
      cell_hierarchy[level] = linear_coordinate_field(
         geometry[level], grids[level], distribution[level],
         amrex::IndexType::TheCellType()
      );
   }
   const std::vector<amrex::GpuArray<amrex::Real, 3>> boundary_position{
      {0.0, 0.0, 0.0}
   };
   const HierarchyInterpolationPlan boundary_plan(
      boundary_position,
      pyfluid_amrex::MeshCartesian,
      pyfluid_amrex::mesh_default_axes_layout(pyfluid_amrex::MeshCartesian),
      geometry,
      grids,
      distribution,
      amrex::IndexType::TheCellType()
   );
   const auto boundary_value = boundary_plan.interpolate(cell_hierarchy, 0, 1);
   require_close(boundary_value[0], 1.0, 2.0e-13);

   std::vector<std::unique_ptr<amrex::MultiFab>> missing_ghost_hierarchy(2);
   for (int level = 0; level < 2; ++level) {
      missing_ghost_hierarchy[level] = linear_coordinate_field(
         geometry[level], grids[level], distribution[level],
         amrex::IndexType::TheCellType(), 0
      );
   }
   bool rejected_missing_ghost = false;
   try {
      static_cast<void>(boundary_plan.interpolate(missing_ghost_hierarchy, 0, 1));
   } catch (std::out_of_range const&) {
      rejected_missing_ghost = true;
   }
   if (!rejected_missing_ghost) {
      throw std::runtime_error("cross-FAB interpolation accepted a missing ghost layer");
   }
}

void check_reflected_cell_interpolation()
{
   const amrex::GpuArray<int, 3> periodic{{0, 0, 0}};
   const amrex::GpuArray<amrex::Real, 3> lower{{0.0, -1.0, -1.0}};
   const amrex::GpuArray<amrex::Real, 3> upper{{1.0, 1.0, 1.0}};
   std::vector<amrex::Geometry> geometry{
      make_geometry(8, periodic, lower, upper)
   };
   amrex::BoxArray level_grids(geometry[0].Domain());
   level_grids.maxSize(4);
   std::vector<amrex::BoxArray> grids{level_grids};
   std::vector<amrex::DistributionMapping> distribution{
      amrex::DistributionMapping(level_grids)
   };
   std::vector<std::unique_ptr<amrex::MultiFab>> hierarchy(1);
   hierarchy[0] = std::make_unique<amrex::MultiFab>(
      grids[0], distribution[0], 2, 1
   );
   const auto spacing = geometry[0].CellSizeArray();
   for (amrex::MFIter mfi(*hierarchy[0], false); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.fabbox();
      const auto values = hierarchy[0]->array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const amrex::Real x = lower[0] + (i + 0.5) * spacing[0];
         values(i, j, k, 0) = 1.0 + x * x;
         values(i, j, k, 1) = x;
      });
   }
   const HierarchyInterpolationPlan plan(
      {{{0.0, 0.0, 0.0}}},
      pyfluid_amrex::MeshCartesian,
      pyfluid_amrex::mesh_default_axes_layout(pyfluid_amrex::MeshCartesian),
      geometry,
      grids,
      distribution,
      amrex::IndexType::TheCellType(),
      std::array<int, 3>{{1, 0, 0}},
      std::array<int, 3>{{0, 0, 0}}
   );
   const auto values = plan.interpolate(hierarchy, 0, 2);
   require_close(values[0], 1.0 + 0.25 * spacing[0] * spacing[0], 2.0e-13);
   require_close(values[1], 0.0, 2.0e-13);
}

void check_coordinate_hierarchy_interpolation(int coordinate)
{
   const amrex::GpuArray<int, 3> periodic{{0, 0, 0}};
   const amrex::GpuArray<amrex::Real, 3> lower = coordinate == pyfluid_amrex::MeshCylindrical
      ? amrex::GpuArray<amrex::Real, 3>{{0.0, 0.0, -1.2}}
      : amrex::GpuArray<amrex::Real, 3>{{0.2, 0.0, 0.0}};
   const amrex::GpuArray<amrex::Real, 3> upper = coordinate == pyfluid_amrex::MeshCylindrical
      ? amrex::GpuArray<amrex::Real, 3>{{2.0, pyfluid_amrex::constants::two_pi, 1.2}}
      : amrex::GpuArray<amrex::Real, 3>{{2.0, pyfluid_amrex::constants::pi, pyfluid_amrex::constants::two_pi}};
   std::vector<amrex::Geometry> geometry{make_geometry(32, periodic, lower, upper)};
   amrex::BoxArray level_grids(geometry[0].Domain());
   level_grids.maxSize(16);
   std::vector<amrex::BoxArray> grids{level_grids};
   std::vector<amrex::DistributionMapping> distribution{
      amrex::DistributionMapping(level_grids)
   };
   const auto surface = ParametricSurface::sphere({0.0, 0.0, 0.0}, 1.0, 16);
   std::vector<amrex::GpuArray<amrex::Real, 3>> positions;
   for (auto const& point : surface.points()) {
      positions.push_back(point.position);
   }
   for (amrex::IndexType centering : {
           amrex::IndexType::TheNodeType(), amrex::IndexType::TheCellType()}) {
      std::vector<std::unique_ptr<amrex::MultiFab>> hierarchy(1);
      hierarchy[0] = linear_coordinate_field(
         geometry[0], grids[0], distribution[0], centering
      );
      std::vector<amrex::Real> values;
      try {
         const HierarchyInterpolationPlan plan(
            positions,
            coordinate,
            pyfluid_amrex::mesh_default_axes_layout(coordinate),
            geometry,
            grids,
            distribution,
            centering
         );
         values = plan.interpolate(hierarchy, 0, 1);
      } catch (std::exception const& error) {
         throw std::runtime_error(
            "coordinate hierarchy interpolation failed for coordinate="
            + std::to_string(coordinate) + " centering="
            + (centering.nodeCentered(0) ? "nodal" : "cell") + ": " + error.what()
         );
      }
      for (std::size_t point = 0; point < positions.size(); ++point) {
         const auto q = pyfluid_amrex::cartesian_to_mesh_coordinate_triplet(
            coordinate, positions[point]
         );
         require_close(values[point], 1.0 + 2.0 * q[0] - 3.0 * q[1] + 0.5 * q[2], 3.0e-13);
      }
   }

   // A source-grid cell center must remain inside its own interpolation
   // domain after the curvilinear -> Cartesian -> curvilinear round trip.
   // Near a spherical/cylindrical axis that round trip can land a few ulps
   // below the first cell-centered coordinate.
   const auto spacing = geometry[0].CellSizeArray();
   const amrex::GpuArray<amrex::Real, 3> first_center{{
      lower[0] + 0.5 * spacing[0],
      lower[1] + 0.5 * spacing[1],
      lower[2] + 0.5 * spacing[2],
   }};
   const auto first_center_cartesian =
      pyfluid_amrex::mesh_coordinate_triplet_to_cartesian(coordinate, first_center);
   std::vector<std::unique_ptr<amrex::MultiFab>> cell_hierarchy(1);
   cell_hierarchy[0] = linear_coordinate_field(
      geometry[0], grids[0], distribution[0], amrex::IndexType::TheCellType()
   );
   const HierarchyInterpolationPlan first_center_plan(
      {first_center_cartesian},
      coordinate,
      pyfluid_amrex::mesh_default_axes_layout(coordinate),
      geometry,
      grids,
      distribution,
      amrex::IndexType::TheCellType()
   );
   const auto first_center_value = first_center_plan.interpolate(cell_hierarchy, 0, 1);
   require_close(
      first_center_value[0],
      1.0 + 2.0 * first_center[0] - 3.0 * first_center[1]
         + 0.5 * first_center[2],
      3.0e-13
   );
}

} // namespace

int main(int argc, char* argv[])
{
   amrex::Initialize(argc, argv);
   int status = 0;
   try {
      check_surface_geometry();
      check_spin_weighted_multipoles();
      check_azimuthal_modes();
      check_coordinate_round_trip();
      check_amr_interpolation();
      check_reflected_cell_interpolation();
      check_coordinate_hierarchy_interpolation(pyfluid_amrex::MeshCylindrical);
      check_coordinate_hierarchy_interpolation(pyfluid_amrex::MeshSpherical);
      amrex::Print() << "PASS surface geometry, multipoles, and AMR interpolation\n";
   } catch (std::exception const& error) {
      amrex::Print() << "FAIL " << error.what() << '\n';
      status = 1;
   }
   amrex::Finalize();
   return status;
}
