#include <AMReX.H>
#include <AMReX_BCRec.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_RealBox.H>

#include "pyfluid_amrex/multipatch.hpp"
#include "pyfluid_amrex/initial_data/sampling.hpp"
#include "pyfluid_amrex/patch_maps.hpp"
#include "pyfluid_amrex/patch_grid.hpp"
#include "pyfluid_amrex/patch_layout_builder.hpp"
#include "pyfluid_amrex/patch_domain.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace {

constexpr int NumComponents = 3;
constexpr int NumGhost = 2;
constexpr int RefRatio = 2;
constexpr int Thornburg04PatchCount = 1 + 2 * AMREX_SPACEDIM;
constexpr int Thornburg13PatchCount = 1 + 4 * AMREX_SPACEDIM;

pyfluid_amrex::PatchCartesianAxes xz_patch_cartesian_axes()
{
   pyfluid_amrex::PatchCartesianAxes axes = pyfluid_amrex::default_patch_cartesian_axes();
#if (AMREX_SPACEDIM >= 2)
   axes.component[0] = 0;
   axes.component[1] = 2;
#endif
   return axes;
}

amrex::Real analytic_value(amrex::IntVect const& cell, int component)
{
   amrex::Real value = 0.25 * static_cast<amrex::Real>(component + 1);
   value += 0.013 * static_cast<amrex::Real>(cell[0] + 1);
#if (AMREX_SPACEDIM >= 2)
   value += 0.017 * static_cast<amrex::Real>(cell[1] + 2);
#endif
#if (AMREX_SPACEDIM == 3)
   value += 0.019 * static_cast<amrex::Real>(cell[2] + 3);
#endif
   return value;
}

pyfluid_amrex::PatchDomainBoxes centered_patch_domain()
{
   pyfluid_amrex::PatchDomainBoxes boxes;
   boxes.count = 1;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      boxes.small[dir] = 2;
      boxes.big[dir] = 5;
   }
   return boxes;
}

pyfluid_amrex::PatchCoordinate shifted_patch_coordinate(
   pyfluid_amrex::PatchCoordinate coordinate,
   int component,
   amrex::Real offset
)
{
   pyfluid_amrex::set_patch_coordinate_component(
      coordinate,
      component,
      pyfluid_amrex::patch_coordinate_component(coordinate, component) + offset
   );
   return coordinate;
}

amrex::Real patch_jacobian_finite_difference_error(
   pyfluid_amrex::PatchMap const& map,
   pyfluid_amrex::PatchCoordinate const& local
)
{
   constexpr amrex::Real eps = 1.0e-6;
   const pyfluid_amrex::PatchMapJacobian jacobian =
      pyfluid_amrex::patch_local_to_global_jacobian(map, local);
   amrex::Real max_error = 0.0;
   for (int local_component = 0; local_component < 3; ++local_component) {
      const pyfluid_amrex::PatchCoordinate plus =
         pyfluid_amrex::patch_local_to_global(map, shifted_patch_coordinate(local, local_component, eps));
      const pyfluid_amrex::PatchCoordinate minus =
         pyfluid_amrex::patch_local_to_global(map, shifted_patch_coordinate(local, local_component, -eps));
      for (int global_component = 0; global_component < 3; ++global_component) {
         const amrex::Real finite_difference = (
            pyfluid_amrex::patch_coordinate_component(plus, global_component)
            - pyfluid_amrex::patch_coordinate_component(minus, global_component)
         ) / (2.0 * eps);
         const amrex::Real analytic =
            pyfluid_amrex::patch_map_jacobian_component(jacobian, global_component, local_component);
         max_error = std::max(max_error, std::abs(finite_difference - analytic));
      }
   }
   return max_error;
}

pyfluid_amrex::PatchDomainFace centered_patch_domain_face(int dir, int face_index)
{
   amrex::Box domain(
      amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
      amrex::IntVect(AMREX_D_DECL(7, 7, 7))
   );
   const auto lo = amrex::lbound(domain);
   const auto hi = amrex::ubound(domain);
   amrex::IntVect face_cell(AMREX_D_DECL(3, 3, 3));
   face_cell[dir] = face_index;
   int k = 0;
#if (AMREX_SPACEDIM == 3)
   k = face_cell[2];
#endif
   return pyfluid_amrex::patch_domain_face(
      centered_patch_domain(),
      lo,
      hi,
      dir,
      face_cell[0],
#if (AMREX_SPACEDIM >= 2)
      face_cell[1],
#else
      0,
#endif
      k
   );
}

int run_patch_domain_face_checks()
{
   int return_code = 0;
   const auto check_face = [&](int dir, int face_index, bool expect_interior, bool expect_wall, char const* label) {
      const pyfluid_amrex::PatchDomainFace face = centered_patch_domain_face(dir, face_index);
      const bool interior = pyfluid_amrex::patch_domain_face_is_interior(face);
      const bool wall = pyfluid_amrex::patch_domain_face_is_wall(face);
      if (interior != expect_interior || wall != expect_wall) {
         amrex::Print() << "patch-domain face classifier failed for " << label
                        << " dir=" << dir
                        << " face_index=" << face_index
                        << " interior=" << interior
                        << " wall=" << wall << "\n";
         return_code = 1;
      }
   };
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      check_face(dir, 4, true, false, "interior face");
      check_face(dir, 2, false, true, "low wall face");
      check_face(dir, 6, false, true, "high wall face");
      check_face(dir, 1, false, false, "exterior inactive face");
   }
   if (return_code == 0) {
      amrex::Print() << "patch-domain face classifier checks passed\n";
   }
   return return_code;
}

int run_patch_coordinate_map_checks()
{
   int return_code = 0;
   const auto check_map = [&](int coordinate_map, bool expect_cartesian, bool expect_thornburg) {
      const bool supported = pyfluid_amrex::patch_coordinate_map_supported(coordinate_map);
      const bool cartesian = pyfluid_amrex::patch_coordinate_map_is_cartesian(coordinate_map);
      const bool thornburg = pyfluid_amrex::patch_coordinate_map_is_thornburg(coordinate_map);
      if (!supported || cartesian != expect_cartesian || thornburg != expect_thornburg) {
         amrex::Print() << "patch coordinate map check failed for "
                        << pyfluid_amrex::patch_coordinate_map_name(coordinate_map)
                        << " supported=" << supported
                        << " cartesian=" << cartesian
                        << " thornburg=" << thornburg << "\n";
         return_code = 1;
      }
   };
   check_map(pyfluid_amrex::PatchMapCartesianIdentity, true, false);
   check_map(pyfluid_amrex::PatchMapThornburgCentralCartesian, true, true);
   check_map(pyfluid_amrex::PatchMapThornburgShellPlusX, false, true);
   check_map(pyfluid_amrex::PatchMapThornburgShellMinusX, false, true);
   check_map(pyfluid_amrex::PatchMapThornburgShellPlusY, false, true);
   check_map(pyfluid_amrex::PatchMapThornburgShellMinusY, false, true);
   check_map(pyfluid_amrex::PatchMapThornburgShellPlusZ, false, true);
   check_map(pyfluid_amrex::PatchMapThornburgShellMinusZ, false, true);
   check_map(pyfluid_amrex::PatchMapCylindricalShellPlusX, false, false);
   check_map(pyfluid_amrex::PatchMapCylindricalShellMinusX, false, false);
   check_map(pyfluid_amrex::PatchMapCylindricalShellPlusY, false, false);
   check_map(pyfluid_amrex::PatchMapCylindricalShellMinusY, false, false);

   pyfluid_amrex::PatchLayout layout;
   layout.global_box = amrex::Box(
      amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
      amrex::IntVect(AMREX_D_DECL(7, 7, 7))
   );
   amrex::Box central = layout.global_box;
   amrex::Box outer = layout.global_box;
   central.setBig(0, 3);
   outer.setSmall(0, 4);
   layout.patches.push_back(
      pyfluid_amrex::make_patch_descriptor(
         0,
         central,
         central,
         0,
         pyfluid_amrex::PatchMapThornburgCentralCartesian
      )
   );
   layout.patches.push_back(
      pyfluid_amrex::make_patch_descriptor(
         1,
         outer,
         outer,
         1,
         pyfluid_amrex::PatchMapThornburgShellPlusX
      )
   );
   pyfluid_amrex::validate_patch_layout(layout);
   if (return_code == 0) {
      amrex::Print() << "patch coordinate map topology checks passed\n";
   }
   return return_code;
}

int run_patch_coordinate_round_trip_checks()
{
   int return_code = 0;
   const auto check_local_round_trip = [&](
      pyfluid_amrex::PatchMap const& map,
      pyfluid_amrex::PatchCoordinate const& local,
      char const* label
   ) {
      const pyfluid_amrex::PatchCoordinate global = pyfluid_amrex::patch_local_to_global(map, local);
      const pyfluid_amrex::PatchCoordinate local_round_trip =
         pyfluid_amrex::patch_global_to_local(map, global);
      const amrex::Real local_error =
         pyfluid_amrex::patch_coordinate_max_abs_difference(local, local_round_trip);
      const pyfluid_amrex::PatchCoordinate global_round_trip =
         pyfluid_amrex::patch_local_to_global(map, pyfluid_amrex::patch_global_to_local(map, global));
      const amrex::Real global_error =
         pyfluid_amrex::patch_coordinate_max_abs_difference(global, global_round_trip);
      if (local_error > 1.0e-14 || global_error > 1.0e-14) {
         amrex::Print() << "patch coordinate round-trip check failed for "
                        << label
                        << " map=" << pyfluid_amrex::patch_coordinate_map_name(map.coordinate_map)
                        << " local_error=" << local_error
                        << " global_error=" << global_error << "\n";
         return_code = 1;
      }
      const amrex::Real jacobian_error = patch_jacobian_finite_difference_error(map, local);
      if (jacobian_error > 1.0e-7) {
         amrex::Print() << "patch coordinate Jacobian check failed for "
                        << label
                        << " map=" << pyfluid_amrex::patch_coordinate_map_name(map.coordinate_map)
                        << " jacobian_error=" << jacobian_error << "\n";
         return_code = 1;
      }
   };
   const auto check_system_round_trips = [&](
      pyfluid_amrex::PatchMapSystem const& system,
      char const* label
   ) {
      for (int patch_id = 0; patch_id < static_cast<int>(system.maps.size()); ++patch_id) {
         pyfluid_amrex::PatchMap const& map = system.maps[patch_id];
         pyfluid_amrex::PatchCoordinate local{};
         if (pyfluid_amrex::patch_coordinate_map_is_cartesian(map.coordinate_map)) {
            local = pyfluid_amrex::make_patch_coordinate(0.125, -0.25, 0.375);
         } else if (pyfluid_amrex::patch_coordinate_map_is_thornburg_shell(map.coordinate_map)) {
            local = pyfluid_amrex::make_patch_coordinate(
               0.125,
               -0.25,
               0.5 * (map.radial_inner + map.radial_outer)
            );
         } else {
            local = pyfluid_amrex::make_patch_coordinate(
               0.125,
               -0.375,
               0.5 * (map.radial_inner + map.radial_outer)
            );
         }
         check_local_round_trip(map, local, label);
         const pyfluid_amrex::PatchMapPoint point =
            pyfluid_amrex::patch_system_local_to_global(system, patch_id, local);
         const int owner_patch = pyfluid_amrex::patch_system_owner_patch_id(system, point.global);
         if (owner_patch != patch_id) {
            amrex::Print() << "patch owner check failed for " << label
                           << " expected_patch=" << patch_id
                           << " owner_patch=" << owner_patch << "\n";
            return_code = 1;
         }
      }
   };
   const auto check_global_owner = [&](
      pyfluid_amrex::PatchMapSystem const& system,
      pyfluid_amrex::PatchCoordinate const& global,
      int expected_patch,
      char const* label
   ) {
      const pyfluid_amrex::PatchMapPoint point = pyfluid_amrex::patch_system_global_to_local(system, global);
      const pyfluid_amrex::PatchCoordinate global_round_trip =
         pyfluid_amrex::patch_local_to_global(system.maps[point.patch_id], point.local);
      const amrex::Real global_error =
         pyfluid_amrex::patch_coordinate_max_abs_difference(global, global_round_trip);
      if (point.patch_id != expected_patch || global_error > 1.0e-14) {
         amrex::Print() << "patch system owner/round-trip check failed for " << label
                        << " expected_patch=" << expected_patch
                        << " actual_patch=" << point.patch_id
                        << " global_error=" << global_error << "\n";
         return_code = 1;
      }
   };

   const pyfluid_amrex::PatchMap identity_maps[] = {
      pyfluid_amrex::make_patch_map(
         pyfluid_amrex::PatchMapCartesianIdentity,
         pyfluid_amrex::make_patch_coordinate(0.0, 0.0, 0.0),
         0.0,
         0.0
      ),
      pyfluid_amrex::make_patch_map(
         pyfluid_amrex::PatchMapCartesianIdentity,
         pyfluid_amrex::make_patch_coordinate(1.25, -0.75, 0.5),
         0.0,
         0.0
      ),
   };
   const pyfluid_amrex::PatchCoordinate identity_points[] = {
      pyfluid_amrex::make_patch_coordinate(0.0, 0.0, 0.0),
      pyfluid_amrex::make_patch_coordinate(0.125, -0.25, 0.375),
      pyfluid_amrex::make_patch_coordinate(2.5, 1.75, -3.25),
      pyfluid_amrex::make_patch_coordinate(-0.03125, 4.0, 0.0625),
   };
   for (pyfluid_amrex::PatchMap const& map : identity_maps) {
      for (pyfluid_amrex::PatchCoordinate const& local : identity_points) {
         check_local_round_trip(map, local, "cartesian_identity");
      }
   }

   const pyfluid_amrex::PatchCoordinate origin =
      pyfluid_amrex::make_patch_coordinate(0.25, -0.5, 0.75);
   const pyfluid_amrex::PatchMapSystem thornburg04 =
      pyfluid_amrex::make_thornburg04_patch_map_system(1.0, 4.0, origin);
   const pyfluid_amrex::PatchMapSystem thornburg13 =
      pyfluid_amrex::make_thornburg13_patch_map_system(1.0, 2.5, 5.0, origin);
   const pyfluid_amrex::PatchMapSystem cylindrical5 =
      pyfluid_amrex::make_cylindrical5_patch_map_system(1.0, 4.0, origin);
   const pyfluid_amrex::PatchMapSystem cylindrical9 =
      pyfluid_amrex::make_cylindrical9_patch_map_system(1.0, 2.5, 5.0, origin);
   check_system_round_trips(thornburg04, "thornburg04");
   check_system_round_trips(thornburg13, "thornburg13");
   check_system_round_trips(cylindrical5, "cylindrical5");
   check_system_round_trips(cylindrical9, "cylindrical9");
#if (AMREX_SPACEDIM == 2)
   const pyfluid_amrex::PatchMapSystem thornburg04_xz =
      pyfluid_amrex::make_thornburg04_patch_map_system(1.0, 4.0, origin, xz_patch_cartesian_axes());
   check_system_round_trips(thornburg04_xz, "thornburg04_xz");
#endif

   check_global_owner(thornburg04, pyfluid_amrex::patch_coordinate_add(origin, {0.2, 0.1, 0.1}), 0, "thornburg04 central");
   check_global_owner(thornburg04, pyfluid_amrex::patch_coordinate_add(origin, {2.0, 0.2, 0.3}), 1, "thornburg04 plus x");
   check_global_owner(thornburg04, pyfluid_amrex::patch_coordinate_add(origin, {-2.0, 0.2, 0.3}), 2, "thornburg04 minus x");
#if (AMREX_SPACEDIM >= 2)
   check_global_owner(thornburg04, pyfluid_amrex::patch_coordinate_add(origin, {0.2, 2.0, 0.3}), 3, "thornburg04 plus y");
   check_global_owner(thornburg04, pyfluid_amrex::patch_coordinate_add(origin, {0.2, -2.0, 0.3}), 4, "thornburg04 minus y");
#endif
#if (AMREX_SPACEDIM == 3)
   check_global_owner(thornburg04, pyfluid_amrex::patch_coordinate_add(origin, {0.2, 0.3, 2.0}), 5, "thornburg04 plus z");
   check_global_owner(thornburg04, pyfluid_amrex::patch_coordinate_add(origin, {0.2, 0.3, -2.0}), 6, "thornburg04 minus z");
#endif
#if (AMREX_SPACEDIM == 2)
   check_global_owner(thornburg04_xz, pyfluid_amrex::patch_coordinate_add(origin, {0.2, 0.0, 2.0}), 3, "thornburg04 xz plus z");
   check_global_owner(thornburg04_xz, pyfluid_amrex::patch_coordinate_add(origin, {0.2, 0.0, -2.0}), 4, "thornburg04 xz minus z");
#endif

   check_global_owner(thornburg13, pyfluid_amrex::patch_coordinate_add(origin, {2.0, 0.2, 0.3}), 1, "thornburg13 inner plus x");
   check_global_owner(
      thornburg13,
      pyfluid_amrex::patch_coordinate_add(origin, {4.0, 0.2, 0.3}),
      1 + 2 * AMREX_SPACEDIM,
      "thornburg13 outer plus x"
   );
#if (AMREX_SPACEDIM == 3)
   check_global_owner(thornburg13, pyfluid_amrex::patch_coordinate_add(origin, {0.2, 0.3, -4.0}), 12, "thornburg13 outer minus z");
#endif

   check_global_owner(cylindrical5, pyfluid_amrex::patch_coordinate_add(origin, {0.2, 0.1, 1.5}), 0, "cylindrical5 central");
   check_global_owner(cylindrical5, pyfluid_amrex::patch_coordinate_add(origin, {2.0, 0.2, 1.5}), 1, "cylindrical5 plus x");
   check_global_owner(cylindrical5, pyfluid_amrex::patch_coordinate_add(origin, {-2.0, 0.2, 1.5}), 2, "cylindrical5 minus x");
   check_global_owner(cylindrical5, pyfluid_amrex::patch_coordinate_add(origin, {0.2, 2.0, 1.5}), 3, "cylindrical5 plus y");
   check_global_owner(cylindrical5, pyfluid_amrex::patch_coordinate_add(origin, {0.2, -2.0, 1.5}), 4, "cylindrical5 minus y");

   check_global_owner(cylindrical9, pyfluid_amrex::patch_coordinate_add(origin, {2.0, 0.2, 1.5}), 1, "cylindrical9 inner plus x");
   check_global_owner(cylindrical9, pyfluid_amrex::patch_coordinate_add(origin, {4.0, 0.2, 1.5}), 5, "cylindrical9 outer plus x");
   check_global_owner(cylindrical9, pyfluid_amrex::patch_coordinate_add(origin, {0.2, -4.0, 1.5}), 8, "cylindrical9 outer minus y");

   if (return_code == 0) {
      amrex::Print() << "patch coordinate round-trip checks passed\n";
   }
   return return_code;
}

amrex::IntVect test_patch_central_cells()
{
#if (AMREX_SPACEDIM == 1)
   return amrex::IntVect(8);
#elif (AMREX_SPACEDIM == 2)
   return amrex::IntVect(8, 10);
#else
   return amrex::IntVect(8, 10, 12);
#endif
}

int run_patch_grid_builder_checks()
{
   int return_code = 0;
   const pyfluid_amrex::PatchCoordinate origin =
      pyfluid_amrex::make_patch_coordinate(0.125, -0.25, 0.375);
   pyfluid_amrex::PatchGridResolution resolution;
   resolution.central_cells = test_patch_central_cells();
   resolution.angular_cells = 6;
   resolution.radial_cells = 5;
   resolution.outer_radial_cells = 7;
   resolution.axial_cells = 9;
   resolution.axial_lo = -1.25;
   resolution.axial_hi = 1.75;

   const auto check_grid_system = [&](
      pyfluid_amrex::PatchGridSystem const& grid_system,
      int expected_patches,
      char const* label
   ) {
      if (static_cast<int>(grid_system.patches.size()) != expected_patches
          || static_cast<int>(grid_system.map_system.maps.size()) != expected_patches) {
         amrex::Print() << "patch-grid count check failed for " << label
                        << " patches=" << grid_system.patches.size()
                        << " maps=" << grid_system.map_system.maps.size()
                        << " expected=" << expected_patches << "\n";
         return_code = 1;
         return;
      }
      for (int patch_id = 0; patch_id < expected_patches; ++patch_id) {
         const pyfluid_amrex::PatchGridDescriptor& patch = grid_system.patches[patch_id];
         if (patch.patch_id != patch_id || patch.map.coordinate_map != grid_system.map_system.maps[patch_id].coordinate_map) {
            amrex::Print() << "patch-grid id/map check failed for " << label
                           << " patch=" << patch_id << "\n";
            return_code = 1;
         }
         amrex::IntVect cell(AMREX_D_DECL(0, 0, 0));
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            cell[dir] = patch.cells[dir] / 2;
            if (patch_grid_cell_size(patch, dir) <= 0.0) {
               amrex::Print() << "patch-grid spacing check failed for " << label
                              << " patch=" << patch_id
                              << " dir=" << dir << "\n";
               return_code = 1;
            }
         }
         const pyfluid_amrex::PatchCoordinate local =
            pyfluid_amrex::patch_grid_cell_center(patch, cell);
         const pyfluid_amrex::PatchMapPoint point =
            pyfluid_amrex::patch_system_local_to_global(grid_system.map_system, patch_id, local);
         const int owner_patch =
            pyfluid_amrex::patch_system_owner_patch_id(grid_system.map_system, point.global);
         if (owner_patch != patch_id) {
            amrex::Print() << "patch-grid owner check failed for " << label
                           << " patch=" << patch_id
                           << " owner=" << owner_patch << "\n";
            return_code = 1;
         }
      }
   };

   check_grid_system(
      pyfluid_amrex::make_thornburg04_patch_grid_system(1.0, 4.0, resolution, origin),
      Thornburg04PatchCount,
      "thornburg04"
   );
   check_grid_system(
      pyfluid_amrex::make_thornburg13_patch_grid_system(1.0, 2.5, 5.0, resolution, origin),
      Thornburg13PatchCount,
      "thornburg13"
   );
#if (AMREX_SPACEDIM == 2)
   check_grid_system(
      pyfluid_amrex::make_thornburg04_patch_grid_system(1.0, 4.0, resolution, origin, xz_patch_cartesian_axes()),
      Thornburg04PatchCount,
      "thornburg04_xz"
   );
   check_grid_system(
      pyfluid_amrex::make_thornburg13_patch_grid_system(1.0, 2.5, 5.0, resolution, origin, xz_patch_cartesian_axes()),
      Thornburg13PatchCount,
      "thornburg13_xz"
   );
#endif
   check_grid_system(
      pyfluid_amrex::make_cylindrical5_patch_grid_system(1.0, 4.0, resolution, origin),
      5,
      "cylindrical5"
   );
   check_grid_system(
      pyfluid_amrex::make_cylindrical9_patch_grid_system(1.0, 2.5, 5.0, resolution, origin),
      9,
      "cylindrical9"
   );

   if (return_code == 0) {
      amrex::Print() << "patch-grid builder checks passed\n";
   }
   return return_code;
}

amrex::Real smooth_physical_sample_value(
   pyfluid_amrex::PatchCoordinate const& physical,
   pyfluid_amrex::PatchCoordinate const& origin
)
{
   const pyfluid_amrex::PatchCoordinate centered = pyfluid_amrex::patch_coordinate_subtract(physical, origin);
   const amrex::Real radius2 = centered.q1 * centered.q1 + centered.q2 * centered.q2 + centered.q3 * centered.q3;
   return 1.0
      + 0.25 * std::exp(-0.5 * radius2 / 0.37)
      + 0.031 * std::sin(0.7 * physical.q1 - 0.2 * physical.q2 + 0.11 * physical.q3);
}

int run_mixed_patch_initial_data_sampling_checks()
{
   int return_code = 0;
   const pyfluid_amrex::PatchCoordinate origin =
      pyfluid_amrex::make_patch_coordinate(0.125, -0.25, 0.375);
   pyfluid_amrex::PatchGridResolution resolution;
   resolution.central_cells = test_patch_central_cells();
   resolution.angular_cells = 6;
   resolution.radial_cells = 5;
   resolution.outer_radial_cells = 7;

   const auto check_owned_or_context_sample = [&](
      pyfluid_amrex::PatchGridSystem const& grid_system,
      pyfluid_amrex::PatchGridDescriptor const& patch,
      int owner_patch
   ) {
      if (owner_patch == patch.patch_id) {
         return true;
      }
      if (pyfluid_amrex::patch_coordinate_map_is_cartesian(patch.map.coordinate_map)) {
         return owner_patch >= 0 && owner_patch < static_cast<int>(grid_system.patches.size());
      }
      return false;
   };

   const auto check_grid_system = [&](
      pyfluid_amrex::PatchGridSystem const& grid_system,
      char const* label
   ) {
      for (pyfluid_amrex::PatchGridDescriptor const& patch : grid_system.patches) {
         amrex::IntVect small(AMREX_D_DECL(0, 0, 0));
         amrex::IntVect big = patch.cells;
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            big[dir] -= 1;
         }
         const amrex::Box patch_box(small, big);
         for (amrex::BoxIterator iterator(patch_box); iterator.ok(); ++iterator) {
            const amrex::IntVect& cell = iterator();
            const pyfluid_amrex::InitialDataSamplePoint sample =
               pyfluid_amrex::patch_grid_cell_center_initial_data_sample_point(patch, cell);
            const pyfluid_amrex::PatchCoordinate round_trip_global =
               pyfluid_amrex::patch_local_to_global(patch.map, sample.mesh);
            const amrex::Real global_error =
               pyfluid_amrex::patch_coordinate_max_abs_difference(sample.physical, round_trip_global);
            const int owner_patch =
               pyfluid_amrex::patch_system_owner_patch_id(grid_system.map_system, sample.physical);
            const amrex::Real value = smooth_physical_sample_value(sample.physical, origin);
            if (global_error > 1.0e-14
                || !check_owned_or_context_sample(grid_system, patch, owner_patch)
                || !std::isfinite(value)) {
               amrex::Print() << "mixed-patch IC sampling check failed for " << label
                              << " patch=" << patch.patch_id
                              << " owner=" << owner_patch
                              << " global_error=" << global_error
                              << " value=" << value << "\n";
               return_code = 1;
               return;
            }
         }
      }
   };

   check_grid_system(
      pyfluid_amrex::make_thornburg04_patch_grid_system(1.0, 4.0, resolution, origin),
      "thornburg04"
   );
   check_grid_system(
      pyfluid_amrex::make_thornburg13_patch_grid_system(1.0, 2.5, 5.0, resolution, origin),
      "thornburg13"
   );
#if (AMREX_SPACEDIM == 2)
   check_grid_system(
      pyfluid_amrex::make_thornburg04_patch_grid_system(1.0, 4.0, resolution, origin, xz_patch_cartesian_axes()),
      "thornburg04_xz"
   );
   check_grid_system(
      pyfluid_amrex::make_thornburg13_patch_grid_system(1.0, 2.5, 5.0, resolution, origin, xz_patch_cartesian_axes()),
      "thornburg13_xz"
   );
#endif
   check_grid_system(
      pyfluid_amrex::make_cylindrical5_patch_grid_system(1.0, 4.0, resolution, origin),
      "cylindrical5"
   );
   check_grid_system(
      pyfluid_amrex::make_cylindrical9_patch_grid_system(1.0, 2.5, 5.0, resolution, origin),
      "cylindrical9"
   );

   if (return_code == 0) {
      amrex::Print() << "mixed-patch IC sampling checks passed\n";
   }
   return return_code;
}

int run_mixed_patch_logical_layout_checks()
{
   int return_code = 0;
   const pyfluid_amrex::PatchCoordinate origin =
      pyfluid_amrex::make_patch_coordinate(0.125, -0.25, 0.375);
   pyfluid_amrex::PatchGridResolution resolution;
   resolution.central_cells = test_patch_central_cells();
   resolution.angular_cells = 6;
   resolution.radial_cells = 5;
   resolution.outer_radial_cells = 7;
   resolution.axial_cells = 9;
   resolution.axial_lo = -1.25;
   resolution.axial_hi = 1.75;

   const auto check_owned_or_context_sample = [&](
      pyfluid_amrex::PatchGridSystem const& grid_system,
      pyfluid_amrex::PatchGridDescriptor const& grid_patch,
      int owner_patch
   ) {
      if (owner_patch == grid_patch.patch_id) {
         return true;
      }
      if (pyfluid_amrex::patch_coordinate_map_is_cartesian(grid_patch.map.coordinate_map)) {
         return owner_patch >= 0 && owner_patch < static_cast<int>(grid_system.patches.size());
      }
      return false;
   };

   const auto check_grid_system = [&](
      pyfluid_amrex::PatchGridSystem const& grid_system,
      char const* label
   ) {
      const pyfluid_amrex::PatchLayout layout =
         pyfluid_amrex::make_patch_layout_from_grid_system(grid_system);
      if (layout.patches.size() != grid_system.patches.size()) {
         amrex::Print() << "mixed-patch logical-layout count check failed for "
                        << label
                        << " layout_patches=" << layout.patches.size()
                        << " grid_patches=" << grid_system.patches.size() << "\n";
         return_code = 1;
         return;
      }

      amrex::Long layout_points = 0;
      amrex::Long grid_points = 0;
      for (pyfluid_amrex::PatchDescriptor const& layout_patch : layout.patches) {
         const pyfluid_amrex::PatchGridDescriptor& grid_patch =
            pyfluid_amrex::patch_grid_descriptor_for_layout_patch(grid_system, layout_patch);
         layout_points += layout_patch.owned_box.numPts();
         grid_points += pyfluid_amrex::patch_grid_cell_count(grid_patch);
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (layout_patch.owned_box.length(dir) != grid_patch.cells[dir]) {
               amrex::Print() << "mixed-patch logical-layout shape check failed for "
                              << label
                              << " patch=" << layout_patch.patch_id
                              << " dir=" << dir
                              << " layout_length=" << layout_patch.owned_box.length(dir)
                              << " grid_cells=" << grid_patch.cells[dir] << "\n";
               return_code = 1;
            }
         }

         for (amrex::BoxIterator iterator(layout_patch.owned_box); iterator.ok(); ++iterator) {
            const amrex::IntVect& layout_cell = iterator();
            const amrex::IntVect local_cell =
               pyfluid_amrex::patch_layout_cell_to_patch_local_cell(layout_patch, layout_cell);
            if (!pyfluid_amrex::patch_grid_index_box(grid_patch).contains(local_cell)) {
               amrex::Print() << "mixed-patch logical-layout local-cell check failed for "
                              << label
                              << " patch=" << layout_patch.patch_id << "\n";
               return_code = 1;
               return;
            }
            const pyfluid_amrex::InitialDataSamplePoint layout_sample =
               pyfluid_amrex::patch_layout_cell_center_initial_data_sample_point(
                  grid_patch,
                  layout_patch,
                  layout_cell
               );
            const pyfluid_amrex::InitialDataSamplePoint grid_sample =
               pyfluid_amrex::patch_grid_cell_center_initial_data_sample_point(grid_patch, local_cell);
            const amrex::Real mesh_error =
               pyfluid_amrex::patch_coordinate_max_abs_difference(layout_sample.mesh, grid_sample.mesh);
            const amrex::Real physical_error =
               pyfluid_amrex::patch_coordinate_max_abs_difference(layout_sample.physical, grid_sample.physical);
            const int owner_patch =
               pyfluid_amrex::patch_system_owner_patch_id(grid_system.map_system, layout_sample.physical);
            if (
               mesh_error > 1.0e-14
               || physical_error > 1.0e-14
               || !check_owned_or_context_sample(grid_system, grid_patch, owner_patch)
            ) {
               amrex::Print() << "mixed-patch logical-layout sampling check failed for "
                              << label
                              << " patch=" << layout_patch.patch_id
                              << " owner=" << owner_patch
                              << " mesh_error=" << mesh_error
                              << " physical_error=" << physical_error << "\n";
               return_code = 1;
               return;
            }
         }
      }
      if (layout_points != grid_points) {
         amrex::Print() << "mixed-patch logical-layout cell-count check failed for "
                        << label
                        << " layout_points=" << layout_points
                        << " grid_points=" << grid_points << "\n";
         return_code = 1;
      }
   };

   check_grid_system(
      pyfluid_amrex::make_thornburg04_patch_grid_system(1.0, 4.0, resolution, origin),
      "thornburg04"
   );
   check_grid_system(
      pyfluid_amrex::make_thornburg13_patch_grid_system(1.0, 2.5, 5.0, resolution, origin),
      "thornburg13"
   );
#if (AMREX_SPACEDIM == 2)
   check_grid_system(
      pyfluid_amrex::make_thornburg04_patch_grid_system(1.0, 4.0, resolution, origin, xz_patch_cartesian_axes()),
      "thornburg04_xz"
   );
   check_grid_system(
      pyfluid_amrex::make_thornburg13_patch_grid_system(1.0, 2.5, 5.0, resolution, origin, xz_patch_cartesian_axes()),
      "thornburg13_xz"
   );
#endif
   check_grid_system(
      pyfluid_amrex::make_cylindrical5_patch_grid_system(1.0, 4.0, resolution, origin),
      "cylindrical5"
   );
   check_grid_system(
      pyfluid_amrex::make_cylindrical9_patch_grid_system(1.0, 2.5, 5.0, resolution, origin),
      "cylindrical9"
   );

   if (return_code == 0) {
      amrex::Print() << "mixed-patch logical-layout checks passed\n";
   }
   return return_code;
}

void fill_valid_cells(amrex::MultiFab& data)
{
   data.setVal(-1.0e200);
   for (amrex::MFIter mfi(data); mfi.isValid(); ++mfi) {
      amrex::FArrayBox& fab = data[mfi];
      for (amrex::BoxIterator iterator(mfi.validbox()); iterator.ok(); ++iterator) {
         const amrex::IntVect& cell = iterator();
         for (int component = 0; component < NumComponents; ++component) {
            fab(cell, component) = analytic_value(cell, component);
         }
      }
   }
}

amrex::Array<amrex::Real, AMREX_SPACEDIM> domain_center(amrex::Box const& box)
{
   amrex::Array<amrex::Real, AMREX_SPACEDIM> center{};
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      center[dir] = 0.5 * static_cast<amrex::Real>(box.smallEnd(dir) + box.bigEnd(dir));
   }
   return center;
}

amrex::Real wave_packet_value_at_position(
   amrex::Array<amrex::Real, AMREX_SPACEDIM> const& position,
   int component,
   amrex::Array<amrex::Real, AMREX_SPACEDIM> const& center,
   amrex::Real width
)
{
   amrex::Real radius_squared = 0.0;
   amrex::Real phase = 0.17 * static_cast<amrex::Real>(component + 1);
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const amrex::Real distance = position[dir] - center[dir];
      radius_squared += distance * distance;
      phase += 0.11 * static_cast<amrex::Real>(dir + 1) * (position[dir] + 1.0);
   }
   const amrex::Real packet = std::exp(-radius_squared / (2.0 * width * width));
   return 1.0 + 0.08 * static_cast<amrex::Real>(component + 1) * packet * (1.0 + 0.1 * std::sin(phase));
}

amrex::Real wave_packet_value(
   amrex::IntVect const& cell,
   int component,
   amrex::Array<amrex::Real, AMREX_SPACEDIM> const& center,
   amrex::Real width
)
{
   amrex::Array<amrex::Real, AMREX_SPACEDIM> position{};
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      position[dir] = static_cast<amrex::Real>(cell[dir]);
   }
   return wave_packet_value_at_position(position, component, center, width);
}

void fill_wave_packet_scaled(
   amrex::MultiFab& data,
   amrex::Array<amrex::Real, AMREX_SPACEDIM> const& center,
   amrex::Real width,
   int refinement_ratio
)
{
   data.setVal(-1.0e200);
   for (amrex::MFIter mfi(data); mfi.isValid(); ++mfi) {
      amrex::FArrayBox& fab = data[mfi];
      for (amrex::BoxIterator iterator(mfi.validbox()); iterator.ok(); ++iterator) {
         const amrex::IntVect& cell = iterator();
         amrex::Array<amrex::Real, AMREX_SPACEDIM> position{};
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            position[dir] = (static_cast<amrex::Real>(cell[dir]) + 0.5)
                            / static_cast<amrex::Real>(refinement_ratio) - 0.5;
         }
         for (int component = 0; component < NumComponents; ++component) {
            fab(cell, component) = wave_packet_value_at_position(position, component, center, width);
         }
      }
   }
}

void fill_wave_packet(
   amrex::MultiFab& data,
   amrex::Array<amrex::Real, AMREX_SPACEDIM> const& center,
   amrex::Real width
)
{
   fill_wave_packet_scaled(data, center, width, 1);
}

amrex::Real max_error_on_box(amrex::MultiFab const& data, amrex::Box const& box)
{
   amrex::Real max_error = 0.0;
   for (amrex::MFIter mfi(data); mfi.isValid(); ++mfi) {
      amrex::FArrayBox const& fab = data[mfi];
      amrex::Box covered = box;
      covered &= fab.box();
      if (!covered.ok()) {
         continue;
      }
      for (amrex::BoxIterator iterator(covered); iterator.ok(); ++iterator) {
         const amrex::IntVect& cell = iterator();
         for (int component = 0; component < NumComponents; ++component) {
            const amrex::Real expected = analytic_value(cell, component);
            max_error = std::max(max_error, std::abs(fab(cell, component) - expected));
         }
      }
   }
   return max_error;
}

amrex::Real max_difference_on_box(
   amrex::MultiFab const& actual,
   amrex::MultiFab const& expected,
   amrex::Box const& box
)
{
   amrex::MultiFab expected_on_actual(actual.boxArray(), actual.DistributionMap(), actual.nComp(), 0);
   expected_on_actual.setVal(-1.0e200);
   expected_on_actual.ParallelCopy(expected, 0, 0, actual.nComp(), 0, 0);

   amrex::Real max_error = 0.0;
   for (amrex::MFIter actual_mfi(actual); actual_mfi.isValid(); ++actual_mfi) {
      amrex::FArrayBox const& actual_fab = actual[actual_mfi];
      amrex::FArrayBox const& expected_fab = expected_on_actual[actual_mfi];
      amrex::Box covered = box;
      covered &= actual_fab.box();
      if (!covered.ok()) {
         continue;
      }
      for (amrex::BoxIterator iterator(covered); iterator.ok(); ++iterator) {
         const amrex::IntVect& cell = iterator();
         for (int component = 0; component < NumComponents; ++component) {
            max_error = std::max(max_error, std::abs(actual_fab(cell, component) - expected_fab(cell, component)));
         }
      }
   }
   return max_error;
}

amrex::MultiFab make_patch_data(amrex::Box const& patch_box)
{
   amrex::BoxArray boxes(patch_box);
   amrex::DistributionMapping distribution(boxes);
   return amrex::MultiFab(boxes, distribution, NumComponents, NumGhost);
}

amrex::MultiFab make_patch_data(amrex::BoxArray const& boxes, int ghost_cells)
{
   amrex::DistributionMapping distribution(boxes);
   return amrex::MultiFab(boxes, distribution, NumComponents, ghost_cells);
}

amrex::MultiFab make_output_data_like(amrex::MultiFab const& state)
{
   return amrex::MultiFab(state.boxArray(), state.DistributionMap(), NumComponents, 0);
}

amrex::MultiFab make_state_data_like(amrex::MultiFab const& state)
{
   return amrex::MultiFab(state.boxArray(), state.DistributionMap(), NumComponents, state.nGrowVect());
}

void fill_fine_boundary_from_coarse(
   amrex::MultiFab& fine_state,
   amrex::Box const& owned_fine_box,
   amrex::MultiFab const& coarse_state,
   amrex::IntVect const& ratio
)
{
   const amrex::Box coarse_domain = coarse_state.boxArray().minimalBox();
   amrex::BoxArray coarse_footprint = fine_state.boxArray();
   coarse_footprint.grow(fine_state.nGrowVect());
   coarse_footprint.coarsen(ratio);
   amrex::MultiFab coarse_on_fine(coarse_footprint, fine_state.DistributionMap(), NumComponents, 0);
   coarse_on_fine.setVal(-1.0e200);
   coarse_on_fine.ParallelCopy(coarse_state, 0, 0, NumComponents, 0, 0);

   for (amrex::MFIter mfi(fine_state); mfi.isValid(); ++mfi) {
      amrex::FArrayBox& fine_fab = fine_state[mfi];
      amrex::FArrayBox const& coarse_fab = coarse_on_fine[mfi];
      for (amrex::BoxIterator iterator(fine_fab.box()); iterator.ok(); ++iterator) {
         const amrex::IntVect& fine_cell = iterator();
         if (owned_fine_box.contains(fine_cell)) {
            continue;
         }
         const amrex::IntVect coarse_cell = amrex::coarsen(fine_cell, ratio);
         if (!coarse_domain.contains(coarse_cell)) {
            continue;
         }
         for (int component = 0; component < NumComponents; ++component) {
            fine_fab(fine_cell, component) = coarse_fab(coarse_cell, component);
         }
      }
   }
}

amrex::MultiFab make_owned_coarse_snapshot(
   pyfluid_amrex::CartesianMultipatchLayout const& coarse_layout,
   amrex::Vector<amrex::MultiFab> const& patch_state
)
{
   amrex::BoxArray boxes = pyfluid_amrex::cartesian_patch_box_array(coarse_layout);
   amrex::DistributionMapping distribution(boxes);
   amrex::MultiFab snapshot(boxes, distribution, NumComponents, 0);
   snapshot.setVal(-1.0e200);
   for (int patch_index = 0; patch_index < static_cast<int>(coarse_layout.patches.size()); ++patch_index) {
      snapshot.ParallelCopy(patch_state[patch_index], 0, 0, NumComponents, 0, 0);
   }
   return snapshot;
}

void fill_fine_boundary_from_coarse(
   amrex::MultiFab& fine_state,
   amrex::Box const& owned_fine_box,
   pyfluid_amrex::CartesianMultipatchLayout const& coarse_layout,
   amrex::Vector<amrex::MultiFab> const& coarse_patch_state,
   amrex::IntVect const& ratio
)
{
   amrex::MultiFab coarse_snapshot = make_owned_coarse_snapshot(coarse_layout, coarse_patch_state);
   fill_fine_boundary_from_coarse(fine_state, owned_fine_box, coarse_snapshot, ratio);
}

amrex::MultiFab make_owned_copy(amrex::MultiFab const& source, amrex::Box const& owned_box)
{
   amrex::BoxArray boxes(owned_box);
   amrex::DistributionMapping distribution(boxes);
   amrex::MultiFab owned_copy(boxes, distribution, NumComponents, 0);
   owned_copy.ParallelCopy(source, 0, 0, NumComponents, 0, 0);
   return owned_copy;
}

void finite_volume_reference_update(
   amrex::MultiFab& updated,
   amrex::MultiFab const& state,
   amrex::Box const& update_region
)
{
   updated.setVal(-1.0e200);
   for (amrex::MFIter mfi(updated); mfi.isValid(); ++mfi) {
      amrex::FArrayBox& updated_fab = updated[mfi];
      amrex::FArrayBox const& state_fab = state[mfi];
      amrex::Box box = mfi.validbox();
      box &= update_region;
      if (!box.ok()) {
         continue;
      }
      for (amrex::BoxIterator iterator(box); iterator.ok(); ++iterator) {
         const amrex::IntVect& cell = iterator();
         for (int component = 0; component < NumComponents; ++component) {
            amrex::Real flux_difference = 0.0;
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
               amrex::IntVect lower_cell = cell;
               amrex::IntVect upper_cell = cell;
               lower_cell[dir] -= 1;
               upper_cell[dir] += 1;
               const amrex::Real wave_fraction = 0.05 + 0.01 * static_cast<amrex::Real>(dir + component);
               const amrex::Real flux_lower = 0.5 * (state_fab(lower_cell, component) + state_fab(cell, component));
               const amrex::Real flux_upper = 0.5 * (state_fab(cell, component) + state_fab(upper_cell, component));
               flux_difference += wave_fraction * (flux_upper - flux_lower);
            }
            updated_fab(cell, component) = state_fab(cell, component) - flux_difference;
         }
      }
   }
}

void upwind_advection_update(
   amrex::MultiFab& updated,
   amrex::MultiFab const& state,
   amrex::Box const& update_region,
   amrex::Array<amrex::Real, AMREX_SPACEDIM> const& courant_numbers
)
{
   updated.ParallelCopy(state, 0, 0, NumComponents, 0, 0);
   for (amrex::MFIter mfi(updated); mfi.isValid(); ++mfi) {
      amrex::FArrayBox& updated_fab = updated[mfi];
      amrex::FArrayBox const& state_fab = state[mfi];
      amrex::Box box = mfi.validbox();
      box &= update_region;
      if (!box.ok()) {
         continue;
      }
      for (amrex::BoxIterator iterator(box); iterator.ok(); ++iterator) {
         const amrex::IntVect& cell = iterator();
         for (int component = 0; component < NumComponents; ++component) {
            amrex::Real value = state_fab(cell, component);
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
               const amrex::Real courant_number = courant_numbers[dir];
               if (courant_number == 0.0) {
                  continue;
               }
               amrex::IntVect upwind_cell = cell;
               if (courant_number > 0.0) {
                  upwind_cell[dir] -= 1;
                  value -= courant_number * (state_fab(cell, component) - state_fab(upwind_cell, component));
               } else {
                  upwind_cell[dir] += 1;
                  value -= courant_number * (state_fab(upwind_cell, component) - state_fab(cell, component));
               }
            }
            updated_fab(cell, component) = value;
         }
      }
   }
}

int run_layout_ghost_check(pyfluid_amrex::CartesianMultipatchLayout const& layout, std::string const& label)
{
   amrex::Vector<amrex::MultiFab> patch_storage;
   patch_storage.reserve(layout.patches.size());
   amrex::Vector<amrex::MultiFab*> patch_data;
   for (auto const& patch : layout.patches) {
      patch_storage.push_back(make_patch_data(patch.patch_box));
      fill_valid_cells(patch_storage.back());
      patch_data.push_back(&patch_storage.back());
   }

   pyfluid_amrex::fill_aligned_cartesian_neighbor_ghosts(
      patch_data,
      layout,
      0,
      0,
      NumComponents,
      amrex::IntVect(NumGhost)
   );

   amrex::Real local_max_error = 0.0;
   for (int destination_index = 0; destination_index < static_cast<int>(layout.patches.size()); ++destination_index) {
      for (int source_index = 0; source_index < static_cast<int>(layout.patches.size()); ++source_index) {
         if (source_index == destination_index) {
            continue;
         }
         const amrex::Box overlap = pyfluid_amrex::cartesian_neighbor_overlap_box(
            layout.patches[destination_index],
            layout.patches[source_index],
            amrex::IntVect(NumGhost)
         );
         if (overlap.ok()) {
            local_max_error = std::max(local_max_error, max_error_on_box(patch_storage[destination_index], overlap));
         }
      }
   }

   amrex::ParallelDescriptor::ReduceRealMax(local_max_error);
   if (local_max_error != 0.0) {
      amrex::Print() << "multipatch " << label << " ghost check failed max_error="
                     << local_max_error << "\n";
      return 1;
   }
   amrex::Print() << "multipatch " << label << " ghost check passed\n";
   return 0;
}

int run_layout_update_check(pyfluid_amrex::CartesianMultipatchLayout const& layout, std::string const& label)
{
   amrex::Box update_region = layout.global_box;
   update_region.grow(-1);

   amrex::MultiFab single_state = make_patch_data(layout.global_box);
   fill_valid_cells(single_state);
   amrex::MultiFab single_updated = make_output_data_like(single_state);
   finite_volume_reference_update(single_updated, single_state, update_region);

   amrex::Vector<amrex::MultiFab> patch_state;
   amrex::Vector<amrex::MultiFab> patch_updated;
   patch_state.reserve(layout.patches.size());
   patch_updated.reserve(layout.patches.size());
   amrex::Vector<amrex::MultiFab*> patch_state_ptrs;
   for (auto const& patch : layout.patches) {
      patch_state.push_back(make_patch_data(patch.patch_box));
      patch_updated.push_back(make_output_data_like(patch_state.back()));
      fill_valid_cells(patch_state.back());
      patch_state_ptrs.push_back(&patch_state.back());
   }

   pyfluid_amrex::fill_aligned_cartesian_neighbor_ghosts(
      patch_state_ptrs,
      layout,
      0,
      0,
      NumComponents,
      amrex::IntVect(NumGhost)
   );
   for (int patch_index = 0; patch_index < static_cast<int>(layout.patches.size()); ++patch_index) {
      amrex::Box patch_update_region = layout.patches[patch_index].owned_box;
      patch_update_region &= update_region;
      finite_volume_reference_update(patch_updated[patch_index], patch_state[patch_index], patch_update_region);
   }

   amrex::Real local_max_error = 0.0;
   for (int patch_index = 0; patch_index < static_cast<int>(layout.patches.size()); ++patch_index) {
      amrex::Box compare_box = layout.patches[patch_index].owned_box;
      compare_box &= update_region;
      if (compare_box.ok()) {
         local_max_error = std::max(
            local_max_error,
            max_difference_on_box(patch_updated[patch_index], single_updated, compare_box)
         );
      }
   }
   amrex::ParallelDescriptor::ReduceRealMax(local_max_error);
   if (local_max_error != 0.0) {
      amrex::Print() << "multipatch " << label << " update check failed max_error="
                     << local_max_error << "\n";
      return 1;
   }
   amrex::Print() << "multipatch " << label << " update check passed\n";
   return 0;
}

int run_layout_checks(pyfluid_amrex::CartesianMultipatchLayout const& layout, std::string const& label)
{
   int return_code = 0;
   return_code = std::max(return_code, run_layout_ghost_check(layout, label));
   return_code = std::max(return_code, run_layout_update_check(layout, label));
   return return_code;
}

int run_wave_packet_crossing_check(
   pyfluid_amrex::CartesianMultipatchLayout const& layout,
   std::string const& label,
   amrex::Array<amrex::Real, AMREX_SPACEDIM> const& initial_center,
   amrex::Array<amrex::Real, AMREX_SPACEDIM> const& courant_numbers,
   int step_count
)
{
   constexpr amrex::Real packet_width = 2.5;

   amrex::Box update_region = layout.global_box;
   update_region.grow(-1);

   amrex::MultiFab single_state = make_patch_data(layout.global_box);
   amrex::MultiFab single_next = make_state_data_like(single_state);
   fill_wave_packet(single_state, initial_center, packet_width);

   amrex::Vector<amrex::MultiFab> patch_state;
   amrex::Vector<amrex::MultiFab> patch_next;
   patch_state.reserve(layout.patches.size());
   patch_next.reserve(layout.patches.size());
   amrex::Vector<amrex::MultiFab*> patch_state_ptrs;
   for (auto const& patch : layout.patches) {
      patch_state.push_back(make_patch_data(patch.patch_box));
      patch_next.push_back(make_state_data_like(patch_state.back()));
      fill_wave_packet(patch_state.back(), initial_center, packet_width);
      patch_state_ptrs.push_back(&patch_state.back());
   }

   for (int step = 0; step < step_count; ++step) {
      upwind_advection_update(single_next, single_state, update_region, courant_numbers);
      single_state.ParallelCopy(single_next, 0, 0, NumComponents, 0, 0);

      pyfluid_amrex::fill_aligned_cartesian_neighbor_ghosts(
         patch_state_ptrs,
         layout,
         0,
         0,
         NumComponents,
         amrex::IntVect(NumGhost)
      );
      for (int patch_index = 0; patch_index < static_cast<int>(layout.patches.size()); ++patch_index) {
         amrex::Box patch_update_region = layout.patches[patch_index].owned_box;
         patch_update_region &= update_region;
         upwind_advection_update(patch_next[patch_index], patch_state[patch_index], patch_update_region, courant_numbers);
      }
      for (int patch_index = 0; patch_index < static_cast<int>(layout.patches.size()); ++patch_index) {
         patch_state[patch_index].ParallelCopy(patch_next[patch_index], 0, 0, NumComponents, 0, 0);
      }
   }

   amrex::Real local_max_error = 0.0;
   for (int patch_index = 0; patch_index < static_cast<int>(layout.patches.size()); ++patch_index) {
      local_max_error = std::max(
         local_max_error,
         max_difference_on_box(patch_state[patch_index], single_state, layout.patches[patch_index].owned_box)
      );
   }
   amrex::ParallelDescriptor::ReduceRealMax(local_max_error);
   if (local_max_error > 1.0e-13) {
      amrex::Print() << "multipatch " << label << " wave-packet crossing check failed max_error="
                     << local_max_error << "\n";
      return 1;
   }
   amrex::Print() << "multipatch " << label << " wave-packet crossing check passed max_error="
                  << local_max_error << "\n";
   return 0;
}

amrex::Box check_global_box()
{
   amrex::IntVect small(AMREX_D_DECL(0, 0, 0));
   amrex::IntVect big(AMREX_D_DECL(63, 31, 15));
#if (AMREX_SPACEDIM == 2)
   big[1] = 47;
#endif
   return amrex::Box(small, big);
}

int run_aligned_split_checks(int split_direction)
{
   const auto layout = pyfluid_amrex::make_centered_cartesian_split_layout(check_global_box(), split_direction);
   int return_code = run_layout_checks(layout, "aligned two-patch split direction " + std::to_string(split_direction));

   amrex::Array<amrex::Real, AMREX_SPACEDIM> center = domain_center(layout.global_box);
   center[split_direction] = static_cast<amrex::Real>(layout.patches[1].patch_box.smallEnd(split_direction)) - 3.0;
   amrex::Array<amrex::Real, AMREX_SPACEDIM> courant_numbers{};
   courant_numbers.fill(0.0);
   courant_numbers[split_direction] = 0.45;
   return_code = std::max(
      return_code,
      run_wave_packet_crossing_check(
         layout,
         "aligned two-patch split direction " + std::to_string(split_direction),
         center,
         courant_numbers,
         14
      )
   );
   return return_code;
}

int run_overlapping_split_checks(int split_direction)
{
   const auto layout = pyfluid_amrex::make_overlapping_cartesian_split_layout(
      check_global_box(),
      split_direction,
      check_global_box().smallEnd(split_direction) + check_global_box().length(split_direction) / 2,
      NumGhost
   );
   const std::string label = "overlapping two-patch split direction " + std::to_string(split_direction);
   int return_code = run_layout_checks(layout, label);

   amrex::Array<amrex::Real, AMREX_SPACEDIM> center = domain_center(layout.global_box);
   center[split_direction] = static_cast<amrex::Real>(layout.patches[1].owned_box.smallEnd(split_direction)) - 3.0;
   amrex::Array<amrex::Real, AMREX_SPACEDIM> courant_numbers{};
   courant_numbers.fill(0.0);
   courant_numbers[split_direction] = 0.45;
   return_code = std::max(
      return_code,
      run_wave_packet_crossing_check(layout, label, center, courant_numbers, 14)
   );
   return return_code;
}

int run_aligned_tiled_checks()
{
   amrex::IntVect patch_counts(AMREX_D_DECL(1, 1, 1));
   patch_counts[0] = 2;
   patch_counts[1] = 2;
   const auto layout = pyfluid_amrex::make_aligned_cartesian_tiled_layout(check_global_box(), patch_counts);
   int return_code = run_layout_checks(layout, "aligned four-patch tile");

   amrex::Array<amrex::Real, AMREX_SPACEDIM> center = domain_center(layout.global_box);
   center[0] = static_cast<amrex::Real>(layout.global_box.smallEnd(0) + layout.global_box.length(0) / 2) - 3.0;
   center[1] = static_cast<amrex::Real>(layout.global_box.smallEnd(1) + layout.global_box.length(1) / 2) - 3.0;
   amrex::Array<amrex::Real, AMREX_SPACEDIM> courant_numbers{};
   courant_numbers.fill(0.0);
   courant_numbers[0] = 0.30;
   courant_numbers[1] = 0.25;
   return_code = std::max(
      return_code,
      run_wave_packet_crossing_check(layout, "aligned four-patch tile", center, courant_numbers, 14)
   );
   return return_code;
}

pyfluid_amrex::CartesianMultipatchLayout make_t_junction_layout()
{
   const amrex::Box global_box = check_global_box();
   const int split_x = global_box.smallEnd(0) + global_box.length(0) / 2;
   const int split_y = global_box.smallEnd(1) + global_box.length(1) / 2;

   amrex::Box left_patch = global_box;
   left_patch.setBig(0, split_x - 1);

   amrex::Box right_low_patch = global_box;
   right_low_patch.setSmall(0, split_x);
   right_low_patch.setBig(1, split_y - 1);

   amrex::Box right_high_patch = global_box;
   right_high_patch.setSmall(0, split_x);
   right_high_patch.setSmall(1, split_y);

   return pyfluid_amrex::make_cartesian_layout_from_boxes(
      global_box,
      amrex::Vector<amrex::Box>{left_patch, right_low_patch, right_high_patch}
   );
}

int run_t_junction_checks()
{
   const auto layout = make_t_junction_layout();
   int return_code = run_layout_checks(layout, "three-patch T-junction");

   amrex::Array<amrex::Real, AMREX_SPACEDIM> center = domain_center(layout.global_box);
   center[0] = static_cast<amrex::Real>(layout.patches[1].patch_box.smallEnd(0)) - 3.0;
   center[1] = static_cast<amrex::Real>(layout.patches[2].patch_box.smallEnd(1)) - 3.0;
   amrex::Array<amrex::Real, AMREX_SPACEDIM> courant_numbers{};
   courant_numbers.fill(0.0);
   courant_numbers[0] = 0.30;
   courant_numbers[1] = 0.25;
   return_code = std::max(
      return_code,
      run_wave_packet_crossing_check(layout, "three-patch T-junction", center, courant_numbers, 14)
   );
   return return_code;
}

amrex::Geometry make_periodic_geometry(amrex::Box const& domain)
{
   amrex::RealBox real_box(AMREX_D_DECL(0.0, 0.0, 0.0), AMREX_D_DECL(1.0, 1.0, 1.0));
   amrex::Array<int, AMREX_SPACEDIM> periodic{AMREX_D_DECL(1, 1, 1)};
   return amrex::Geometry(domain, real_box, amrex::CoordSys::cartesian, periodic);
}

amrex::Vector<amrex::BCRec> periodic_bcs()
{
   amrex::Vector<amrex::BCRec> bcs(NumComponents);
   for (int component = 0; component < NumComponents; ++component) {
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         bcs[component].setLo(dir, amrex::BCType::int_dir);
         bcs[component].setHi(dir, amrex::BCType::int_dir);
      }
   }
   return bcs;
}

amrex::BoxArray refine_box_array(amrex::BoxArray const& boxes, amrex::IntVect const& ratio)
{
   amrex::BoxArray refined = boxes;
   refined.refine(ratio);
   return refined;
}

int run_amr_transfer_check(
   pyfluid_amrex::CartesianMultipatchLayout const& coarse_layout,
   amrex::Vector<int> const& refined_patch_ids,
   std::string const& label
)
{
   const amrex::IntVect ratio(RefRatio);
   const amrex::BoxArray coarse_boxes = pyfluid_amrex::cartesian_patch_box_array(coarse_layout);
   const amrex::BoxArray selected_coarse_boxes =
      pyfluid_amrex::cartesian_selected_patch_box_array(coarse_layout, refined_patch_ids);
   const amrex::BoxArray fine_boxes = refine_box_array(selected_coarse_boxes, ratio);

   amrex::MultiFab coarse_state = make_patch_data(coarse_boxes, 0);
   fill_valid_cells(coarse_state);
   amrex::MultiFab fine_state = make_patch_data(fine_boxes, 0);
   fine_state.setVal(-1.0e200);

   const amrex::Box fine_global_box = amrex::refine(coarse_layout.global_box, ratio);
   const amrex::Geometry coarse_geometry = make_periodic_geometry(coarse_layout.global_box);
   const amrex::Geometry fine_geometry = make_periodic_geometry(fine_global_box);
   auto bcs = periodic_bcs();
   amrex::GpuBndryFuncFab<amrex::FabFillNoOp> boundary_func;
   amrex::PhysBCFunct<amrex::GpuBndryFuncFab<amrex::FabFillNoOp>> coarse_bc(coarse_geometry, bcs, boundary_func);
   amrex::PhysBCFunct<amrex::GpuBndryFuncFab<amrex::FabFillNoOp>> fine_bc(fine_geometry, bcs, boundary_func);

   amrex::InterpFromCoarseLevel(
      fine_state,
      amrex::IntVect::TheZeroVector(),
      0.0,
      coarse_state,
      0,
      0,
      NumComponents,
      coarse_geometry,
      fine_geometry,
      coarse_bc,
      0,
      fine_bc,
      0,
      ratio,
      &amrex::cell_cons_interp,
      bcs,
      0
   );

   amrex::MultiFab restricted_state = make_patch_data(selected_coarse_boxes, 0);
   restricted_state.setVal(-1.0e200);
   amrex::average_down(fine_state, restricted_state, fine_geometry, coarse_geometry, 0, NumComponents, ratio);

   amrex::Real local_max_error = 0.0;
   for (int patch_id : refined_patch_ids) {
      local_max_error = std::max(
         local_max_error,
         max_difference_on_box(restricted_state, coarse_state, coarse_layout.patches[patch_id].patch_box)
      );
   }
   amrex::ParallelDescriptor::ReduceRealMax(local_max_error);
   if (local_max_error > 1.0e-13) {
      amrex::Print() << "multipatch " << label << " AMR transfer check failed max_error="
                     << local_max_error << "\n";
      return 1;
   }
   amrex::Print() << "multipatch " << label << " AMR transfer check passed max_error="
                  << local_max_error << "\n";
   return 0;
}

int run_amr_packet_evolution_check(
   pyfluid_amrex::CartesianMultipatchLayout const& coarse_layout,
   int refined_patch_id,
   std::string const& label,
   int advection_direction = 0,
   bool start_inside_refined_patch = false
)
{
   constexpr int step_count = 10;
   constexpr amrex::Real packet_width = 3.0;
   const amrex::IntVect ratio(RefRatio);
   const pyfluid_amrex::CartesianPatchDescriptor& refined_patch = coarse_layout.patches[refined_patch_id];
   if (advection_direction < 0 || advection_direction >= AMREX_SPACEDIM) {
      amrex::Abort("mixed-resolution multipatch packet test direction is outside the executable dimension");
   }
   const amrex::Box owned_fine_box = amrex::refine(refined_patch.owned_box, ratio);
   const amrex::Box patch_fine_box = amrex::refine(refined_patch.patch_box, ratio);

   amrex::Box coarse_update_region = coarse_layout.global_box;
   coarse_update_region.grow(-1);
   amrex::Box fine_update_region = owned_fine_box;
   fine_update_region &= amrex::refine(coarse_update_region, ratio);

   amrex::Array<amrex::Real, AMREX_SPACEDIM> center = domain_center(coarse_layout.global_box);
   center[advection_direction] = start_inside_refined_patch
      ? static_cast<amrex::Real>(refined_patch.owned_box.smallEnd(advection_direction)) + 4.0
      : static_cast<amrex::Real>(refined_patch.owned_box.smallEnd(advection_direction)) - 4.0;
   amrex::Array<amrex::Real, AMREX_SPACEDIM> coarse_courant_numbers{};
   coarse_courant_numbers.fill(0.0);
   coarse_courant_numbers[advection_direction] = start_inside_refined_patch ? -0.35 : 0.35;
   amrex::Array<amrex::Real, AMREX_SPACEDIM> fine_courant_numbers = coarse_courant_numbers;

   amrex::MultiFab reference_coarse = make_patch_data(coarse_layout.global_box);
   amrex::MultiFab reference_coarse_next = make_state_data_like(reference_coarse);
   fill_wave_packet(reference_coarse, center, packet_width);

   amrex::MultiFab reference_fine = make_patch_data(amrex::BoxArray(owned_fine_box), NumGhost);
   amrex::MultiFab reference_fine_next = make_state_data_like(reference_fine);
   fill_wave_packet_scaled(reference_fine, center, packet_width, RefRatio);

   amrex::Vector<amrex::MultiFab> patch_coarse;
   amrex::Vector<amrex::MultiFab> patch_coarse_next;
   patch_coarse.reserve(coarse_layout.patches.size());
   patch_coarse_next.reserve(coarse_layout.patches.size());
   amrex::Vector<amrex::MultiFab*> patch_coarse_ptrs;
   for (auto const& patch : coarse_layout.patches) {
      patch_coarse.push_back(make_patch_data(patch.patch_box));
      patch_coarse_next.push_back(make_state_data_like(patch_coarse.back()));
      fill_wave_packet(patch_coarse.back(), center, packet_width);
      patch_coarse_ptrs.push_back(&patch_coarse.back());
   }

   amrex::MultiFab patch_fine = make_patch_data(patch_fine_box);
   amrex::MultiFab patch_fine_next = make_state_data_like(patch_fine);
   fill_wave_packet_scaled(patch_fine, center, packet_width, RefRatio);

   const amrex::Box fine_global_box = amrex::refine(coarse_layout.global_box, ratio);
   const amrex::Geometry coarse_geometry = make_periodic_geometry(coarse_layout.global_box);
   const amrex::Geometry fine_geometry = make_periodic_geometry(fine_global_box);

   for (int step = 0; step < step_count; ++step) {
      fill_fine_boundary_from_coarse(reference_fine, owned_fine_box, reference_coarse, ratio);
      upwind_advection_update(reference_coarse_next, reference_coarse, coarse_update_region, coarse_courant_numbers);
      upwind_advection_update(reference_fine_next, reference_fine, fine_update_region, fine_courant_numbers);
      reference_coarse.ParallelCopy(reference_coarse_next, 0, 0, NumComponents, 0, 0);
      reference_fine.ParallelCopy(reference_fine_next, 0, 0, NumComponents, 0, 0);
      amrex::average_down(reference_fine, reference_coarse, fine_geometry, coarse_geometry, 0, NumComponents, ratio);

      pyfluid_amrex::fill_aligned_cartesian_neighbor_ghosts(
         patch_coarse_ptrs,
         coarse_layout,
         0,
         0,
         NumComponents,
         amrex::IntVect(NumGhost)
      );
      fill_fine_boundary_from_coarse(patch_fine, owned_fine_box, coarse_layout, patch_coarse, ratio);
      for (int patch_index = 0; patch_index < static_cast<int>(coarse_layout.patches.size()); ++patch_index) {
         amrex::Box patch_update_region = coarse_layout.patches[patch_index].owned_box;
         patch_update_region &= coarse_update_region;
         upwind_advection_update(
            patch_coarse_next[patch_index],
            patch_coarse[patch_index],
            patch_update_region,
            coarse_courant_numbers
         );
      }
      upwind_advection_update(patch_fine_next, patch_fine, fine_update_region, fine_courant_numbers);
      for (int patch_index = 0; patch_index < static_cast<int>(coarse_layout.patches.size()); ++patch_index) {
         patch_coarse[patch_index].ParallelCopy(patch_coarse_next[patch_index], 0, 0, NumComponents, 0, 0);
      }
      patch_fine.ParallelCopy(patch_fine_next, 0, 0, NumComponents, 0, 0);

      amrex::MultiFab owned_fine_for_average = make_owned_copy(patch_fine, owned_fine_box);
      amrex::average_down(
         owned_fine_for_average,
         patch_coarse[refined_patch_id],
         fine_geometry,
         coarse_geometry,
         0,
         NumComponents,
         ratio
      );
   }

   amrex::Real local_max_error = 0.0;
   for (int patch_index = 0; patch_index < static_cast<int>(coarse_layout.patches.size()); ++patch_index) {
      local_max_error = std::max(
         local_max_error,
         max_difference_on_box(patch_coarse[patch_index], reference_coarse, coarse_layout.patches[patch_index].owned_box)
      );
   }
   local_max_error = std::max(
      local_max_error,
      max_difference_on_box(patch_fine, reference_fine, owned_fine_box)
   );
   amrex::ParallelDescriptor::ReduceRealMax(local_max_error);
   if (local_max_error > 1.0e-13) {
      amrex::Print() << "multipatch " << label << " AMR packet evolution check failed max_error="
                     << local_max_error << "\n";
      return 1;
   }
   amrex::Print() << "multipatch " << label << " AMR packet evolution check passed max_error="
                  << local_max_error << "\n";
   return 0;
}

int run_amr_transfer_checks()
{
   const auto aligned_layout = pyfluid_amrex::make_aligned_cartesian_tiled_layout(
      check_global_box(),
      amrex::IntVect(AMREX_D_DECL(2, 2, 1))
   );
   int return_code = 0;
   return_code = std::max(
      return_code,
      run_amr_transfer_check(aligned_layout, amrex::Vector<int>{0}, "one refined patch")
   );

   amrex::Vector<int> all_patch_ids;
   all_patch_ids.reserve(aligned_layout.patches.size());
   for (int patch_id = 0; patch_id < static_cast<int>(aligned_layout.patches.size()); ++patch_id) {
      all_patch_ids.push_back(patch_id);
   }
   return_code = std::max(
      return_code,
      run_amr_transfer_check(aligned_layout, all_patch_ids, "all refined patches")
   );

   const auto overlapping_layout = pyfluid_amrex::make_overlapping_cartesian_split_layout(
      check_global_box(),
      0,
      check_global_box().smallEnd(0) + check_global_box().length(0) / 2,
      NumGhost
   );
   return_code = std::max(
      return_code,
      run_amr_transfer_check(overlapping_layout, amrex::Vector<int>{1}, "overlapping split refined high patch")
   );
   return_code = std::max(
      return_code,
      run_amr_packet_evolution_check(aligned_layout, 1, "aligned split refined high patch")
   );
   return_code = std::max(
      return_code,
      run_amr_packet_evolution_check(overlapping_layout, 1, "overlapping split refined high patch")
   );

   for (int split_direction = 0; split_direction < AMREX_SPACEDIM; ++split_direction) {
      const int split_index =
         check_global_box().smallEnd(split_direction) + check_global_box().length(split_direction) / 2;
      const auto aligned_split = pyfluid_amrex::make_aligned_cartesian_split_layout(
         check_global_box(),
         split_direction,
         split_index
      );
      const auto overlapping_split = pyfluid_amrex::make_overlapping_cartesian_split_layout(
         check_global_box(),
         split_direction,
         split_index,
         NumGhost
      );
      return_code = std::max(
         return_code,
         run_amr_packet_evolution_check(
            aligned_split,
            1,
            "aligned mixed-resolution split direction " + std::to_string(split_direction) + " coarse-to-fine",
            split_direction,
            false
         )
      );
      return_code = std::max(
         return_code,
         run_amr_packet_evolution_check(
            aligned_split,
            1,
            "aligned mixed-resolution split direction " + std::to_string(split_direction) + " fine-to-coarse",
            split_direction,
            true
         )
      );
      return_code = std::max(
         return_code,
         run_amr_packet_evolution_check(
            overlapping_split,
            1,
            "overlapping mixed-resolution split direction " + std::to_string(split_direction) + " coarse-to-fine",
            split_direction,
            false
         )
      );
      return_code = std::max(
         return_code,
         run_amr_packet_evolution_check(
            overlapping_split,
            1,
            "overlapping mixed-resolution split direction " + std::to_string(split_direction) + " fine-to-coarse",
            split_direction,
            true
         )
      );
   }

   const auto t_junction_layout = make_t_junction_layout();
   return_code = std::max(
      return_code,
      run_amr_transfer_check(t_junction_layout, amrex::Vector<int>{1}, "T-junction one refined branch patch")
   );
   return_code = std::max(
      return_code,
      run_amr_packet_evolution_check(
         t_junction_layout,
         1,
         "T-junction mixed-resolution branch coarse-to-fine",
         0,
         false
      )
   );
   return_code = std::max(
      return_code,
      run_amr_packet_evolution_check(
         t_junction_layout,
         1,
         "T-junction mixed-resolution branch fine-to-coarse",
         0,
         true
      )
   );
   return return_code;
}

} // namespace

int main(int argc, char* argv[])
{
   amrex::Initialize(argc, argv);
   int return_code = 0;
   {
#if (AMREX_SPACEDIM < 2)
      amrex::Abort("multipatch_check is intended for 2D and 3D executables");
#else
      return_code = std::max(return_code, run_patch_domain_face_checks());
      return_code = std::max(return_code, run_patch_coordinate_map_checks());
      return_code = std::max(return_code, run_patch_coordinate_round_trip_checks());
      return_code = std::max(return_code, run_patch_grid_builder_checks());
      return_code = std::max(return_code, run_mixed_patch_initial_data_sampling_checks());
      return_code = std::max(return_code, run_mixed_patch_logical_layout_checks());
      for (int split_direction = 0; split_direction < AMREX_SPACEDIM; ++split_direction) {
         return_code = std::max(return_code, run_aligned_split_checks(split_direction));
         return_code = std::max(return_code, run_overlapping_split_checks(split_direction));
      }
      return_code = std::max(return_code, run_aligned_tiled_checks());
      return_code = std::max(return_code, run_t_junction_checks());
      return_code = std::max(return_code, run_amr_transfer_checks());
#endif
   }
   amrex::Finalize();
   return return_code;
}
