#pragma once

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/patch_maps.hpp"

namespace pyfluid_amrex {

struct PatchLocalAxes {
   amrex::GpuArray<int, AMREX_SPACEDIM> component{};
};

struct PatchGridDescriptor {
   int patch_id = -1;
   PatchMap map{};
   PatchCoordinate local_lo{};
   PatchCoordinate local_hi{};
   PatchLocalAxes local_axes{};
   amrex::IntVect cells{AMREX_D_DECL(1, 1, 1)};
};

struct PatchGridSystem {
   PatchMapSystem map_system{};
   amrex::Vector<PatchGridDescriptor> patches;
};

struct PatchGridResolution {
   amrex::IntVect central_cells{AMREX_D_DECL(16, 16, 16)};
   int angular_cells = 16;
   int radial_cells = 16;
   int outer_radial_cells = 16;
   int axial_cells = 16;
   amrex::Real axial_lo = -1.0;
   amrex::Real axial_hi = 1.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchLocalAxes make_patch_local_axes(int c0, int c1 = 1, int c2 = 2) noexcept
{
   PatchLocalAxes axes{};
   axes.component[0] = c0;
#if (AMREX_SPACEDIM >= 2)
   axes.component[1] = c1;
#else
   amrex::ignore_unused(c1);
#endif
#if (AMREX_SPACEDIM == 3)
   axes.component[2] = c2;
#else
   amrex::ignore_unused(c2);
#endif
   return axes;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int patch_local_component_for_dir(PatchLocalAxes const& axes, int dir) noexcept
{
   return axes.component[dir];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real patch_grid_cell_size(PatchGridDescriptor const& patch, int dir) noexcept
{
   const int component = patch_local_component_for_dir(patch.local_axes, dir);
   return (
      patch_coordinate_component(patch.local_hi, component)
      - patch_coordinate_component(patch.local_lo, component)
   ) / static_cast<amrex::Real>(patch.cells[dir]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_grid_cell_center(PatchGridDescriptor const& patch, amrex::IntVect const& cell) noexcept
{
   PatchCoordinate local = make_patch_coordinate(
      0.5 * (patch.local_lo.q1 + patch.local_hi.q1),
      0.5 * (patch.local_lo.q2 + patch.local_hi.q2),
      0.5 * (patch.local_lo.q3 + patch.local_hi.q3)
   );
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const int component = patch_local_component_for_dir(patch.local_axes, dir);
      const amrex::Real coordinate = patch_coordinate_component(patch.local_lo, component)
         + (static_cast<amrex::Real>(cell[dir]) + 0.5) * patch_grid_cell_size(patch, dir);
      set_patch_coordinate_component(local, component, coordinate);
   }
   return local;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool patch_grid_contains_global_point(
   PatchGridDescriptor const& patch,
   PatchCoordinate const& global,
   amrex::Real tolerance = 0.0
) noexcept
{
   const PatchCoordinate local = patch_global_to_local(patch.map, global);
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const int component = patch_local_component_for_dir(patch.local_axes, dir);
      const amrex::Real coordinate = patch_coordinate_component(local, component);
      const amrex::Real lo = patch_coordinate_component(patch.local_lo, component);
      const amrex::Real hi = patch_coordinate_component(patch.local_hi, component);
      if (coordinate < lo - tolerance || coordinate > hi + tolerance) {
         return false;
      }
   }
   return true;
}

inline amrex::IntVect shell_cells(int angular_cells, int radial_cells)
{
#if (AMREX_SPACEDIM == 1)
   return amrex::IntVect(angular_cells);
#elif (AMREX_SPACEDIM == 2)
   return amrex::IntVect(angular_cells, radial_cells);
#else
   return amrex::IntVect(angular_cells, angular_cells, radial_cells);
#endif
}

inline amrex::IntVect cylindrical_shell_cells(int angular_cells, int axial_cells, int radial_cells)
{
#if (AMREX_SPACEDIM == 1)
   amrex::ignore_unused(axial_cells, radial_cells);
   return amrex::IntVect(angular_cells);
#elif (AMREX_SPACEDIM == 2)
   amrex::ignore_unused(axial_cells);
   return amrex::IntVect(angular_cells, radial_cells);
#else
   return amrex::IntVect(angular_cells, axial_cells, radial_cells);
#endif
}

inline PatchLocalAxes cylindrical_shell_local_axes()
{
#if (AMREX_SPACEDIM == 1)
   return make_patch_local_axes(2);
#elif (AMREX_SPACEDIM == 2)
   return make_patch_local_axes(0, 2);
#else
   return make_patch_local_axes(0, 1, 2);
#endif
}

inline PatchLocalAxes patch_local_axes_from_cartesian(PatchCartesianAxes const& axes)
{
#if (AMREX_SPACEDIM == 1)
   return make_patch_local_axes(patch_cartesian_axis(axes, 0));
#elif (AMREX_SPACEDIM == 2)
   return make_patch_local_axes(patch_cartesian_axis(axes, 0), patch_cartesian_axis(axes, 1));
#else
   return make_patch_local_axes(
      patch_cartesian_axis(axes, 0),
      patch_cartesian_axis(axes, 1),
      patch_cartesian_axis(axes, 2)
   );
#endif
}

inline PatchCoordinate patch_coordinate_with_active_bounds(
   PatchCartesianAxes const& axes,
   amrex::Real value0,
   amrex::Real value1 = 0.0,
   amrex::Real value2 = 0.0
)
{
   PatchCoordinate coordinate = make_patch_coordinate(0.0, 0.0, 0.0);
   set_patch_coordinate_component(coordinate, patch_cartesian_axis(axes, 0), value0);
#if (AMREX_SPACEDIM >= 2)
   set_patch_coordinate_component(coordinate, patch_cartesian_axis(axes, 1), value1);
#else
   amrex::ignore_unused(value1);
#endif
#if (AMREX_SPACEDIM == 3)
   set_patch_coordinate_component(coordinate, patch_cartesian_axis(axes, 2), value2);
#else
   amrex::ignore_unused(value2);
#endif
   return coordinate;
}

inline PatchCoordinate spherical_central_local_bound(
   PatchCartesianAxes const& axes,
   amrex::Real radius,
   amrex::Real sign
)
{
#if (AMREX_SPACEDIM == 1)
   return patch_coordinate_with_active_bounds(axes, sign * radius);
#elif (AMREX_SPACEDIM == 2)
   return patch_coordinate_with_active_bounds(axes, sign * radius, sign * radius);
#else
   return patch_coordinate_with_active_bounds(axes, sign * radius, sign * radius, sign * radius);
#endif
}

inline PatchCoordinate cylindrical_central_local_bound(
   PatchCartesianAxes const& axes,
   amrex::Real radius,
   amrex::Real axial,
   amrex::Real sign
)
{
#if (AMREX_SPACEDIM == 1)
   amrex::ignore_unused(axial);
   return patch_coordinate_with_active_bounds(axes, sign * radius);
#elif (AMREX_SPACEDIM == 2)
   amrex::ignore_unused(axial);
   return patch_coordinate_with_active_bounds(axes, sign * radius, sign * radius);
#else
   return patch_coordinate_with_active_bounds(axes, sign * radius, sign * radius, axial);
#endif
}

inline int thornburg_tangent_component_for_plane(int primary_axis, int other_axis)
{
   if (primary_axis == 0) {
      return other_axis == 2 ? 0 : 1;
   }
   if (primary_axis == 1) {
      return other_axis == 2 ? 0 : 1;
   }
   return other_axis == 1 ? 0 : 1;
}

inline PatchLocalAxes spherical_shell_local_axes(
   PatchMap const& map,
   PatchCartesianAxes const& axes
)
{
#if (AMREX_SPACEDIM == 1)
   amrex::ignore_unused(map, axes);
   return make_patch_local_axes(2);
#elif (AMREX_SPACEDIM == 2)
   const int primary_axis = patch_shell_primary_axis(map.coordinate_map);
   const int axis0 = patch_cartesian_axis(axes, 0);
   const int axis1 = patch_cartesian_axis(axes, 1);
   const int other_axis = primary_axis == axis0 ? axis1 : axis0;
   return make_patch_local_axes(thornburg_tangent_component_for_plane(primary_axis, other_axis), 2);
#else
   amrex::ignore_unused(map, axes);
   return make_patch_local_axes(0, 1, 2);
#endif
}

inline void append_patch_grid(
   PatchGridSystem& grid_system,
   int patch_id,
   PatchMap const& map,
   PatchCoordinate local_lo,
   PatchCoordinate local_hi,
   PatchLocalAxes local_axes,
   amrex::IntVect const& cells
)
{
   if (patch_id != static_cast<int>(grid_system.patches.size())) {
      amrex::Abort("patch-grid ids must be dense and ordered");
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (cells[dir] <= 0) {
         amrex::Abort("patch-grid cell counts must be positive in every active direction");
      }
      const int component = patch_local_component_for_dir(local_axes, dir);
      if (patch_coordinate_component(local_hi, component) <= patch_coordinate_component(local_lo, component)) {
         amrex::Abort("patch-grid local extents must be ordered in every active direction");
      }
   }
   grid_system.patches.push_back(PatchGridDescriptor{patch_id, map, local_lo, local_hi, local_axes, cells});
}

inline PatchGridSystem make_thornburg04_patch_grid_system(
   amrex::Real inner_radius,
   amrex::Real outer_radius,
   PatchGridResolution const& resolution,
   PatchCoordinate origin = {},
   PatchCartesianAxes active_axes = default_patch_cartesian_axes()
)
{
   PatchGridSystem grid_system;
   grid_system.map_system = make_thornburg04_patch_map_system(inner_radius, outer_radius, origin, active_axes);
   append_patch_grid(
      grid_system,
      0,
      grid_system.map_system.maps[0],
      spherical_central_local_bound(active_axes, inner_radius, -1.0),
      spherical_central_local_bound(active_axes, inner_radius, 1.0),
      patch_local_axes_from_cartesian(active_axes),
      resolution.central_cells
   );
   const amrex::IntVect cells = shell_cells(resolution.angular_cells, resolution.radial_cells);
   for (int patch_id = 1; patch_id < static_cast<int>(grid_system.map_system.maps.size()); ++patch_id) {
      PatchMap const& map = grid_system.map_system.maps[patch_id];
      append_patch_grid(
         grid_system,
         patch_id,
         map,
         make_patch_coordinate(-0.25 * constants::pi, -0.25 * constants::pi, inner_radius),
         make_patch_coordinate(0.25 * constants::pi, 0.25 * constants::pi, outer_radius),
         spherical_shell_local_axes(map, active_axes),
         cells
      );
   }
   return grid_system;
}

inline PatchGridSystem make_thornburg13_patch_grid_system(
   amrex::Real inner_radius,
   amrex::Real medium_radius,
   amrex::Real outer_radius,
   PatchGridResolution const& resolution,
   PatchCoordinate origin = {},
   PatchCartesianAxes active_axes = default_patch_cartesian_axes()
)
{
   PatchGridSystem grid_system;
   grid_system.map_system = make_thornburg13_patch_map_system(inner_radius, medium_radius, outer_radius, origin, active_axes);
   append_patch_grid(
      grid_system,
      0,
      grid_system.map_system.maps[0],
      spherical_central_local_bound(active_axes, inner_radius, -1.0),
      spherical_central_local_bound(active_axes, inner_radius, 1.0),
      patch_local_axes_from_cartesian(active_axes),
      resolution.central_cells
   );
   for (int patch_id = 1; patch_id < static_cast<int>(grid_system.map_system.maps.size()); ++patch_id) {
      const bool outer_patch = patch_id >= 1 + 2 * AMREX_SPACEDIM;
      PatchMap const& map = grid_system.map_system.maps[patch_id];
      append_patch_grid(
         grid_system,
         patch_id,
         map,
         make_patch_coordinate(
            -0.25 * constants::pi,
            -0.25 * constants::pi,
            outer_patch ? medium_radius : inner_radius
         ),
         make_patch_coordinate(
            0.25 * constants::pi,
            0.25 * constants::pi,
            outer_patch ? outer_radius : medium_radius
         ),
         spherical_shell_local_axes(map, active_axes),
         shell_cells(
            resolution.angular_cells,
            outer_patch ? resolution.outer_radial_cells : resolution.radial_cells
         )
      );
   }
   return grid_system;
}

inline PatchGridSystem make_cylindrical5_patch_grid_system(
   amrex::Real inner_radius,
   amrex::Real outer_radius,
   PatchGridResolution const& resolution,
   PatchCoordinate origin = {},
   PatchCartesianAxes active_axes = default_patch_cartesian_axes()
)
{
   validate_cylindrical_patch_cartesian_axes(active_axes);
   PatchGridSystem grid_system;
   grid_system.map_system = make_cylindrical5_patch_map_system(inner_radius, outer_radius, origin, active_axes);
   append_patch_grid(
      grid_system,
      0,
      grid_system.map_system.maps[0],
      cylindrical_central_local_bound(active_axes, inner_radius, resolution.axial_lo, -1.0),
      cylindrical_central_local_bound(active_axes, inner_radius, resolution.axial_hi, 1.0),
      patch_local_axes_from_cartesian(active_axes),
      resolution.central_cells
   );
   const PatchLocalAxes local_axes = cylindrical_shell_local_axes();
   const amrex::IntVect cells = cylindrical_shell_cells(
      resolution.angular_cells,
      resolution.axial_cells,
      resolution.radial_cells
   );
   for (int patch_id = 1; patch_id < static_cast<int>(grid_system.map_system.maps.size()); ++patch_id) {
      append_patch_grid(
         grid_system,
         patch_id,
         grid_system.map_system.maps[patch_id],
         make_patch_coordinate(-0.25 * constants::pi, resolution.axial_lo, inner_radius),
         make_patch_coordinate(0.25 * constants::pi, resolution.axial_hi, outer_radius),
         local_axes,
         cells
      );
   }
   return grid_system;
}

inline PatchGridSystem make_cylindrical9_patch_grid_system(
   amrex::Real inner_radius,
   amrex::Real medium_radius,
   amrex::Real outer_radius,
   PatchGridResolution const& resolution,
   PatchCoordinate origin = {},
   PatchCartesianAxes active_axes = default_patch_cartesian_axes()
)
{
   validate_cylindrical_patch_cartesian_axes(active_axes);
   PatchGridSystem grid_system;
   grid_system.map_system = make_cylindrical9_patch_map_system(inner_radius, medium_radius, outer_radius, origin, active_axes);
   append_patch_grid(
      grid_system,
      0,
      grid_system.map_system.maps[0],
      cylindrical_central_local_bound(active_axes, inner_radius, resolution.axial_lo, -1.0),
      cylindrical_central_local_bound(active_axes, inner_radius, resolution.axial_hi, 1.0),
      patch_local_axes_from_cartesian(active_axes),
      resolution.central_cells
   );
   const PatchLocalAxes local_axes = cylindrical_shell_local_axes();
   for (int patch_id = 1; patch_id < static_cast<int>(grid_system.map_system.maps.size()); ++patch_id) {
      const bool outer_patch = patch_id >= 5;
      append_patch_grid(
         grid_system,
         patch_id,
         grid_system.map_system.maps[patch_id],
         make_patch_coordinate(
            -0.25 * constants::pi,
            resolution.axial_lo,
            outer_patch ? medium_radius : inner_radius
         ),
         make_patch_coordinate(
            0.25 * constants::pi,
            resolution.axial_hi,
            outer_patch ? outer_radius : medium_radius
         ),
         local_axes,
         cylindrical_shell_cells(
            resolution.angular_cells,
            resolution.axial_cells,
            outer_patch ? resolution.outer_radial_cells : resolution.radial_cells
         )
      );
   }
   return grid_system;
}

} // namespace pyfluid_amrex
