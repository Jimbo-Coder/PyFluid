#pragma once

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "pyfluid_amrex/initial_data/common.hpp"
#include "pyfluid_amrex/patch_grid.hpp"
#include "pyfluid_amrex/patch_layout_builder.hpp"
#include "pyfluid_amrex/patch_maps.hpp"

namespace pyfluid_amrex {

struct InitialDataSamplePoint {
   PatchCoordinate mesh{};
   PatchCoordinate physical{};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real initial_data_coordinate_for_axis(
   MeshAxes const& axes,
   PatchCoordinate const& mesh,
   int target_axis,
   amrex::Real default_value
) noexcept
{
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (mesh_axis(axes, dir) == target_axis) {
         return patch_coordinate_component(mesh, dir);
      }
   }
   return default_value;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate mesh_coordinate_to_physical_point(
   int mesh_coordinate,
   MeshAxes const& axes,
   PatchCoordinate const& mesh
) noexcept
{
   amrex::GpuArray<amrex::Real, 3> q{};
   const amrex::GpuArray<int, 3> canonical_axes = mesh_canonical_vector_axes(mesh_coordinate);
   for (int component = 0; component < 3; ++component) {
      q[component] = initial_data_coordinate_for_axis(
         axes,
         mesh,
         canonical_axes[component],
         mesh_default_coordinate_for_axis(canonical_axes[component])
      );
   }
   const auto physical = mesh_coordinate_triplet_to_cartesian(mesh_coordinate, q);
   return {physical[0], physical[1], physical[2]};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
InitialDataSamplePoint initial_data_sample_point(
   int mesh_coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& offsets,
   int i,
   int j,
   int k
) noexcept
{
   PatchCoordinate mesh{};
   set_patch_coordinate_component(mesh, 0, problo[0] + (static_cast<amrex::Real>(i) + offsets[0]) * dx[0]);
#if (AMREX_SPACEDIM >= 2)
   set_patch_coordinate_component(mesh, 1, problo[1] + (static_cast<amrex::Real>(j) + offsets[1]) * dx[1]);
#else
   amrex::ignore_unused(j);
#endif
#if (AMREX_SPACEDIM == 3)
   set_patch_coordinate_component(mesh, 2, problo[2] + (static_cast<amrex::Real>(k) + offsets[2]) * dx[2]);
#else
   amrex::ignore_unused(k);
#endif
   return {mesh, mesh_coordinate_to_physical_point(mesh_coordinate, axes, mesh)};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
InitialDataSamplePoint cell_center_initial_data_sample_point(
   int mesh_coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> offsets{};
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      offsets[dir] = 0.5;
   }
   return initial_data_sample_point(mesh_coordinate, axes, problo, dx, offsets, i, j, k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
InitialDataSamplePoint face_center_initial_data_sample_point(
   int mesh_coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int face_dir,
   int i,
   int j,
   int k
) noexcept
{
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> offsets{};
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      offsets[dir] = dir == face_dir ? 0.0 : 0.5;
   }
   return initial_data_sample_point(mesh_coordinate, axes, problo, dx, offsets, i, j, k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
InitialDataSamplePoint patch_grid_cell_center_initial_data_sample_point(
   PatchGridDescriptor const& patch,
   amrex::IntVect const& cell
) noexcept
{
   const PatchCoordinate local = patch_grid_cell_center(patch, cell);
   return {local, patch_local_to_global(patch.map, local)};
}

inline InitialDataSamplePoint patch_layout_cell_center_initial_data_sample_point(
   PatchGridDescriptor const& grid_patch,
   PatchDescriptor const& layout_patch,
   amrex::IntVect const& layout_cell
)
{
   const PatchCoordinate local = patch_layout_cell_center(grid_patch, layout_patch, layout_cell);
   return {local, patch_local_to_global(grid_patch.map, local)};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real initial_data_physical_distance2(
   int mesh_coordinate,
   MeshAxes const& axes,
   PatchCoordinate const& mesh,
   PatchCoordinate const& physical,
   PatchCoordinate const& center,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi,
   amrex::GpuArray<int, AMREX_SPACEDIM> const& periodic
) noexcept
{
   if (mesh_coordinate == MeshCartesian) {
      amrex::Real distance2 = 0.0;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         const int axis = mesh_axis(axes, dir);
         int component = 0;
         if (axis == MeshAxisY) {
            component = 1;
         } else if (axis == MeshAxisZ) {
            component = 2;
         }
         const amrex::Real delta = coordinate_delta(
            patch_coordinate_component(mesh, dir),
            patch_coordinate_component(center, component),
            problo[dir],
            probhi[dir],
            periodic[dir] != 0
         );
         distance2 += delta * delta;
      }
      return distance2;
   }

   const amrex::Real dx_physical = physical.q1 - center.q1;
   const amrex::Real dy_physical = physical.q2 - center.q2;
   const amrex::Real dz_physical = physical.q3 - center.q3;
   amrex::ignore_unused(axes, mesh, problo, probhi, periodic);
   return dx_physical * dx_physical + dy_physical * dy_physical + dz_physical * dz_physical;
}

} // namespace pyfluid_amrex
