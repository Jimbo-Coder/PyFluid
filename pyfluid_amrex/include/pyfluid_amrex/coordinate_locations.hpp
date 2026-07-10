#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/mesh_geometry.hpp"

namespace pyfluid_amrex {

static constexpr amrex::Real PyFluidFourPi =
   static_cast<amrex::Real>(12.5663706143591729538505735331180115367886775975);
static constexpr amrex::Real PyFluidTwoPi =
   static_cast<amrex::Real>(6.28318530717958647692528676655900576839433879875);
static constexpr amrex::Real PyFluidHalfPi =
   static_cast<amrex::Real>(1.57079632679489661923132169163975144209858469969);

struct MeshAxisCellBounds {
   amrex::Real lo = 0.0;
   amrex::Real hi = 0.0;
   amrex::Real width = 0.0;
   int dir = -1;
   bool active = false;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real active_or_unit_width(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx, int dir) noexcept
{
   if (dir == 0) {
      return dx[0];
   }
#if (AMREX_SPACEDIM >= 2)
   if (dir == 1) {
      return dx[1];
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (dir == 2) {
      return dx[2];
   }
#endif
   return 1.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mesh_default_coordinate_for_axis(int axis) noexcept
{
   if (axis == MeshAxisR) {
      return 1.0;
   }
   if (axis == MeshAxisTheta) {
      return PyFluidHalfPi;
   }
   return 0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real runtime_coordinate_for_axis(
   MeshAxes const& axes,
   int target_axis,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k,
   bool nodal0,
   bool nodal1,
   bool nodal2
) noexcept
{
   if (mesh_axis(axes, 0) == target_axis) {
      const amrex::Real offset = nodal0 ? 0.0 : 0.5;
      return problo[0] + (static_cast<amrex::Real>(i) + offset) * dx[0];
   }
#if (AMREX_SPACEDIM >= 2)
   if (mesh_axis(axes, 1) == target_axis) {
      const amrex::Real offset = nodal1 ? 0.0 : 0.5;
      return problo[1] + (static_cast<amrex::Real>(j) + offset) * dx[1];
   }
#else
   amrex::ignore_unused(j, nodal1);
#endif
#if (AMREX_SPACEDIM == 3)
   if (mesh_axis(axes, 2) == target_axis) {
      const amrex::Real offset = nodal2 ? 0.0 : 0.5;
      return problo[2] + (static_cast<amrex::Real>(k) + offset) * dx[2];
   }
#else
   amrex::ignore_unused(k, nodal2);
#endif
   return mesh_default_coordinate_for_axis(target_axis);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real runtime_coordinate_for_axis(
   MeshAxes const& axes,
   int target_axis,
   int face_dir,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   return runtime_coordinate_for_axis(
      axes,
      target_axis,
      problo,
      dx,
      i,
      j,
      k,
      face_dir == 0,
      face_dir == 1,
      face_dir == 2
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MeshAxisCellBounds runtime_axis_cell_bounds(
   MeshAxes const& axes,
   int target_axis,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   const int dir = mesh_direction_for_axis(axes, target_axis);
   if (dir < 0) {
      const amrex::Real coordinate_value = mesh_default_coordinate_for_axis(target_axis);
      return {coordinate_value, coordinate_value, 0.0, -1, false};
   }
   const amrex::Real lo = runtime_coordinate_for_axis(
      axes,
      target_axis,
      problo,
      dx,
      i,
      j,
      k,
      dir == 0,
      dir == 1,
      dir == 2
   );
   return {lo, lo + dx[dir], dx[dir], dir, true};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> mesh_coordinate_triplet_at_location(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k,
   bool nodal0,
   bool nodal1,
   bool nodal2
) noexcept
{
   if (coordinate == MeshCylindrical) {
      return {
         runtime_coordinate_for_axis(axes, MeshAxisR, problo, dx, i, j, k, nodal0, nodal1, nodal2),
         runtime_coordinate_for_axis(axes, MeshAxisPhi, problo, dx, i, j, k, nodal0, nodal1, nodal2),
         runtime_coordinate_for_axis(axes, MeshAxisZ, problo, dx, i, j, k, nodal0, nodal1, nodal2),
      };
   }
   if (coordinate == MeshSpherical) {
      return {
         runtime_coordinate_for_axis(axes, MeshAxisR, problo, dx, i, j, k, nodal0, nodal1, nodal2),
         runtime_coordinate_for_axis(axes, MeshAxisTheta, problo, dx, i, j, k, nodal0, nodal1, nodal2),
         runtime_coordinate_for_axis(axes, MeshAxisPhi, problo, dx, i, j, k, nodal0, nodal1, nodal2),
      };
   }
   return {
      runtime_coordinate_for_axis(axes, MeshAxisX, problo, dx, i, j, k, nodal0, nodal1, nodal2),
      runtime_coordinate_for_axis(axes, MeshAxisY, problo, dx, i, j, k, nodal0, nodal1, nodal2),
      runtime_coordinate_for_axis(axes, MeshAxisZ, problo, dx, i, j, k, nodal0, nodal1, nodal2),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mesh_width_for_axis(
   MeshAxes const& axes,
   int target_axis,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx
) noexcept
{
   if (mesh_axis(axes, 0) == target_axis) {
      return dx[0];
   }
#if (AMREX_SPACEDIM >= 2)
   if (mesh_axis(axes, 1) == target_axis) {
      return dx[1];
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (mesh_axis(axes, 2) == target_axis) {
      return dx[2];
   }
#endif
   return 1.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> cartesian_cell_center(
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   return {
      problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0],
#if (AMREX_SPACEDIM >= 2)
      problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1],
#else
      0.0,
#endif
#if (AMREX_SPACEDIM == 3)
      problo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2],
#else
      0.0,
#endif
   };
}

} // namespace pyfluid_amrex
