#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/physical_geometry.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct MeshAxisCellBounds {
   amrex::Real lo = 0.0_rt;
   amrex::Real hi = 0.0_rt;
   amrex::Real width = 0.0_rt;
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
   return 1.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mesh_default_coordinate_for_axis(int axis) noexcept
{
   if (axis == MeshAxisR) {
      return 1.0_rt;
   }
   if (axis == MeshAxisTheta) {
      return constants::half_pi;
   }
   return 0.0_rt;
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
      const amrex::Real offset = nodal0 ? 0.0_rt : 0.5_rt;
      return problo[0] + (static_cast<amrex::Real>(i) + offset) * dx[0];
   }
#if (AMREX_SPACEDIM >= 2)
   if (mesh_axis(axes, 1) == target_axis) {
      const amrex::Real offset = nodal1 ? 0.0_rt : 0.5_rt;
      return problo[1] + (static_cast<amrex::Real>(j) + offset) * dx[1];
   }
#else
   amrex::ignore_unused(j, nodal1);
#endif
#if (AMREX_SPACEDIM == 3)
   if (mesh_axis(axes, 2) == target_axis) {
      const amrex::Real offset = nodal2 ? 0.0_rt : 0.5_rt;
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
      return {coordinate_value, coordinate_value, 0.0_rt, -1, false};
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
amrex::GpuArray<amrex::Real, 3> mesh_coordinate_triplet_to_cartesian(
   int coordinate,
   amrex::GpuArray<amrex::Real, 3> const& q
) noexcept
{
   if (coordinate == MeshCylindrical) {
      return {q[0] * std::cos(q[1]), q[0] * std::sin(q[1]), q[2]};
   }
   if (coordinate == MeshSpherical) {
      const amrex::Real sin_theta = std::sin(q[1]);
      return {
         q[0] * sin_theta * std::cos(q[2]),
         q[0] * sin_theta * std::sin(q[2]),
         q[0] * std::cos(q[1]),
      };
   }
   return q;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> cartesian_to_mesh_coordinate_triplet(
   int coordinate,
   amrex::GpuArray<amrex::Real, 3> const& x
) noexcept
{
   if (coordinate == MeshCylindrical) {
      amrex::Real phi = std::atan2(x[1], x[0]);
      if (phi < 0.0_rt) {
         phi += constants::two_pi;
      }
      return {cartesian_cylindrical_radius(x[0], x[1]), phi, x[2]};
   }
   if (coordinate == MeshSpherical) {
      const amrex::Real radius = cartesian_radius(x[0], x[1], x[2]);
      amrex::Real phi = std::atan2(x[1], x[0]);
      if (phi < 0.0_rt) {
         phi += constants::two_pi;
      }
      const amrex::Real theta = radius > 0.0_rt
         ? std::acos(amrex::max(-1.0_rt, amrex::min(1.0_rt, x[2] / radius)))
         : 0.0_rt;
      return {radius, theta, phi};
   }
   return x;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> mesh_triplet_to_runtime_coordinates(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, 3> const& q
) noexcept
{
   const auto canonical_axes = mesh_canonical_vector_axes(coordinate);
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> runtime{};
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const int axis = mesh_axis(axes, dir);
      for (int component = 0; component < 3; ++component) {
         if (axis == canonical_axes[component]) {
            runtime[dir] = q[component];
            break;
         }
      }
   }
   return runtime;
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
   return 1.0_rt;
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
      problo[0] + (static_cast<amrex::Real>(i) + 0.5_rt) * dx[0],
#if (AMREX_SPACEDIM >= 2)
      problo[1] + (static_cast<amrex::Real>(j) + 0.5_rt) * dx[1],
#else
      0.0_rt,
#endif
#if (AMREX_SPACEDIM == 3)
      problo[2] + (static_cast<amrex::Real>(k) + 0.5_rt) * dx[2],
#else
      0.0_rt,
#endif
   };
}

} // namespace pyfluid_amrex
