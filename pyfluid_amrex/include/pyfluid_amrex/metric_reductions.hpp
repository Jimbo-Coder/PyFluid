#pragma once

#include "pyfluid_amrex/metric_provider.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

// Evolution uses matching face/volume measures per suppressed angular
// extent. Integral diagnostics restore that extent on BOTH cells and faces.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mesh_reduction_symmetry_factor(int coordinate, MeshAxes const& axes) noexcept
{
   if (!mesh_has_axis(axes, MeshAxisR) || mesh_has_axis(axes, MeshAxisPhi)) {
      return 1.0_rt;
   }
   if (coordinate == MeshSpherical) {
      return mesh_has_axis(axes, MeshAxisTheta) ? constants::two_pi : constants::four_pi;
   }
   return coordinate == MeshCylindrical ? constants::two_pi : 1.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mesh_cell_volume_for_reduction(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i, int j, int k
) noexcept
{
   return mesh_reduction_symmetry_factor(coordinate, axes)
      * mesh_cell_volume_scale(coordinate, axes, problo, dx, i, j, k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mesh_face_area_for_reduction(
   int direction,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i, int j, int k
) noexcept
{
   return mesh_reduction_symmetry_factor(coordinate, axes)
      * mesh_face_area_scale(direction, coordinate, axes, problo, dx, i, j, k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mesh_cell_volume_for_reduction(
   int coordinate,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   return mesh_cell_volume_for_reduction(coordinate, mesh_default_axes_layout(coordinate), problo, dx, i, j, k);
}

} // namespace pyfluid_amrex
