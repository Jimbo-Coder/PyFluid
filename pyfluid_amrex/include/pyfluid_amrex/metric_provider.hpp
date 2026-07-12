#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "pyfluid_amrex/coordinate_locations.hpp"
#include "pyfluid_amrex/kernels_generated/coordinate_metrics.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

struct CartesianUniformMetric {
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{};
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> inv_dx{};
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> face_area{};
   amrex::Real cell_volume = 1.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
CartesianUniformMetric make_cartesian_uniform_metric(
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx
) noexcept
{
   CartesianUniformMetric metric;
   metric.dx = dx;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      metric.inv_dx[dir] = 1.0 / dx[dir];
   }
   const amrex::Real dx0 = active_or_unit_width(dx, 0);
   const amrex::Real dx1 = active_or_unit_width(dx, 1);
   const amrex::Real dx2 = active_or_unit_width(dx, 2);
   metric.cell_volume = generated::cartesian_cell_volume(dx0, dx1, dx2);
   metric.face_area[0] = generated::cartesian_face_area_x(dx1, dx2);
#if (AMREX_SPACEDIM >= 2)
   metric.face_area[1] = generated::cartesian_face_area_y(dx0, dx2);
#endif
#if (AMREX_SPACEDIM == 3)
   metric.face_area[2] = generated::cartesian_face_area_z(dx0, dx1);
#endif
   return metric;
}

inline CartesianUniformMetric make_cartesian_uniform_metric(amrex::Geometry const& geom)
{
   return make_cartesian_uniform_metric(geom.CellSizeArray());
}

inline amrex::Real cartesian_cell_volume(amrex::Geometry const& geom)
{
   return make_cartesian_uniform_metric(geom).cell_volume;
}

inline amrex::Real cartesian_face_area(amrex::Geometry const& geom, int dir)
{
   return make_cartesian_uniform_metric(geom).face_area[dir];
}

template <int Dir>
inline amrex::Real cartesian_face_area(amrex::Geometry const& geom)
{
   static_assert(Dir >= 0 && Dir < AMREX_SPACEDIM);
   return make_cartesian_uniform_metric(geom).face_area[Dir];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cartesian_cell_volume(
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx
) noexcept
{
   const amrex::Real dx0 = active_or_unit_width(dx, 0);
   const amrex::Real dx1 = active_or_unit_width(dx, 1);
   const amrex::Real dx2 = active_or_unit_width(dx, 2);
   return generated::cartesian_cell_volume(dx0, dx1, dx2);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cylindrical_radial_shell_volume_scale(amrex::Real rlo, amrex::Real rhi) noexcept
{
   return generated::cylindrical_radial_shell_volume_scale(rlo, rhi);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cylindrical_radial_shell_volume(amrex::Real rlo, amrex::Real rhi) noexcept
{
   return PyFluidTwoPi * cylindrical_radial_shell_volume_scale(rlo, rhi);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cylindrical_radial_face_area_scale(amrex::Real r) noexcept
{
   return generated::cylindrical_radial_face_area_scale(r);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real spherical_radial_shell_volume_scale(amrex::Real rlo, amrex::Real rhi) noexcept
{
   return generated::spherical_radial_shell_volume_scale(rlo, rhi);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real spherical_radial_shell_volume(amrex::Real rlo, amrex::Real rhi) noexcept
{
   return PyFluidFourPi * spherical_radial_shell_volume_scale(rlo, rhi);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real spherical_radial_face_area_scale(amrex::Real r) noexcept
{
   return generated::spherical_radial_face_area_scale(r);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real radial_shell_volume_scale(int coordinate, amrex::Real rlo, amrex::Real rhi) noexcept
{
   if (coordinate == MeshSpherical) {
      return spherical_radial_shell_volume_scale(rlo, rhi);
   }
   return cylindrical_radial_shell_volume_scale(rlo, rhi);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real radial_face_area_scale(int coordinate, amrex::Real r) noexcept
{
   if (coordinate == MeshSpherical) {
      return spherical_radial_face_area_scale(r);
   }
   return cylindrical_radial_face_area_scale(r);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real radial_shell_volume_for_reduction(int coordinate, amrex::Real rlo, amrex::Real rhi) noexcept
{
   if (coordinate == MeshSpherical) {
      return spherical_radial_shell_volume(rlo, rhi);
   }
   return cylindrical_radial_shell_volume(rlo, rhi);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real spherical_polar_weight(amrex::Real theta_lo, amrex::Real theta_hi) noexcept
{
   return generated::spherical_polar_weight(theta_lo, theta_hi);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real curvilinear_cell_volume_scale(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
#if (AMREX_SPACEDIM == 1)
   if (!mesh_has_axis(axes, MeshAxisR)) {
      const int axis = mesh_axis(axes, 0);
      const amrex::Real r = mesh_default_coordinate_for_axis(MeshAxisR);
      const amrex::Real theta = mesh_default_coordinate_for_axis(MeshAxisTheta);
      if (coordinate == MeshCylindrical && axis == MeshAxisPhi) {
         return r * dx[0];
      }
      if (coordinate == MeshSpherical && axis == MeshAxisTheta) {
         return r * dx[0];
      }
      if (coordinate == MeshSpherical && axis == MeshAxisPhi) {
         return r * std::sin(theta) * dx[0];
      }
      return dx[0];
   }
#endif
#if (AMREX_SPACEDIM == 2)
   if (!mesh_has_axis(axes, MeshAxisR)) {
      if (coordinate == MeshCylindrical) {
         return generated::cylindrical_phiz_cell_volume_scale(
            mesh_default_coordinate_for_axis(MeshAxisR),
            mesh_width_for_axis(axes, MeshAxisZ, dx),
            mesh_width_for_axis(axes, MeshAxisPhi, dx)
         );
      }
      if (coordinate == MeshSpherical) {
         const MeshAxisCellBounds theta = runtime_axis_cell_bounds(axes, MeshAxisTheta, problo, dx, i, j, k);
         return generated::spherical_thetaphi_cell_volume_scale(
            mesh_default_coordinate_for_axis(MeshAxisR),
            theta.lo,
            theta.hi,
            mesh_width_for_axis(axes, MeshAxisPhi, dx)
         );
      }
   }
#endif
   const MeshAxisCellBounds radius = runtime_axis_cell_bounds(axes, MeshAxisR, problo, dx, i, j, k);
#if (AMREX_SPACEDIM == 1)
   amrex::ignore_unused(j, k);
   return radial_shell_volume_scale(coordinate, radius.lo, radius.hi);
#elif (AMREX_SPACEDIM == 2)
   amrex::ignore_unused(k);
   if (coordinate == MeshSpherical) {
      if (mesh_axis(axes, 1) == MeshAxisPhi) {
         return generated::spherical_rphi_cell_volume_scale(radius.lo, radius.hi, PyFluidHalfPi, dx[1]);
      }
      const MeshAxisCellBounds theta = runtime_axis_cell_bounds(axes, MeshAxisTheta, problo, dx, i, j, k);
      return generated::spherical_rtheta_cell_volume_scale(radius.lo, radius.hi, theta.lo, theta.hi);
   }
   if (mesh_axis(axes, 1) == MeshAxisPhi) {
      return generated::cylindrical_rphiz_cell_volume_scale(radius.lo, radius.hi, 1.0, dx[1]);
   }
   return generated::cylindrical_rz_cell_volume_scale(radius.lo, radius.hi, dx[1]);
#else
   if (coordinate == MeshSpherical) {
      const MeshAxisCellBounds theta = runtime_axis_cell_bounds(axes, MeshAxisTheta, problo, dx, i, j, k);
      return generated::spherical_rthetaphi_cell_volume_scale(radius.lo, radius.hi, theta.lo, theta.hi, dx[2]);
   }
   const amrex::Real z_width = mesh_width_for_axis(axes, MeshAxisZ, dx);
   const amrex::Real phi_width = mesh_width_for_axis(axes, MeshAxisPhi, dx);
   return generated::cylindrical_rphiz_cell_volume_scale(radius.lo, radius.hi, z_width, phi_width);
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real curvilinear_cell_volume_scale(
   int coordinate,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   return curvilinear_cell_volume_scale(coordinate, mesh_default_axes_layout(coordinate), problo, dx, i, j, k);
}

template <int Dir>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real curvilinear_face_area_scale(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   static_assert(Dir >= 0 && Dir < AMREX_SPACEDIM);
#if (AMREX_SPACEDIM == 1)
   if (!mesh_has_axis(axes, MeshAxisR)) {
      return 1.0;
   }
#endif
#if (AMREX_SPACEDIM == 2)
   if (!mesh_has_axis(axes, MeshAxisR)) {
      const int axis = mesh_axis(axes, Dir);
      const amrex::Real r = mesh_default_coordinate_for_axis(MeshAxisR);
      if (coordinate == MeshCylindrical) {
         if (axis == MeshAxisZ) {
            return generated::cylindrical_phiz_axial_face_area_scale(
               r,
               mesh_width_for_axis(axes, MeshAxisPhi, dx)
            );
         }
         if (axis == MeshAxisPhi) {
            return generated::cylindrical_phiz_azimuthal_face_area_scale(
               mesh_width_for_axis(axes, MeshAxisZ, dx)
            );
         }
      }
      if (coordinate == MeshSpherical) {
         if (axis == MeshAxisTheta) {
            const amrex::Real theta = runtime_axis_cell_bounds(axes, MeshAxisTheta, problo, dx, i, j, k).lo;
            return generated::spherical_thetaphi_theta_face_area_scale(
               r,
               theta,
               mesh_width_for_axis(axes, MeshAxisPhi, dx)
            );
         }
         if (axis == MeshAxisPhi) {
            const MeshAxisCellBounds theta = runtime_axis_cell_bounds(axes, MeshAxisTheta, problo, dx, i, j, k);
            return generated::spherical_thetaphi_azimuthal_face_area_scale(
               r,
               theta.lo,
               theta.hi
            );
         }
      }
   }
#endif
   if constexpr (Dir == 0) {
      const amrex::Real r = runtime_axis_cell_bounds(axes, MeshAxisR, problo, dx, i, j, k).lo;
#if (AMREX_SPACEDIM == 1)
      amrex::ignore_unused(j, k);
      return radial_face_area_scale(coordinate, r);
#elif (AMREX_SPACEDIM == 2)
      amrex::ignore_unused(k);
      if (coordinate == MeshSpherical) {
         if (mesh_axis(axes, 1) == MeshAxisPhi) {
            return generated::spherical_rphi_radial_face_area_scale(r, PyFluidHalfPi, dx[1]);
         }
         const MeshAxisCellBounds theta = runtime_axis_cell_bounds(axes, MeshAxisTheta, problo, dx, i, j, k);
         return generated::spherical_rtheta_radial_face_area_scale(r, theta.lo, theta.hi);
      }
      if (mesh_axis(axes, 1) == MeshAxisPhi) {
         return generated::cylindrical_rphiz_radial_face_area_scale(r, 1.0, dx[1]);
      }
      return generated::cylindrical_rz_radial_face_area_scale(r, dx[1]);
#else
      if (coordinate == MeshSpherical) {
         const MeshAxisCellBounds theta = runtime_axis_cell_bounds(axes, MeshAxisTheta, problo, dx, i, j, k);
         return generated::spherical_rthetaphi_radial_face_area_scale(r, theta.lo, theta.hi, dx[2]);
      }
      const amrex::Real z_width = mesh_width_for_axis(axes, MeshAxisZ, dx);
      const amrex::Real phi_width = mesh_width_for_axis(axes, MeshAxisPhi, dx);
      return generated::cylindrical_rphiz_radial_face_area_scale(r, z_width, phi_width);
#endif
   } else {
#if (AMREX_SPACEDIM >= 2)
      const MeshAxisCellBounds radius = runtime_axis_cell_bounds(axes, MeshAxisR, problo, dx, i, j, k);
      if constexpr (Dir == 1) {
         amrex::ignore_unused(k);
         const int axis = mesh_axis(axes, Dir);
         if (coordinate == MeshSpherical) {
            if (axis == MeshAxisPhi) {
               return generated::spherical_rphi_azimuthal_face_area_scale(radius.lo, radius.hi);
            }
            const amrex::Real theta = runtime_axis_cell_bounds(axes, MeshAxisTheta, problo, dx, i, j, k).lo;
#if (AMREX_SPACEDIM == 2)
            return generated::spherical_rtheta_theta_face_area_scale(radius.lo, radius.hi, theta);
#else
            return generated::spherical_rthetaphi_theta_face_area_scale(radius.lo, radius.hi, theta, dx[2]);
#endif
         }
#if (AMREX_SPACEDIM == 2)
         if (axis == MeshAxisPhi) {
            return generated::cylindrical_rphiz_azimuthal_face_area_scale(radius.lo, radius.hi, 1.0);
         }
         return generated::cylindrical_rz_axial_face_area_scale(radius.lo, radius.hi);
#else
         if (axis == MeshAxisPhi) {
            const amrex::Real z_width = mesh_width_for_axis(axes, MeshAxisZ, dx);
            return generated::cylindrical_rphiz_azimuthal_face_area_scale(radius.lo, radius.hi, z_width);
         }
         const amrex::Real phi_width = mesh_width_for_axis(axes, MeshAxisPhi, dx);
         return generated::cylindrical_rphiz_axial_face_area_scale(radius.lo, radius.hi, phi_width);
#endif
      }
#if (AMREX_SPACEDIM == 3)
      if constexpr (Dir == 2) {
         const int axis = mesh_axis(axes, Dir);
         if (coordinate == MeshSpherical) {
            const MeshAxisCellBounds theta = runtime_axis_cell_bounds(axes, MeshAxisTheta, problo, dx, i, j, k);
            return generated::spherical_rthetaphi_azimuthal_face_area_scale(radius.lo, radius.hi, theta.lo, theta.hi);
         }
         if (axis == MeshAxisPhi) {
            const amrex::Real z_width = mesh_width_for_axis(axes, MeshAxisZ, dx);
            return generated::cylindrical_rphiz_azimuthal_face_area_scale(radius.lo, radius.hi, z_width);
         }
         const amrex::Real phi_width = mesh_width_for_axis(axes, MeshAxisPhi, dx);
         return generated::cylindrical_rphiz_axial_face_area_scale(radius.lo, radius.hi, phi_width);
      }
#endif
      return 1.0;
#else
      amrex::ignore_unused(coordinate, problo, dx, i, j, k);
      return 1.0;
#endif
   }
}

template <int Dir>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real curvilinear_face_area_scale(
   int coordinate,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   return curvilinear_face_area_scale<Dir>(coordinate, mesh_default_axes_layout(coordinate), problo, dx, i, j, k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real curvilinear_face_area_scale(
   int dir,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   if (dir == 0) {
      return curvilinear_face_area_scale<0>(coordinate, axes, problo, dx, i, j, k);
   }
#if (AMREX_SPACEDIM >= 2)
   if (dir == 1) {
      return curvilinear_face_area_scale<1>(coordinate, axes, problo, dx, i, j, k);
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (dir == 2) {
      return curvilinear_face_area_scale<2>(coordinate, axes, problo, dx, i, j, k);
   }
#endif
   return 1.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real curvilinear_face_area_scale(
   int dir,
   int coordinate,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   return curvilinear_face_area_scale(dir, coordinate, mesh_default_axes_layout(coordinate), problo, dx, i, j, k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real curvilinear_edge_length_scale(
   int edge_dir,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   if (coordinate == MeshCartesian) {
      if (edge_dir == 0) {
         return generated::cartesian_edge_length_x(dx[0]);
      }
#if (AMREX_SPACEDIM >= 2)
      if (edge_dir == 1) {
         return generated::cartesian_edge_length_y(dx[1]);
      }
#endif
#if (AMREX_SPACEDIM == 3)
      if (edge_dir == 2) {
         return generated::cartesian_edge_length_z(dx[2]);
      }
#endif
      return 1.0;
   }
#if (AMREX_SPACEDIM == 2)
   const int axis = mesh_vector_axis_for_component(coordinate, axes, edge_dir);
   const MeshAxisCellBounds radius = runtime_axis_cell_bounds(axes, MeshAxisR, problo, dx, i, j, k);
   const amrex::Real r = runtime_coordinate_for_axis(axes, MeshAxisR, -1, problo, dx, i, j, k);
   if (coordinate == MeshCylindrical) {
      if (axis == MeshAxisR) {
         if (!radius.active) {
            return 1.0;
         }
         return generated::cylindrical_rphiz_radial_edge_length_scale(radius.lo, radius.hi);
      }
      if (axis == MeshAxisZ) {
         return generated::cylindrical_rphiz_axial_edge_length_scale(mesh_width_for_axis(axes, MeshAxisZ, dx));
      }
      if (axis == MeshAxisPhi) {
         return generated::cylindrical_rphiz_azimuthal_edge_length_scale(
            r,
            mesh_width_for_axis(axes, MeshAxisPhi, dx)
         );
      }
   }
   if (coordinate == MeshSpherical) {
      if (axis == MeshAxisR) {
         if (!radius.active) {
            return 1.0;
         }
         return generated::spherical_rthetaphi_radial_edge_length_scale(radius.lo, radius.hi);
      }
      const amrex::Real theta = runtime_coordinate_for_axis(axes, MeshAxisTheta, -1, problo, dx, i, j, k);
      if (axis == MeshAxisTheta) {
         return generated::spherical_rthetaphi_theta_edge_length_scale(
            r,
            0.0,
            mesh_width_for_axis(axes, MeshAxisTheta, dx)
         );
      }
      if (axis == MeshAxisPhi) {
         return generated::spherical_rthetaphi_azimuthal_edge_length_scale(
            r,
            theta,
            mesh_width_for_axis(axes, MeshAxisPhi, dx)
         );
      }
   }
   amrex::ignore_unused(k);
#elif (AMREX_SPACEDIM == 3)
   const MeshAxisCellBounds radius = runtime_axis_cell_bounds(axes, MeshAxisR, problo, dx, i, j, k);
   if (coordinate == MeshCylindrical) {
      const int axis = mesh_axis(axes, edge_dir);
      if (axis == MeshAxisR) {
         return generated::cylindrical_rphiz_radial_edge_length_scale(radius.lo, radius.hi);
      }
      if (axis == MeshAxisZ) {
         return generated::cylindrical_rphiz_axial_edge_length_scale(dx[edge_dir]);
      }
      if (axis == MeshAxisPhi) {
         return generated::cylindrical_rphiz_azimuthal_edge_length_scale(radius.lo, dx[edge_dir]);
      }
   }
   if (coordinate == MeshSpherical) {
      if (edge_dir == 0) {
         return generated::spherical_rthetaphi_radial_edge_length_scale(radius.lo, radius.hi);
      }
      if (edge_dir == 1) {
         const MeshAxisCellBounds theta = runtime_axis_cell_bounds(axes, MeshAxisTheta, problo, dx, i, j, k);
         return generated::spherical_rthetaphi_theta_edge_length_scale(radius.lo, theta.lo, theta.hi);
      }
      if (edge_dir == 2) {
         amrex::ignore_unused(k);
         const amrex::Real theta = runtime_axis_cell_bounds(axes, MeshAxisTheta, problo, dx, i, j, k).lo;
         return generated::spherical_rthetaphi_azimuthal_edge_length_scale(radius.lo, theta, dx[2]);
      }
   }
#else
   amrex::ignore_unused(edge_dir, coordinate, axes, problo, dx, i, j, k);
#endif
   return 1.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real curvilinear_edge_length_scale(
   int edge_dir,
   int coordinate,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   return curvilinear_edge_length_scale(edge_dir, coordinate, mesh_default_axes_layout(coordinate), problo, dx, i, j, k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cartesian_face_area_scale(
   int dir,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx
) noexcept
{
   const amrex::Real dx0 = active_or_unit_width(dx, 0);
   const amrex::Real dx1 = active_or_unit_width(dx, 1);
   const amrex::Real dx2 = active_or_unit_width(dx, 2);
   if (dir == 0) {
      return generated::cartesian_face_area_x(dx1, dx2);
   }
#if (AMREX_SPACEDIM >= 2)
   if (dir == 1) {
      return generated::cartesian_face_area_y(dx0, dx2);
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (dir == 2) {
      return generated::cartesian_face_area_z(dx0, dx1);
   }
#endif
   return 1.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mesh_face_area_scale(
   int dir,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   if (coordinate == MeshCartesian) {
      amrex::ignore_unused(axes, problo, i, j, k);
      return cartesian_face_area_scale(dir, dx);
   }
   return curvilinear_face_area_scale(dir, coordinate, axes, problo, dx, i, j, k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mesh_edge_length_scale(
   int edge_dir,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   return curvilinear_edge_length_scale(edge_dir, coordinate, axes, problo, dx, i, j, k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mesh_cell_volume_scale(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   if (coordinate == MeshCartesian) {
      amrex::ignore_unused(axes, problo, i, j, k);
      return cartesian_cell_volume(dx);
   }
   return curvilinear_cell_volume_scale(coordinate, axes, problo, dx, i, j, k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mesh_cell_metric_density_factor(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   const amrex::Real coordinate_volume = cartesian_cell_volume(dx);
   const amrex::Real physical_volume = mesh_cell_volume_scale(coordinate, axes, problo, dx, i, j, k);
   return amrex::Math::abs(physical_volume) / amrex::max(coordinate_volume, 1.0e-300);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real vector_component_for_axis(
   int coordinate,
   MeshAxes const& axes,
   int target_axis,
   amrex::Real v1,
   amrex::Real v2,
   amrex::Real v3
) noexcept
{
   if (mesh_vector_axis_for_component(coordinate, axes, 0) == target_axis) {
      return v1;
   }
   if (mesh_vector_axis_for_component(coordinate, axes, 1) == target_axis) {
      return v2;
   }
   if (mesh_vector_axis_for_component(coordinate, axes, 2) == target_axis) {
      return v3;
   }
   return 0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 mesh_basis_vector_to_cartesian(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, 3> const& q,
   StateVector3 const& vector
) noexcept
{
   if (coordinate == MeshCylindrical) {
      const amrex::Real vr = vector_component_for_axis(coordinate, axes, MeshAxisR, vector.x, vector.y, vector.z);
      const amrex::Real vphi = vector_component_for_axis(coordinate, axes, MeshAxisPhi, vector.x, vector.y, vector.z);
      const amrex::Real vz = vector_component_for_axis(coordinate, axes, MeshAxisZ, vector.x, vector.y, vector.z);
      const amrex::Real cos_phi = std::cos(q[1]);
      const amrex::Real sin_phi = std::sin(q[1]);
      return {vr * cos_phi - vphi * sin_phi, vr * sin_phi + vphi * cos_phi, vz};
   }
   if (coordinate == MeshSpherical) {
      const amrex::Real vr = vector_component_for_axis(coordinate, axes, MeshAxisR, vector.x, vector.y, vector.z);
      const amrex::Real vtheta = vector_component_for_axis(coordinate, axes, MeshAxisTheta, vector.x, vector.y, vector.z);
      const amrex::Real vphi = vector_component_for_axis(coordinate, axes, MeshAxisPhi, vector.x, vector.y, vector.z);
      const amrex::Real sin_theta = std::sin(q[1]);
      const amrex::Real cos_theta = std::cos(q[1]);
      const amrex::Real sin_phi = std::sin(q[2]);
      const amrex::Real cos_phi = std::cos(q[2]);
      return {
         vr * sin_theta * cos_phi + vtheta * cos_theta * cos_phi - vphi * sin_phi,
         vr * sin_theta * sin_phi + vtheta * cos_theta * sin_phi + vphi * cos_phi,
         vr * cos_theta - vtheta * sin_theta,
      };
   }
   return {
      vector_component_for_axis(coordinate, axes, MeshAxisX, vector.x, vector.y, vector.z),
      vector_component_for_axis(coordinate, axes, MeshAxisY, vector.x, vector.y, vector.z),
      vector_component_for_axis(coordinate, axes, MeshAxisZ, vector.x, vector.y, vector.z),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 cartesian_vector_to_mesh_basis(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, 3> const& q,
   StateVector3 const& vector
) noexcept
{
   StateVector3 canonical{};
   if (coordinate == MeshCylindrical) {
      const amrex::Real cos_phi = std::cos(q[1]);
      const amrex::Real sin_phi = std::sin(q[1]);
      canonical = {
         vector.x * cos_phi + vector.y * sin_phi,
         -vector.x * sin_phi + vector.y * cos_phi,
         vector.z,
      };
   } else if (coordinate == MeshSpherical) {
      const amrex::Real sin_theta = std::sin(q[1]);
      const amrex::Real cos_theta = std::cos(q[1]);
      const amrex::Real sin_phi = std::sin(q[2]);
      const amrex::Real cos_phi = std::cos(q[2]);
      canonical = {
         vector.x * sin_theta * cos_phi + vector.y * sin_theta * sin_phi + vector.z * cos_theta,
         vector.x * cos_theta * cos_phi + vector.y * cos_theta * sin_phi - vector.z * sin_theta,
         -vector.x * sin_phi + vector.y * cos_phi,
      };
   } else {
      canonical = vector;
   }

   const amrex::GpuArray<int, 3> canonical_axes = mesh_canonical_vector_axes(coordinate);
   StateVector3 mesh{};
   for (int component = 0; component < 3; ++component) {
      const int axis = mesh_vector_axis_for_component(coordinate, axes, component);
      amrex::Real value = 0.0;
      for (int canonical_component = 0; canonical_component < 3; ++canonical_component) {
         if (canonical_axes[canonical_component] == axis) {
            value = canonical_component == 0
               ? canonical.x
               : (canonical_component == 1 ? canonical.y : canonical.z);
            break;
         }
      }
      if (component == 0) {
         mesh.x = value;
      } else if (component == 1) {
         mesh.y = value;
      } else {
         mesh.z = value;
      }
   }
   return mesh;
}

struct MeshFaceMetric {
   StateVector3 unit_normal{};
   amrex::Real area = 0.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real vector_norm(StateVector3 const& vector) noexcept
{
   return std::sqrt(vector.squared_norm());
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 normalized_vector(StateVector3 const& vector) noexcept
{
   const amrex::Real magnitude = vector_norm(vector);
   return magnitude > 0.0 ? vector / magnitude : StateVector3{};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 aligned_vector(StateVector3 vector, StateVector3 const& direction) noexcept
{
   return vector.dot(direction) < 0.0 ? -vector : vector;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 mesh_coordinate_basis_vector(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, 3> const& q,
   int dir
) noexcept
{
   StateVector3 mesh{};
   if (dir == 0) {
      mesh.x = 1.0;
   }
#if (AMREX_SPACEDIM >= 2)
   else if (dir == 1) {
      mesh.y = 1.0;
   }
#endif
#if (AMREX_SPACEDIM == 3)
   else {
      mesh.z = 1.0;
   }
#endif
   return mesh_basis_vector_to_cartesian(coordinate, axes, q, mesh);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void shift_mesh_coordinate(
   amrex::GpuArray<amrex::Real, 3>& q,
   int coordinate,
   MeshAxes const& axes,
   int dir,
   amrex::Real offset
) noexcept
{
   const int axis = mesh_axis(axes, dir);
   const int component = mesh_canonical_axis_index(coordinate, axis);
   if (component >= 0) {
      q[component] += offset;
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MeshFaceMetric mesh_physical_face_metric(
   int dir,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   const auto face_q = mesh_coordinate_triplet_at_location(
      coordinate,
      axes,
      problo,
      dx,
      i,
      j,
      k,
      dir == 0,
      dir == 1,
      dir == 2
   );
   const StateVector3 desired_normal = mesh_coordinate_basis_vector(coordinate, axes, face_q, dir);
#if (AMREX_SPACEDIM == 1)
   return {normalized_vector(desired_normal), mesh_face_area_scale(dir, coordinate, axes, problo, dx, i, j, k)};
#elif (AMREX_SPACEDIM == 2)
   const int tangent_dir = dir == 0 ? 1 : 0;
   auto lo_q = face_q;
   auto hi_q = face_q;
   shift_mesh_coordinate(lo_q, coordinate, axes, tangent_dir, -0.5 * dx[tangent_dir]);
   shift_mesh_coordinate(hi_q, coordinate, axes, tangent_dir, 0.5 * dx[tangent_dir]);
   const auto lo = mesh_coordinate_triplet_to_cartesian(coordinate, lo_q);
   const auto hi = mesh_coordinate_triplet_to_cartesian(coordinate, hi_q);
   const StateVector3 edge{hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
   const StateVector3 surface_normal = normalized_vector(desired_normal.cross(edge));
   const StateVector3 area_vector = aligned_vector(edge.cross(surface_normal), desired_normal);
   return {normalized_vector(area_vector), vector_norm(area_vector)};
#else
   int tangent0 = 1;
   int tangent1 = 2;
   if (dir == 1) {
      tangent0 = 2;
      tangent1 = 0;
   } else if (dir == 2) {
      tangent0 = 0;
      tangent1 = 1;
   }
   auto q00 = face_q;
   auto q10 = face_q;
   auto q11 = face_q;
   auto q01 = face_q;
   shift_mesh_coordinate(q00, coordinate, axes, tangent0, -0.5 * dx[tangent0]);
   shift_mesh_coordinate(q00, coordinate, axes, tangent1, -0.5 * dx[tangent1]);
   shift_mesh_coordinate(q10, coordinate, axes, tangent0, 0.5 * dx[tangent0]);
   shift_mesh_coordinate(q10, coordinate, axes, tangent1, -0.5 * dx[tangent1]);
   shift_mesh_coordinate(q11, coordinate, axes, tangent0, 0.5 * dx[tangent0]);
   shift_mesh_coordinate(q11, coordinate, axes, tangent1, 0.5 * dx[tangent1]);
   shift_mesh_coordinate(q01, coordinate, axes, tangent0, -0.5 * dx[tangent0]);
   shift_mesh_coordinate(q01, coordinate, axes, tangent1, 0.5 * dx[tangent1]);
   const auto p00 = mesh_coordinate_triplet_to_cartesian(coordinate, q00);
   const auto p10 = mesh_coordinate_triplet_to_cartesian(coordinate, q10);
   const auto p11 = mesh_coordinate_triplet_to_cartesian(coordinate, q11);
   const auto p01 = mesh_coordinate_triplet_to_cartesian(coordinate, q01);
   const StateVector3 edge10{p10[0] - p00[0], p10[1] - p00[1], p10[2] - p00[2]};
   const StateVector3 edge11{p11[0] - p00[0], p11[1] - p00[1], p11[2] - p00[2]};
   const StateVector3 edge01{p01[0] - p00[0], p01[1] - p00[1], p01[2] - p00[2]};
   const StateVector3 area_vector = aligned_vector(
      (edge10.cross(edge11) + edge11.cross(edge01)) * 0.5, desired_normal
   );
   return {normalized_vector(area_vector), vector_norm(area_vector)};
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real curvilinear_momentum_curvature_source(
   int coordinate,
   MeshAxes const& axes,
   int component,
   amrex::Real rho,
   amrex::Real v1,
   amrex::Real v2,
   amrex::Real v3,
   amrex::Real radius,
   amrex::Real theta
) noexcept
{
   const amrex::Real inv_radius = 1.0 / amrex::max(radius, 1.0e-300);
   const int component_axis = mesh_vector_axis_for_component(coordinate, axes, component - MOM1);
   if (coordinate == MeshCylindrical) {
      const amrex::Real vr = vector_component_for_axis(coordinate, axes, MeshAxisR, v1, v2, v3);
      const amrex::Real vphi = vector_component_for_axis(coordinate, axes, MeshAxisPhi, v1, v2, v3);
      const amrex::Real vz = vector_component_for_axis(coordinate, axes, MeshAxisZ, v1, v2, v3);
      if (component_axis == MeshAxisR) {
         return generated::cylindrical_momentum_curvature_source_mom1(rho, vr, vphi, vz, inv_radius);
      }
      if (component_axis == MeshAxisPhi) {
         return generated::cylindrical_momentum_curvature_source_mom2(rho, vr, vphi, vz, inv_radius);
      }
      if (component_axis == MeshAxisZ) {
         return generated::cylindrical_momentum_curvature_source_mom3(rho, vr, vphi, vz, inv_radius);
      }
      return 0.0;
   }
   if (coordinate == MeshSpherical) {
      const amrex::Real vr = vector_component_for_axis(coordinate, axes, MeshAxisR, v1, v2, v3);
      const amrex::Real vtheta = vector_component_for_axis(coordinate, axes, MeshAxisTheta, v1, v2, v3);
      const amrex::Real vphi = vector_component_for_axis(coordinate, axes, MeshAxisPhi, v1, v2, v3);
      const amrex::Real sin_theta = std::sin(theta);
      const amrex::Real cot_theta = std::cos(theta) / amrex::max(std::abs(sin_theta), 1.0e-300);
      if (component_axis == MeshAxisR) {
         return generated::spherical_momentum_curvature_source_mom1(rho, vr, vtheta, vphi, inv_radius, cot_theta);
      }
      if (component_axis == MeshAxisTheta) {
         return generated::spherical_momentum_curvature_source_mom2(rho, vr, vtheta, vphi, inv_radius, cot_theta);
      }
      if (component_axis == MeshAxisPhi) {
         return generated::spherical_momentum_curvature_source_mom3(rho, vr, vtheta, vphi, inv_radius, cot_theta);
      }
   }
   return 0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real curvilinear_mhd_stress_curvature_source(
   int coordinate,
   MeshAxes const& axes,
   int component,
   amrex::Real rho,
   amrex::Real v1,
   amrex::Real v2,
   amrex::Real v3,
   amrex::Real b1,
   amrex::Real b2,
   amrex::Real b3,
   amrex::Real radius,
   amrex::Real theta
) noexcept
{
   const amrex::Real inv_radius = 1.0 / amrex::max(radius, 1.0e-300);
   const int component_axis = mesh_vector_axis_for_component(coordinate, axes, component - MOM1);
   if (coordinate == MeshCylindrical) {
      const amrex::Real vr = vector_component_for_axis(coordinate, axes, MeshAxisR, v1, v2, v3);
      const amrex::Real vphi = vector_component_for_axis(coordinate, axes, MeshAxisPhi, v1, v2, v3);
      const amrex::Real vz = vector_component_for_axis(coordinate, axes, MeshAxisZ, v1, v2, v3);
      const amrex::Real br = vector_component_for_axis(coordinate, axes, MeshAxisR, b1, b2, b3);
      const amrex::Real bphi = vector_component_for_axis(coordinate, axes, MeshAxisPhi, b1, b2, b3);
      const amrex::Real bz = vector_component_for_axis(coordinate, axes, MeshAxisZ, b1, b2, b3);
      if (component_axis == MeshAxisR) {
         return generated::cylindrical_mhd_stress_curvature_source_mom1(
            rho, vr, vphi, vz, br, bphi, bz, inv_radius
         );
      }
      if (component_axis == MeshAxisPhi) {
         return generated::cylindrical_mhd_stress_curvature_source_mom2(rho, vr, vphi, vz, br, bphi, bz, inv_radius);
      }
      if (component_axis == MeshAxisZ) {
         return generated::cylindrical_mhd_stress_curvature_source_mom3(rho, vr, vphi, vz, br, bphi, bz, inv_radius);
      }
      return 0.0;
   }
   if (coordinate == MeshSpherical) {
      const amrex::Real vr = vector_component_for_axis(coordinate, axes, MeshAxisR, v1, v2, v3);
      const amrex::Real vtheta = vector_component_for_axis(coordinate, axes, MeshAxisTheta, v1, v2, v3);
      const amrex::Real vphi = vector_component_for_axis(coordinate, axes, MeshAxisPhi, v1, v2, v3);
      const amrex::Real br = vector_component_for_axis(coordinate, axes, MeshAxisR, b1, b2, b3);
      const amrex::Real btheta = vector_component_for_axis(coordinate, axes, MeshAxisTheta, b1, b2, b3);
      const amrex::Real bphi = vector_component_for_axis(coordinate, axes, MeshAxisPhi, b1, b2, b3);
      const amrex::Real sin_theta = std::sin(theta);
      const amrex::Real cot_theta = std::cos(theta) / amrex::max(std::abs(sin_theta), 1.0e-300);
      if (component_axis == MeshAxisR) {
         return generated::spherical_mhd_stress_curvature_source_mom1(
            rho, vr, vtheta, vphi, br, btheta, bphi, inv_radius, cot_theta
         );
      }
      if (component_axis == MeshAxisTheta) {
         return generated::spherical_mhd_stress_curvature_source_mom2(
            rho, vr, vtheta, vphi, br, btheta, bphi, inv_radius, cot_theta
         );
      }
      if (component_axis == MeshAxisPhi) {
         return generated::spherical_mhd_stress_curvature_source_mom3(
            rho, vr, vtheta, vphi, br, btheta, bphi, inv_radius, cot_theta
         );
      }
   }
   return 0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real curvilinear_momentum_curvature_source(
   int coordinate,
   int component,
   amrex::Real rho,
   amrex::Real vr,
   amrex::Real v2,
   amrex::Real v3,
   amrex::Real radius,
   amrex::Real theta
) noexcept
{
   return curvilinear_momentum_curvature_source(
      coordinate, mesh_default_axes_layout(coordinate), component, rho, vr, v2, v3, radius, theta
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mesh_cell_volume_for_reduction(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
#if (AMREX_SPACEDIM == 1)
   amrex::ignore_unused(j, k);
   if (coordinate == MeshCylindrical || coordinate == MeshSpherical) {
      if (!mesh_has_axis(axes, MeshAxisR)) {
         return curvilinear_cell_volume_scale(coordinate, axes, problo, dx, i, j, k);
      }
      const MeshAxisCellBounds radius = runtime_axis_cell_bounds(axes, MeshAxisR, problo, dx, i, j, k);
      return radial_shell_volume_for_reduction(coordinate, radius.lo, radius.hi);
   }
#elif (AMREX_SPACEDIM == 2)
   amrex::ignore_unused(k);
   if (coordinate == MeshCylindrical) {
      if (!mesh_has_axis(axes, MeshAxisR)) {
         return curvilinear_cell_volume_scale(coordinate, axes, problo, dx, i, j, k);
      }
      const MeshAxisCellBounds radius = runtime_axis_cell_bounds(axes, MeshAxisR, problo, dx, i, j, k);
      if (mesh_axis(axes, 1) == MeshAxisPhi) {
         return generated::cylindrical_rphiz_cell_volume_scale(radius.lo, radius.hi, 1.0, dx[1]);
      }
      return PyFluidTwoPi * cylindrical_radial_shell_volume_scale(radius.lo, radius.hi) * dx[1];
   }
   if (coordinate == MeshSpherical) {
      if (!mesh_has_axis(axes, MeshAxisR)) {
         return curvilinear_cell_volume_scale(coordinate, axes, problo, dx, i, j, k);
      }
      const MeshAxisCellBounds radius = runtime_axis_cell_bounds(axes, MeshAxisR, problo, dx, i, j, k);
      if (mesh_axis(axes, 1) == MeshAxisPhi) {
         return generated::spherical_rphi_cell_volume_scale(radius.lo, radius.hi, PyFluidHalfPi, dx[1]);
      }
      const MeshAxisCellBounds theta = runtime_axis_cell_bounds(axes, MeshAxisTheta, problo, dx, i, j, k);
      return PyFluidTwoPi * spherical_radial_shell_volume_scale(radius.lo, radius.hi)
         * spherical_polar_weight(theta.lo, theta.hi);
   }
#else
   if (coordinate == MeshCylindrical || coordinate == MeshSpherical) {
      return curvilinear_cell_volume_scale(coordinate, axes, problo, dx, i, j, k);
   }
#endif
   return cartesian_cell_volume(dx);
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

template <int Coordinate>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> coordinate_scale_factors(
   amrex::Real q1,
   amrex::Real q2,
   amrex::Real q3
) noexcept
{
   if constexpr (Coordinate == MeshCylindrical) {
      return {
         generated::cylindrical_scale_r(q1, q2, q3),
         generated::cylindrical_scale_phi(q1, q2, q3),
         generated::cylindrical_scale_z(q1, q2, q3),
      };
   } else if constexpr (Coordinate == MeshSpherical) {
      return {
         generated::spherical_scale_r(q1, q2, q3),
         generated::spherical_scale_theta(q1, q2, q3),
         generated::spherical_scale_phi(q1, q2, q3),
      };
   } else {
      static_assert(Coordinate == MeshCartesian, "unsupported coordinate specialization");
      return {
         generated::cartesian_scale_x(q1, q2, q3),
         generated::cartesian_scale_y(q1, q2, q3),
         generated::cartesian_scale_z(q1, q2, q3),
      };
   }
}

template <int Coordinate>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real coordinate_volume_scale(amrex::Real q1, amrex::Real q2, amrex::Real q3) noexcept
{
   if constexpr (Coordinate == MeshCylindrical) {
      return generated::cylindrical_volume_scale(q1, q2, q3);
   } else if constexpr (Coordinate == MeshSpherical) {
      return generated::spherical_volume_scale(q1, q2, q3);
   } else {
      static_assert(Coordinate == MeshCartesian, "unsupported coordinate specialization");
      return generated::cartesian_volume_scale(q1, q2, q3);
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> coordinate_scale_factors(
   int coordinate,
   amrex::Real q1,
   amrex::Real q2,
   amrex::Real q3
) noexcept
{
   if (coordinate == MeshCylindrical) {
      return coordinate_scale_factors<MeshCylindrical>(q1, q2, q3);
   }
   if (coordinate == MeshSpherical) {
      return coordinate_scale_factors<MeshSpherical>(q1, q2, q3);
   }
   return coordinate_scale_factors<MeshCartesian>(q1, q2, q3);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real runtime_direction_scale_factor(
   int coordinate,
   MeshAxes const& axes,
   int dir,
   amrex::Real q1,
   amrex::Real q2,
   amrex::Real q3
) noexcept
{
   const auto scale = coordinate_scale_factors(coordinate, q1, q2, q3);
   const int axis = mesh_axis(axes, dir);
   if (coordinate == MeshCylindrical) {
      if (axis == MeshAxisR) {
         return scale[0];
      }
      if (axis == MeshAxisPhi) {
         return scale[1];
      }
      if (axis == MeshAxisZ) {
         return scale[2];
      }
      return 1.0;
   }
   if (axis == MeshAxisX || axis == MeshAxisR) {
      return scale[0];
   }
   if (axis == MeshAxisY || axis == MeshAxisTheta) {
      return scale[1];
   }
   if (axis == MeshAxisZ || axis == MeshAxisPhi) {
      return scale[2];
   }
   return 1.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real runtime_direction_scale_factor(
   int coordinate,
   int dir,
   amrex::Real q1,
   amrex::Real q2,
   amrex::Real q3
) noexcept
{
   return runtime_direction_scale_factor(coordinate, mesh_default_axes_layout(coordinate), dir, q1, q2, q3);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real curvilinear_face_gradient_spacing(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int dir,
   int i,
   int j,
   int k
) noexcept
{
   amrex::Real q1 = runtime_coordinate_for_axis(axes, MeshAxisX, dir, problo, dx, i, j, k);
   amrex::Real q2 = runtime_coordinate_for_axis(axes, MeshAxisY, dir, problo, dx, i, j, k);
   amrex::Real q3 = runtime_coordinate_for_axis(axes, MeshAxisZ, dir, problo, dx, i, j, k);
   if (coordinate == MeshCylindrical) {
      q1 = runtime_coordinate_for_axis(axes, MeshAxisR, dir, problo, dx, i, j, k);
      q2 = runtime_coordinate_for_axis(axes, MeshAxisPhi, dir, problo, dx, i, j, k);
      q3 = runtime_coordinate_for_axis(axes, MeshAxisZ, dir, problo, dx, i, j, k);
   } else if (coordinate == MeshSpherical) {
      q1 = runtime_coordinate_for_axis(axes, MeshAxisR, dir, problo, dx, i, j, k);
      q2 = runtime_coordinate_for_axis(axes, MeshAxisTheta, dir, problo, dx, i, j, k);
      q3 = runtime_coordinate_for_axis(axes, MeshAxisPhi, dir, problo, dx, i, j, k);
   }
   const amrex::Real scale = runtime_direction_scale_factor(coordinate, axes, dir, q1, q2, q3);
   return amrex::max(scale, 1.0e-300) * dx[dir];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real curvilinear_face_gradient_spacing(
   int coordinate,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int dir,
   int i,
   int j,
   int k
) noexcept
{
   return curvilinear_face_gradient_spacing(coordinate, mesh_default_axes_layout(coordinate), problo, dx, dir, i, j, k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mesh_face_gradient_spacing(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int dir,
   int i,
   int j,
   int k
) noexcept
{
   if (coordinate == MeshCartesian) {
      amrex::ignore_unused(axes, problo, i, j, k);
      return dx[dir];
   }
   return curvilinear_face_gradient_spacing(coordinate, axes, problo, dx, dir, i, j, k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real coordinate_volume_scale(int coordinate, amrex::Real q1, amrex::Real q2, amrex::Real q3) noexcept
{
   if (coordinate == MeshCylindrical) {
      return coordinate_volume_scale<MeshCylindrical>(q1, q2, q3);
   }
   if (coordinate == MeshSpherical) {
      return coordinate_volume_scale<MeshSpherical>(q1, q2, q3);
   }
   return coordinate_volume_scale<MeshCartesian>(q1, q2, q3);
}

} // namespace pyfluid_amrex
