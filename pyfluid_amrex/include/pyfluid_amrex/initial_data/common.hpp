#pragma once

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/patch_maps.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real coordinate_delta(
   amrex::Real x,
   amrex::Real x0,
   amrex::Real lo,
   amrex::Real hi,
   bool periodic
) noexcept
{
   amrex::Real dx = x - x0;
   if (periodic) {
      const amrex::Real width = hi - lo;
      if (dx > 0.5_rt * width) {
         dx -= width;
      }
      if (dx < -0.5_rt * width) {
         dx += width;
      }
   }
   return dx;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool finite_wave_mode(amrex::Real mode) noexcept
{
   return mode == mode && mode > -std::numeric_limits<amrex::Real>::max()
      && mode < std::numeric_limits<amrex::Real>::max();
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real normalized_domain_coordinate(amrex::Real q, amrex::Real lo, amrex::Real hi) noexcept
{
   const amrex::Real width = hi - lo;
   return width > 0.0_rt ? (q - lo) / width : q;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool directional_wave_modes_enabled(amrex::Real mode1, amrex::Real mode2, amrex::Real mode3) noexcept
{
   return finite_wave_mode(mode1) || finite_wave_mode(mode2) || finite_wave_mode(mode3);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real wave_phase_coordinate(
   PatchCoordinate const& mesh,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi,
   amrex::Real mode1,
   amrex::Real mode2,
   amrex::Real mode3
) noexcept
{
   if (!directional_wave_modes_enabled(mode1, mode2, mode3)) {
      amrex::Real coordinate_sum = 0.0_rt;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         coordinate_sum += patch_coordinate_component(mesh, dir);
      }
      return coordinate_sum;
   }

   amrex::Real phase = 0.0_rt;
   if (finite_wave_mode(mode1)) {
      phase += mode1 * normalized_domain_coordinate(patch_coordinate_component(mesh, 0), problo[0], probhi[0]);
   }
#if (AMREX_SPACEDIM >= 2)
   if (finite_wave_mode(mode2)) {
      phase += mode2 * normalized_domain_coordinate(patch_coordinate_component(mesh, 1), problo[1], probhi[1]);
   }
#else
   amrex::ignore_unused(mode2);
#endif
#if (AMREX_SPACEDIM == 3)
   if (finite_wave_mode(mode3)) {
      phase += mode3 * normalized_domain_coordinate(patch_coordinate_component(mesh, 2), problo[2], probhi[2]);
   }
#else
   amrex::ignore_unused(mode3);
#endif
   return phase;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real bump_distance2(
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::Real x,
   amrex::Real y,
   amrex::Real z,
   amrex::Real x0,
   amrex::Real y0,
   amrex::Real z0,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi,
   amrex::GpuArray<int, AMREX_SPACEDIM> const& periodic
) noexcept
{
   const amrex::Real dx0 = coordinate_delta(x, x0, problo[0], probhi[0], periodic[0] != 0);
#if (AMREX_SPACEDIM == 1)
   amrex::ignore_unused(y, z, y0, z0, problo, probhi, periodic);
   return dx0 * dx0;
#elif (AMREX_SPACEDIM == 2)
   if (mesh_coordinate == MeshSpherical) {
      const amrex::Real x_phys = x * std::sin(y);
      const amrex::Real z_phys = x * std::cos(y);
      const amrex::Real x0_phys = x0 * std::sin(y0);
      const amrex::Real z0_phys = x0 * std::cos(y0);
      const amrex::Real dx_phys = x_phys - x0_phys;
      const amrex::Real dz_phys = z_phys - z0_phys;
      amrex::ignore_unused(z, z0);
      return dx_phys * dx_phys + dz_phys * dz_phys;
   }
   if (mesh_coordinate == MeshCylindrical && mesh_axis(mesh_axes, 1) == MeshAxisPhi) {
      const amrex::Real x_phys = x * std::cos(y);
      const amrex::Real y_phys = x * std::sin(y);
      const amrex::Real x0_phys = x0 * std::cos(y0);
      const amrex::Real y0_phys = x0 * std::sin(y0);
      const amrex::Real dx_phys = x_phys - x0_phys;
      const amrex::Real dy_phys = y_phys - y0_phys;
      amrex::ignore_unused(z, z0);
      return dx_phys * dx_phys + dy_phys * dy_phys;
   }
   const amrex::Real dy = coordinate_delta(y, y0, problo[1], probhi[1], periodic[1] != 0);
   amrex::ignore_unused(z, z0);
   return dx0 * dx0 + dy * dy;
#else
   if (mesh_coordinate == MeshSpherical) {
      const amrex::Real x_phys = x * std::sin(y) * std::cos(z);
      const amrex::Real y_phys = x * std::sin(y) * std::sin(z);
      const amrex::Real z_phys = x * std::cos(y);
      const amrex::Real x0_phys = x0 * std::sin(y0) * std::cos(z0);
      const amrex::Real y0_phys = x0 * std::sin(y0) * std::sin(z0);
      const amrex::Real z0_phys = x0 * std::cos(y0);
      const amrex::Real dx_phys = x_phys - x0_phys;
      const amrex::Real dy_phys = y_phys - y0_phys;
      const amrex::Real dz_phys = z_phys - z0_phys;
      return dx_phys * dx_phys + dy_phys * dy_phys + dz_phys * dz_phys;
   }
   if (mesh_coordinate == MeshCylindrical) {
      const bool dir1_is_phi = mesh_axis(mesh_axes, 1) == MeshAxisPhi;
      const amrex::Real phi = dir1_is_phi ? y : z;
      const amrex::Real phi0 = dir1_is_phi ? y0 : z0;
      const amrex::Real z_coord = dir1_is_phi ? z : y;
      const amrex::Real z0_coord = dir1_is_phi ? z0 : y0;
      const amrex::Real zlo = dir1_is_phi ? problo[2] : problo[1];
      const amrex::Real zhi = dir1_is_phi ? probhi[2] : probhi[1];
      const bool zperiodic = dir1_is_phi ? (periodic[2] != 0) : (periodic[1] != 0);
      const amrex::Real x_phys = x * std::cos(phi);
      const amrex::Real y_phys = x * std::sin(phi);
      const amrex::Real x0_phys = x0 * std::cos(phi0);
      const amrex::Real y0_phys = x0 * std::sin(phi0);
      const amrex::Real dz = coordinate_delta(z_coord, z0_coord, zlo, zhi, zperiodic);
      const amrex::Real dx_phys = x_phys - x0_phys;
      const amrex::Real dy_phys = y_phys - y0_phys;
      return dx_phys * dx_phys + dy_phys * dy_phys + dz * dz;
   }
   const amrex::Real dy = coordinate_delta(y, y0, problo[1], probhi[1], periodic[1] != 0);
   const amrex::Real dz = coordinate_delta(z, z0, problo[2], probhi[2], periodic[2] != 0);
   return dx0 * dx0 + dy * dy + dz * dz;
#endif
}

} // namespace pyfluid_amrex
