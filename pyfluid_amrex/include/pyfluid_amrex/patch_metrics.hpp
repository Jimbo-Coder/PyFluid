#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/patch_grid.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/patch_layout_builder.hpp"
#include "pyfluid_amrex/patch_maps.hpp"

#include <cmath>

namespace pyfluid_amrex {

using PhysicalVector = Vector3;

struct PatchFaceMetric {
   PhysicalVector unit_normal{};
   amrex::Real area = 1.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PhysicalVector physical_vector(
   amrex::Real x,
   amrex::Real y = 0.0,
   amrex::Real z = 0.0
) noexcept
{
   return {x, y, z};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real dot(PhysicalVector a, PhysicalVector b) noexcept
{
   return a.dot(b);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PhysicalVector cross(PhysicalVector a, PhysicalVector b) noexcept
{
   return a.cross(b);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real norm(PhysicalVector a) noexcept
{
   return std::sqrt(dot(a, a));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PhysicalVector scale(PhysicalVector a, amrex::Real s) noexcept
{
   return a * s;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PhysicalVector normalized(PhysicalVector a) noexcept
{
   const amrex::Real magnitude = norm(a);
   if (magnitude <= 0.0 || !amrex::Math::isfinite(magnitude)) {
      return {1.0, 0.0, 0.0};
   }
   return scale(a, 1.0 / magnitude);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PhysicalVector align_with(PhysicalVector vector, PhysicalVector direction) noexcept
{
   return dot(vector, direction) < 0.0 ? scale(vector, -1.0) : vector;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PhysicalVector patch_basis_vector(
   PatchGridDescriptor const& patch,
   PatchCoordinate const& local,
   int dir
) noexcept
{
   const PatchMapJacobian jacobian = patch_local_to_global_jacobian(patch.map, local);
   const int local_component = patch_local_component_for_dir(patch.local_axes, dir);
   return {
      patch_map_jacobian_component(jacobian, 0, local_component),
      patch_map_jacobian_component(jacobian, 1, local_component),
      patch_map_jacobian_component(jacobian, 2, local_component),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_shifted_coordinate(
   PatchGridDescriptor const& patch,
   PatchCoordinate local,
   int dir,
   amrex::Real cell_offsets
) noexcept
{
   const int local_component = patch_local_component_for_dir(patch.local_axes, dir);
   const amrex::Real coordinate = patch_coordinate_component(local, local_component)
      + cell_offsets * patch_grid_cell_size(patch, dir);
   set_patch_coordinate_component(local, local_component, coordinate);
   return local;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real patch_cell_volume(
   PatchGridDescriptor const& patch,
   PatchCoordinate const& local
) noexcept
{
   const PhysicalVector e0 = patch_basis_vector(patch, local, 0);
   const amrex::Real dq0 = patch_grid_cell_size(patch, 0);
#if (AMREX_SPACEDIM == 1)
   return norm(e0) * dq0;
#elif (AMREX_SPACEDIM == 2)
   const PhysicalVector e1 = patch_basis_vector(patch, local, 1);
   const amrex::Real dq1 = patch_grid_cell_size(patch, 1);
   return norm(cross(e0, e1)) * dq0 * dq1;
#else
   const PhysicalVector e1 = patch_basis_vector(patch, local, 1);
   const PhysicalVector e2 = patch_basis_vector(patch, local, 2);
   const amrex::Real dq1 = patch_grid_cell_size(patch, 1);
   const amrex::Real dq2 = patch_grid_cell_size(patch, 2);
   return amrex::Math::abs(dot(e0, cross(e1, e2))) * dq0 * dq1 * dq2;
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchFaceMetric patch_face_metric(
   PatchGridDescriptor const& patch,
   PatchCoordinate const& local,
   int dir
) noexcept
{
#if (AMREX_SPACEDIM == 1)
   amrex::ignore_unused(patch, local);
   return {physical_vector(1.0, 0.0, 0.0), 1.0};
#elif (AMREX_SPACEDIM == 2)
   const PhysicalVector e0 = patch_basis_vector(patch, local, 0);
   const PhysicalVector e1 = patch_basis_vector(patch, local, 1);
   const PhysicalVector surface_normal = normalized(cross(e0, e1));
   const int tangent_dir = dir == 0 ? 1 : 0;
   const PatchCoordinate lo = patch_shifted_coordinate(patch, local, tangent_dir, -0.5);
   const PatchCoordinate hi = patch_shifted_coordinate(patch, local, tangent_dir, 0.5);
   const PatchCoordinate global_lo = patch_local_to_global(patch.map, lo);
   const PatchCoordinate global_hi = patch_local_to_global(patch.map, hi);
   const PhysicalVector edge = physical_vector(
      global_hi.q1 - global_lo.q1,
      global_hi.q2 - global_lo.q2,
      global_hi.q3 - global_lo.q3
   );
   if (dir == 0) {
      const PhysicalVector area_vector = align_with(
         cross(edge, surface_normal),
         e0
      );
      return {normalized(area_vector), norm(area_vector)};
   }
   const PhysicalVector area_vector = align_with(
      cross(surface_normal, edge),
      e1
   );
   return {normalized(area_vector), norm(area_vector)};
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
   PatchCoordinate c00 = patch_shifted_coordinate(patch, local, tangent0, -0.5);
   c00 = patch_shifted_coordinate(patch, c00, tangent1, -0.5);
   PatchCoordinate c10 = patch_shifted_coordinate(patch, local, tangent0, 0.5);
   c10 = patch_shifted_coordinate(patch, c10, tangent1, -0.5);
   PatchCoordinate c11 = patch_shifted_coordinate(patch, local, tangent0, 0.5);
   c11 = patch_shifted_coordinate(patch, c11, tangent1, 0.5);
   PatchCoordinate c01 = patch_shifted_coordinate(patch, local, tangent0, -0.5);
   c01 = patch_shifted_coordinate(patch, c01, tangent1, 0.5);
   const PatchCoordinate g00 = patch_local_to_global(patch.map, c00);
   const PatchCoordinate g10 = patch_local_to_global(patch.map, c10);
   const PatchCoordinate g11 = patch_local_to_global(patch.map, c11);
   const PatchCoordinate g01 = patch_local_to_global(patch.map, c01);
   const PhysicalVector p00 = physical_vector(g00.q1, g00.q2, g00.q3);
   const PhysicalVector p10 = physical_vector(g10.q1, g10.q2, g10.q3);
   const PhysicalVector p11 = physical_vector(g11.q1, g11.q2, g11.q3);
   const PhysicalVector p01 = physical_vector(g01.q1, g01.q2, g01.q3);
   const PhysicalVector triangle0 = cross(
      physical_vector(p10.x - p00.x, p10.y - p00.y, p10.z - p00.z),
      physical_vector(p11.x - p00.x, p11.y - p00.y, p11.z - p00.z)
   );
   const PhysicalVector triangle1 = cross(
      physical_vector(p11.x - p00.x, p11.y - p00.y, p11.z - p00.z),
      physical_vector(p01.x - p00.x, p01.y - p00.y, p01.z - p00.z)
   );
   PhysicalVector area_vector = scale(
      physical_vector(
         triangle0.x + triangle1.x,
         triangle0.y + triangle1.y,
         triangle0.z + triangle1.z
      ),
      0.5
   );
   area_vector = align_with(area_vector, patch_basis_vector(patch, local, dir));
   return {normalized(area_vector), norm(area_vector)};
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_layout_cell_center_coordinate(
   PatchGridDescriptor const& grid_patch,
   PatchDescriptor const& layout_patch,
   int i,
   int j,
   int k
) noexcept
{
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> offsets{};
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      offsets[dir] = 0.5;
   }
   return patch_layout_local_coordinate(
      grid_patch,
      layout_patch,
      amrex::IntVect(AMREX_D_DECL(i, j, k)),
      offsets
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_layout_face_coordinate(
   PatchGridDescriptor const& grid_patch,
   PatchDescriptor const& layout_patch,
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
   return patch_layout_local_coordinate(
      grid_patch,
      layout_patch,
      amrex::IntVect(AMREX_D_DECL(i, j, k)),
      offsets
   );
}

} // namespace pyfluid_amrex
