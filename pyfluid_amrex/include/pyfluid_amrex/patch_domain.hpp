#pragma once

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <array>

#include "pyfluid_amrex/mesh_limits.hpp"

namespace pyfluid_amrex {

struct PatchDomainBoxes {
   int count = 0;
   std::array<int, MaxMeshPatchBoxValues> small{};
   std::array<int, MaxMeshPatchBoxValues> big{};
};

struct PatchDomainFace {
   int li = 0;
   int lj = 0;
   int lk = 0;
   int ri = 0;
   int rj = 0;
   int rk = 0;
   bool left_inside_rectangular_domain = false;
   bool right_inside_rectangular_domain = false;
   bool left_inside_patch_domain = false;
   bool right_inside_patch_domain = false;
};

inline amrex::Vector<amrex::Box> patch_domain_box_vector(PatchDomainBoxes const& boxes)
{
   amrex::Vector<amrex::Box> result;
   result.reserve(boxes.count);
   for (int patch = 0; patch < boxes.count; ++patch) {
      amrex::IntVect small(AMREX_D_DECL(0, 0, 0));
      amrex::IntVect big(AMREX_D_DECL(0, 0, 0));
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         const int idx = patch * AMREX_SPACEDIM + dir;
         small[dir] = boxes.small[idx];
         big[dir] = boxes.big[idx];
      }
      result.emplace_back(small, big);
   }
   return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool patch_domain_contains_cell(
   PatchDomainBoxes const& boxes,
   int i,
   int j,
   int k
) noexcept
{
   for (int patch = 0; patch < boxes.count; ++patch) {
      const int base = patch * AMREX_SPACEDIM;
      bool inside = i >= boxes.small[base] && i <= boxes.big[base];
#if (AMREX_SPACEDIM >= 2)
      inside = inside && j >= boxes.small[base + 1] && j <= boxes.big[base + 1];
#else
      (void)j;
#endif
#if (AMREX_SPACEDIM == 3)
      inside = inside && k >= boxes.small[base + 2] && k <= boxes.big[base + 2];
#else
      (void)k;
#endif
      if (inside) {
         return true;
      }
   }
   return false;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool patch_domain_cell_inside_rectangular_domain(
   amrex::Dim3 const& lo,
   amrex::Dim3 const& hi,
   int i,
   int j,
   int k
) noexcept
{
   bool inside = i >= lo.x && i <= hi.x;
#if (AMREX_SPACEDIM >= 2)
   inside = inside && j >= lo.y && j <= hi.y;
#else
   (void)j;
#endif
#if (AMREX_SPACEDIM == 3)
   inside = inside && k >= lo.z && k <= hi.z;
#else
   (void)k;
#endif
   return inside;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchDomainFace patch_domain_face(
   PatchDomainBoxes const& boxes,
   amrex::Dim3 const& rectangular_lo,
   amrex::Dim3 const& rectangular_hi,
   int dir,
   int i,
   int j,
   int k
) noexcept
{
   PatchDomainFace face;
   face.li = i;
   face.lj = j;
   face.lk = k;
   face.ri = i;
   face.rj = j;
   face.rk = k;
   if (dir == 0) {
      face.li = i - 1;
   }
#if (AMREX_SPACEDIM >= 2)
   else if (dir == 1) {
      face.lj = j - 1;
   }
#endif
#if (AMREX_SPACEDIM == 3)
   else {
      face.lk = k - 1;
   }
#else
   (void)face.lk;
   (void)face.rk;
#endif

   face.left_inside_rectangular_domain = patch_domain_cell_inside_rectangular_domain(
      rectangular_lo,
      rectangular_hi,
      face.li,
      face.lj,
      face.lk
   );
   face.right_inside_rectangular_domain = patch_domain_cell_inside_rectangular_domain(
      rectangular_lo,
      rectangular_hi,
      face.ri,
      face.rj,
      face.rk
   );
   face.left_inside_patch_domain =
      face.left_inside_rectangular_domain
      && patch_domain_contains_cell(boxes, face.li, face.lj, face.lk);
   face.right_inside_patch_domain =
      face.right_inside_rectangular_domain
      && patch_domain_contains_cell(boxes, face.ri, face.rj, face.rk);
   return face;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool patch_domain_face_is_interior(PatchDomainFace const& face) noexcept
{
   return face.left_inside_patch_domain && face.right_inside_patch_domain;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool patch_domain_face_is_wall(PatchDomainFace const& face) noexcept
{
   return face.left_inside_patch_domain != face.right_inside_patch_domain;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool patch_domain_face_is_interior(
   PatchDomainBoxes const& boxes,
   amrex::Dim3 const& rectangular_lo,
   amrex::Dim3 const& rectangular_hi,
   int dir,
   int i,
   int j,
   int k
) noexcept
{
   return patch_domain_face_is_interior(
      patch_domain_face(boxes, rectangular_lo, rectangular_hi, dir, i, j, k)
   );
}

} // namespace pyfluid_amrex
