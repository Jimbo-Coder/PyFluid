#pragma once

#include <AMReX_Array4.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuLaunch.H>

#include "numerics/solid_flux_stencil.hpp"
#include "pyfluid_amrex/generated/field_ids.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/reconstruction/scalar.hpp"

namespace pyfluid_amrex {

// Component views keep this launch independent of the hydro/MHD layout and
// Riemann/EoS choice. Mass flux is already in the face's stored area convention.
// Keep host traversal out of each fluid launcher; device point work stays local
// to its kernel. Inlining this host loop regressed the measured MP5/HLLC path.
template <int ReconMethod, int Direction>
AMREX_NO_INLINE inline void fill_passive_flux_tile(
   amrex::Box const& face_box,
   amrex::Array4<amrex::Real> const& flux,
   amrex::Array4<amrex::Real const> const& primitive,
   int passive_component,
   int mesh_coordinate = MeshCartesian,
   MeshAxes mesh_axes = mesh_default_axes_layout(MeshCartesian),
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo = {},
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = {},
   bool skip_zero_area_metric_faces = false
)
{
   const amrex::Array4<amrex::Real> dye_flux(flux, passive_component, 1);
   const amrex::Array4<amrex::Real const> dye(primitive, passive_component, 1);
   amrex::ParallelFor(face_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      if (skip_zero_area_metric_faces
          && mesh_face_area_scale(Direction, mesh_coordinate, mesh_axes, problo, dx, i, j, k) == 0.0_rt) {
         dye_flux(i, j, k, 0) = 0.0_rt;
         return;
      }
      int li = i;
      int lj = j;
      int lk = k;
      if constexpr (Direction == 0) {
         li = i - 1;
      }
#if (AMREX_SPACEDIM >= 2)
      else if constexpr (Direction == 1) {
         lj = j - 1;
      }
#endif
#if (AMREX_SPACEDIM == 3)
      else {
         lk = k - 1;
      }
#endif
      const ScalarFaceValues face_dye = primitive_component_face_values<ReconMethod, Direction, 0>(
         dye, li, lj, lk, i, j, k
      );
      const amrex::Real mass_flux = flux(i, j, k, RHO);
      dye_flux(i, j, k, 0) = mass_flux * (mass_flux >= 0.0_rt ? face_dye.left : face_dye.right);
   });
}

AMREX_NO_INLINE inline void fill_zero_passive_flux_tile(
   amrex::Box const& face_box,
   amrex::Array4<amrex::Real> const& flux,
   int passive_component
)
{
   const amrex::Array4<amrex::Real> dye_flux(flux, passive_component, 1);
   amrex::ParallelFor(face_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      dye_flux(i, j, k, 0) = 0.0_rt;
   });
}

// The caller reuses the physics tile classification. This mixed-tile launch
// follows the same stencil fallback and one-sided wall value as the base flux.
template <int ReconMethod, int Direction>
AMREX_NO_INLINE inline void fill_solid_passive_flux_tile(
   amrex::Box const& face_box,
   amrex::Array4<amrex::Real> const& flux,
   amrex::Array4<amrex::Real const> const& primitive,
   amrex::Array4<int const> const& solid,
   amrex::Box const& domain,
   int passive_component
)
{
   const amrex::Array4<amrex::Real> dye_flux(flux, passive_component, 1);
   const amrex::Array4<amrex::Real const> dye(primitive, passive_component, 1);
   const auto lo = domain.smallEnd();
   const auto hi = domain.bigEnd();
   amrex::ParallelFor(face_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      int li = i;
      int lj = j;
      int lk = k;
      if constexpr (Direction == 0) {
         li = i - 1;
      }
#if (AMREX_SPACEDIM >= 2)
      else if constexpr (Direction == 1) {
         lj = j - 1;
      }
#endif
#if (AMREX_SPACEDIM == 3)
      else {
         lk = k - 1;
      }
#endif
      bool left_in_domain = li >= lo[0] && li <= hi[0];
      bool right_in_domain = i >= lo[0] && i <= hi[0];
#if (AMREX_SPACEDIM >= 2)
      left_in_domain = left_in_domain && lj >= lo[1] && lj <= hi[1];
      right_in_domain = right_in_domain && j >= lo[1] && j <= hi[1];
#endif
#if (AMREX_SPACEDIM == 3)
      left_in_domain = left_in_domain && lk >= lo[2] && lk <= hi[2];
      right_in_domain = right_in_domain && k >= lo[2] && k <= hi[2];
#endif
      const bool left_solid = left_in_domain && solid(li, lj, lk, 0) != 0;
      const bool right_solid = right_in_domain && solid(i, j, k, 0) != 0;
      const amrex::Real mass_flux = flux(i, j, k, RHO);
      if (!left_solid && !right_solid) {
         ScalarFaceValues face_dye;
         if (solid_reconstruction_stencil_is_fluid<ReconMethod, Direction>(solid, li, lj, lk)) {
            face_dye = primitive_component_face_values<ReconMethod, Direction, 0>(dye, li, lj, lk, i, j, k);
         } else {
            face_dye = primitive_component_face_values<ReconPWC, Direction, 0>(dye, li, lj, lk, i, j, k);
         }
         dye_flux(i, j, k, 0) = mass_flux * (mass_flux >= 0.0_rt ? face_dye.left : face_dye.right);
      } else if (left_solid && right_solid) {
         dye_flux(i, j, k, 0) = 0.0_rt;
      } else {
         const int fi = left_solid ? i : li;
         const int fj = left_solid ? j : lj;
         const int fk = left_solid ? k : lk;
         dye_flux(i, j, k, 0) = mass_flux * dye(fi, fj, fk, 0);
      }
   });
}

} // namespace pyfluid_amrex
