#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include <array>

#include "pyfluid_amrex/patch_domain.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

template <int NumComponents, int Direction>
inline void apply_patch_union_hydro_wall_flux_direction(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   amrex::Geometry const& geometry,
   PatchDomainBoxes const& patch_domain
)
{
   amrex::MultiFab& flux = *fluxes[Direction];
   const amrex::Box domain = geometry.Domain();
   const auto domain_lo = amrex::lbound(domain);
   const auto domain_hi = amrex::ubound(domain);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto primitive_values = primitive.const_array(mfi);
      const auto flux_values = flux.array(mfi);
      amrex::ParallelFor(box, [patch_domain, domain_lo, domain_hi, primitive_values, flux_values] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const PatchDomainFace face = patch_domain_face(
            patch_domain, domain_lo, domain_hi, Direction, i, j, k
         );
         if (!patch_domain_face_is_wall(face)) {
            return;
         }

         const int fluid_i = face.left_inside_patch_domain ? face.li : face.ri;
         const int fluid_j = face.left_inside_patch_domain ? face.lj : face.rj;
         const int fluid_k = face.left_inside_patch_domain ? face.lk : face.rk;
         for (int component = 0; component < NumComponents; ++component) {
            flux_values(i, j, k, component) = 0.0;
         }
         if constexpr (Direction == 0) {
            flux_values(i, j, k, MOM1) = primitive_values(fluid_i, fluid_j, fluid_k, ENERGY);
         }
#if (AMREX_SPACEDIM >= 2)
         else if constexpr (Direction == 1) {
            flux_values(i, j, k, MOM2) = primitive_values(fluid_i, fluid_j, fluid_k, ENERGY);
         }
#endif
#if (AMREX_SPACEDIM == 3)
         else {
            flux_values(i, j, k, MOM3) = primitive_values(fluid_i, fluid_j, fluid_k, ENERGY);
         }
#endif
      });
   }
   }
}

template <int NumComponents>
inline void apply_patch_union_hydro_wall_fluxes(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   amrex::Geometry const& geometry,
   PatchDomainBoxes const& patch_domain
)
{
   apply_patch_union_hydro_wall_flux_direction<NumComponents, 0>(
      fluxes, primitive, geometry, patch_domain
   );
#if (AMREX_SPACEDIM >= 2)
   apply_patch_union_hydro_wall_flux_direction<NumComponents, 1>(
      fluxes, primitive, geometry, patch_domain
   );
#endif
#if (AMREX_SPACEDIM == 3)
   apply_patch_union_hydro_wall_flux_direction<NumComponents, 2>(
      fluxes, primitive, geometry, patch_domain
   );
#endif
}

} // namespace pyfluid_amrex
