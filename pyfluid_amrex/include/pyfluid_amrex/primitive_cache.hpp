#pragma once

#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/mask_regions.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

template <int NComp, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_primitive_cache_cell(
   amrex::Array4<amrex::Real const> const& state,
   amrex::Array4<amrex::Real> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& eos
) noexcept
{
   if constexpr (has_mhd_components<NComp>()) {
      const MHDPrimitive q = primitive_from_conserved<EOSMethod>(load_mhd_state(state, i, j, k), eos);
      primitive(i, j, k, RHO) = q.rho;
      primitive(i, j, k, MOM1) = q.v1;
      primitive(i, j, k, MOM2) = q.v2;
      primitive(i, j, k, MOM3) = q.v3;
      primitive(i, j, k, ENERGY) = q.pressure;
      primitive(i, j, k, B1) = q.b1;
      primitive(i, j, k, B2) = q.b2;
      primitive(i, j, k, B3) = q.b3;
   } else {
      const HydroPrimitive q = primitive_from_conserved<EOSMethod>(load_hydro_state(state, i, j, k), eos);
      primitive(i, j, k, RHO) = q.rho;
      primitive(i, j, k, MOM1) = q.v1;
      primitive(i, j, k, MOM2) = q.v2;
      primitive(i, j, k, MOM3) = q.v3;
      primitive(i, j, k, ENERGY) = q.pressure;
   }
   for (int q = base_component_count<NComp>(); q < NComp; ++q) {
      primitive(i, j, k, q) = state(i, j, k, q) / amrex::max(state(i, j, k, RHO), 1.0e-300);
   }
}

template <int NComp, int EOSMethod>
inline void fill_primitive_cache(amrex::MultiFab& primitive, amrex::MultiFab const& state, EOSParams const& eos)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(primitive); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.fabbox();
      const auto s = state.const_array(mfi);
      const auto p = primitive.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         store_primitive_cache_cell<NComp, EOSMethod>(s, p, i, j, k, eos);
      });
   }
   }
}

template <int NComp, int EOSMethod>
inline void fill_primitive_cache_masked(
   amrex::MultiFab& primitive,
   amrex::MultiFab const& state,
   amrex::iMultiFab const& primitive_needed,
   EOSParams const& eos,
   amrex::Vector<int> const* primitive_needed_kind = nullptr
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(primitive); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.fabbox();
      const auto s = state.const_array(mfi);
      const auto needed = primitive_needed.const_array(mfi);
      const auto p = primitive.array(mfi);
      const int kind = cached_mask_kind(primitive_needed_kind, mfi, primitive_needed, bx);
      if (kind == MaskKindZero) {
         continue;
      }
      if (kind == MaskKindNonzero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            store_primitive_cache_cell<NComp, EOSMethod>(s, p, i, j, k, eos);
         });
         continue;
      }
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (needed(i, j, k, 0) == 0) {
            return;
         }
         store_primitive_cache_cell<NComp, EOSMethod>(s, p, i, j, k, eos);
      });
   }
   }
}

} // namespace pyfluid_amrex
