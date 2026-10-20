#pragma once

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>

#include "pyfluid_amrex/method_codes.hpp"

namespace pyfluid_amrex {

template <int ReconMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int solid_reconstruction_offset_lo() noexcept
{
   if constexpr (is_five_point_reconstruction(ReconMethod)) {
      return -2;
   } else if constexpr (ReconMethod == ReconPWL) {
      return -1;
   } else {
      return 0;
   }
}

template <int ReconMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int solid_reconstruction_offset_hi() noexcept
{
   if constexpr (is_five_point_reconstruction(ReconMethod)) {
      return 3;
   } else if constexpr (ReconMethod == ReconPWL) {
      return 2;
   } else {
      return 1;
   }
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool solid_mask_at_offset(
   amrex::Array4<int const> const& solid,
   int i,
   int j,
   int k,
   int offset
) noexcept
{
   if constexpr (Direction == 0) {
      return solid(i + offset, j, k, 0) != 0;
   }
#if (AMREX_SPACEDIM >= 2)
   else if constexpr (Direction == 1) {
      return solid(i, j + offset, k, 0) != 0;
   }
#endif
   return solid(i, j, k + offset, 0) != 0;
}

template <int ReconMethod, int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool solid_reconstruction_stencil_is_fluid(
   amrex::Array4<int const> const& solid,
   int i,
   int j,
   int k
) noexcept
{
   if constexpr (ReconMethod == ReconPWC) {
      return true;
   }
   for (int offset = solid_reconstruction_offset_lo<ReconMethod>();
        offset <= solid_reconstruction_offset_hi<ReconMethod>();
        ++offset) {
      if (solid_mask_at_offset<Direction>(solid, i, j, k, offset)) {
         return false;
      }
   }
   return true;
}

} // namespace pyfluid_amrex
