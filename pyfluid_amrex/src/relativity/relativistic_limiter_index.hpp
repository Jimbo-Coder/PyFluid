#pragma once

#include <AMReX_GpuQualifiers.H>

namespace pyfluid_amrex::limiter_detail {

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void left_cell(int i, int j, int k, int& il, int& jl, int& kl) noexcept
{
   il = i;
   jl = j;
   kl = k;
   if constexpr (Direction == 0) {
      --il;
   } else if constexpr (Direction == 1) {
      --jl;
   } else {
      --kl;
   }
}

} // namespace pyfluid_amrex::limiter_detail
