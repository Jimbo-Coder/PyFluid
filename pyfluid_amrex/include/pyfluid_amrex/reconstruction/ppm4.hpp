#pragma once

#include "pyfluid_amrex/reconstruction/parabolic.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

namespace reconstruction_detail {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real clamp_between(
   amrex::Real value,
   amrex::Real first,
   amrex::Real second
) noexcept
{
   return amrex::max(
      amrex::min(first, second),
      amrex::min(amrex::max(first, second), value)
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ppm4_left(
   amrex::Real vmm,
   amrex::Real vm,
   amrex::Real v0,
   amrex::Real vp,
   amrex::Real vpp
) noexcept
{
   amrex::Real left =
      clamp_between(parabolic_right_interface(vmm, vm, v0, vp), vm, v0);
   amrex::Real right =
      clamp_between(parabolic_right_interface(vm, v0, vp, vpp), v0, vp);

   const amrex::Real right_delta = right - v0;
   const amrex::Real left_delta = left - v0;
   if (right_delta * left_delta >= 0.0_rt) {
      return v0;
   }
   if (amrex::Math::abs(right_delta) >= 2.0_rt * amrex::Math::abs(left_delta)) {
      right = v0 - 2.0_rt * left_delta;
   }
   return right;
}

} // namespace reconstruction_detail

template <>
struct FivePointReconstruction<ReconPPM4> {
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   static amrex::Real left(
      amrex::Real vmm,
      amrex::Real vm,
      amrex::Real v0,
      amrex::Real vp,
      amrex::Real vpp
   ) noexcept
   {
      return reconstruction_detail::ppm4_left(vmm, vm, v0, vp, vpp);
   }
};

template <>
struct FivePointReconstruction<ReconPPM4Char>
   : FivePointReconstruction<ReconPPM4> {};

} // namespace pyfluid_amrex
