#pragma once

#include "pyfluid_amrex/reconstruction/parabolic.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

namespace reconstruction_detail {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real limited_second_derivative(
   amrex::Real candidate,
   amrex::Real first,
   amrex::Real second
) noexcept
{
   if (
      (candidate > 0.0_rt && first > 0.0_rt && second > 0.0_rt)
      || (candidate < 0.0_rt && first < 0.0_rt && second < 0.0_rt)
   ) {
      const amrex::Real magnitude = amrex::min(
         amrex::Math::abs(candidate),
         1.25_rt * amrex::min(
            amrex::Math::abs(first),
            amrex::Math::abs(second)
         )
      );
      return candidate > 0.0_rt ? magnitude : -magnitude;
   }
   return 0.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real limited_second_derivative(
   amrex::Real candidate,
   amrex::Real center,
   amrex::Real left,
   amrex::Real right
) noexcept
{
   if (
      (candidate > 0.0_rt && center > 0.0_rt && left > 0.0_rt && right > 0.0_rt)
      || (candidate < 0.0_rt && center < 0.0_rt && left < 0.0_rt && right < 0.0_rt)
   ) {
      const amrex::Real neighbor_magnitude = amrex::min(
         amrex::Math::abs(center),
         amrex::min(amrex::Math::abs(left), amrex::Math::abs(right))
      );
      const amrex::Real magnitude = amrex::min(
         amrex::Math::abs(candidate),
         1.25_rt * neighbor_magnitude
      );
      return candidate > 0.0_rt ? magnitude : -magnitude;
   }
   return 0.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ppmx_left(
   amrex::Real vmm,
   amrex::Real vm,
   amrex::Real v0,
   amrex::Real vp,
   amrex::Real vpp
) noexcept
{
   amrex::Real left = parabolic_right_interface(vmm, vm, v0, vp);
   amrex::Real right = parabolic_right_interface(vm, v0, vp, vpp);

   amrex::Real center_second = 3.0_rt * ((vm + v0) - 2.0_rt * left);
   amrex::Real left_second = (vmm + v0) - 2.0_rt * vm;
   amrex::Real right_second = (vm + vp) - 2.0_rt * v0;
   amrex::Real limited_second =
      limited_second_derivative(center_second, left_second, right_second);
   if ((vm - left) * (v0 - left) > 0.0_rt) {
      left = 0.5_rt * (v0 + vm) - limited_second / 6.0_rt;
   }

   center_second = 3.0_rt * ((v0 + vp) - 2.0_rt * right);
   left_second = right_second;
   right_second = (v0 + vpp) - 2.0_rt * vp;
   limited_second =
      limited_second_derivative(center_second, left_second, right_second);
   if ((v0 - right) * (vp - right) > 0.0_rt) {
      right = 0.5_rt * (v0 + vp) - limited_second / 6.0_rt;
   }

   const amrex::Real parabolic_monotonicity =
      (right - v0) * (v0 - left);
   const amrex::Real cell_monotonicity =
      (vm - v0) * (v0 - vp);
   if (parabolic_monotonicity <= 0.0_rt || cell_monotonicity <= 0.0_rt) {
      const amrex::Real parabolic_second =
         6.0_rt * (left + right - 2.0_rt * v0);
      const amrex::Real cell_second = (vm + vp) - 2.0_rt * v0;
      const amrex::Real previous_second = (vmm + v0) - 2.0_rt * vm;
      const amrex::Real next_second = (v0 + vpp) - 2.0_rt * vp;
      limited_second = limited_second_derivative(
         parabolic_second,
         cell_second,
         previous_second,
         next_second
      );
      const amrex::Real scale = amrex::max(
         amrex::Math::abs(vm),
         amrex::max(amrex::Math::abs(v0), amrex::Math::abs(vp))
      );
      amrex::Real limiter_ratio = 0.0_rt;
      if (
         amrex::Math::abs(parabolic_second)
         > 1.0e-12_rt * scale
      ) {
         limiter_ratio = limited_second / parabolic_second;
      }
      return v0 + (right - v0) * limiter_ratio;
   }

   const amrex::Real right_delta = right - v0;
   const amrex::Real left_delta = left - v0;
   if (amrex::Math::abs(right_delta) >= 2.0_rt * amrex::Math::abs(left_delta)) {
      right = v0 - 2.0_rt * left_delta;
   }
   return right;
}

} // namespace reconstruction_detail

template <>
struct FivePointReconstruction<ReconPPMX> {
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   static amrex::Real left(
      amrex::Real vmm,
      amrex::Real vm,
      amrex::Real v0,
      amrex::Real vp,
      amrex::Real vpp
   ) noexcept
   {
      return reconstruction_detail::ppmx_left(vmm, vm, v0, vp, vpp);
   }
};

template <>
struct FivePointReconstruction<ReconPPMXChar>
   : FivePointReconstruction<ReconPPMX> {};

} // namespace pyfluid_amrex
