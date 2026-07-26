#pragma once

#include "pyfluid_amrex/reconstruction/parabolic.hpp"

namespace pyfluid_amrex {

namespace reconstruction_detail {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real limited_second_derivative(
   amrex::Real candidate,
   amrex::Real first,
   amrex::Real second
) noexcept
{
   if (
      (candidate > 0.0 && first > 0.0 && second > 0.0)
      || (candidate < 0.0 && first < 0.0 && second < 0.0)
   ) {
      const amrex::Real magnitude = amrex::min(
         amrex::Math::abs(candidate),
         1.25 * amrex::min(
            amrex::Math::abs(first),
            amrex::Math::abs(second)
         )
      );
      return candidate > 0.0 ? magnitude : -magnitude;
   }
   return 0.0;
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
      (candidate > 0.0 && center > 0.0 && left > 0.0 && right > 0.0)
      || (candidate < 0.0 && center < 0.0 && left < 0.0 && right < 0.0)
   ) {
      const amrex::Real neighbor_magnitude = amrex::min(
         amrex::Math::abs(center),
         amrex::min(amrex::Math::abs(left), amrex::Math::abs(right))
      );
      const amrex::Real magnitude = amrex::min(
         amrex::Math::abs(candidate),
         1.25 * neighbor_magnitude
      );
      return candidate > 0.0 ? magnitude : -magnitude;
   }
   return 0.0;
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

   amrex::Real center_second = 3.0 * ((vm + v0) - 2.0 * left);
   amrex::Real left_second = (vmm + v0) - 2.0 * vm;
   amrex::Real right_second = (vm + vp) - 2.0 * v0;
   amrex::Real limited_second =
      limited_second_derivative(center_second, left_second, right_second);
   if ((vm - left) * (v0 - left) > 0.0) {
      left = 0.5 * (v0 + vm) - limited_second / 6.0;
   }

   center_second = 3.0 * ((v0 + vp) - 2.0 * right);
   left_second = right_second;
   right_second = (v0 + vpp) - 2.0 * vp;
   limited_second =
      limited_second_derivative(center_second, left_second, right_second);
   if ((v0 - right) * (vp - right) > 0.0) {
      right = 0.5 * (v0 + vp) - limited_second / 6.0;
   }

   const amrex::Real parabolic_monotonicity =
      (right - v0) * (v0 - left);
   const amrex::Real cell_monotonicity =
      (vm - v0) * (v0 - vp);
   if (parabolic_monotonicity <= 0.0 || cell_monotonicity <= 0.0) {
      const amrex::Real parabolic_second =
         6.0 * (left + right - 2.0 * v0);
      const amrex::Real cell_second = (vm + vp) - 2.0 * v0;
      const amrex::Real previous_second = (vmm + v0) - 2.0 * vm;
      const amrex::Real next_second = (v0 + vpp) - 2.0 * vp;
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
      amrex::Real limiter_ratio = 0.0;
      if (
         amrex::Math::abs(parabolic_second)
         > 1.0e-12 * scale
      ) {
         limiter_ratio = limited_second / parabolic_second;
      }
      return v0 + (right - v0) * limiter_ratio;
   }

   const amrex::Real right_delta = right - v0;
   const amrex::Real left_delta = left - v0;
   if (amrex::Math::abs(right_delta) >= 2.0 * amrex::Math::abs(left_delta)) {
      right = v0 - 2.0 * left_delta;
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
