#pragma once

#include "pyfluid_amrex/reconstruction/scalar.hpp"

namespace pyfluid_amrex {

namespace reconstruction_detail {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ppm3_left(
   amrex::Real vmm,
   amrex::Real vm,
   amrex::Real v0,
   amrex::Real vp,
   amrex::Real vpp
) noexcept
{
   const amrex::Real slope_m = mc_slope(vmm, vm, v0);
   const amrex::Real slope_0 = mc_slope(vm, v0, vp);
   const amrex::Real slope_p = mc_slope(v0, vp, vpp);
   amrex::Real left = 0.5 * (vm + v0) + (slope_m - slope_0) / 6.0;
   amrex::Real right = 0.5 * (v0 + vp) + (slope_0 - slope_p) / 6.0;

   const amrex::Real right_delta = right - v0;
   const amrex::Real left_delta = left - v0;
   if (right_delta * left_delta >= 0.0) {
      return v0;
   }

   const amrex::Real interface_delta = right - left;
   const amrex::Real interface_delta_squared =
      interface_delta * interface_delta;
   const amrex::Real parabolic_curvature =
      6.0 * interface_delta * (v0 - 0.5 * (left + right));
   if (parabolic_curvature > interface_delta_squared) {
      left = 3.0 * v0 - 2.0 * right;
   } else if (parabolic_curvature < -interface_delta_squared) {
      right = 3.0 * v0 - 2.0 * left;
   }
   return right;
}

} // namespace reconstruction_detail

template <>
struct FivePointReconstruction<ReconPPM3> {
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   static amrex::Real left(
      amrex::Real vmm,
      amrex::Real vm,
      amrex::Real v0,
      amrex::Real vp,
      amrex::Real vpp
   ) noexcept
   {
      return reconstruction_detail::ppm3_left(vmm, vm, v0, vp, vpp);
   }
};

template <>
struct FivePointReconstruction<ReconPPM3Char>
   : FivePointReconstruction<ReconPPM3> {};

} // namespace pyfluid_amrex
