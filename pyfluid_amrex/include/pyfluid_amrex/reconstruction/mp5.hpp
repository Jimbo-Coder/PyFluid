#pragma once

#include "pyfluid_amrex/reconstruction/scalar.hpp"

namespace pyfluid_amrex {

namespace reconstruction_detail {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real minmod4(
   amrex::Real a,
   amrex::Real b,
   amrex::Real c,
   amrex::Real d
) noexcept
{
   if (a > 0.0 && b > 0.0 && c > 0.0 && d > 0.0) {
      return amrex::min(amrex::min(a, b), amrex::min(c, d));
   }
   if (a < 0.0 && b < 0.0 && c < 0.0 && d < 0.0) {
      return amrex::max(amrex::max(a, b), amrex::max(c, d));
   }
   return 0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mp5_left(
   amrex::Real vmm,
   amrex::Real vm,
   amrex::Real v0,
   amrex::Real vp,
   amrex::Real vpp
) noexcept
{
   constexpr amrex::Real alpha = 4.0;
   constexpr amrex::Real epsilon = 1.0e-12;
   const amrex::Real unlimited =
      (2.0 * vmm - 13.0 * vm + 47.0 * v0 + 27.0 * vp - 3.0 * vpp) / 60.0;
   const amrex::Real monotone = v0 + minmod(vp - v0, alpha * (v0 - vm));
   const amrex::Real scale = amrex::max(
      amrex::max(amrex::Math::abs(vmm), amrex::Math::abs(vm)),
      amrex::max(
         amrex::Math::abs(v0),
         amrex::max(amrex::Math::abs(vp), amrex::Math::abs(vpp))
      )
   );
   if ((unlimited - v0) * (unlimited - monotone)
       <= epsilon * amrex::max(1.0, scale * scale)) {
      return unlimited;
   }

   const amrex::Real dm = vmm - 2.0 * vm + v0;
   const amrex::Real d0 = vm - 2.0 * v0 + vp;
   const amrex::Real dp = v0 - 2.0 * vp + vpp;
   const amrex::Real d_m4_left = minmod4(4.0 * dm - d0, 4.0 * d0 - dm, dm, d0);
   const amrex::Real d_m4_right = minmod4(4.0 * d0 - dp, 4.0 * dp - d0, d0, dp);
   const amrex::Real upper_limit = v0 + alpha * (v0 - vm);
   const amrex::Real average = 0.5 * (v0 + vp);
   const amrex::Real median = average - 0.5 * d_m4_right;
   const amrex::Real large_curvature =
      v0 + 0.5 * (v0 - vm) + (4.0 / 3.0) * d_m4_left;
   const amrex::Real lower_bound = amrex::max(
      amrex::min(amrex::min(v0, vp), median),
      amrex::min(amrex::min(v0, upper_limit), large_curvature)
   );
   const amrex::Real upper_bound = amrex::min(
      amrex::max(amrex::max(v0, vp), median),
      amrex::max(amrex::max(v0, upper_limit), large_curvature)
   );
   return unlimited + minmod(lower_bound - unlimited, upper_bound - unlimited);
}

} // namespace reconstruction_detail

template <>
struct FivePointReconstruction<ReconMP5> {
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   static amrex::Real left(
      amrex::Real vmm,
      amrex::Real vm,
      amrex::Real v0,
      amrex::Real vp,
      amrex::Real vpp
   ) noexcept
   {
      return reconstruction_detail::mp5_left(vmm, vm, v0, vp, vpp);
   }
};

template <>
struct FivePointReconstruction<ReconMP5Char>
   : FivePointReconstruction<ReconMP5> {};

} // namespace pyfluid_amrex
