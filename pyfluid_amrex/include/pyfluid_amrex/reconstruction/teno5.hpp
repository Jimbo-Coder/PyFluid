#pragma once

#include "pyfluid_amrex/reconstruction/scalar.hpp"

namespace pyfluid_amrex {

namespace reconstruction_detail {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real teno5_left(
   amrex::Real vmm,
   amrex::Real vm,
   amrex::Real v0,
   amrex::Real vp,
   amrex::Real vpp
) noexcept
{
   constexpr amrex::Real epsilon = 1.0e-40;
   constexpr amrex::Real cutoff = 1.0e-5;
   const amrex::Real d0 = vmm - 2.0 * vm + v0;
   const amrex::Real e0 = vmm - 4.0 * vm + 3.0 * v0;
   const amrex::Real d1 = vm - 2.0 * v0 + vp;
   const amrex::Real e1 = vm - vp;
   const amrex::Real d2 = v0 - 2.0 * vp + vpp;
   const amrex::Real e2 = 3.0 * v0 - 4.0 * vp + vpp;
   const amrex::Real beta0 = (13.0 / 12.0) * d0 * d0 + 0.25 * e0 * e0;
   const amrex::Real beta1 = (13.0 / 12.0) * d1 * d1 + 0.25 * e1 * e1;
   const amrex::Real beta2 = (13.0 / 12.0) * d2 * d2 + 0.25 * e2 * e2;
   const amrex::Real tau5 = amrex::Math::abs(beta0 - beta2);
   const amrex::Real r0 = 1.0 + tau5 / (beta0 + epsilon);
   const amrex::Real r1 = 1.0 + tau5 / (beta1 + epsilon);
   const amrex::Real r2 = 1.0 + tau5 / (beta2 + epsilon);
   const amrex::Real g0 = r0 * r0 * r0 * r0 * r0 * r0;
   const amrex::Real g1 = r1 * r1 * r1 * r1 * r1 * r1;
   const amrex::Real g2 = r2 * r2 * r2 * r2 * r2 * r2;
   const amrex::Real inv_gamma = 1.0 / (g0 + g1 + g2);
   const amrex::Real a0 = (g0 * inv_gamma >= cutoff) ? 0.1 : 0.0;
   const amrex::Real a1 = (g1 * inv_gamma >= cutoff) ? 0.6 : 0.0;
   const amrex::Real a2 = (g2 * inv_gamma >= cutoff) ? 0.3 : 0.0;
   const amrex::Real inv_alpha = 1.0 / (a0 + a1 + a2);
   const amrex::Real p0 = (2.0 * vmm - 7.0 * vm + 11.0 * v0) / 6.0;
   const amrex::Real p1 = (-vm + 5.0 * v0 + 2.0 * vp) / 6.0;
   const amrex::Real p2 = (2.0 * v0 + 5.0 * vp - vpp) / 6.0;
   return (a0 * p0 + a1 * p1 + a2 * p2) * inv_alpha;
}

} // namespace reconstruction_detail

template <>
struct FivePointReconstruction<ReconTENO5> {
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   static amrex::Real left(
      amrex::Real vmm,
      amrex::Real vm,
      amrex::Real v0,
      amrex::Real vp,
      amrex::Real vpp
   ) noexcept
   {
      return reconstruction_detail::teno5_left(vmm, vm, v0, vp, vpp);
   }
};

template <>
struct FivePointReconstruction<ReconTENO5Char>
   : FivePointReconstruction<ReconTENO5> {};

} // namespace pyfluid_amrex
