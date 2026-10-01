#pragma once

#include "pyfluid_amrex/reconstruction/scalar.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

namespace reconstruction_detail {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real wenoz_left(
   amrex::Real vmm,
   amrex::Real vm,
   amrex::Real v0,
   amrex::Real vp,
   amrex::Real vpp
) noexcept
{
   constexpr amrex::Real epsilon = 1.0e-8_rt;
   const amrex::Real d0 = vmm - 2.0_rt * vm + v0;
   const amrex::Real e0 = vmm - 4.0_rt * vm + 3.0_rt * v0;
   const amrex::Real d1 = vm - 2.0_rt * v0 + vp;
   const amrex::Real e1 = vm - vp;
   const amrex::Real d2 = v0 - 2.0_rt * vp + vpp;
   const amrex::Real e2 = 3.0_rt * v0 - 4.0_rt * vp + vpp;
   const amrex::Real beta0 = (13.0_rt / 12.0_rt) * d0 * d0 + 0.25_rt * e0 * e0;
   const amrex::Real beta1 = (13.0_rt / 12.0_rt) * d1 * d1 + 0.25_rt * e1 * e1;
   const amrex::Real beta2 = (13.0_rt / 12.0_rt) * d2 * d2 + 0.25_rt * e2 * e2;
   const amrex::Real tau5 = amrex::Math::abs(beta0 - beta2);
   const amrex::Real r0 = tau5 / (epsilon + beta0);
   const amrex::Real r1 = tau5 / (epsilon + beta1);
   const amrex::Real r2 = tau5 / (epsilon + beta2);
   const amrex::Real a0 = 0.1_rt * (1.0_rt + r0 * r0);
   const amrex::Real a1 = 0.6_rt * (1.0_rt + r1 * r1);
   const amrex::Real a2 = 0.3_rt * (1.0_rt + r2 * r2);
   const amrex::Real inv = 1.0_rt / (a0 + a1 + a2);
   const amrex::Real p0 = (2.0_rt * vmm - 7.0_rt * vm + 11.0_rt * v0) / 6.0_rt;
   const amrex::Real p1 = (-vm + 5.0_rt * v0 + 2.0_rt * vp) / 6.0_rt;
   const amrex::Real p2 = (2.0_rt * v0 + 5.0_rt * vp - vpp) / 6.0_rt;
   return (a0 * p0 + a1 * p1 + a2 * p2) * inv;
}

} // namespace reconstruction_detail

template <>
struct FivePointReconstruction<ReconWENOZ> {
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   static amrex::Real left(
      amrex::Real vmm,
      amrex::Real vm,
      amrex::Real v0,
      amrex::Real vp,
      amrex::Real vpp
   ) noexcept
   {
      return reconstruction_detail::wenoz_left(vmm, vm, v0, vp, vpp);
   }
};

template <>
struct FivePointReconstruction<ReconWENOZChar>
   : FivePointReconstruction<ReconWENOZ> {};

} // namespace pyfluid_amrex
