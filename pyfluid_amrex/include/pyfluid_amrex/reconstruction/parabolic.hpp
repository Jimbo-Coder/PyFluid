#pragma once

#include "pyfluid_amrex/reconstruction/scalar.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

namespace reconstruction_detail {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real parabolic_right_interface(
   amrex::Real vm,
   amrex::Real v0,
   amrex::Real vp,
   amrex::Real vpp
) noexcept
{
   return (7.0_rt * (v0 + vp) - (vm + vpp)) / 12.0_rt;
}

} // namespace reconstruction_detail

template <>
struct FivePointReconstruction<ReconParabolic> {
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   static amrex::Real left(
      amrex::Real vmm,
      amrex::Real vm,
      amrex::Real v0,
      amrex::Real vp,
      amrex::Real vpp
   ) noexcept
   {
      static_cast<void>(vmm);
      return reconstruction_detail::parabolic_right_interface(vm, v0, vp, vpp);
   }
};

template <>
struct FivePointReconstruction<ReconParabolicChar>
   : FivePointReconstruction<ReconParabolic> {};

} // namespace pyfluid_amrex
