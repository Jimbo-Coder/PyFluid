#pragma once

#include <cmath>

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/centering_representation.hpp"
#include "pyfluid_amrex/generated/field_ids.hpp"

namespace pyfluid_amrex {

// Z4c chi and lapse are positive scalars. Interpolate their logarithms using
// the selected polynomial stencil, then exponentiate at the target. This is
// not a source repair: nonpositive or nonfinite source values remain invalid.
struct Z4cPositiveScalarCenteringRepresentation {
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   static bool transforms(int component) noexcept
   {
      return component == Z4C_CHI || component == Z4C_LAPSE;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   static amrex::Real encode(int component, amrex::Real value) noexcept
   {
      return transforms(component) ? std::log(value) : value;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   static amrex::Real decode(int component, amrex::Real value) noexcept
   {
      return transforms(component) ? std::exp(value) : value;
   }
};

} // namespace pyfluid_amrex
