#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace pyfluid_amrex {

// RK state coefficients sum to one, as checked exactly by the catalog.
// Combine the small stage difference and RHS before adding the base state.
// This preserves a constant state exactly and avoids rounding two weighted
// copies of an O(1) background independently at every stage.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real affine_state_update(
   amrex::Real base,
   amrex::Real input,
   amrex::Real input_weight,
   amrex::Real increment
) noexcept
{
   return base + (input_weight * (input - base) + increment);
}

} // namespace pyfluid_amrex
