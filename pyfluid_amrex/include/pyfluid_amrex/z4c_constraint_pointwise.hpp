#pragma once

#include "pyfluid_amrex/kernels_generated/z4c/constraints.hpp"
#include "pyfluid_amrex/z4c_derivatives.hpp"

namespace pyfluid_amrex {

template <int Order, int Dim, typename Array, typename SpacingArray>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cConstraints evaluate_z4c_constraints_at(
   Array const& field,
   int i,
   int j,
   int k,
   SpacingArray const& inv_spacing,
   Z4cMatter const& matter
) noexcept
{
   const Z4cState state = load_z4c_state(field, i, j, k);
   const Z4cSpatialDerivatives derivatives =
      gather_z4c_spatial_derivatives<Order, Dim, Z4cDerivativePurpose::Constraints>(
         field, i, j, k, inv_spacing, state
      );
   return generated::evaluate_z4c_constraints(state, derivatives, matter);
}

template <int Order, int Dim, typename Array, typename SpacingArray>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cConstraints evaluate_vacuum_z4c_constraints_at(
   Array const& field,
   int i,
   int j,
   int k,
   SpacingArray const& inv_spacing
) noexcept
{
   return evaluate_z4c_constraints_at<Order, Dim>(
      field, i, j, k, inv_spacing, vacuum_z4c_matter()
   );
}

} // namespace pyfluid_amrex
