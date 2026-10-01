#pragma once

#include "pyfluid_amrex/z4c_excision_derivatives.hpp"

namespace pyfluid_amrex {

// Inner lambdas inherit the enclosing host/device execution space. They are
// not extended launch lambdas (CUDA forbids generic extended HD lambdas).
// Dispatch only the differential operator. Equation kernels and component
// assembly stay shared across accuracy orders and coordinate systems.
// Keep this gather out of the large generated RHS optimizer region: inlining
// every order there multiplies compile work without removing stencil loads.
template <Z4cDerivativePurpose Purpose = Z4cDerivativePurpose::Evolution,
          typename Array, typename Spacing>
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE inline
Z4cSpatialDerivatives gather_spacetime_derivatives(
   int order, Array const& field, int i, int j, int k,
   Spacing const& inverse_spacing, Z4cState const& state,
   ExcisionGridView const& excision = {}) noexcept
{
   return dispatch_spatial_order_device(order, [=](auto order_tag) {
      constexpr int spatial_order = decltype(order_tag)::value;
      if (excision.region.enabled()) {
         const auto plan = excision.derivative_stencil(
            i, j, k, upwind_dissipation_radius(spatial_order));
         return gather_z4c_spatial_derivatives<spatial_order, AMREX_SPACEDIM, Purpose>(
            field, i, j, k, inverse_spacing, state,
            ExcisionZ4cDerivativeOperator<spatial_order>{plan});
      }
      return gather_z4c_spatial_derivatives<spatial_order, AMREX_SPACEDIM, Purpose>(
         field, i, j, k, inverse_spacing, state);
   });
}

template <int Direction, typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real spacetime_dissipation(
   int order, Array const& field, ExcisionGridView const& excision,
   int i, int j, int k, int component, amrex::Real inverse_spacing) noexcept
{
   return dispatch_spatial_order_device(order, [=](auto order_tag) {
      constexpr int spatial_order = decltype(order_tag)::value;
      if (excision.region.enabled()) {
         return generated::masked_kreiss_oliger<spatial_order, Direction>(
            field, excision, i, j, k, component, inverse_spacing);
      }
      return generated::kreiss_oliger<spatial_order, Direction>(
         field, i, j, k, component, inverse_spacing);
   });
}

} // namespace pyfluid_amrex
