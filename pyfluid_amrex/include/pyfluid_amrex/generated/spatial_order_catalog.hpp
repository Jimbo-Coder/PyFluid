#pragma once

// Generated from schema/numerics.toml. Do not edit by hand.
#include <AMReX_GpuQualifiers.H>
#include <AMReX.H>
#include <type_traits>

namespace pyfluid_amrex {

inline constexpr int maximum_spatial_order = 10;

AMREX_GPU_HOST_DEVICE constexpr bool is_supported_spatial_order(int order) noexcept
{
   return order == 2 || order == 4 || order == 6 || order == 8 || order == 10;
}

// Host orchestration may call non-device operations; point kernels use the
// host/device variant. Both instantiate the same catalogued set of orders.
template <typename Function>
inline
auto dispatch_spatial_order(int order, Function const& function)
{
   switch (order) {
      case 2: return function(std::integral_constant<int, 2>{});
      case 4: return function(std::integral_constant<int, 4>{});
      case 6: return function(std::integral_constant<int, 6>{});
      case 8: return function(std::integral_constant<int, 8>{});
      case 10: return function(std::integral_constant<int, 10>{});
      default: amrex::Abort("unsupported spatial stencil order");
   }
   return function(std::integral_constant<int, 2>{});
}

template <typename Function>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto dispatch_spatial_order_device(int order, Function const& function)
{
   switch (order) {
      case 2: return function(std::integral_constant<int, 2>{});
      case 4: return function(std::integral_constant<int, 4>{});
      case 6: return function(std::integral_constant<int, 6>{});
      case 8: return function(std::integral_constant<int, 8>{});
      case 10: return function(std::integral_constant<int, 10>{});
      default: amrex::Abort("unsupported spatial stencil order");
   }
   return function(std::integral_constant<int, 2>{});
}

// Stencil support only: these do not advertise whole-solver accuracy.
AMREX_GPU_HOST_DEVICE constexpr int centered_stencil_radius(int order) noexcept
{
   return order / 2;
}

AMREX_GPU_HOST_DEVICE constexpr int upwind_dissipation_radius(int order) noexcept
{
   return order / 2 + 1;
}

AMREX_GPU_HOST_DEVICE constexpr int one_sided_stencil_points(int order, int derivative) noexcept
{
   return order + derivative;
}

} // namespace pyfluid_amrex
