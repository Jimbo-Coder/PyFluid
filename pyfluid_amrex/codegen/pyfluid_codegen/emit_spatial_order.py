"""Small shared order contract; no physics kernels or method products."""

from .numerical_catalog import SPATIAL_ORDERS


def render_spatial_order_header() -> str:
    condition = " || ".join(f"order == {order}" for order in SPATIAL_ORDERS)
    dispatches = "\n\n".join(
        _dispatch_cpp(name, qualifiers)
        for name, qualifiers in (
            ("dispatch_spatial_order", "inline"),
            ("dispatch_spatial_order_device", "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE"),
        )
    )
    return f'''#pragma once

// Generated from schema/numerics.toml. Do not edit by hand.
#include <AMReX_GpuQualifiers.H>
#include <AMReX.H>
#include <type_traits>

namespace pyfluid_amrex {{

inline constexpr int maximum_spatial_order = {max(SPATIAL_ORDERS)};

AMREX_GPU_HOST_DEVICE constexpr bool is_supported_spatial_order(int order) noexcept
{{
   return {condition};
}}

// Host orchestration may call non-device operations; point kernels use the
// host/device variant. Both instantiate the same catalogued set of orders.
{dispatches}

// Stencil support only: these do not advertise whole-solver accuracy.
AMREX_GPU_HOST_DEVICE constexpr int centered_stencil_radius(int order) noexcept
{{
   return order / 2;
}}

AMREX_GPU_HOST_DEVICE constexpr int upwind_dissipation_radius(int order) noexcept
{{
   return order / 2 + 1;
}}

AMREX_GPU_HOST_DEVICE constexpr int one_sided_stencil_points(int order, int derivative) noexcept
{{
   return order + derivative;
}}

}} // namespace pyfluid_amrex
'''


def spatial_order_assertion_cpp() -> str:
    """Guard the orders actually emitted in a kernel, even with a stale file.

    Checking only the separately generated catalog can admit a newly added
    order into an old kernel's zero-return fallback after an incomplete build.
    """
    condition = " || ".join(f"Order == {order}" for order in SPATIAL_ORDERS)
    return f'   static_assert({condition}, "stencil order absent; regenerate numerical kernels");'


def _dispatch_cpp(name: str, qualifiers: str) -> str:
    cases = "\n".join(
        f"      case {order}: return function(std::integral_constant<int, {order}>{{}});"
        for order in SPATIAL_ORDERS
    )
    return f'''template <typename Function>
{qualifiers}
auto {name}(int order, Function const& function)
{{
   switch (order) {{
{cases}
      default: amrex::Abort("unsupported spatial stencil order");
   }}
   return function(std::integral_constant<int, {SPATIAL_ORDERS[0]}>{{}});
}}'''
