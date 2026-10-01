#pragma once

#include "pyfluid_amrex/excision_geometry.hpp"
#include "pyfluid_amrex/kernels_generated/boundary_differential_ops.hpp"
#include "pyfluid_amrex/z4c_derivatives.hpp"
#include "pyfluid_amrex/z4c_characteristics.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

// Cao and Hilditch (1111.2177), puncture-gauge principal speeds. The normal
// points from the hole into the exterior. Positive margin means every mode
// travels into the hole. This is stronger than apparent-horizon containment.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real z4c_excision_outflow_margin(Z4cState const& state,
   amrex::GpuArray<amrex::Real, 3> const& normal,
   Z4cEvolutionParameters const& parameters, Z4cLapseGauge lapse_gauge,
   Z4cShiftGauge shift_gauge) noexcept
{
   const auto metric = generated::z4c_spacetime_metric(state);
   const amrex::Real speed_squared = z4c_maximum_speed_squared(
      state, parameters, lapse_gauge, shift_gauge);
   const amrex::Real inverse_normal_squared = metric.inv_gamma11 * normal[0] * normal[0]
      + metric.inv_gamma22 * normal[1] * normal[1] + metric.inv_gamma33 * normal[2] * normal[2]
      + 2.0_rt * (metric.inv_gamma12 * normal[0] * normal[1]
         + metric.inv_gamma13 * normal[0] * normal[2] + metric.inv_gamma23 * normal[1] * normal[2]);
   return metric.shift1 * normal[0] + metric.shift2 * normal[1] + metric.shift3 * normal[2]
      - metric.lapse * std::sqrt(speed_squared * inverse_normal_squared);
}

// Supplies stencil choice to the shared component/derivative assembler.
// No Z4c equation or component map is duplicated for the excision boundary.
template <int Order>
struct ExcisionZ4cDerivativeOperator {
   ExcisionStencilPlan plan{};

   AMREX_GPU_HOST_DEVICE
   explicit ExcisionZ4cDerivativeOperator(ExcisionStencilPlan stencil) : plan(stencil) {}

   AMREX_GPU_HOST_DEVICE
   explicit ExcisionZ4cDerivativeOperator(amrex::GpuArray<int, 3> sides)
   {
      plan.sides = sides;
      plan.mixed_sides = {{sides[0], sides[1], sides[0], sides[2], sides[1], sides[2]}};
   }

   template <int Direction, typename Array>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real first(Array const& field, int i, int j, int k, int component,
                    amrex::Real scale) const noexcept
   {
      if (plan.sides[Direction] == 0) {
         return generated::centered_first<Order, Direction>(
            field, i, j, k, component, scale);
      }
      return generated::one_sided_first<Order, Direction>(
         field, i, j, k, component, plan.sides[Direction], scale);
   }

   template <int Direction, typename Array>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real second(Array const& field, int i, int j, int k, int component,
                     amrex::Real scale) const noexcept
   {
      if (plan.sides[Direction] == 0) {
         return generated::centered_second<Order, Direction>(
            field, i, j, k, component, scale);
      }
      return generated::one_sided_second<Order, Direction>(
         field, i, j, k, component, plan.sides[Direction], scale);
   }

   template <int Direction, typename Array>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real advection(Array const& field, int i, int j, int k, int component,
                        amrex::Real shift, amrex::Real scale) const noexcept
   {
      if (plan.sides[Direction] == 0) {
         return CenteredZ4cDerivativeOperator<Order>{}.template advection<Direction>(
            field, i, j, k, component, shift, scale);
      }
      return generated::one_sided_first<Order, Direction>(
         field, i, j, k, component, plan.sides[Direction], scale);
   }

   template <int Direction1, int Direction2, typename Array>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real mixed(Array const& field, int i, int j, int k, int component,
                    amrex::Real scale) const noexcept
   {
      static_assert(Direction1 < Direction2);
      constexpr int pair = Direction1 == 0 ? Direction2 - 1 : 2;
      const int side1 = plan.mixed_sides[2 * pair];
      const int side2 = plan.mixed_sides[2 * pair + 1];
      if (side1 == 0 && side2 == 0) {
         return generated::centered_mixed<Order, Direction1, Direction2>(
            field, i, j, k, component, scale);
      }
      if (side1 != 0 && side2 != 0) {
         return generated::one_sided_mixed<Order, Direction1, Direction2>(
            field, i, j, k, component, side1, side2, scale);
      }
      const auto inner = [=] AMREX_GPU_HOST_DEVICE(int a, int b, int c, int n) noexcept {
         return side2 == 0 ? generated::centered_first<Order, Direction2>(field, a, b, c, n, 1.0_rt)
            : generated::one_sided_first<Order, Direction2>(field, a, b, c, n, side2, 1.0_rt);
      };
      return side1 == 0 ? generated::centered_first<Order, Direction1>(inner, i, j, k, component, scale)
         : generated::one_sided_first<Order, Direction1>(inner, i, j, k, component, side1, scale);
   }
};

} // namespace pyfluid_amrex
