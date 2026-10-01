#pragma once

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace pyfluid_amrex {

using namespace amrex::literals;

// Lagrange interpolation/extrapolation on consecutive integer lattice points.
// A shortened line (for example N x M x 1) uses all available distinct nodes.
template <int MaxPoints>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, MaxPoints> uniform_lagrange_weights(
   amrex::Real target, int first_index, int num_points = MaxPoints
) noexcept
{
   amrex::GpuArray<amrex::Real, MaxPoints> weights{};
   for (int point = 0; point < num_points; ++point) {
      amrex::Real weight = 1.0_rt;
      for (int other = 0; other < num_points; ++other) {
         if (other != point) {
            weight *= (target - static_cast<amrex::Real>(first_index + other))
               / static_cast<amrex::Real>(point - other);
         }
      }
      weights[point] = weight;
   }
   return weights;
}

// Match value, first derivative and second derivative at both endpoints.
// Derivatives are with respect to the physical interpolation coordinate.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real quintic_hermite(
   amrex::Real fraction, amrex::Real width,
   amrex::Real left_value, amrex::Real right_value,
   amrex::Real left_first, amrex::Real right_first,
   amrex::Real left_second, amrex::Real right_second) noexcept
{
   const amrex::Real x = fraction;
   const amrex::Real x2 = x * x;
   const amrex::Real x3 = x2 * x;
   const amrex::Real right_weight = x3 * (10.0_rt + x * (-15.0_rt + 6.0_rt * x));
   const amrex::Real left_slope_weight = x + x3 * (-6.0_rt + x * (8.0_rt - 3.0_rt * x));
   const amrex::Real right_slope_weight = x3 * (-4.0_rt + x * (7.0_rt - 3.0_rt * x));
   const amrex::Real left_curvature_weight = 0.5_rt * x2 * (1.0_rt + x * (-3.0_rt + x * (3.0_rt - x)));
   const amrex::Real right_curvature_weight = 0.5_rt * x3 * (1.0_rt + x * (-2.0_rt + x));
   return left_value + right_weight * (right_value - left_value)
      + width * (left_slope_weight * left_first + right_slope_weight * right_first)
      + width * width * (left_curvature_weight * left_second + right_curvature_weight * right_second);
}

} // namespace pyfluid_amrex
