#pragma once

#include <AMReX_Algorithm.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

namespace pyfluid_amrex {

struct BracketedRootResult {
   amrex::Real root = 0.0;
   amrex::Real residual = 0.0;
   amrex::Real bracket_width = 0.0;
   int iterations = 0;
   bool converged = false;
   bool finite = true;
};

struct RootBracket {
   amrex::Real lower = 0.0;
   amrex::Real upper = 0.0;
   amrex::Real lower_residual = 0.0;
   amrex::Real upper_residual = 0.0;
   bool found = false;
   bool finite = true;
};

struct RootSystem2 {
   amrex::Real residual1 = 0.0;
   amrex::Real residual2 = 0.0;
   amrex::Real jacobian11 = 0.0;
   amrex::Real jacobian12 = 0.0;
   amrex::Real jacobian21 = 0.0;
   amrex::Real jacobian22 = 0.0;
};

template <typename Function>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RootBracket find_geometric_root_bracket(
   Function&& function,
   amrex::Real first,
   amrex::Real bound,
   amrex::Real multiplier,
   int max_steps
) noexcept
{
   RootBracket bracket{};
   if (!(first > 0.0) || !(bound > 0.0) || !(multiplier > 0.0)
       || multiplier == 1.0) {
      bracket.finite = false;
      return bracket;
   }
   amrex::Real previous = first;
   amrex::Real previous_residual = function(previous);
   for (int step = 0; step < max_steps; ++step) {
      amrex::Real current = previous * multiplier;
      if (multiplier > 1.0) {
         current = amrex::min(current, bound);
      } else {
         current = amrex::max(current, bound);
      }
      const amrex::Real current_residual = function(current);
      if (amrex::Math::isfinite(previous_residual)
          && amrex::Math::isfinite(current_residual)
          && (previous_residual == 0.0 || current_residual == 0.0
              || previous_residual * current_residual < 0.0)) {
         if (previous < current) {
            bracket = {
               previous,
               current,
               previous_residual,
               current_residual,
               true,
               true,
            };
         } else {
            bracket = {
               current,
               previous,
               current_residual,
               previous_residual,
               true,
               true,
            };
         }
         return bracket;
      }
      if (current == bound) {
         break;
      }
      previous = current;
      previous_residual = current_residual;
   }
   return bracket;
}

// Illinois regula falsi preserves the bracket while avoiding the endpoint
// stagnation of ordinary false position. Stop receives (root, bracket width).
template <typename Function, typename Stop>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
BracketedRootResult solve_bracketed_illinois(
   Function&& function,
   Stop&& stop,
   amrex::Real lower,
   amrex::Real upper,
   amrex::Real lower_residual,
   amrex::Real upper_residual,
   int max_iterations
) noexcept
{
   BracketedRootResult result{};
   if (!amrex::Math::isfinite(lower_residual)
       || !amrex::Math::isfinite(upper_residual)) {
      result.finite = false;
      return result;
   }
   if (lower_residual == 0.0) {
      result.root = lower;
      result.converged = true;
      return result;
   }
   if (upper_residual == 0.0) {
      result.root = upper;
      result.converged = true;
      return result;
   }
   if (!(lower < upper) || lower_residual * upper_residual > 0.0) {
      return result;
   }

   int retained_side = 0;
   amrex::Real root = 0.5 * (lower + upper);
   amrex::Real residual = function(root);
   for (int iteration = 1; iteration <= max_iterations; ++iteration) {
      result.iterations = iteration;
      const amrex::Real denominator = upper_residual - lower_residual;
      amrex::Real candidate = denominator != 0.0
         ? (lower * upper_residual - upper * lower_residual) / denominator
         : 0.5 * (lower + upper);
      if (!(candidate > lower && candidate < upper)
          || !amrex::Math::isfinite(candidate)) {
         candidate = 0.5 * (lower + upper);
      }
      root = candidate;
      residual = function(root);
      if (!amrex::Math::isfinite(residual)) {
         result.finite = false;
         return result;
      }

      const amrex::Real width = upper - lower;
      if (residual == 0.0 || stop(root, width)) {
         result.root = root;
         result.residual = residual;
         result.bracket_width = width;
         result.converged = true;
         return result;
      }

      if ((lower_residual < 0.0 && residual > 0.0)
          || (lower_residual > 0.0 && residual < 0.0)) {
         upper = root;
         upper_residual = residual;
         if (retained_side == -1) {
            lower_residual *= 0.5;
         }
         retained_side = -1;
      } else {
         lower = root;
         lower_residual = residual;
         if (retained_side == 1) {
            upper_residual *= 0.5;
         }
         retained_side = 1;
      }
   }

   result.root = root;
   result.residual = residual;
   result.bracket_width = upper - lower;
   return result;
}

} // namespace pyfluid_amrex
