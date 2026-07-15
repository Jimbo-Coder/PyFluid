#pragma once

#include "pyfluid_amrex/c2p_policy.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic_fluid.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"

#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <limits>

namespace pyfluid_amrex {

struct RHDRecoveryResult {
   RHDPrimitive primitive{};
   C2PAttemptResult attempt{};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool finite_rhd_conserved(const RHDConserved& conserved) noexcept
{
   return amrex::Math::isfinite(conserved.dens)
      && amrex::Math::isfinite(conserved.mom1)
      && amrex::Math::isfinite(conserved.mom2)
      && amrex::Math::isfinite(conserved.mom3)
      && amrex::Math::isfinite(conserved.tau);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool admissible_rhd_primitive(
   const RHDPrimitive& primitive,
   const SpacetimeMetric& metric
) noexcept
{
   const amrex::Real velocity_squared = spatial_vector_norm_squared(
      primitive.v1, primitive.v2, primitive.v3, metric
   );
   return amrex::Math::isfinite(primitive.rho)
      && amrex::Math::isfinite(primitive.eps)
      && amrex::Math::isfinite(primitive.pressure)
      && amrex::Math::isfinite(velocity_squared)
      && primitive.rho > 0.0
      && primitive.eps >= 0.0
      && primitive.pressure >= 0.0
      && velocity_squared < 1.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool admissible_rhd_primitive(const RHDPrimitive& primitive) noexcept
{
   return admissible_rhd_primitive(primitive, minkowski_spacetime_metric());
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srhd_palenzuela_residual(
   const RHDRecoveryState& recovery,
   amrex::Real x,
   amrex::Real gamma
) noexcept
{
   return generated::srhd_palenzuela_residual(recovery, x, gamma);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srhd_palenzuela_residual_derivative(
   const RHDRecoveryState& recovery,
   amrex::Real x,
   amrex::Real gamma
) noexcept
{
   return generated::srhd_palenzuela_residual_derivative(recovery, x, gamma);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDRecoveryResult recover_srhd_palenzuela_1d(
   const RHDConserved& conserved,
   const SpacetimeMetric& metric,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   RHDRecoveryResult result{};
   result.attempt.method = C2PMethod::Palenzuela1D;
   const RHDRecoveryState recovery = generated::rhd_recovery_state(
      conserved, metric
   );

   if (!finite_rhd_conserved(conserved) || !(recovery.dens > 0.0)
       || !amrex::Math::isfinite(recovery.momentum_squared)
       || !(recovery.dens + recovery.tau > 0.0)) {
      result.attempt.status = C2PStatus::InvalidConservedState;
      return result;
   }
   if (!amrex::Math::isfinite(gamma) || !(gamma > 1.0) || !(gamma <= 2.0)) {
      result.attempt.status = C2PStatus::InvalidEquationOfState;
      return result;
   }

   const amrex::Real q = recovery.tau / recovery.dens;
   const amrex::Real r = recovery.momentum_squared
      / (recovery.dens * recovery.dens);
   const amrex::Real margin =
      64.0 * std::numeric_limits<amrex::Real>::epsilon();
   const amrex::Real causal_lower = std::sqrt(r) * (1.0 + margin);
   amrex::Real lower = 1.0 + q;
   if (causal_lower > lower) {
      lower = causal_lower;
   }
   amrex::Real upper = 2.0 * (1.0 + q);
   if (!(upper > lower)) {
      result.attempt.status = C2PStatus::NoBracket;
      return result;
   }

   amrex::Real lower_residual = srhd_palenzuela_residual(recovery, lower, gamma);
   amrex::Real upper_residual = srhd_palenzuela_residual(recovery, upper, gamma);
   if (!amrex::Math::isfinite(lower_residual)
       || !amrex::Math::isfinite(upper_residual)) {
      result.attempt.status = C2PStatus::NonfiniteResidual;
      return result;
   }
   if (lower_residual * upper_residual > 0.0) {
      result.attempt.status = C2PStatus::NoBracket;
      return result;
   }

   amrex::Real x = 0.5 * (lower + upper);
   amrex::Real x_residual = srhd_palenzuela_residual(recovery, x, gamma);
   for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
      result.attempt.iterations = iteration;
      if (!amrex::Math::isfinite(x_residual)) {
         result.attempt.status = C2PStatus::NonfiniteResidual;
         return result;
      }

      const amrex::Real tolerance = options.absolute_tolerance
         + options.relative_tolerance * amrex::max(1.0, amrex::Math::abs(x));
      if (amrex::Math::abs(x_residual) <= tolerance
          || upper - lower <= tolerance) {
         result.primitive = generated::srhd_primitive_from_palenzuela(
            recovery,
            x,
            gamma
         );
         result.attempt.residual = amrex::Math::abs(x_residual);
         result.attempt.status = admissible_rhd_primitive(result.primitive, metric)
            ? C2PStatus::Success
            : C2PStatus::UnphysicalPrimitiveState;
         return result;
      }

      if ((lower_residual <= 0.0 && x_residual >= 0.0)
          || (lower_residual >= 0.0 && x_residual <= 0.0)) {
         upper = x;
         upper_residual = x_residual;
      } else {
         lower = x;
         lower_residual = x_residual;
      }

      amrex::Real candidate = 0.5 * (lower + upper);
      const amrex::Real derivative = srhd_palenzuela_residual_derivative(
         recovery,
         x,
         gamma
      );
      if (amrex::Math::isfinite(derivative) && derivative != 0.0) {
         const amrex::Real newton_candidate = x - x_residual / derivative;
         if (newton_candidate > lower && newton_candidate < upper) {
            candidate = newton_candidate;
         }
      }
      x = candidate;
      x_residual = srhd_palenzuela_residual(recovery, x, gamma);
   }

   result.attempt.status = C2PStatus::DidNotConverge;
   result.attempt.residual = amrex::Math::abs(x_residual);
   return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srhd_palenzuela_residual(
   const RHDConserved& conserved,
   amrex::Real x,
   amrex::Real gamma
) noexcept
{
   const RHDRecoveryState recovery = generated::rhd_recovery_state(
      conserved, minkowski_spacetime_metric()
   );
   return srhd_palenzuela_residual(recovery, x, gamma);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srhd_palenzuela_residual_derivative(
   const RHDConserved& conserved,
   amrex::Real x,
   amrex::Real gamma
) noexcept
{
   const RHDRecoveryState recovery = generated::rhd_recovery_state(
      conserved, minkowski_spacetime_metric()
   );
   return srhd_palenzuela_residual_derivative(recovery, x, gamma);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDRecoveryResult recover_srhd_palenzuela_1d(
   const RHDConserved& conserved,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   return recover_srhd_palenzuela_1d(
      conserved, minkowski_spacetime_metric(), gamma, options
   );
}

} // namespace pyfluid_amrex
