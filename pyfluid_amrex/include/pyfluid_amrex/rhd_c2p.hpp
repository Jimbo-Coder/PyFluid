#pragma once

#include "pyfluid_amrex/c2p_policy.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/c2p_rhd.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"

#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <limits>

namespace pyfluid_amrex {

using namespace amrex::literals;

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
bool canonical_rhd_vacuum(const RHDConserved& conserved) noexcept
{
   return conserved.dens == 0.0_rt
      && conserved.mom1 == 0.0_rt
      && conserved.mom2 == 0.0_rt
      && conserved.mom3 == 0.0_rt
      && conserved.tau == 0.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool canonical_rhd_vacuum(const RHDPrimitive& primitive) noexcept
{
   return primitive.rho == 0.0_rt
      && primitive.eps == 0.0_rt
      && primitive.pressure == 0.0_rt
      && primitive.v1 == 0.0_rt
      && primitive.v2 == 0.0_rt
      && primitive.v3 == 0.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rhd_conserved_admissibility_margin(
   const RHDConserved& conserved,
   const SpacetimeMetric& metric
) noexcept
{
   const RHDRecoveryState recovery = generated::rhd_recovery_state(
      conserved, metric
   );
   return recovery.dens + recovery.tau - std::sqrt(
      recovery.dens * recovery.dens + recovery.momentum_squared
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool admissible_rhd_primitive(
   const RHDPrimitive& primitive,
   const SpacetimeMetric& metric
) noexcept
{
   if (canonical_rhd_vacuum(primitive)) {
      return true;
   }
   const amrex::Real velocity_squared = spatial_vector_norm_squared(
      primitive.v1, primitive.v2, primitive.v3, metric
   );
   return amrex::Math::isfinite(primitive.rho)
      && amrex::Math::isfinite(primitive.eps)
      && amrex::Math::isfinite(primitive.pressure)
      && amrex::Math::isfinite(velocity_squared)
      && primitive.rho > 0.0_rt
      && primitive.eps >= 0.0_rt
      && primitive.pressure >= 0.0_rt
      && velocity_squared < 1.0_rt;
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
   if (finite_rhd_conserved(conserved)
       && canonical_rhd_vacuum(conserved)) {
      result.attempt.status = C2PStatus::Success;
      return result;
   }
   const RHDRecoveryState recovery = generated::rhd_recovery_state(
      conserved, metric
   );

   if (!finite_rhd_conserved(conserved) || !(recovery.dens > 0.0_rt)
       || !amrex::Math::isfinite(recovery.momentum_squared)
       || !(recovery.dens + recovery.tau > 0.0_rt)) {
      result.attempt.status = C2PStatus::InvalidConservedState;
      return result;
   }
   if (!amrex::Math::isfinite(gamma) || !(gamma > 1.0_rt) || !(gamma <= 2.0_rt)) {
      result.attempt.status = C2PStatus::InvalidEquationOfState;
      return result;
   }

   const amrex::Real q = recovery.tau / recovery.dens;
   const amrex::Real r = recovery.momentum_squared
      / (recovery.dens * recovery.dens);
   const amrex::Real margin =
      64.0_rt * std::numeric_limits<amrex::Real>::epsilon();
   const amrex::Real causal_lower = std::sqrt(r) * (1.0_rt + margin);
   amrex::Real lower = 1.0_rt + q;
   if (causal_lower > lower) {
      lower = causal_lower;
   }
   amrex::Real upper = 2.0_rt * (1.0_rt + q);
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
   if (lower_residual * upper_residual > 0.0_rt) {
      result.attempt.status = C2PStatus::NoBracket;
      return result;
   }

   amrex::Real x = 0.5_rt * (lower + upper);
   amrex::Real x_residual = srhd_palenzuela_residual(recovery, x, gamma);
   for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
      result.attempt.iterations = iteration;
      if (!amrex::Math::isfinite(x_residual)) {
         result.attempt.status = C2PStatus::NonfiniteResidual;
         return result;
      }

      const amrex::Real tolerance = options.absolute_tolerance
         + options.relative_tolerance * amrex::max(1.0_rt, amrex::Math::abs(x));
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

      if ((lower_residual <= 0.0_rt && x_residual >= 0.0_rt)
          || (lower_residual >= 0.0_rt && x_residual <= 0.0_rt)) {
         upper = x;
         upper_residual = x_residual;
      } else {
         lower = x;
         lower_residual = x_residual;
      }

      amrex::Real candidate = 0.5_rt * (lower + upper);
      const amrex::Real derivative = srhd_palenzuela_residual_derivative(
         recovery,
         x,
         gamma
      );
      if (amrex::Math::isfinite(derivative) && derivative != 0.0_rt) {
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

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDRecoveryResult recover_rhd(
   const RHDConserved& conserved,
   const SpacetimeMetric& metric,
   const EOSParams& eos,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   if constexpr (EOSMethod == EOSIdealGas) {
      return recover_srhd_palenzuela_1d(conserved, metric, eos.gamma, options);
   } else {
      if constexpr (EOSMethod == EOSRuntime) {
         if (eos.method == EOSIdealGas) {
            return recover_srhd_palenzuela_1d(
               conserved, metric, eos.gamma, options
            );
         }
      }
      RHDRecoveryResult result{};
      result.attempt.method = C2PMethod::Palenzuela1D;
      if (finite_rhd_conserved(conserved)
          && canonical_rhd_vacuum(conserved)) {
         result.attempt.status = C2PStatus::Success;
         return result;
      }
      const RHDRecoveryState recovery = generated::rhd_recovery_state(
         conserved, metric
      );
      if (!finite_rhd_conserved(conserved) || !(recovery.dens > 0.0_rt)
          || !amrex::Math::isfinite(recovery.momentum_squared)
          || !(recovery.dens + recovery.tau > 0.0_rt)) {
         result.attempt.status = C2PStatus::InvalidConservedState;
         return result;
      }

      const amrex::Real q = recovery.tau / recovery.dens;
      const amrex::Real momentum_ratio_squared = recovery.momentum_squared
         / (recovery.dens * recovery.dens);
      const amrex::Real margin =
         64.0_rt * std::numeric_limits<amrex::Real>::epsilon();
      amrex::Real lower = amrex::max(
         1.0_rt + q,
         std::sqrt(momentum_ratio_squared) * (1.0_rt + margin)
      );
      lower = amrex::max(lower, 1.0_rt + margin);

      const auto residual = [=] AMREX_GPU_HOST_DEVICE(amrex::Real x) noexcept {
         const amrex::Real velocity_squared = momentum_ratio_squared / (x * x);
         if (!(velocity_squared >= 0.0_rt) || !(velocity_squared < 1.0_rt)) {
            return std::numeric_limits<amrex::Real>::quiet_NaN();
         }
         const amrex::Real inverse_lorentz = std::sqrt(1.0_rt - velocity_squared);
         const amrex::Real rho = recovery.dens * inverse_lorentz;
         const amrex::Real pressure_from_energy = recovery.dens * (x - 1.0_rt - q);
         const amrex::Real eps = generated::rhd_specific_internal_energy_from_palenzuela(
            recovery, x
         );
         return (
            relativistic_pressure<EOSMethod>(rho, eps, eos)
            - pressure_from_energy
         ) / recovery.dens;
      };

      amrex::Real lower_residual = residual(lower);
      amrex::Real upper = amrex::max(2.0_rt * lower, lower + 1.0_rt);
      amrex::Real upper_residual = residual(upper);
      for (int expansion = 0;
           expansion < 64
              && amrex::Math::isfinite(lower_residual)
              && amrex::Math::isfinite(upper_residual)
              && lower_residual * upper_residual > 0.0_rt;
           ++expansion) {
         upper *= 2.0_rt;
         upper_residual = residual(upper);
      }
      if (!amrex::Math::isfinite(lower_residual)
          || !amrex::Math::isfinite(upper_residual)) {
         result.attempt.status = C2PStatus::NonfiniteResidual;
         return result;
      }
      if (lower_residual * upper_residual > 0.0_rt) {
         result.attempt.status = C2PStatus::NoBracket;
         return result;
      }

      amrex::Real x = 0.5_rt * (lower + upper);
      amrex::Real x_residual = residual(x);
      for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
         result.attempt.iterations = iteration;
         if (!amrex::Math::isfinite(x_residual)) {
            result.attempt.status = C2PStatus::NonfiniteResidual;
            return result;
         }
         const amrex::Real tolerance = options.absolute_tolerance
            + options.relative_tolerance * amrex::max(1.0_rt, amrex::Math::abs(x));
         if (amrex::Math::abs(x_residual) <= tolerance
             || upper - lower <= tolerance) {
            const amrex::Real velocity_squared = momentum_ratio_squared / (x * x);
            const amrex::Real inverse_lorentz = std::sqrt(1.0_rt - velocity_squared);
            const amrex::Real rho = recovery.dens * inverse_lorentz;
            const amrex::Real eps = generated::rhd_specific_internal_energy_from_palenzuela(
               recovery, x
            );
            const amrex::Real pressure = relativistic_pressure<EOSMethod>(rho, eps, eos);
            result.primitive = {
               rho,
               eps,
               pressure,
               recovery.mom1 / (recovery.dens * x),
               recovery.mom2 / (recovery.dens * x),
               recovery.mom3 / (recovery.dens * x),
            };
            result.attempt.residual = amrex::Math::abs(x_residual);
            result.attempt.status = admissible_rhd_primitive(result.primitive, metric)
               ? C2PStatus::Success
               : C2PStatus::UnphysicalPrimitiveState;
            return result;
         }
         if ((lower_residual <= 0.0_rt && x_residual >= 0.0_rt)
             || (lower_residual >= 0.0_rt && x_residual <= 0.0_rt)) {
            upper = x;
            upper_residual = x_residual;
         } else {
            lower = x;
            lower_residual = x_residual;
         }
         const amrex::Real denominator = upper_residual - lower_residual;
         const amrex::Real secant = denominator != 0.0_rt
            ? upper - upper_residual * (upper - lower) / denominator
            : 0.5_rt * (lower + upper);
         x = secant > lower && secant < upper
            ? secant
            : 0.5_rt * (lower + upper);
         x_residual = residual(x);
      }
      result.attempt.status = C2PStatus::DidNotConverge;
      result.attempt.residual = amrex::Math::abs(x_residual);
      return result;
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDRecoveryResult recover_rhd_runtime(
   int eos_method,
   const RHDConserved& conserved,
   const SpacetimeMetric& metric,
   const EOSParams& eos,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   static_cast<void>(eos_method);
   return recover_rhd<EOSRuntime>(conserved, metric, eos, options);
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
