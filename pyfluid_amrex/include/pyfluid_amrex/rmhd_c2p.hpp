#pragma once

#include "pyfluid_amrex/c2p_policy.hpp"
#include "pyfluid_amrex/c2p_root.hpp"
#include "pyfluid_amrex/kernels_generated/eos.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/c2p_rmhd.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"

#include <AMReX_Algorithm.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace pyfluid_amrex {

struct RMHDRecoveryResult {
   RMHDPrimitive primitive{};
   C2PAttemptResult attempt{};
   int attempt_count = 0;

   [[nodiscard]] AMREX_GPU_HOST_DEVICE constexpr bool succeeded() const noexcept
   {
      return attempt.succeeded();
   }
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool finite_rmhd_conserved(const RMHDConserved& conserved) noexcept
{
   return amrex::Math::isfinite(conserved.dens)
      && amrex::Math::isfinite(conserved.mom1)
      && amrex::Math::isfinite(conserved.mom2)
      && amrex::Math::isfinite(conserved.mom3)
      && amrex::Math::isfinite(conserved.tau)
      && amrex::Math::isfinite(conserved.b1)
      && amrex::Math::isfinite(conserved.b2)
      && amrex::Math::isfinite(conserved.b3);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rmhd_velocity_squared(
   const RMHDPrimitive& primitive,
   const SpacetimeMetric& metric
) noexcept
{
   return spatial_vector_norm_squared(
      primitive.v1, primitive.v2, primitive.v3, metric
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool admissible_rmhd_primitive(
   const RMHDPrimitive& primitive,
   const SpacetimeMetric& metric
) noexcept
{
   const amrex::Real velocity_squared = rmhd_velocity_squared(primitive, metric);
   return amrex::Math::isfinite(primitive.rho)
      && amrex::Math::isfinite(primitive.eps)
      && amrex::Math::isfinite(primitive.pressure)
      && amrex::Math::isfinite(velocity_squared)
      && amrex::Math::isfinite(primitive.b1)
      && amrex::Math::isfinite(primitive.b2)
      && amrex::Math::isfinite(primitive.b3)
      && primitive.rho > 0.0
      && primitive.eps >= 0.0
      && primitive.pressure >= 0.0
      && velocity_squared < 1.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool admissible_rmhd_primitive(const RMHDPrimitive& primitive) noexcept
{
   return admissible_rmhd_primitive(primitive, minkowski_spacetime_metric());
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool finite_rmhd_recovery_state(const RMHDRecoveryState& recovery) noexcept
{
   return amrex::Math::isfinite(recovery.dens)
      && amrex::Math::isfinite(recovery.mom1)
      && amrex::Math::isfinite(recovery.mom2)
      && amrex::Math::isfinite(recovery.mom3)
      && amrex::Math::isfinite(recovery.tau)
      && amrex::Math::isfinite(recovery.b1)
      && amrex::Math::isfinite(recovery.b2)
      && amrex::Math::isfinite(recovery.b3)
      && amrex::Math::isfinite(recovery.momentum_squared)
      && amrex::Math::isfinite(recovery.magnetic_squared)
      && amrex::Math::isfinite(recovery.magnetic_momentum);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real scaled_c2p_component_residual(
   amrex::Real recovered,
   amrex::Real conserved
) noexcept
{
   return amrex::Math::abs(recovered - conserved) / amrex::max(
      1.0,
      amrex::max(amrex::Math::abs(recovered), amrex::Math::abs(conserved))
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rmhd_conserved_residual(
   const RMHDPrimitive& primitive,
   const RMHDConserved& conserved,
   const SpacetimeMetric& metric
) noexcept
{
   const RMHDConserved recovered = generated::rmhd_primitive_to_conserved(
      primitive, metric
   );
   amrex::Real residual = scaled_c2p_component_residual(
      recovered.dens, conserved.dens
   );
   residual = amrex::max(
      residual, scaled_c2p_component_residual(recovered.mom1, conserved.mom1)
   );
   residual = amrex::max(
      residual, scaled_c2p_component_residual(recovered.mom2, conserved.mom2)
   );
   residual = amrex::max(
      residual, scaled_c2p_component_residual(recovered.mom3, conserved.mom3)
   );
   residual = amrex::max(
      residual, scaled_c2p_component_residual(recovered.tau, conserved.tau)
   );
   residual = amrex::max(
      residual, scaled_c2p_component_residual(recovered.b1, conserved.b1)
   );
   residual = amrex::max(
      residual, scaled_c2p_component_residual(recovered.b2, conserved.b2)
   );
   return amrex::max(
      residual, scaled_c2p_component_residual(recovered.b3, conserved.b3)
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rmhd_conserved_residual(
   const RMHDPrimitive& primitive,
   const RMHDConserved& conserved
) noexcept
{
   return rmhd_conserved_residual(
      primitive, conserved, minkowski_spacetime_metric()
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void validate_rmhd_recovery(
   RMHDRecoveryResult& result,
   const RMHDConserved& conserved,
   const SpacetimeMetric& metric,
   const C2PRecoveryOptions& options
) noexcept
{
   if (!result.attempt.succeeded()) {
      return;
   }
   if (!admissible_rmhd_primitive(result.primitive, metric)) {
      result.attempt.status = C2PStatus::UnphysicalPrimitiveState;
      return;
   }
   result.attempt.conserved_residual = rmhd_conserved_residual(
      result.primitive, conserved, metric
   );
   if (!amrex::Math::isfinite(result.attempt.conserved_residual)
       || result.attempt.conserved_residual > options.conserved_tolerance) {
      result.attempt.status = C2PStatus::ConservedResidualMismatch;
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RootSystem2 srmhd_noble_2d_system(
   const RMHDRecoveryState& recovery,
   amrex::Real z,
   amrex::Real velocity_squared,
   amrex::Real gamma
) noexcept
{
   return generated::srmhd_noble_2d_system(
      recovery,
      z,
      velocity_squared,
      gamma
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srmhd_noble_2d_residual_norm(
   const RootSystem2& system,
   const RMHDRecoveryState& recovery,
   amrex::Real z
) noexcept
{
   const amrex::Real momentum_scale = amrex::max(
      1.0,
      amrex::max(
         recovery.momentum_squared,
         (recovery.magnetic_squared + z) * (recovery.magnetic_squared + z)
      )
   );
   const amrex::Real energy_scale = amrex::max(
      1.0,
      amrex::Math::abs(recovery.tau) + recovery.dens
         + recovery.magnetic_squared
         + amrex::Math::abs(z)
   );
   return amrex::max(
      amrex::Math::abs(system.residual1) / momentum_scale,
      amrex::Math::abs(system.residual2) / energy_scale
   );
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE inline
RMHDRecoveryResult recover_srmhd_noble_2d(
   const RMHDConserved& conserved,
   const SpacetimeMetric& metric,
   const RMHDPrimitive& seed,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   RMHDRecoveryResult result{};
   result.attempt.method = C2PMethod::Noble2D;
   result.attempt_count = 1;
   const RMHDRecoveryState recovery = generated::rmhd_recovery_state(
      conserved, metric
   );
   if (!finite_rmhd_conserved(conserved)
       || !finite_rmhd_recovery_state(recovery) || !(recovery.dens > 0.0)
       || !admissible_rmhd_primitive(seed, metric)) {
      result.attempt.status = C2PStatus::InvalidConservedState;
      return result;
   }
   if (!amrex::Math::isfinite(gamma) || !(gamma > 1.0) || !(gamma <= 2.0)) {
      result.attempt.status = C2PStatus::InvalidEquationOfState;
      return result;
   }

   const amrex::Real seed_velocity_squared = rmhd_velocity_squared(seed, metric);
   const amrex::Real seed_lorentz_squared = 1.0 / (1.0 - seed_velocity_squared);
   const amrex::Real seed_enthalpy =
      generated::relativistic_ideal_gas_specific_enthalpy(seed.eps, gamma);
   amrex::Real z = seed.rho * seed_enthalpy * seed_lorentz_squared;
   amrex::Real velocity_squared = seed_velocity_squared;
   RootSystem2 system = srmhd_noble_2d_system(
      recovery, z, velocity_squared, gamma
   );
   amrex::Real residual_norm = srmhd_noble_2d_residual_norm(
      system, recovery, z
   );
   constexpr amrex::Real velocity_limit =
      1.0 - 128.0 * std::numeric_limits<amrex::Real>::epsilon();

   for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
      result.attempt.iterations = iteration;
      result.attempt.residual = residual_norm;
      if (!amrex::Math::isfinite(residual_norm)) {
         result.attempt.status = C2PStatus::NonfiniteResidual;
         return result;
      }
      if (residual_norm <= options.relative_tolerance) {
         result.primitive = generated::srmhd_primitive_from_noble_2d(
            recovery, z, velocity_squared, gamma
         );
         result.attempt.status = admissible_rmhd_primitive(
            result.primitive, metric
         )
            ? C2PStatus::Success
            : C2PStatus::UnphysicalPrimitiveState;
         if constexpr (ValidateConserved) {
            validate_rmhd_recovery(result, conserved, metric, options);
         }
         return result;
      }

      const amrex::Real determinant = system.jacobian11 * system.jacobian22
         - system.jacobian12 * system.jacobian21;
      if (!amrex::Math::isfinite(determinant) || determinant == 0.0) {
         result.attempt.status = C2PStatus::NonfiniteResidual;
         return result;
      }
      const amrex::Real delta_z = (
         -system.residual1 * system.jacobian22
         + system.jacobian12 * system.residual2
      ) / determinant;
      const amrex::Real delta_velocity_squared = (
         system.jacobian21 * system.residual1
         - system.jacobian11 * system.residual2
      ) / determinant;

      bool accepted = false;
      amrex::Real damping = 1.0;
      for (int backtrack = 0; backtrack < 24; ++backtrack) {
         const amrex::Real candidate_z = z + damping * delta_z;
         const amrex::Real candidate_velocity_squared =
            velocity_squared + damping * delta_velocity_squared;
         if (candidate_z > 0.0 && candidate_velocity_squared >= 0.0
             && candidate_velocity_squared < velocity_limit) {
            const RootSystem2 candidate_system = srmhd_noble_2d_system(
               recovery,
               candidate_z,
               candidate_velocity_squared,
               gamma
            );
            const amrex::Real candidate_norm = srmhd_noble_2d_residual_norm(
               candidate_system, recovery, candidate_z
            );
            if (amrex::Math::isfinite(candidate_norm)
                && candidate_norm < residual_norm) {
               z = candidate_z;
               velocity_squared = candidate_velocity_squared;
               system = candidate_system;
               residual_norm = candidate_norm;
               accepted = true;
               break;
            }
         }
         damping *= 0.5;
      }
      if (!accepted) {
         result.attempt.status = C2PStatus::DidNotConverge;
         return result;
      }
   }

   result.attempt.status = C2PStatus::DidNotConverge;
   return result;
}

template <bool ValidateConserved, typename Function, typename Reconstruct>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRecoveryResult recover_srmhd_bracketed_candidate(
   C2PMethod method,
   const RMHDConserved& conserved,
   const SpacetimeMetric& metric,
   const RootBracket& bracket,
   Function&& function,
   Reconstruct&& reconstruct,
   const C2PRecoveryOptions& options
) noexcept
{
   RMHDRecoveryResult result{};
   result.attempt.method = method;
   result.attempt_count = 1;
   if (!bracket.finite) {
      result.attempt.status = C2PStatus::NonfiniteResidual;
      return result;
   }
   if (!bracket.found) {
      result.attempt.status = C2PStatus::NoBracket;
      return result;
   }
   const auto root = solve_bracketed_illinois(
      static_cast<Function&&>(function),
      [=] AMREX_GPU_HOST_DEVICE(amrex::Real value, amrex::Real width) noexcept {
         const amrex::Real tolerance = options.absolute_tolerance
            + options.relative_tolerance
               * amrex::max(1.0, amrex::Math::abs(value));
         return width <= tolerance;
      },
      bracket.lower,
      bracket.upper,
      bracket.lower_residual,
      bracket.upper_residual,
      options.max_iterations
   );
   result.attempt.iterations = root.iterations;
   result.attempt.residual = amrex::Math::abs(root.residual);
   if (!root.finite) {
      result.attempt.status = C2PStatus::NonfiniteResidual;
      return result;
   }
   if (!root.converged) {
      result.attempt.status = C2PStatus::DidNotConverge;
      return result;
   }
   result.primitive = reconstruct(root.root);
   result.attempt.status = admissible_rmhd_primitive(result.primitive, metric)
      ? C2PStatus::Success
      : C2PStatus::UnphysicalPrimitiveState;
   if constexpr (ValidateConserved) {
      validate_rmhd_recovery(result, conserved, metric, options);
   }
   return result;
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE inline
RMHDRecoveryResult recover_srmhd_palenzuela_1d_entropy(
   const RMHDConserved& conserved,
   const SpacetimeMetric& metric,
   const RMHDEntropyInput& entropy,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   RMHDRecoveryResult invalid{};
   invalid.attempt.method = C2PMethod::Palenzuela1DEntropy;
   invalid.attempt_count = 1;
   const RMHDRecoveryState recovery = generated::rmhd_recovery_state(
      conserved, metric
   );
   if (!finite_rmhd_conserved(conserved)
       || !finite_rmhd_recovery_state(recovery) || !(recovery.dens > 0.0)
       || !amrex::Math::isfinite(entropy.conserved_entropy)
       || !(entropy.conserved_entropy > 0.0)) {
      invalid.attempt.status = C2PStatus::InvalidConservedState;
      return invalid;
   }
   if (!amrex::Math::isfinite(gamma) || !(gamma > 1.0) || !(gamma <= 2.0)) {
      invalid.attempt.status = C2PStatus::InvalidEquationOfState;
      return invalid;
   }
   const amrex::Real conserved_entropy = entropy.conserved_entropy
      / metric.sqrt_det_gamma;
   const auto residual = [=] AMREX_GPU_HOST_DEVICE(amrex::Real x) noexcept {
      return generated::srmhd_palenzuela_entropy_residual(
         recovery,
         conserved_entropy,
         x,
         gamma
      );
   };
   const amrex::Real q = recovery.tau / recovery.dens;
   const amrex::Real s = recovery.magnetic_squared / recovery.dens;
   const amrex::Real tiny = 64.0
      * std::numeric_limits<amrex::Real>::epsilon();
   amrex::Real lower = amrex::max(1.0 + tiny, 1.0 + q - s);
   amrex::Real upper = 2.0 + 2.0 * q - s;
   if (!(upper > lower)) {
      invalid.attempt.status = C2PStatus::NoBracket;
      return invalid;
   }
   amrex::Real lower_residual = residual(lower);
   amrex::Real upper_residual = residual(upper);
   if (!amrex::Math::isfinite(upper_residual)) {
      invalid.attempt.status = C2PStatus::NonfiniteResidual;
      return invalid;
   }
   if (!amrex::Math::isfinite(lower_residual)) {
      amrex::Real invalid_endpoint = lower;
      amrex::Real valid_endpoint = upper;
      for (int iteration = 0; iteration < 64; ++iteration) {
         const amrex::Real candidate = 0.5 * (
            invalid_endpoint + valid_endpoint
         );
         const amrex::Real candidate_residual = residual(candidate);
         if (amrex::Math::isfinite(candidate_residual)) {
            valid_endpoint = candidate;
            lower_residual = candidate_residual;
         } else {
            invalid_endpoint = candidate;
         }
      }
      lower = valid_endpoint;
   }
   if (lower_residual * upper_residual > 0.0) {
      upper = 3.0 + 3.0 * q - 1.5 * s;
      if (!(upper > lower)) {
         invalid.attempt.status = C2PStatus::NoBracket;
         return invalid;
      }
      upper_residual = residual(upper);
   }
   const RootBracket bracket{
      lower,
      upper,
      lower_residual,
      upper_residual,
      amrex::Math::isfinite(lower_residual)
         && amrex::Math::isfinite(upper_residual)
         && lower_residual * upper_residual <= 0.0,
      amrex::Math::isfinite(lower_residual)
         && amrex::Math::isfinite(upper_residual),
   };
   return recover_srmhd_bracketed_candidate<ValidateConserved>(
      C2PMethod::Palenzuela1DEntropy,
      conserved,
      metric,
      bracket,
      residual,
      [=] AMREX_GPU_HOST_DEVICE(amrex::Real x) noexcept {
         return generated::srmhd_primitive_from_palenzuela_entropy(
            recovery, conserved_entropy, x, gamma
         );
      },
      options
   );
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE inline
RMHDRecoveryResult recover_srmhd_noble_1d_entropy(
   const RMHDConserved& conserved,
   const SpacetimeMetric& metric,
   const RMHDPrimitive& seed,
   const RMHDEntropyInput& entropy,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   RMHDRecoveryResult invalid{};
   invalid.attempt.method = C2PMethod::Noble1DEntropy;
   invalid.attempt_count = 1;
   const RMHDRecoveryState recovery = generated::rmhd_recovery_state(
      conserved, metric
   );
   if (!finite_rmhd_conserved(conserved)
       || !finite_rmhd_recovery_state(recovery) || !(recovery.dens > 0.0)
       || !admissible_rmhd_primitive(seed, metric)
       || !amrex::Math::isfinite(entropy.conserved_entropy)
       || !(entropy.conserved_entropy > 0.0)) {
      invalid.attempt.status = C2PStatus::InvalidConservedState;
      return invalid;
   }
   if (!amrex::Math::isfinite(gamma) || !(gamma > 1.0) || !(gamma <= 2.0)) {
      invalid.attempt.status = C2PStatus::InvalidEquationOfState;
      return invalid;
   }
   const amrex::Real conserved_entropy = entropy.conserved_entropy
      / metric.sqrt_det_gamma;
   const auto residual = [=] AMREX_GPU_HOST_DEVICE(amrex::Real density) noexcept {
      return generated::srmhd_noble_entropy_residual(
         recovery,
         conserved_entropy,
         density,
         gamma
      );
   };
   const amrex::Real upper_density = recovery.dens;
   const amrex::Real lower_density = amrex::max(
      std::numeric_limits<amrex::Real>::min(), recovery.dens * 1.0e-16
   );
   const amrex::Real seed_density = amrex::min(
      upper_density, amrex::max(lower_density, seed.rho)
   );
   RootBracket bracket = find_geometric_root_bracket(
      residual, seed_density, upper_density, 2.0, 64
   );
   if (!bracket.found) {
      bracket = find_geometric_root_bracket(
         residual, seed_density, lower_density, 0.5, 64
      );
   }
   return recover_srmhd_bracketed_candidate<ValidateConserved>(
      C2PMethod::Noble1DEntropy,
      conserved,
      metric,
      bracket,
      residual,
      [=] AMREX_GPU_HOST_DEVICE(amrex::Real density) noexcept {
         return generated::srmhd_primitive_from_noble_entropy(
            recovery, conserved_entropy, density, gamma
         );
      },
      options
   );
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE inline
RMHDRecoveryResult recover_srmhd_font_1d(
   const RMHDConserved& conserved,
   const SpacetimeMetric& metric,
   const ColdPolytropeEquationOfState& eos,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   RMHDRecoveryResult invalid{};
   invalid.attempt.method = C2PMethod::Font1D;
   invalid.attempt_count = 1;
   const RMHDRecoveryState recovery = generated::rmhd_recovery_state(
      conserved, metric
   );
   if (!finite_rmhd_conserved(conserved)
       || !finite_rmhd_recovery_state(recovery) || !(recovery.dens > 0.0)) {
      invalid.attempt.status = C2PStatus::InvalidConservedState;
      return invalid;
   }
   if (!amrex::Math::isfinite(eos.polytropic_k)
       || !amrex::Math::isfinite(eos.polytropic_gamma)
       || !(eos.polytropic_k > 0.0) || !(eos.polytropic_gamma > 1.0)
       || !(eos.polytropic_gamma <= 2.0)) {
      invalid.attempt.status = C2PStatus::InvalidEquationOfState;
      return invalid;
   }
   const auto residual = [=] AMREX_GPU_HOST_DEVICE(amrex::Real density) noexcept {
      return generated::srmhd_font_cold_residual(
         recovery,
         density,
         eos.polytropic_k,
         eos.polytropic_gamma
      );
   };
   const amrex::Real upper_density = recovery.dens;
   const amrex::Real lower_density = amrex::max(
      std::numeric_limits<amrex::Real>::min(), recovery.dens * 1.0e-16
   );
   const RootBracket bracket = find_geometric_root_bracket(
      residual, upper_density, lower_density, 0.5, 64
   );
   return recover_srmhd_bracketed_candidate<ValidateConserved>(
      C2PMethod::Font1D,
      conserved,
      metric,
      bracket,
      residual,
      [=] AMREX_GPU_HOST_DEVICE(amrex::Real density) noexcept {
         return generated::srmhd_primitive_from_font_cold(
            recovery, density, eos.polytropic_k, eos.polytropic_gamma
         );
      },
      options
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srmhd_palenzuela_residual(
   const RMHDRecoveryState& recovery,
   amrex::Real x,
   amrex::Real gamma
) noexcept
{
   return generated::srmhd_palenzuela_residual(recovery, x, gamma);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srmhd_palenzuela_residual_derivative(
   const RMHDRecoveryState& recovery,
   amrex::Real x,
   amrex::Real gamma
) noexcept
{
   return generated::srmhd_palenzuela_residual_derivative(
      recovery, x, gamma
   );
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRecoveryResult recover_srmhd_palenzuela_1d(
   const RMHDConserved& conserved,
   const SpacetimeMetric& metric,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   RMHDRecoveryResult result{};
   result.attempt.method = C2PMethod::Palenzuela1D;
   result.attempt_count = 1;
   const RMHDRecoveryState recovery = generated::rmhd_recovery_state(
      conserved, metric
   );
   if (!finite_rmhd_conserved(conserved)
       || !finite_rmhd_recovery_state(recovery) || !(recovery.dens > 0.0)) {
      result.attempt.status = C2PStatus::InvalidConservedState;
      return result;
   }
   if (!amrex::Math::isfinite(gamma) || !(gamma > 1.0) || !(gamma <= 2.0)) {
      result.attempt.status = C2PStatus::InvalidEquationOfState;
      return result;
   }

   const amrex::Real q = recovery.tau / recovery.dens;
   const amrex::Real s = recovery.magnetic_squared / recovery.dens;
   const amrex::Real tiny = 64.0 * std::numeric_limits<amrex::Real>::epsilon();
   amrex::Real lower = amrex::max(tiny, 1.0 + q - s);
   amrex::Real upper = 2.0 + 2.0 * q - s;
   if (!(upper > lower)) {
      result.attempt.status = C2PStatus::NoBracket;
      return result;
   }

   amrex::Real lower_residual = srmhd_palenzuela_residual(recovery, lower, gamma);
   amrex::Real upper_residual = srmhd_palenzuela_residual(recovery, upper, gamma);
   if (!amrex::Math::isfinite(upper_residual)) {
      result.attempt.status = C2PStatus::NonfiniteResidual;
      return result;
   }
   if (!amrex::Math::isfinite(lower_residual)) {
      amrex::Real invalid_endpoint = lower;
      amrex::Real valid_endpoint = upper;
      for (int iteration = 0; iteration < 64; ++iteration) {
         const amrex::Real candidate = 0.5 * (
            invalid_endpoint + valid_endpoint
         );
         const amrex::Real candidate_residual = srmhd_palenzuela_residual(
            recovery,
            candidate,
            gamma
         );
         if (amrex::Math::isfinite(candidate_residual)) {
            valid_endpoint = candidate;
            lower_residual = candidate_residual;
         } else {
            invalid_endpoint = candidate;
         }
      }
      lower = valid_endpoint;
   }
   if (lower_residual * upper_residual > 0.0) {
      upper = 3.0 + 3.0 * q - 1.5 * s;
      if (!(upper > lower)) {
         result.attempt.status = C2PStatus::NoBracket;
         return result;
      }
      upper_residual = srmhd_palenzuela_residual(recovery, upper, gamma);
      if (!amrex::Math::isfinite(upper_residual)) {
         result.attempt.status = C2PStatus::NonfiniteResidual;
         return result;
      }
   }
   if (lower_residual * upper_residual > 0.0) {
      result.attempt.status = C2PStatus::NoBracket;
      return result;
   }

   amrex::Real x = 0.5 * (lower + upper);
   amrex::Real x_residual = srmhd_palenzuela_residual(recovery, x, gamma);
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
         result.primitive = generated::srmhd_primitive_from_palenzuela(
            recovery, x, gamma
         );
         result.attempt.residual = amrex::Math::abs(x_residual);
         result.attempt.status = admissible_rmhd_primitive(
            result.primitive, metric
         )
            ? C2PStatus::Success
            : C2PStatus::UnphysicalPrimitiveState;
         if constexpr (ValidateConserved) {
            validate_rmhd_recovery(result, conserved, metric, options);
         }
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
      const amrex::Real derivative = srmhd_palenzuela_residual_derivative(
         recovery, x, gamma
      );
      if (amrex::Math::isfinite(derivative) && derivative != 0.0) {
         const amrex::Real newton_candidate = x - x_residual / derivative;
         if (newton_candidate > lower && newton_candidate < upper) {
            candidate = newton_candidate;
         }
      }
      x = candidate;
      x_residual = srmhd_palenzuela_residual(recovery, x, gamma);
   }

   result.attempt.status = C2PStatus::DidNotConverge;
   result.attempt.residual = amrex::Math::abs(x_residual);
   return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srmhd_reprimand_residual(
   const RMHDRecoveryState& recovery,
   amrex::Real mu,
   amrex::Real gamma
) noexcept
{
   return generated::srmhd_reprimand_residual(recovery, mu, gamma);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srmhd_reprimand_upper_residual(
   const RMHDRecoveryState& recovery,
   amrex::Real mu,
   amrex::Real gamma
) noexcept
{
   return generated::srmhd_reprimand_upper_residual(recovery, mu, gamma);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srmhd_reprimand_upper_residual_derivative(
   const RMHDRecoveryState& recovery,
   amrex::Real mu,
   amrex::Real gamma
) noexcept
{
   return generated::srmhd_reprimand_upper_residual_derivative(
      recovery, mu, gamma
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srmhd_reprimand_lorentz_factor(
   const RMHDRecoveryState& recovery,
   amrex::Real mu,
   amrex::Real gamma
) noexcept
{
   return generated::srmhd_reprimand_lorentz_factor(recovery, mu, gamma);
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE inline
RMHDRecoveryResult recover_srmhd_reprimand(
   const RMHDConserved& conserved,
   const SpacetimeMetric& metric,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   RMHDRecoveryResult result{};
   result.attempt.method = C2PMethod::RePrimAnd;
   result.attempt_count = 1;
   const RMHDRecoveryState recovery = generated::rmhd_recovery_state(
      conserved, metric
   );
   if (!finite_rmhd_conserved(conserved)
       || !finite_rmhd_recovery_state(recovery) || !(recovery.dens > 0.0)) {
      result.attempt.status = C2PStatus::InvalidConservedState;
      return result;
   }
   if (!amrex::Math::isfinite(gamma) || !(gamma > 1.0) || !(gamma <= 2.0)) {
      result.attempt.status = C2PStatus::InvalidEquationOfState;
      return result;
   }

   amrex::Real lower = 0.0;
   amrex::Real upper = 1.0;
   const amrex::Real normalized_momentum_squared =
      recovery.momentum_squared / (recovery.dens * recovery.dens);
   if (normalized_momentum_squared >= 1.0) {
      amrex::Real bracket_lower = 1.0 / std::sqrt(
         1.0 + normalized_momentum_squared
      );
      amrex::Real bracket_upper = 1.0;
      amrex::Real trial = 0.5 * (bracket_lower + bracket_upper);
      for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
         const amrex::Real residual = srmhd_reprimand_upper_residual(
            recovery, trial, gamma
         );
         if (!amrex::Math::isfinite(residual)) {
            result.attempt.status = C2PStatus::NonfiniteResidual;
            return result;
         }
         if (residual > 0.0) {
            bracket_upper = trial;
         } else {
            bracket_lower = trial;
         }
         const amrex::Real tolerance = options.absolute_tolerance
            + options.relative_tolerance * amrex::max(1.0, amrex::Math::abs(trial));
         if (bracket_upper - bracket_lower <= tolerance) {
            break;
         }
         amrex::Real candidate = 0.5 * (bracket_lower + bracket_upper);
         const amrex::Real derivative = srmhd_reprimand_upper_residual_derivative(
            recovery, trial, gamma
         );
         if (amrex::Math::isfinite(derivative) && derivative != 0.0) {
            const amrex::Real newton_candidate = trial - residual / derivative;
            if (newton_candidate > bracket_lower && newton_candidate < bracket_upper) {
               candidate = newton_candidate;
            }
         }
         trial = candidate;
      }
      upper = bracket_upper * (
         1.0 + 64.0 * std::numeric_limits<amrex::Real>::epsilon()
      );
   }
   amrex::Real lower_residual = srmhd_reprimand_residual(
      recovery, lower, gamma
   );
   amrex::Real upper_residual = srmhd_reprimand_residual(
      recovery, upper, gamma
   );
   if (!amrex::Math::isfinite(lower_residual)
       || !amrex::Math::isfinite(upper_residual)) {
      result.attempt.status = C2PStatus::NonfiniteResidual;
      return result;
   }
   if (lower_residual * upper_residual > 0.0) {
      result.attempt.status = C2PStatus::NoBracket;
      return result;
   }

   const auto root = solve_bracketed_illinois(
      [=] AMREX_GPU_HOST_DEVICE(amrex::Real mu) noexcept {
         return srmhd_reprimand_residual(recovery, mu, gamma);
      },
      [=] AMREX_GPU_HOST_DEVICE(amrex::Real mu, amrex::Real width) noexcept {
         const amrex::Real lorentz_factor = srmhd_reprimand_lorentz_factor(
            recovery, mu, gamma
         );
         const amrex::Real relative_floor =
            std::numeric_limits<amrex::Real>::epsilon()
            * amrex::Math::abs(mu);
         const amrex::Real tolerance = amrex::max(
            relative_floor,
            options.relative_tolerance * amrex::Math::abs(mu)
               / (lorentz_factor * lorentz_factor)
         );
         return width <= tolerance;
      },
      lower,
      upper,
      lower_residual,
      upper_residual,
      options.max_iterations
   );
   result.attempt.iterations = root.iterations;
   result.attempt.residual = amrex::Math::abs(root.residual);
   if (!root.finite) {
      result.attempt.status = C2PStatus::NonfiniteResidual;
      return result;
   }
   if (!root.converged) {
      result.attempt.status = C2PStatus::DidNotConverge;
      return result;
   }
   result.primitive = generated::srmhd_primitive_from_reprimand(
      recovery, root.root, gamma
   );
   result.attempt.status = admissible_rmhd_primitive(result.primitive, metric)
      ? C2PStatus::Success
      : C2PStatus::UnphysicalPrimitiveState;
   if constexpr (ValidateConserved) {
      validate_rmhd_recovery(result, conserved, metric, options);
   }
   return result;
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRecoveryResult recover_srmhd(
   const RMHDConserved& conserved,
   const SpacetimeMetric& metric,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   RMHDRecoveryResult recovered = recover_srmhd_reprimand<ValidateConserved>(
      conserved, metric, gamma, options
   );
   if (recovered.succeeded()) AMREX_LIKELY {
      return recovered;
   }
   const int primary_iterations = recovered.attempt.iterations;
   recovered = recover_srmhd_palenzuela_1d<ValidateConserved>(
      conserved, metric, gamma, options
   );
   recovered.attempt_count = 2;
   recovered.attempt.iterations += primary_iterations;
   return recovered;
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRecoveryResult recover_srmhd(
   const RMHDConserved& conserved,
   const SpacetimeMetric& metric,
   const RMHDPrimitive& seed,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   RMHDRecoveryResult recovered = recover_srmhd_noble_2d<ValidateConserved>(
      conserved, metric, seed, gamma, options
   );
   if (recovered.succeeded()) AMREX_LIKELY {
      return recovered;
   }
   int total_iterations = recovered.attempt.iterations;
   recovered = recover_srmhd_reprimand<ValidateConserved>(
      conserved, metric, gamma, options
   );
   recovered.attempt_count = 2;
   recovered.attempt.iterations += total_iterations;
   if (recovered.succeeded()) {
      return recovered;
   }
   total_iterations = recovered.attempt.iterations;
   recovered = recover_srmhd_palenzuela_1d<ValidateConserved>(
      conserved, metric, gamma, options
   );
   recovered.attempt_count = 3;
   recovered.attempt.iterations += total_iterations;
   return recovered;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srmhd_palenzuela_residual(
   const RMHDConserved& conserved,
   amrex::Real x,
   amrex::Real gamma
) noexcept
{
   const RMHDRecoveryState recovery = generated::rmhd_recovery_state(
      conserved, minkowski_spacetime_metric()
   );
   return srmhd_palenzuela_residual(recovery, x, gamma);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srmhd_palenzuela_residual_derivative(
   const RMHDConserved& conserved,
   amrex::Real x,
   amrex::Real gamma
) noexcept
{
   const RMHDRecoveryState recovery = generated::rmhd_recovery_state(
      conserved, minkowski_spacetime_metric()
   );
   return srmhd_palenzuela_residual_derivative(recovery, x, gamma);
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRecoveryResult recover_srmhd_noble_2d(
   const RMHDConserved& conserved,
   const RMHDPrimitive& seed,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   return recover_srmhd_noble_2d<ValidateConserved>(
      conserved, minkowski_spacetime_metric(), seed, gamma, options
   );
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRecoveryResult recover_srmhd_palenzuela_1d_entropy(
   const RMHDConserved& conserved,
   const RMHDEntropyInput& entropy,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   return recover_srmhd_palenzuela_1d_entropy<ValidateConserved>(
      conserved, minkowski_spacetime_metric(), entropy, gamma, options
   );
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRecoveryResult recover_srmhd_noble_1d_entropy(
   const RMHDConserved& conserved,
   const RMHDPrimitive& seed,
   const RMHDEntropyInput& entropy,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   return recover_srmhd_noble_1d_entropy<ValidateConserved>(
      conserved, minkowski_spacetime_metric(), seed, entropy, gamma, options
   );
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRecoveryResult recover_srmhd_font_1d(
   const RMHDConserved& conserved,
   const ColdPolytropeEquationOfState& eos,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   return recover_srmhd_font_1d<ValidateConserved>(
      conserved, minkowski_spacetime_metric(), eos, options
   );
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRecoveryResult recover_srmhd_palenzuela_1d(
   const RMHDConserved& conserved,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   return recover_srmhd_palenzuela_1d<ValidateConserved>(
      conserved, minkowski_spacetime_metric(), gamma, options
   );
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRecoveryResult recover_srmhd_reprimand(
   const RMHDConserved& conserved,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   return recover_srmhd_reprimand<ValidateConserved>(
      conserved, minkowski_spacetime_metric(), gamma, options
   );
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRecoveryResult recover_srmhd(
   const RMHDConserved& conserved,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   return recover_srmhd<ValidateConserved>(
      conserved, minkowski_spacetime_metric(), gamma, options
   );
}

template <bool ValidateConserved = false>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRecoveryResult recover_srmhd(
   const RMHDConserved& conserved,
   const RMHDPrimitive& seed,
   amrex::Real gamma,
   const C2PRecoveryOptions& options = {}
) noexcept
{
   return recover_srmhd<ValidateConserved>(
      conserved, minkowski_spacetime_metric(), seed, gamma, options
   );
}

} // namespace pyfluid_amrex
