#pragma once

#include <AMReX_GpuQualifiers.H>

#include <cmath>
#include <limits>

#include "pyfluid_amrex/generated/initial_data_method_catalog.hpp"
#include "pyfluid_amrex/kernels_generated/initial_data.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

// The stationary circular Euler integral is shared by stars and detached
// tori.  Object-specific owners select the integration constant and matter
// support; this pointwise law owner contains neither choice.
struct CircularRotationParameters {
   RotationLaw rotation_law = RotationLaw::Uniform;
   amrex::Real angular_velocity = 0.0_rt;
   amrex::Real differential_rotation_length = 10.0_rt;
   amrex::Real differential_rotation_length_b = 10.0_rt;
   amrex::Real rotation_power_q = 1.0_rt;
   amrex::Real injection_energy = 1.0_rt;
};

struct CircularRotationPoint {
   amrex::Real log_specific_enthalpy = 0.0_rt;
   amrex::Real angular_velocity = 0.0_rt;
   amrex::Real v1 = 0.0_rt;
   amrex::Real v2 = 0.0_rt;
   amrex::Real v3 = 0.0_rt;
   amrex::Real velocity_squared = 0.0_rt;
   bool timelike = true;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
CircularRotationPoint circular_rotation_point_from_tangent(
   CircularRotationParameters const& parameters,
   amrex::Real tangent1,
   amrex::Real tangent2,
   amrex::Real tangent3,
   SpacetimeMetric const& metric
) noexcept
{
   amrex::Real angular_velocity = parameters.angular_velocity;
   amrex::Real rotation_integral = 0.0_rt;
   if (parameters.rotation_law != RotationLaw::Uniform
       && parameters.angular_velocity != 0.0_rt) {
      const amrex::Real tangent_squared = spatial_vector_norm_squared(
         tangent1, tangent2, tangent3, metric
      );
      const amrex::Real shift_tangent =
         tangent1 * (
            metric.gamma11 * metric.shift1 + metric.gamma12 * metric.shift2
            + metric.gamma13 * metric.shift3
         )
         + tangent2 * (
            metric.gamma12 * metric.shift1 + metric.gamma22 * metric.shift2
            + metric.gamma23 * metric.shift3
         )
         + tangent3 * (
            metric.gamma13 * metric.shift1 + metric.gamma23 * metric.shift2
            + metric.gamma33 * metric.shift3
         );
      const amrex::Real shift_squared = spatial_vector_norm_squared(
         metric.shift1, metric.shift2, metric.shift3, metric
      );
      if (tangent_squared > 64.0_rt * std::numeric_limits<amrex::Real>::epsilon()) {
         const amrex::Real rotation_sign = parameters.angular_velocity > 0.0_rt
            ? 1.0_rt : -1.0_rt;
         auto metric_specific_angular_momentum =
            [=](amrex::Real omega) noexcept {
            const amrex::Real denominator =
               metric.lapse * metric.lapse - shift_squared
               - 2.0_rt * omega * shift_tangent
               - omega * omega * tangent_squared;
            if (!(denominator > 0.0_rt)) {
               return std::numeric_limits<amrex::Real>::quiet_NaN();
            }
            return (omega * tangent_squared + shift_tangent) / denominator;
         };
         const bool uryu_plus_13 =
            parameters.rotation_law == RotationLaw::UryuPlus13;
         const bool uryu_j_envelope =
            parameters.rotation_law == RotationLaw::UryuJEnvelopeP1;
         if (uryu_plus_13 || uryu_j_envelope) {
            const amrex::Real omega_center = std::abs(
               parameters.angular_velocity
            );
            const amrex::Real momentum_scale =
               parameters.differential_rotation_length
               * parameters.differential_rotation_length * omega_center;
            auto law_omega = [=](
               amrex::Real specific_angular_momentum
            ) noexcept {
               return uryu_plus_13
                  ? generated::uryu_plus_13_angular_velocity(
                       specific_angular_momentum,
                       omega_center,
                       parameters.differential_rotation_length,
                       parameters.differential_rotation_length_b
                    )
                  : generated::uryu_j_envelope_p1_angular_velocity(
                       specific_angular_momentum,
                       omega_center,
                       parameters.differential_rotation_length,
                       parameters.differential_rotation_length_b
                    );
            };
            auto root_function = [=](
               amrex::Real specific_angular_momentum
            ) noexcept {
               const amrex::Real omega = rotation_sign * law_omega(
                  specific_angular_momentum
               );
               return rotation_sign
                  * metric_specific_angular_momentum(omega)
                  - specific_angular_momentum;
            };
            amrex::Real lower_momentum = 0.0_rt;
            amrex::Real lower_value = root_function(lower_momentum);
            amrex::Real upper_momentum = 0.0_rt;
            bool bracketed = false;
            for (int sample = 1; sample <= 32 && !bracketed; ++sample) {
               upper_momentum = momentum_scale
                  * static_cast<amrex::Real>(sample) / 32.0_rt;
               const amrex::Real upper_value = root_function(upper_momentum);
               if (!std::isfinite(lower_value)
                   || !std::isfinite(upper_value)) {
                  return {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, false};
               }
               if (lower_value >= 0.0_rt && upper_value <= 0.0_rt) {
                  bracketed = true;
               } else {
                  lower_momentum = upper_momentum;
                  lower_value = upper_value;
               }
            }
            if (uryu_plus_13) {
               for (int sample = 0; sample < 64 && !bracketed; ++sample) {
                  upper_momentum *= 1.25_rt;
                  const amrex::Real upper_value = root_function(
                     upper_momentum
                  );
                  if (!std::isfinite(lower_value)
                      || !std::isfinite(upper_value)) {
                     return {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, false};
                  }
                  if (lower_value >= 0.0_rt && upper_value <= 0.0_rt) {
                     bracketed = true;
                  } else {
                     lower_momentum = upper_momentum;
                     lower_value = upper_value;
                  }
               }
            }
            if (!bracketed) {
               return {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, false};
            }
            for (int iteration = 0; iteration < 56; ++iteration) {
               const amrex::Real midpoint = 0.5_rt * (
                  lower_momentum + upper_momentum
               );
               if (root_function(midpoint) >= 0.0_rt) {
                  lower_momentum = midpoint;
               } else {
                  upper_momentum = midpoint;
               }
            }
            const amrex::Real specific_angular_momentum = 0.5_rt * (
               lower_momentum + upper_momentum
            );
            angular_velocity = rotation_sign * law_omega(
               specific_angular_momentum
            );
            rotation_integral = uryu_plus_13
               ? generated::uryu_plus_13_rotation_integral(
                    specific_angular_momentum,
                    omega_center,
                    parameters.differential_rotation_length,
                    parameters.differential_rotation_length_b
                 )
               : generated::uryu_j_envelope_p1_rotation_integral(
                    specific_angular_momentum,
                    omega_center,
                    parameters.differential_rotation_length,
                    parameters.differential_rotation_length_b
                 );
         } else {
            auto root_function = [=](
               amrex::Real omega
            ) noexcept {
               const amrex::Real candidate_metric_j =
                  metric_specific_angular_momentum(omega);
               // The established j-constant/q-power bracket may begin beyond
               // the local timelike interval and bisect back into it.  Keep
               // that outer trial on the positive side of the bracket; the
               // Uryu branch above instead brackets in specific angular
               // momentum and rejects nonfinite trials explicitly.
               const amrex::Real metric_j = std::isfinite(candidate_metric_j)
                  ? candidate_metric_j
                  : std::numeric_limits<amrex::Real>::max();
               const amrex::Real law_j = parameters.rotation_law
                     == RotationLaw::JConstant
                  ? generated::j_constant_specific_angular_momentum(
                       omega,
                       parameters.angular_velocity,
                       parameters.differential_rotation_length
                    )
                  : generated::q_power_specific_angular_momentum(
                       omega,
                       parameters.angular_velocity,
                       parameters.differential_rotation_length,
                       parameters.rotation_power_q
                    );
               return rotation_sign * (metric_j - law_j);
            };
            amrex::Real lower_scale = 1.0e-10_rt;
            amrex::Real upper_scale = 1.0_rt;
            const amrex::Real lower_value = root_function(
               lower_scale * parameters.angular_velocity
            );
            const amrex::Real upper_value = root_function(
               upper_scale * parameters.angular_velocity
            );
            if (!(lower_value <= 0.0_rt && upper_value >= 0.0_rt)) {
               return {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, false};
            }
            for (int iteration = 0; iteration < 48; ++iteration) {
               const amrex::Real midpoint_scale = 0.5_rt * (
                  lower_scale + upper_scale
               );
               if (root_function(
                      midpoint_scale * parameters.angular_velocity
                   ) <= 0.0_rt) {
                  lower_scale = midpoint_scale;
               } else {
                  upper_scale = midpoint_scale;
               }
            }
            angular_velocity = 0.5_rt * (lower_scale + upper_scale)
               * parameters.angular_velocity;
            rotation_integral = parameters.rotation_law
                  == RotationLaw::JConstant
               ? generated::j_constant_rotation_integral(
                    angular_velocity,
                    parameters.angular_velocity,
                    parameters.differential_rotation_length
                 )
               : (
                    std::abs(parameters.rotation_power_q - 2.0_rt) < 1.0e-12_rt
                       ? generated::q_power_two_rotation_integral(
                            std::abs(angular_velocity),
                            std::abs(parameters.angular_velocity),
                            parameters.differential_rotation_length
                         )
                       : generated::q_power_rotation_integral(
                            std::abs(angular_velocity),
                            std::abs(parameters.angular_velocity),
                            parameters.differential_rotation_length,
                            parameters.rotation_power_q
                         )
                 );
         }
      }
   }
   const amrex::Real v1 = (
      angular_velocity * tangent1 + metric.shift1
   ) / metric.lapse;
   const amrex::Real v2 = (
      angular_velocity * tangent2 + metric.shift2
   ) / metric.lapse;
   const amrex::Real v3 = (
      angular_velocity * tangent3 + metric.shift3
   ) / metric.lapse;
   const amrex::Real velocity_squared = spatial_vector_norm_squared(
      v1, v2, v3, metric
   );
   const bool timelike = metric.lapse > 0.0_rt
      && velocity_squared >= 0.0_rt && velocity_squared < 1.0_rt
      && std::isfinite(velocity_squared);
   const amrex::Real log_specific_enthalpy = timelike
      ? generated::uniform_rotation_log_specific_enthalpy(
           parameters.injection_energy, metric.lapse, velocity_squared
        ) - rotation_integral
      : 0.0_rt;
   return {
      log_specific_enthalpy,
      angular_velocity,
      v1,
      v2,
      v3,
      velocity_squared,
      timelike && std::isfinite(log_specific_enthalpy),
   };
}

} // namespace pyfluid_amrex
