#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <limits>

#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/generated/method_catalog.hpp"
#include "pyfluid_amrex/initial_data/atmosphere.hpp"
#include "pyfluid_amrex/initial_data/elliptic.hpp"
#include "pyfluid_amrex/kernels_generated/initial_data.hpp"
#include "pyfluid_amrex/patch_maps.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"

namespace pyfluid_amrex {

struct RelativisticInitialDataView;
struct RunConfig;

struct RotatingStarParameters {
   bool active = false;
   RotationLaw rotation_law = RotationLaw::Uniform;
   amrex::Real angular_velocity = 0.0;
   amrex::Real differential_rotation_length = 10.0;
   amrex::Real rotation_power_q = 1.0;
   PatchCoordinate rotation_axis{0.0, 0.0, 1.0};
   PatchCoordinate rotation_center{};
   amrex::Real injection_energy = 1.0;
   amrex::Real central_log_specific_enthalpy = 0.0;
   amrex::Real matter_relaxation = 0.5;
   amrex::Real relative_tolerance = 1.0e-8;
   int max_iterations = 32;
   int continuation_steps = 1;
   amrex::Vector<amrex::Real> continuation_targets;
   InitialAtmosphereParameters atmosphere{};
};

struct RotatingStarPoint {
   amrex::Real log_specific_enthalpy = 0.0;
   amrex::Real v1 = 0.0;
   amrex::Real v2 = 0.0;
   amrex::Real v3 = 0.0;
   amrex::Real velocity_squared = 0.0;
   bool timelike = true;
};

struct RotatingStarUpdateSummary {
   amrex::Real maximum_density = 0.0;
   amrex::Real maximum_velocity = 0.0;
   amrex::Real boundary_density_excess = 0.0;
   amrex::Real injection_energy = 1.0;
   amrex::Real equatorial_radius = 0.0;
   amrex::Real polar_radius = 0.0;
   amrex::Real equatorial_log_enthalpy_gradient = 0.0;
   amrex::Real polar_log_enthalpy_gradient = 0.0;
   amrex::Real mass_shedding_indicator = 1.0;
   amrex::Real adm_mass = 0.0;
   amrex::Real baryonic_mass = 0.0;
   amrex::Real angular_momentum = 0.0;
   amrex::Real dimensionless_spin = 0.0;
};

struct RotatingStarSurfaceDiagnostic {
   amrex::Real equatorial_log_enthalpy_gradient = 0.0;
   amrex::Real polar_log_enthalpy_gradient = 0.0;
   amrex::Real mass_shedding_indicator = 1.0;
};

RotatingStarParameters rotating_star_parameters(
   RunConfig const& cfg,
   RelativisticInitialDataView const& provider
);

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RotatingStarPoint rotating_star_point_from_tangent(
   RotatingStarParameters const& parameters,
   amrex::Real tangent1,
   amrex::Real tangent2,
   amrex::Real tangent3,
   SpacetimeMetric const& metric
) noexcept
{
   amrex::Real angular_velocity = parameters.angular_velocity;
   amrex::Real rotation_integral = 0.0;
   if (parameters.rotation_law != RotationLaw::Uniform
       && parameters.angular_velocity != 0.0) {
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
      if (tangent_squared > 64.0 * std::numeric_limits<amrex::Real>::epsilon()) {
         const amrex::Real rotation_sign = parameters.angular_velocity > 0.0
            ? 1.0 : -1.0;
         auto root_function = [=] AMREX_GPU_DEVICE(amrex::Real omega) noexcept {
            const amrex::Real denominator =
               metric.lapse * metric.lapse - shift_squared
               - 2.0 * omega * shift_tangent
               - omega * omega * tangent_squared;
            if (!(denominator > 0.0)) {
               return std::numeric_limits<amrex::Real>::max();
            }
            const amrex::Real metric_j =
               (omega * tangent_squared + shift_tangent) / denominator;
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
         amrex::Real lower_scale = 1.0e-10;
         amrex::Real upper_scale = 1.0;
         const amrex::Real lower_value = root_function(
            lower_scale * parameters.angular_velocity
         );
         const amrex::Real upper_value = root_function(
            upper_scale * parameters.angular_velocity
         );
         if (!(lower_value <= 0.0 && upper_value >= 0.0)) {
            return {0.0, 0.0, 0.0, 0.0, 0.0, false};
         }
         for (int iteration = 0; iteration < 48; ++iteration) {
            const amrex::Real midpoint_scale = 0.5 * (
               lower_scale + upper_scale
            );
            if (root_function(
                   midpoint_scale * parameters.angular_velocity
                ) <= 0.0) {
               lower_scale = midpoint_scale;
            } else {
               upper_scale = midpoint_scale;
            }
         }
         angular_velocity = 0.5 * (lower_scale + upper_scale)
            * parameters.angular_velocity;
      }
      rotation_integral = parameters.rotation_law == RotationLaw::JConstant
         ? generated::j_constant_rotation_integral(
              angular_velocity,
              parameters.angular_velocity,
              parameters.differential_rotation_length
           )
         : (
              std::abs(parameters.rotation_power_q - 2.0) < 1.0e-12
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
   const bool timelike = metric.lapse > 0.0
      && velocity_squared >= 0.0 && velocity_squared < 1.0
      && std::isfinite(velocity_squared);
   const amrex::Real log_specific_enthalpy = timelike
      ? generated::uniform_rotation_log_specific_enthalpy(
           parameters.injection_energy, metric.lapse, velocity_squared
        ) - rotation_integral
      : 0.0;
   return {
      log_specific_enthalpy,
      v1,
      v2,
      v3,
      velocity_squared,
      timelike && std::isfinite(log_specific_enthalpy),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RotatingStarPoint rotating_star_point(
   RotatingStarParameters const& parameters,
   PatchCoordinate const& physical,
   SpacetimeMetric const& metric
) noexcept
{
   const amrex::Real displacement1 = physical.q1 - parameters.rotation_center.q1;
   const amrex::Real displacement2 = physical.q2 - parameters.rotation_center.q2;
   const amrex::Real displacement3 = physical.q3 - parameters.rotation_center.q3;
   return rotating_star_point_from_tangent(
      parameters,
      parameters.rotation_axis.q2 * displacement3
         - parameters.rotation_axis.q3 * displacement2,
      parameters.rotation_axis.q3 * displacement1
         - parameters.rotation_axis.q1 * displacement3,
      parameters.rotation_axis.q1 * displacement2
         - parameters.rotation_axis.q2 * displacement1,
      metric
   );
}

// The reduced spherical owner stores vectors in the physical orthonormal
// (r, theta, phi) frame.  For rotation about the symmetry axis its Killing
// tangent therefore has only the phi component r sin(theta).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RotatingStarPoint rotating_star_axisymmetric_spherical_point(
   RotatingStarParameters const& parameters,
   amrex::Real radius,
   amrex::Real theta,
   SpacetimeMetric const& metric
) noexcept
{
   return rotating_star_point_from_tangent(
      parameters, 0.0, 0.0, radius * std::sin(theta), metric
   );
}

RotatingStarUpdateSummary update_rotating_star_primitives(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   int eos_method,
   EOSParams const& eos,
   RotatingStarParameters const& parameters
);

RotatingStarSurfaceDiagnostic rotating_star_surface_diagnostic(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   RotatingStarParameters const& parameters
);

RotatingStarUpdateSummary update_axisymmetric_spherical_rotating_star_primitives(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   int eos_method,
   EOSParams const& eos,
   RotatingStarParameters const& parameters
);

RotatingStarSurfaceDiagnostic
axisymmetric_spherical_rotating_star_surface_diagnostic(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   RotatingStarParameters const& parameters
);

} // namespace pyfluid_amrex
