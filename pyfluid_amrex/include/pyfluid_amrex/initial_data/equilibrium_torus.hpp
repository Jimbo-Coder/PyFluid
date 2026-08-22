#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>
#include <limits>

#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/generated/initial_data_method_catalog.hpp"
#include "pyfluid_amrex/initial_data/atmosphere.hpp"
#include "pyfluid_amrex/initial_data/circular_rotation.hpp"
#include "pyfluid_amrex/initial_data/elliptic.hpp"
#include "pyfluid_amrex/kernels_generated/initial_data.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"

namespace pyfluid_amrex {

struct RelativisticInitialDataView;
struct RunConfig;

enum class EquilibriumTorusGravity {
   TestFluid,
   Coupled,
};

struct EquilibriumTorusParameters {
   bool active = false;
   EquilibriumTorusGravity gravity = EquilibriumTorusGravity::TestFluid;
   DiskRotationLaw rotation_law = DiskRotationLaw::ConstantL;
   amrex::Real specific_angular_momentum = 0.0;
   amrex::Real angular_velocity = 0.0;
   amrex::Real differential_rotation_length = 10.0;
   amrex::Real rotation_power_q = 1.0;
   amrex::Real inner_radius = 0.0;
   amrex::Real axis1 = 0.0;
   amrex::Real axis2 = 0.0;
   amrex::Real axis3 = 1.0;
   amrex::Real center1 = 0.0;
   amrex::Real center2 = 0.0;
   amrex::Real center3 = 0.0;
   amrex::Real matter_relaxation = 0.5;
   bool continue_existing_matter = false;
   InitialAtmosphereParameters atmosphere{};
};

struct EquilibriumTorusPoint {
   amrex::Real log_specific_enthalpy = 0.0;
   amrex::Real angular_velocity = 0.0;
   amrex::Real minus_covariant_time_velocity = 0.0;
   amrex::Real v1 = 0.0;
   amrex::Real v2 = 0.0;
   amrex::Real v3 = 0.0;
   amrex::Real velocity_squared = 0.0;
   bool timelike = false;
};

struct EquilibriumTorusUpdateSummary {
   amrex::Real maximum_density = 0.0;
   amrex::Real maximum_velocity = 0.0;
   amrex::Real inner_radius = 0.0;
   amrex::Real outer_radius = 0.0;
   amrex::Real density_maximum_radius = 0.0;
   amrex::Real maximum_log_specific_enthalpy = 0.0;
   amrex::Real boundary_density_excess = 0.0;
   amrex::Real baryonic_mass = 0.0;
   amrex::Real angular_momentum = 0.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
EquilibriumTorusPoint equilibrium_torus_point_from_tangent(
   EquilibriumTorusParameters const& parameters,
   amrex::Real inner_normalization,
   amrex::Real tangent1,
   amrex::Real tangent2,
   amrex::Real tangent3,
   SpacetimeMetric const& metric
) noexcept
{
   if (parameters.rotation_law != DiskRotationLaw::ConstantL) {
      RotationLaw circular_law = RotationLaw::Uniform;
      if (parameters.rotation_law == DiskRotationLaw::JConstant) {
         circular_law = RotationLaw::JConstant;
      } else if (parameters.rotation_law == DiskRotationLaw::QPower) {
         circular_law = RotationLaw::QPower;
      }
      const CircularRotationPoint point = circular_rotation_point_from_tangent(
         {
            circular_law,
            parameters.angular_velocity,
            parameters.differential_rotation_length,
            parameters.differential_rotation_length,
            parameters.rotation_power_q,
            inner_normalization,
         },
         tangent1,
         tangent2,
         tangent3,
         metric
      );
      return {
         point.log_specific_enthalpy,
         point.angular_velocity,
         0.0,
         point.v1,
         point.v2,
         point.v3,
         point.velocity_squared,
         point.timelike,
      };
   }
   const amrex::Real shift_squared = spatial_vector_norm_squared(
      metric.shift1, metric.shift2, metric.shift3, metric
   );
   const amrex::Real metric_tt = -metric.lapse * metric.lapse
      + shift_squared;
   const amrex::Real metric_t_phi =
      tangent1 * (
         metric.gamma11 * metric.shift1
         + metric.gamma12 * metric.shift2
         + metric.gamma13 * metric.shift3
      )
      + tangent2 * (
         metric.gamma12 * metric.shift1
         + metric.gamma22 * metric.shift2
         + metric.gamma23 * metric.shift3
      )
      + tangent3 * (
         metric.gamma13 * metric.shift1
         + metric.gamma23 * metric.shift2
         + metric.gamma33 * metric.shift3
      );
   const amrex::Real metric_phi_phi = spatial_vector_norm_squared(
      tangent1, tangent2, tangent3, metric
   );
   const amrex::Real denominator = metric_phi_phi
      + parameters.specific_angular_momentum * metric_t_phi;
   const amrex::Real scale = std::max({
      amrex::Real(1.0),
      std::abs(metric_phi_phi),
      std::abs(parameters.specific_angular_momentum * metric_t_phi),
   });
   if (!(metric.lapse > 0.0)
       || std::abs(denominator)
          <= 64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale
       || !(inner_normalization > 0.0)) {
      return {};
   }
   const amrex::Real angular_velocity =
      generated::constant_l_torus_angular_velocity(
         metric_tt,
         metric_t_phi,
         metric_phi_phi,
         parameters.specific_angular_momentum
      );
   const amrex::Real normalization_squared =
      generated::constant_l_torus_normalization_squared(
         metric_tt, metric_t_phi, metric_phi_phi, angular_velocity
      );
   if (!(normalization_squared > 0.0)
       || !std::isfinite(normalization_squared)) {
      return {};
   }
   const amrex::Real minus_covariant_time_velocity =
      generated::constant_l_torus_minus_covariant_time_velocity(
         metric_tt,
         metric_t_phi,
         angular_velocity,
         normalization_squared
      );
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
   const bool timelike = minus_covariant_time_velocity > 0.0
      && velocity_squared >= 0.0 && velocity_squared < 1.0
      && std::isfinite(minus_covariant_time_velocity)
      && std::isfinite(velocity_squared);
   const amrex::Real log_specific_enthalpy = timelike
      ? generated::constant_l_torus_log_specific_enthalpy(
           inner_normalization,
           minus_covariant_time_velocity
        )
      : 0.0;
   return {
      log_specific_enthalpy,
      angular_velocity,
      minus_covariant_time_velocity,
      v1,
      v2,
      v3,
      velocity_squared,
      timelike && std::isfinite(log_specific_enthalpy),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
EquilibriumTorusPoint equilibrium_torus_point(
   EquilibriumTorusParameters const& parameters,
   amrex::Real inner_normalization,
   PatchCoordinate const& physical,
   SpacetimeMetric const& metric
) noexcept
{
   const amrex::Real displacement1 = physical.q1 - parameters.center1;
   const amrex::Real displacement2 = physical.q2 - parameters.center2;
   const amrex::Real displacement3 = physical.q3 - parameters.center3;
   return equilibrium_torus_point_from_tangent(
      parameters,
      inner_normalization,
      parameters.axis2 * displacement3 - parameters.axis3 * displacement2,
      parameters.axis3 * displacement1 - parameters.axis1 * displacement3,
      parameters.axis1 * displacement2 - parameters.axis2 * displacement1,
      metric
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
EquilibriumTorusPoint equilibrium_torus_axisymmetric_spherical_point(
   EquilibriumTorusParameters const& parameters,
   amrex::Real inner_normalization,
   amrex::Real radius,
   amrex::Real theta,
   SpacetimeMetric const& metric
) noexcept
{
   return equilibrium_torus_point_from_tangent(
      parameters,
      inner_normalization,
      0.0,
      0.0,
      radius * std::sin(theta),
      metric
   );
}

EquilibriumTorusParameters equilibrium_torus_parameters(
   RunConfig const& cfg,
   RelativisticInitialDataView const& provider
);

EquilibriumTorusUpdateSummary
update_axisymmetric_spherical_equilibrium_torus_primitives(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   int eos_method,
   EOSParams const& eos,
   EquilibriumTorusParameters const& parameters
);

EquilibriumTorusUpdateSummary update_equilibrium_torus_primitives(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   int eos_method,
   EOSParams const& eos,
   EquilibriumTorusParameters const& parameters
);

} // namespace pyfluid_amrex
