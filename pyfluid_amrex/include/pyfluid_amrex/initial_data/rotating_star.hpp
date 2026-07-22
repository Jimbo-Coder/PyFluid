#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <cmath>

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
   PatchCoordinate rotation_axis{0.0, 0.0, 1.0};
   PatchCoordinate rotation_center{};
   amrex::Real injection_energy = 1.0;
   amrex::Real central_log_specific_enthalpy = 0.0;
   amrex::Real matter_relaxation = 0.5;
   amrex::Real relative_tolerance = 1.0e-8;
   int max_iterations = 32;
   int continuation_steps = 1;
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
};

RotatingStarParameters rotating_star_parameters(
   RunConfig const& cfg,
   RelativisticInitialDataView const& provider
);

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RotatingStarPoint uniform_rotation_point(
   RotatingStarParameters const& parameters,
   PatchCoordinate const& physical,
   SpacetimeMetric const& metric
) noexcept
{
   const amrex::Real displacement1 = physical.q1 - parameters.rotation_center.q1;
   const amrex::Real displacement2 = physical.q2 - parameters.rotation_center.q2;
   const amrex::Real displacement3 = physical.q3 - parameters.rotation_center.q3;
   const amrex::Real tangent1 =
      parameters.rotation_axis.q2 * displacement3
      - parameters.rotation_axis.q3 * displacement2;
   const amrex::Real tangent2 =
      parameters.rotation_axis.q3 * displacement1
      - parameters.rotation_axis.q1 * displacement3;
   const amrex::Real tangent3 =
      parameters.rotation_axis.q1 * displacement2
      - parameters.rotation_axis.q2 * displacement1;
   const amrex::Real v1 = (
      parameters.angular_velocity * tangent1 + metric.shift1
   ) / metric.lapse;
   const amrex::Real v2 = (
      parameters.angular_velocity * tangent2 + metric.shift2
   ) / metric.lapse;
   const amrex::Real v3 = (
      parameters.angular_velocity * tangent3 + metric.shift3
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
        )
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

RotatingStarUpdateSummary update_rotating_star_primitives(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   int eos_method,
   EOSParams const& eos,
   RotatingStarParameters const& parameters
);

} // namespace pyfluid_amrex
