#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <limits>

#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/generated/initial_data_method_catalog.hpp"
#include "pyfluid_amrex/initial_data/atmosphere.hpp"
#include "pyfluid_amrex/initial_data/circular_rotation.hpp"
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
   amrex::Real differential_rotation_length_b = 10.0;
   amrex::Real rotation_power_q = 1.0;
   PatchCoordinate rotation_axis{0.0, 0.0, 1.0};
   PatchCoordinate rotation_center{};
   amrex::Real injection_energy = 1.0;
   amrex::Real central_log_specific_enthalpy = 0.0;
   amrex::Real update_outer_radius =
      std::numeric_limits<amrex::Real>::max();
   amrex::Real matter_relaxation = 0.5;
   amrex::Real relative_tolerance = 1.0e-8;
   int max_iterations = 32;
   int continuation_steps = 1;
   amrex::Vector<amrex::Real> continuation_targets;
   InitialAtmosphereParameters atmosphere{};
};

using RotatingStarPoint = CircularRotationPoint;

struct RotatingStarUpdateSummary {
   amrex::Real maximum_density = 0.0;
   amrex::Real maximum_velocity = 0.0;
   amrex::Real boundary_density_excess = 0.0;
   amrex::Real injection_energy = 1.0;
   amrex::Real equatorial_radius = 0.0;
   amrex::Real polar_radius = 0.0;
   amrex::Real matter_outer_radius = 0.0;
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

// Optional magnetic Euler first-integral closure.  The supplied scalar is the
// covariant azimuthal flux function A_phi in the same Cartesian chart as the
// stellar solve.  It remains solver-private scratch; canonical magnetic
// ownership is the CT face field.
struct PoloidalMagneticBernoulli {
   amrex::Vector<amrex::MultiFab const*> flux_function;
   amrex::Real poloidal_magnetization = 0.0;
   amrex::Real nonlinear_poloidal = 0.0;
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
   return circular_rotation_point_from_tangent(
      {
         parameters.rotation_law,
         parameters.angular_velocity,
         parameters.differential_rotation_length,
         parameters.differential_rotation_length_b,
         parameters.rotation_power_q,
         parameters.injection_energy,
      },
      tangent1,
      tangent2,
      tangent3,
      metric
   );
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
   RotatingStarParameters const& parameters,
   PoloidalMagneticBernoulli const* magnetic_bernoulli = nullptr
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
