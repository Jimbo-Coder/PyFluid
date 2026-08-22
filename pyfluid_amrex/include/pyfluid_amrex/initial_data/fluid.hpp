#pragma once

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>

#include <cmath>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/eos/cold_barotrope.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/initial_data/atmosphere.hpp"
#include "pyfluid_amrex/initial_data/equilibrium_torus.hpp"
#include "pyfluid_amrex/initial_data/provider.hpp"
#include "pyfluid_amrex/initial_data/sampling.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"

namespace pyfluid_amrex {

enum class FluidInitialDataKind : int {
   Uniform,
   SineWave,
   KelvinHelmholtz,
   ShockPlane,
   PeriodicShockPlane,
   FixedMetricHse,
   Atmosphere,
   Provider,
   EquilibriumTorus,
};

struct FluidInitialDataParameters {
   FluidInitialDataKind kind = FluidInitialDataKind::Uniform;
   amrex::Real rho0 = 1.0;
   amrex::Real pressure0 = 1.0;
   amrex::Real amplitude = 0.0;
   amrex::Real wave_number = 1.0;
   amrex::Real layer_width = 0.05;
   amrex::Real sheet_center = 0.5;
   amrex::Real density_amplitude = 0.0;
   amrex::Real v1_0 = 0.0;
   amrex::Real v2_0 = 0.0;
   amrex::Real v3_0 = 0.0;
   amrex::Real b1_0 = 0.0;
   amrex::Real b2_0 = 0.0;
   amrex::Real b3_0 = 0.0;
   amrex::Real shock_x0 = 0.5;
   amrex::Real shock_y0 = 0.5;
   amrex::Real shock_z0 = 0.5;
   amrex::Real shock_n1 = 1.0;
   amrex::Real shock_n2 = 0.0;
   amrex::Real shock_n3 = 0.0;
   amrex::Real rho_left = 1.0;
   amrex::Real pressure_left = 1.0;
   amrex::Real v1_left = 0.0;
   amrex::Real v2_left = 0.0;
   amrex::Real v3_left = 0.0;
   amrex::Real b1_left = 0.0;
   amrex::Real b2_left = 0.0;
   amrex::Real b3_left = 0.0;
   amrex::Real rho_right = 0.125;
   amrex::Real pressure_right = 0.1;
   amrex::Real v1_right = 0.0;
   amrex::Real v2_right = 0.0;
   amrex::Real v3_right = 0.0;
   amrex::Real b1_right = 0.0;
   amrex::Real b2_right = 0.0;
   amrex::Real b3_right = 0.0;
   InitialAtmosphereParameters atmosphere{};
   EquilibriumTorusParameters equilibrium_torus{};
   amrex::Real equilibrium_torus_inner_normalization = 0.0;
   bool provider_radius_is_mesh1 = false;
   EOSParams eos{};
};

struct FluidInitialDataContext {
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo{};
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> probhi{};
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{};
   bool periodic_second_direction = false;
   amrex::Real reference_lapse = 1.0;
};

struct FluidPrimitiveSample {
   RHDPrimitive fluid{};
   amrex::Real b1 = 0.0;
   amrex::Real b2 = 0.0;
   amrex::Real b3 = 0.0;
   SpacetimeMetric metric{};
};

inline FluidInitialDataParameters fluid_initial_data_parameters(
   RunConfig const& cfg,
   RelativisticInitialDataView const& provider
)
{
   FluidInitialDataKind kind = FluidInitialDataKind::ShockPlane;
   if (cfg.ic == "uniform") {
      kind = FluidInitialDataKind::Uniform;
   } else if (cfg.ic == "sin_wave") {
      kind = FluidInitialDataKind::SineWave;
   } else if (cfg.ic == "kelvin_helmholtz") {
      kind = FluidInitialDataKind::KelvinHelmholtz;
   } else if (cfg.ic == "sod_periodic_plane") {
      kind = FluidInitialDataKind::PeriodicShockPlane;
   } else if (cfg.ic == "fixed_metric_hse") {
      kind = FluidInitialDataKind::FixedMetricHse;
   } else if (cfg.ic == "black_hole") {
      kind = FluidInitialDataKind::Atmosphere;
   } else if (cfg.ic == "tov" || cfg.ic == "rotating_star") {
      kind = FluidInitialDataKind::Provider;
   } else if (cfg.ic == "equilibrium_torus") {
      kind = FluidInitialDataKind::EquilibriumTorus;
   }
   FluidInitialDataParameters parameters;
   parameters.kind = kind;
   parameters.rho0 = cfg.rho0;
   parameters.pressure0 = cfg.pressure0;
   parameters.amplitude = cfg.amplitude;
   parameters.wave_number = cfg.wave_number;
   parameters.layer_width = cfg.bump_sigma;
   parameters.sheet_center = cfg.bump_y0;
   parameters.density_amplitude = cfg.kh_density_amplitude;
   parameters.v1_0 = cfg.v1_0;
   parameters.v2_0 = cfg.v2_0;
   parameters.v3_0 = cfg.v3_0;
   parameters.b1_0 = cfg.b1_0;
   parameters.b2_0 = cfg.b2_0;
   parameters.b3_0 = cfg.b3_0;
   parameters.shock_x0 = cfg.shock_x0;
   parameters.shock_y0 = cfg.shock_y0;
   parameters.shock_z0 = cfg.shock_z0;
   parameters.shock_n1 = cfg.shock_nx;
   parameters.shock_n2 = cfg.shock_ny;
   parameters.shock_n3 = cfg.shock_nz;
   parameters.rho_left = cfg.rho_left;
   parameters.pressure_left = cfg.pressure_left;
   parameters.v1_left = cfg.v1_left;
   parameters.v2_left = cfg.v2_left;
   parameters.v3_left = cfg.v3_left;
   parameters.b1_left = std::isfinite(cfg.b1_left) ? cfg.b1_left : cfg.b1_0;
   parameters.b2_left = std::isfinite(cfg.b2_left) ? cfg.b2_left : cfg.b2_0;
   parameters.b3_left = std::isfinite(cfg.b3_left) ? cfg.b3_left : cfg.b3_0;
   parameters.rho_right = cfg.rho_right;
   parameters.pressure_right = cfg.pressure_right;
   parameters.v1_right = cfg.v1_right;
   parameters.v2_right = cfg.v2_right;
   parameters.v3_right = cfg.v3_right;
   parameters.b1_right = std::isfinite(cfg.b1_right) ? cfg.b1_right : cfg.b1_0;
   parameters.b2_right = std::isfinite(cfg.b2_right) ? cfg.b2_right : cfg.b2_0;
   parameters.b3_right = std::isfinite(cfg.b3_right) ? cfg.b3_right : cfg.b3_0;
   parameters.eos = eos_params(cfg);
   parameters.provider_radius_is_mesh1 = kind == FluidInitialDataKind::Provider
      && cfg.mesh_coordinate == "spherical" && provider.tov.active();
   parameters.atmosphere = initial_atmosphere_parameters(cfg, provider);
   parameters.equilibrium_torus = equilibrium_torus_parameters(cfg, provider);
   if (kind == FluidInitialDataKind::EquilibriumTorus) {
      EquilibriumTorusParameters const& torus = parameters.equilibrium_torus;
      const amrex::Real axis[3]{torus.axis1, torus.axis2, torus.axis3};
      int reference_direction = 0;
      for (int direction = 1; direction < 3; ++direction) {
         if (std::abs(axis[direction]) < std::abs(axis[reference_direction])) {
            reference_direction = direction;
         }
      }
      amrex::Real reference[3]{0.0, 0.0, 0.0};
      reference[reference_direction] = 1.0;
      amrex::Real radial[3]{
         axis[1] * reference[2] - axis[2] * reference[1],
         axis[2] * reference[0] - axis[0] * reference[2],
         axis[0] * reference[1] - axis[1] * reference[0],
      };
      const amrex::Real radial_norm = std::sqrt(
         radial[0] * radial[0] + radial[1] * radial[1]
         + radial[2] * radial[2]
      );
      for (amrex::Real& component : radial) {
         component /= radial_norm;
      }
      const PatchCoordinate inner_point{
         torus.center1 + torus.inner_radius * radial[0],
         torus.center2 + torus.inner_radius * radial[1],
         torus.center3 + torus.inner_radius * radial[2],
      };
      const FixedSpacetimeGeometry inner_geometry =
         initial_data_spacetime_geometry(
            provider.fixed_spacetime,
            inner_point.q1,
            inner_point.q2,
            inner_point.q3,
            0.0
         );
      const EquilibriumTorusPoint inner = equilibrium_torus_point(
         torus, 1.0, inner_point, inner_geometry.metric
      );
      if (!inner.timelike) {
         amrex::Abort(
            "equilibrium-torus inner edge does not admit circular timelike flow"
         );
      }
      parameters.equilibrium_torus_inner_normalization =
         torus.rotation_law == DiskRotationLaw::ConstantL
         ? inner.minus_covariant_time_velocity
         : std::exp(-inner.log_specific_enthalpy);
      if (!(parameters.equilibrium_torus_inner_normalization > 0.0)
          || !std::isfinite(
               parameters.equilibrium_torus_inner_normalization
             )) {
         amrex::Abort("equilibrium-torus inner normalization is invalid");
      }
   }
   return parameters;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FluidPrimitiveSample sample_fluid_initial_data(
   FluidInitialDataParameters const& parameters,
   FluidInitialDataContext const& context,
   InitialDataSamplePoint const& sample,
   FixedSpacetimeGeometry const& local_geometry,
   RelativisticInitialDataView const& provider
) noexcept
{
   amrex::Real rho = parameters.rho0;
   amrex::Real pressure = parameters.pressure0;
   amrex::Real v1 = parameters.v1_0;
   amrex::Real v2 = parameters.v2_0;
   amrex::Real v3 = parameters.v3_0;
   amrex::Real b1 = parameters.b1_0;
   amrex::Real b2 = parameters.b2_0;
   amrex::Real b3 = parameters.b3_0;
   SpacetimeMetric metric = local_geometry.metric;
   amrex::Real specific_internal_energy =
      relativistic_specific_internal_energy_from_pressure<EOSMethod>(
         rho, pressure, parameters.eos
      );

   if (parameters.kind == FluidInitialDataKind::Atmosphere) {
      rho = 0.0;
      pressure = 0.0;
      specific_internal_energy = 0.0;
      v1 = 0.0;
      v2 = 0.0;
      v3 = 0.0;
      b1 = 0.0;
      b2 = 0.0;
      b3 = 0.0;
   } else if (parameters.kind == FluidInitialDataKind::Provider) {
      const RelativisticInitialDataSample point =
         parameters.provider_radius_is_mesh1
         ? provider.sample_tov_radius(sample.mesh.q1)
         : provider.sample(sample.physical);
      rho = point.fluid.rho;
      pressure = point.fluid.pressure;
      specific_internal_energy = point.fluid.eps;
      v1 = point.fluid.v1;
      v2 = point.fluid.v2;
      v3 = point.fluid.v3;
      b1 = point.b1;
      b2 = point.b2;
      b3 = point.b3;
      metric = generated::z4c_spacetime_metric(point.spacetime);
   } else if (parameters.kind == FluidInitialDataKind::EquilibriumTorus) {
      const EquilibriumTorusParameters& torus = parameters.equilibrium_torus;
      const amrex::Real displacement1 = sample.physical.q1 - torus.center1;
      const amrex::Real displacement2 = sample.physical.q2 - torus.center2;
      const amrex::Real displacement3 = sample.physical.q3 - torus.center3;
      const amrex::Real radius = std::sqrt(
         displacement1 * displacement1 + displacement2 * displacement2
         + displacement3 * displacement3
      );
      const EquilibriumTorusPoint target = equilibrium_torus_point(
         torus,
         parameters.equilibrium_torus_inner_normalization,
         sample.physical,
         local_geometry.metric
      );
      ColdBarotropeState matter;
      if (radius >= torus.inner_radius && target.timelike
          && target.log_specific_enthalpy > 0.0) {
         matter = cold_barotrope_from_log_enthalpy(
            target.log_specific_enthalpy, parameters.eos
         );
         v1 = target.v1;
         v2 = target.v2;
         v3 = target.v3;
      }
      rho = matter.rho;
      pressure = matter.pressure;
      specific_internal_energy = matter.specific_internal_energy;
   } else if (parameters.kind == FluidInitialDataKind::FixedMetricHse) {
      const amrex::Real reference_enthalpy = 1.0
         + parameters.eos.gamma * parameters.pressure0
         / ((parameters.eos.gamma - 1.0) * parameters.rho0);
      const amrex::Real enthalpy = reference_enthalpy * context.reference_lapse
         / local_geometry.metric.lapse;
      const amrex::Real thermodynamic_ratio = (enthalpy - 1.0)
         / (reference_enthalpy - 1.0);
      rho = parameters.rho0 * std::pow(
         thermodynamic_ratio, 1.0 / (parameters.eos.gamma - 1.0)
      );
      pressure = parameters.pressure0 * std::pow(
         rho / parameters.rho0, parameters.eos.gamma
      );
      v1 = 0.0;
      v2 = 0.0;
      v3 = 0.0;
      specific_internal_energy =
         relativistic_specific_internal_energy_from_pressure<EOSMethod>(
            rho, pressure, parameters.eos
         );
   } else if (parameters.kind == FluidInitialDataKind::KelvinHelmholtz) {
#if (AMREX_SPACEDIM >= 2)
      const amrex::Real x_length = context.probhi[0] - context.problo[0];
      const amrex::Real y_length = context.probhi[1] - context.problo[1];
      const amrex::Real x_normalized =
         (sample.physical.q1 - context.problo[0]) / x_length;
      const amrex::Real width = amrex::max(parameters.layer_width, 2.0 * context.dx[1]);
      amrex::Real stream_profile = 0.0;
      amrex::Real layer_profile = 0.0;
      if (context.periodic_second_direction) {
         const amrex::Real y_low = sample.physical.q2
            - (context.problo[1] + 0.25 * y_length);
         const amrex::Real y_high = sample.physical.q2
            - (context.problo[1] + 0.75 * y_length);
         stream_profile = std::tanh(y_low / width)
            - std::tanh(y_high / width) - 1.0;
         layer_profile = std::exp(-0.5 * y_low * y_low / (width * width))
            + std::exp(-0.5 * y_high * y_high / (width * width));
      } else {
         const amrex::Real y_offset = sample.physical.q2 - parameters.sheet_center;
         stream_profile = std::tanh(y_offset / width);
         layer_profile = std::exp(
            -0.5 * y_offset * y_offset / (width * width)
         );
      }
      rho = parameters.rho0 * (
         1.0 + parameters.density_amplitude * stream_profile
      );
      v1 = parameters.v1_0 * stream_profile;
      v2 = parameters.amplitude
         * std::sin(constants::two_pi * parameters.wave_number * x_normalized)
         * layer_profile;
      v3 = 0.0;
#endif
   } else if (parameters.kind == FluidInitialDataKind::SineWave) {
      const amrex::Real length = context.probhi[0] - context.problo[0];
      const amrex::Real phase = constants::two_pi * parameters.wave_number
         * (sample.physical.q1 - context.problo[0]) / length;
      rho *= 1.0 + parameters.amplitude * std::sin(phase);
   } else if (parameters.kind == FluidInitialDataKind::ShockPlane
              || parameters.kind == FluidInitialDataKind::PeriodicShockPlane) {
      const amrex::Real y = AMREX_SPACEDIM >= 2
         ? sample.physical.q2
         : parameters.shock_y0;
      const amrex::Real z = AMREX_SPACEDIM == 3
         ? sample.physical.q3
         : parameters.shock_z0;
      const amrex::Real signed_distance =
         parameters.shock_n1 * (sample.physical.q1 - parameters.shock_x0)
         + parameters.shock_n2 * (y - parameters.shock_y0)
         + parameters.shock_n3 * (z - parameters.shock_z0);
      const bool left = parameters.kind == FluidInitialDataKind::PeriodicShockPlane
         ? std::sin(constants::two_pi * signed_distance) < 0.0
         : signed_distance < 0.0;
      rho = left ? parameters.rho_left : parameters.rho_right;
      pressure = left ? parameters.pressure_left : parameters.pressure_right;
      v1 = left ? parameters.v1_left : parameters.v1_right;
      v2 = left ? parameters.v2_left : parameters.v2_right;
      v3 = left ? parameters.v3_left : parameters.v3_right;
      b1 = left ? parameters.b1_left : parameters.b1_right;
      b2 = left ? parameters.b2_left : parameters.b2_right;
      b3 = left ? parameters.b3_left : parameters.b3_right;
   }

   const EOSThermodynamicState atmosphere = initial_atmosphere_state<EOSMethod>(
      parameters.atmosphere, sample
   );
   const bool apply_atmosphere = rho < atmosphere.rho;
   if (apply_atmosphere) {
      rho = atmosphere.rho;
      pressure = atmosphere.pressure;
      specific_internal_energy = atmosphere.specific_internal_energy;
      v1 = 0.0;
      v2 = 0.0;
      v3 = 0.0;
   }

   const bool state_was_floored = rho < parameters.eos.rho_floor
      || pressure < parameters.eos.pressure_floor
      || pressure < eos_cold_pressure<EOSMethod>(
         amrex::max(rho, parameters.eos.rho_floor), parameters.eos
      );
   const bool recompute_thermodynamics = !apply_atmosphere && (state_was_floored
      || (parameters.kind != FluidInitialDataKind::Provider
          && parameters.kind != FluidInitialDataKind::FixedMetricHse
          && parameters.kind != FluidInitialDataKind::EquilibriumTorus));
   if (state_was_floored) {
      const EOSThermodynamicState floor_state = eos_project_to_floors<EOSMethod>(
         rho, pressure, parameters.eos
      );
      rho = floor_state.rho;
      specific_internal_energy = floor_state.specific_internal_energy;
      pressure = floor_state.pressure;
   } else if (recompute_thermodynamics) {
      const amrex::Real internal_energy_density =
         eos_internal_energy_density_from_pressure<EOSMethod>(
            rho, pressure, parameters.eos
         );
      specific_internal_energy = internal_energy_density / rho;
      pressure = eos_pressure_from_density_energy<EOSMethod>(
         rho, internal_energy_density, parameters.eos
      );
   }
   return {
      {rho, specific_internal_energy, pressure, v1, v2, v3},
      b1,
      b2,
      b3,
      metric,
   };
}

} // namespace pyfluid_amrex
