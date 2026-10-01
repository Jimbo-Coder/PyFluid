#pragma once

#include <AMReX_GpuQualifiers.H>

#include <cmath>

#include "pyfluid_amrex/azimuthal_geometry.hpp"
#include "pyfluid_amrex/eos/cold_barotrope.hpp"
#include "pyfluid_amrex/generated/initial_data_method_catalog.hpp"
#include "pyfluid_amrex/patch_maps.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct InitialDataPerturbationParameters {
   InitialDataPerturbation kind = InitialDataPerturbation::None;
   amrex::Real pressure_scale = 1.0_rt;
   amrex::Real amplitude = 0.0_rt;
   int mode = 1;
   amrex::Real phase = 0.0_rt;
   amrex::Real reference_radius = 1.0_rt;
   AzimuthalFrame frame{};
   EOSParams eos{};
};

struct InitialDataPerturbationPoint {
   ColdBarotropeState matter{};
   amrex::Real fractional_density_change = 0.0_rt;
   bool valid = true;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ColdBarotropeState cold_barotrope_with_scaled_pressure(
   amrex::Real rho,
   amrex::Real pressure_scale,
   EOSParams const& eos
) noexcept
{
   // A cold barotrope has only one independent thermodynamic variable.
   // Pressure therefore cannot be changed at fixed density while retaining
   // P = P_cold(rho).  Treat the scale as a target pressure, invert the same
   // cold EoS for rho, and return the complete state at that new density.
   const ColdBarotropeState original = cold_barotrope_from_density(rho, eos);
   const amrex::Real target_pressure = pressure_scale * original.pressure;
   amrex::Real lower_density = rho;
   amrex::Real upper_density = rho;
   if (pressure_scale < 1.0_rt) {
      for (int iteration = 0; iteration < 64; ++iteration) {
         lower_density *= 0.5_rt;
         if (cold_barotrope_from_density(lower_density, eos).pressure
             <= target_pressure) {
            break;
         }
      }
   } else if (pressure_scale > 1.0_rt) {
      for (int iteration = 0; iteration < 64; ++iteration) {
         upper_density *= 2.0_rt;
         if (cold_barotrope_from_density(upper_density, eos).pressure
             >= target_pressure) {
            break;
         }
      }
   } else {
      return original;
   }
   for (int iteration = 0; iteration < 64; ++iteration) {
      const amrex::Real midpoint = 0.5_rt * (
         lower_density + upper_density
      );
      if (cold_barotrope_from_density(midpoint, eos).pressure
          < target_pressure) {
         lower_density = midpoint;
      } else {
         upper_density = midpoint;
      }
   }
   return cold_barotrope_from_density(
      0.5_rt * (lower_density + upper_density), eos
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
InitialDataPerturbationPoint initial_data_perturbation_point(
   InitialDataPerturbationParameters const& parameters,
   PatchCoordinate const& physical,
   amrex::Real rho
) noexcept
{
   if (parameters.kind == InitialDataPerturbation::None) {
      return {cold_barotrope_from_density(rho, parameters.eos), 0.0_rt, true};
   }
   if (parameters.kind == InitialDataPerturbation::ColdPressureScale) {
      const ColdBarotropeState matter = cold_barotrope_with_scaled_pressure(
         rho, parameters.pressure_scale, parameters.eos
      );
      return {
         matter,
         matter.rho / rho - 1.0_rt,
         matter.rho > 0.0_rt && std::isfinite(matter.pressure),
      };
   }
   if (parameters.kind == InitialDataPerturbation::DensityRadialFont2002) {
      const amrex::Real relative1 = physical.q1 - parameters.frame.origin[0];
      const amrex::Real relative2 = physical.q2 - parameters.frame.origin[1];
      const amrex::Real relative3 = physical.q3 - parameters.frame.origin[2];
      const amrex::Real radius = std::sqrt(
         relative1 * relative1 + relative2 * relative2
         + relative3 * relative3
      );
      const amrex::Real normalized_radius = radius
         < parameters.reference_radius
         ? radius / parameters.reference_radius
         : 1.0_rt;
      const amrex::Real fractional_change = parameters.amplitude * std::cos(
         1.57079632679489661923_rt * normalized_radius
      );
      const amrex::Real perturbed_density = rho * (1.0_rt + fractional_change);
      // Font et al.'s controlled-collapse perturbation changes the rest-mass
      // density while retaining the equilibrium pressure.  The specific
      // internal energy is then recomputed from the evolution EoS.  Cold-
      // projecting the denser state would instead add pressure support and
      // can drive the unstable model away from collapse.
      const ColdBarotropeState equilibrium = cold_barotrope_from_density(
         rho, parameters.eos
      );
      const amrex::Real internal_energy_density =
         eos_internal_energy_density_from_pressure<EOSRuntime>(
            perturbed_density, equilibrium.pressure, parameters.eos
         );
      const amrex::Real specific_internal_energy =
         internal_energy_density / perturbed_density;
      const ColdBarotropeState matter{
         perturbed_density,
         equilibrium.pressure,
         specific_internal_energy,
         perturbed_density + internal_energy_density,
         relativistic_specific_enthalpy<EOSRuntime>(
            perturbed_density, specific_internal_energy, parameters.eos
         ),
         relativistic_sound_speed_squared<EOSRuntime>(
            perturbed_density, specific_internal_energy, parameters.eos
         ),
      };
      return {
         matter,
         fractional_change,
         perturbed_density > 0.0_rt && matter.pressure >= 0.0_rt
            && matter.energy_density > 0.0_rt
            && matter.specific_enthalpy > 0.0_rt
            && matter.sound_speed_squared >= 0.0_rt
            && matter.sound_speed_squared <= 1.0_rt
            && std::isfinite(matter.specific_internal_energy)
            && std::isfinite(matter.specific_enthalpy)
            && std::isfinite(matter.sound_speed_squared),
      };
   }
   if (parameters.kind != InitialDataPerturbation::DensityAzimuthalEspino2019) {
      return {{}, 0.0_rt, false};
   }
   const AzimuthalCoordinate azimuthal = azimuthal_coordinate(
      parameters.frame,
      {{physical.q1, physical.q2, physical.q3}}
   );
   const amrex::Real fractional_change = parameters.amplitude
      * (azimuthal.cylindrical_radius / parameters.reference_radius)
      * std::sin(
         static_cast<amrex::Real>(parameters.mode) * azimuthal.phi
         + parameters.phase
      );
   const amrex::Real perturbed_density = rho * (1.0_rt + fractional_change);
   const ColdBarotropeState matter = cold_barotrope_from_density(
      perturbed_density, parameters.eos
   );
   return {
      matter,
      fractional_change,
      perturbed_density > 0.0_rt && matter.pressure >= 0.0_rt
         && std::isfinite(matter.specific_internal_energy),
   };
}

} // namespace pyfluid_amrex
