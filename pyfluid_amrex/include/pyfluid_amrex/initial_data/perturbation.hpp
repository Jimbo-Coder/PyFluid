#pragma once

#include <AMReX_GpuQualifiers.H>

#include <cmath>

#include "pyfluid_amrex/azimuthal_geometry.hpp"
#include "pyfluid_amrex/eos/cold_barotrope.hpp"
#include "pyfluid_amrex/generated/initial_data_method_catalog.hpp"
#include "pyfluid_amrex/patch_maps.hpp"

namespace pyfluid_amrex {

struct InitialDataPerturbationParameters {
   InitialDataPerturbation kind = InitialDataPerturbation::None;
   amrex::Real pressure_scale = 1.0;
   amrex::Real amplitude = 0.0;
   int mode = 1;
   amrex::Real phase = 0.0;
   amrex::Real reference_radius = 1.0;
   AzimuthalFrame frame{};
   EOSParams eos{};
};

struct InitialDataPerturbationPoint {
   ColdBarotropeState matter{};
   amrex::Real fractional_density_change = 0.0;
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
   if (pressure_scale < 1.0) {
      for (int iteration = 0; iteration < 64; ++iteration) {
         lower_density *= 0.5;
         if (cold_barotrope_from_density(lower_density, eos).pressure
             <= target_pressure) {
            break;
         }
      }
   } else if (pressure_scale > 1.0) {
      for (int iteration = 0; iteration < 64; ++iteration) {
         upper_density *= 2.0;
         if (cold_barotrope_from_density(upper_density, eos).pressure
             >= target_pressure) {
            break;
         }
      }
   } else {
      return original;
   }
   for (int iteration = 0; iteration < 64; ++iteration) {
      const amrex::Real midpoint = 0.5 * (
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
      0.5 * (lower_density + upper_density), eos
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
      return {cold_barotrope_from_density(rho, parameters.eos), 0.0, true};
   }
   if (parameters.kind == InitialDataPerturbation::ColdPressureScale) {
      const ColdBarotropeState matter = cold_barotrope_with_scaled_pressure(
         rho, parameters.pressure_scale, parameters.eos
      );
      return {
         matter,
         matter.rho / rho - 1.0,
         matter.rho > 0.0 && std::isfinite(matter.pressure),
      };
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
   const amrex::Real perturbed_density = rho * (1.0 + fractional_change);
   const ColdBarotropeState matter = cold_barotrope_from_density(
      perturbed_density, parameters.eos
   );
   return {
      matter,
      fractional_change,
      perturbed_density > 0.0 && matter.pressure >= 0.0
         && std::isfinite(matter.specific_internal_energy),
   };
}

} // namespace pyfluid_amrex
