#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "pyfluid_amrex/eos/equation_of_state.hpp"

namespace pyfluid_amrex {

struct ColdBarotropeState {
   amrex::Real rho = 0.0;
   amrex::Real pressure = 0.0;
   amrex::Real specific_internal_energy = 0.0;
   amrex::Real energy_density = 0.0;
   amrex::Real specific_enthalpy = 1.0;
   amrex::Real sound_speed_squared = 0.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ColdBarotropeState cold_barotrope_from_density(
   amrex::Real rho,
   EOSParams const& eos
) noexcept
{
   if (!(rho > 0.0)) {
      return {};
   }
   const PolytropeSegment segment = polytrope_segment(rho, eos);
   const amrex::Real pressure = eos_cold_pressure<EOSPolytrope>(rho, eos);
   const amrex::Real internal_energy_density =
      eos_cold_energy_density<EOSPolytrope>(rho, eos);
   const amrex::Real energy_density = rho + internal_energy_density;
   const amrex::Real specific_internal_energy = internal_energy_density / rho;
   return {
      rho,
      pressure,
      specific_internal_energy,
      energy_density,
      relativistic_specific_enthalpy<EOSPolytrope>(
         rho, specific_internal_energy, eos
      ),
      segment.gamma * pressure / (energy_density + pressure),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cold_log_specific_enthalpy(
   amrex::Real rho,
   EOSParams const& eos
) noexcept
{
   if (!(rho > 0.0)) {
      return 0.0;
   }
   return std::log(cold_barotrope_from_density(rho, eos).specific_enthalpy);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int cold_polytrope_segment_from_log_enthalpy(
   amrex::Real log_specific_enthalpy,
   EOSParams const& eos
) noexcept
{
   int segment = 0;
   for (int boundary = 0; boundary < eos.polytrope_segments - 1; ++boundary) {
      const amrex::Real break_log_enthalpy = cold_log_specific_enthalpy(
         eos.segment_density_break[boundary], eos
      );
      if (log_specific_enthalpy >= break_log_enthalpy) {
         segment = boundary + 1;
      }
   }
   return segment;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ColdBarotropeState cold_barotrope_from_log_enthalpy(
   amrex::Real log_specific_enthalpy,
   EOSParams const& eos
) noexcept
{
   if (!(log_specific_enthalpy > 0.0)) {
      return {};
   }
   const int segment_index = cold_polytrope_segment_from_log_enthalpy(
      log_specific_enthalpy, eos
   );
   const amrex::Real k = eos.segment_k[segment_index];
   const amrex::Real gamma = eos.segment_gamma[segment_index];
   const amrex::Real energy_offset = eos.segment_energy_offset[segment_index];
   const amrex::Real specific_enthalpy = std::exp(log_specific_enthalpy);
   const amrex::Real density_power =
      (specific_enthalpy - 1.0 - energy_offset)
      * (gamma - 1.0) / (gamma * k);
   const amrex::Real rho = density_power > 0.0
      ? std::pow(density_power, 1.0 / (gamma - 1.0))
      : 0.0;
   if (!(rho > 0.0)) {
      return {};
   }
   ColdBarotropeState state = cold_barotrope_from_density(rho, eos);
   state.specific_enthalpy = specific_enthalpy;
   return state;
}

} // namespace pyfluid_amrex
