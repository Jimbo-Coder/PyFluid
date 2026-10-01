#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "pyfluid_amrex/eos/equation_of_state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct ColdBarotropeState {
   amrex::Real rho = 0.0_rt;
   amrex::Real pressure = 0.0_rt;
   amrex::Real specific_internal_energy = 0.0_rt;
   amrex::Real energy_density = 0.0_rt;
   amrex::Real specific_enthalpy = 1.0_rt;
   amrex::Real sound_speed_squared = 0.0_rt;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool cold_barotrope_supported(EOSParams const& eos) noexcept
{
   return eos.method == EOSPolytrope
      || eos.method == EOSHybridPolytrope
      || eos.method == EOSCompose;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ColdBarotropeState cold_barotrope_from_density(
   amrex::Real rho,
   EOSParams const& eos
) noexcept
{
   if (!(rho > 0.0_rt)) {
      return {};
   }
   if (eos.method == EOSCompose) {
      const TabulatedColdState cold = tabulated_cold_from_density(
         rho, eos.table
      );
      return {
         cold.density,
         cold.pressure,
         cold.energy_density / cold.density - 1.0_rt,
         cold.energy_density,
         cold.specific_enthalpy,
         cold.sound_speed_squared,
      };
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
   if (!(rho > 0.0_rt)) {
      return 0.0_rt;
   }
   if (eos.method == EOSCompose) {
      return tabulated_log_enthalpy_from_density(rho, eos.table);
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
   if (!(log_specific_enthalpy > 0.0_rt)) {
      return {};
   }
   if (eos.method == EOSCompose) {
      const TabulatedColdState cold = tabulated_cold_from_log_enthalpy(
         log_specific_enthalpy, eos.table
      );
      return {
         cold.density,
         cold.pressure,
         cold.energy_density / cold.density - 1.0_rt,
         cold.energy_density,
         cold.specific_enthalpy,
         cold.sound_speed_squared,
      };
   }
   const int segment_index = cold_polytrope_segment_from_log_enthalpy(
      log_specific_enthalpy, eos
   );
   const amrex::Real k = eos.segment_k[segment_index];
   const amrex::Real gamma = eos.segment_gamma[segment_index];
   const amrex::Real energy_offset = eos.segment_energy_offset[segment_index];
   const amrex::Real specific_enthalpy = std::exp(log_specific_enthalpy);
   const amrex::Real density_power =
      (specific_enthalpy - 1.0_rt - energy_offset)
      * (gamma - 1.0_rt) / (gamma * k);
   const amrex::Real rho = density_power > 0.0_rt
      ? std::pow(density_power, 1.0_rt / (gamma - 1.0_rt))
      : 0.0_rt;
   if (!(rho > 0.0_rt)) {
      return {};
   }
   ColdBarotropeState state = cold_barotrope_from_density(rho, eos);
   state.specific_enthalpy = specific_enthalpy;
   return state;
}

} // namespace pyfluid_amrex
