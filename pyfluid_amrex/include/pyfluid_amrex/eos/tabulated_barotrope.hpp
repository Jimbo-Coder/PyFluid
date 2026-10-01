#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "pyfluid_amrex/eos/types.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct TabulatedColdState {
   amrex::Real density = 0.0_rt;
   amrex::Real pressure = 0.0_rt;
   amrex::Real energy_density = 0.0_rt;
   amrex::Real specific_enthalpy = 1.0_rt;
   amrex::Real sound_speed_squared = 0.0_rt;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int tabulated_interval_from_value(
   amrex::Real value,
   amrex::Real const* increasing_values,
   int size
) noexcept
{
   int lower = 0;
   int upper = size - 1;
   while (upper - lower > 1) {
      const int middle = lower + (upper - lower) / 2;
      if (value >= increasing_values[middle]) {
         lower = middle;
      } else {
         upper = middle;
      }
   }
   return lower;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
TabulatedColdState tabulated_cold_from_density(
   amrex::Real density,
   TabulatedEOSView const& table
) noexcept
{
   if (!(density > 0.0_rt) || table.size < 2) {
      return {};
   }
   const amrex::Real log_density = std::log(density);
   const amrex::Real minimum_density = std::exp(table.log_density[0]);
   const amrex::Real minimum_pressure = std::exp(table.log_pressure[0]);
   if (log_density <= table.log_density[0]) {
      const amrex::Real gamma = table.low_density_gamma;
      const amrex::Real pressure = minimum_pressure * std::pow(
         density / minimum_density, gamma
      );
      const amrex::Real energy_density = density
         + pressure / (gamma - 1.0_rt);
      return {
         density,
         pressure,
         energy_density,
         (energy_density + pressure) / density,
         gamma * pressure / (energy_density + pressure),
      };
   }

   const int last = table.size - 1;
   if (log_density >= table.log_density[last]) {
      const amrex::Real reference_density = std::exp(table.log_density[last]);
      const amrex::Real reference_pressure = std::exp(table.log_pressure[last]);
      const amrex::Real reference_enthalpy = std::exp(
         table.log_specific_enthalpy[last]
      );
      const amrex::Real reference_energy =
         reference_density * reference_enthalpy - reference_pressure;
      const amrex::Real sound_speed_squared =
         table.high_density_sound_speed_squared;
      const amrex::Real affine_constant = reference_pressure
         - sound_speed_squared * reference_energy;
      const amrex::Real energy_offset = affine_constant
         / (1.0_rt + sound_speed_squared);
      const amrex::Real energy_density = (
         reference_energy + energy_offset
      ) * std::pow(
         density / reference_density, 1.0_rt + sound_speed_squared
      ) - energy_offset;
      const amrex::Real pressure = reference_pressure
         + sound_speed_squared * (energy_density - reference_energy);
      return {
         density,
         pressure,
         energy_density,
         (energy_density + pressure) / density,
         sound_speed_squared,
      };
   }

   const int left = tabulated_interval_from_value(
      log_density, table.log_density, table.size
   );
   const int right = left + 1;
   const amrex::Real inverse_width = 1.0_rt
      / (table.log_density[right] - table.log_density[left]);
   const amrex::Real fraction = (
      log_density - table.log_density[left]
   ) * inverse_width;
   const amrex::Real pressure_slope = (
      table.log_pressure[right] - table.log_pressure[left]
   ) * inverse_width;
   const amrex::Real enthalpy_slope = (
      table.log_specific_enthalpy[right]
      - table.log_specific_enthalpy[left]
   ) * inverse_width;
   const amrex::Real log_pressure = table.log_pressure[left]
      + fraction * (
         table.log_pressure[right] - table.log_pressure[left]
      );
   const amrex::Real log_enthalpy = table.log_specific_enthalpy[left]
      + fraction * (
         table.log_specific_enthalpy[right]
         - table.log_specific_enthalpy[left]
      );
   const amrex::Real pressure = std::exp(log_pressure);
   const amrex::Real specific_enthalpy = std::exp(log_enthalpy);
   const amrex::Real energy_density = density * specific_enthalpy - pressure;
   const amrex::Real pressure_derivative = pressure_slope * pressure;
   const amrex::Real energy_derivative =
      density * specific_enthalpy * (1.0_rt + enthalpy_slope)
      - pressure_derivative;
   return {
      density,
      pressure,
      energy_density,
      specific_enthalpy,
      pressure_derivative / energy_derivative,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real tabulated_log_enthalpy_from_density(
   amrex::Real density,
   TabulatedEOSView const& table
) noexcept
{
   if (!(density > 0.0_rt) || table.size < 2) {
      return 0.0_rt;
   }
   const amrex::Real log_density = std::log(density);
   if (log_density <= table.log_density[0]) {
      const amrex::Real minimum_density = std::exp(table.log_density[0]);
      const amrex::Real minimum_pressure = std::exp(table.log_pressure[0]);
      const amrex::Real pressure = minimum_pressure * std::pow(
         density / minimum_density, table.low_density_gamma
      );
      return std::log1p(
         table.low_density_gamma / (table.low_density_gamma - 1.0_rt)
         * pressure / density
      );
   }
   const int last = table.size - 1;
   if (log_density >= table.log_density[last]) {
      return std::log(
         tabulated_cold_from_density(density, table).specific_enthalpy
      );
   }
   const int left = tabulated_interval_from_value(
      log_density, table.log_density, table.size
   );
   const int right = left + 1;
   const amrex::Real fraction = (
      log_density - table.log_density[left]
   ) / (table.log_density[right] - table.log_density[left]);
   return table.log_specific_enthalpy[left]
      + fraction * (
         table.log_specific_enthalpy[right]
         - table.log_specific_enthalpy[left]
      );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
TabulatedColdState tabulated_cold_from_log_enthalpy(
   amrex::Real log_specific_enthalpy,
   TabulatedEOSView const& table
) noexcept
{
   if (!(log_specific_enthalpy > 0.0_rt) || table.size < 2) {
      return {};
   }
   const amrex::Real minimum_log_enthalpy = table.log_specific_enthalpy[0];
   if (log_specific_enthalpy <= minimum_log_enthalpy) {
      const amrex::Real minimum_density = std::exp(table.log_density[0]);
      const amrex::Real density = minimum_density * std::pow(
         std::expm1(log_specific_enthalpy)
            / std::expm1(minimum_log_enthalpy),
         1.0_rt / (table.low_density_gamma - 1.0_rt)
      );
      return tabulated_cold_from_density(density, table);
   }

   const int last = table.size - 1;
   if (log_specific_enthalpy >= table.log_specific_enthalpy[last]) {
      const amrex::Real reference_density = std::exp(table.log_density[last]);
      const amrex::Real reference_enthalpy = std::exp(
         table.log_specific_enthalpy[last]
      );
      const amrex::Real specific_enthalpy = std::exp(log_specific_enthalpy);
      const amrex::Real density = reference_density * std::pow(
         specific_enthalpy / reference_enthalpy,
         1.0_rt / table.high_density_sound_speed_squared
      );
      return tabulated_cold_from_density(density, table);
   }

   const int left = tabulated_interval_from_value(
      log_specific_enthalpy,
      table.log_specific_enthalpy,
      table.size
   );
   const int right = left + 1;
   const amrex::Real fraction = (
      log_specific_enthalpy - table.log_specific_enthalpy[left]
   ) / (
      table.log_specific_enthalpy[right]
      - table.log_specific_enthalpy[left]
   );
   const amrex::Real log_density = table.log_density[left]
      + fraction * (table.log_density[right] - table.log_density[left]);
   return tabulated_cold_from_density(std::exp(log_density), table);
}

} // namespace pyfluid_amrex
