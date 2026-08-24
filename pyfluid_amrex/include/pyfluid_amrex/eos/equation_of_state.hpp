#pragma once

#include <AMReX_Array4.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "pyfluid_amrex/eos/tabulated_barotrope.hpp"
#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/kernels_generated/eos.hpp"
#include "pyfluid_amrex/kernels_generated/vector_ops.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

struct PolytropeSegment {
   amrex::Real k;
   amrex::Real gamma;
   amrex::Real energy_offset;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PolytropeSegment polytrope_segment(amrex::Real rho, EOSParams const& eos) noexcept
{
   int segment = 0;
   for (int boundary = 0; boundary < eos.polytrope_segments - 1; ++boundary) {
      if (rho >= eos.segment_density_break[boundary]) {
         segment = boundary + 1;
      }
   }
   return {
      eos.segment_k[segment],
      eos.segment_gamma[segment],
      eos.segment_energy_offset[segment],
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real kinetic_energy_density(amrex::Real rho, StateVector3 const& velocity) noexcept
{
   return 0.5 * rho * dot(velocity, velocity);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real kinetic_energy_density(amrex::Real rho, amrex::Real vx, amrex::Real vy, amrex::Real vz) noexcept
{
   return kinetic_energy_density(rho, StateVector3{vx, vy, vz});
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool eos_is_barotropic(EOSParams const& eos) noexcept
{
   if constexpr (EOSMethod == EOSRuntime) {
      return eos.method == EOSPolytrope;
   }
   return EOSMethod == EOSPolytrope;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eos_cold_pressure(amrex::Real rho, EOSParams const& eos) noexcept
{
   if constexpr (EOSMethod == EOSRuntime) {
      if (eos.method == EOSCompose) {
         return tabulated_cold_from_density(rho, eos.table).pressure;
      }
      if (eos.method == EOSPolytrope || eos.method == EOSHybridPolytrope) {
         return eos_cold_pressure<EOSPolytrope>(rho, eos);
      }
      return 0.0;
   } else if constexpr (EOSMethod == EOSCompose) {
      return tabulated_cold_from_density(rho, eos.table).pressure;
   } else if constexpr (EOSMethod == EOSIdealGas) {
      static_cast<void>(rho);
      static_cast<void>(eos);
      return 0.0;
   } else {
      const PolytropeSegment segment = polytrope_segment(rho, eos);
      return generated::polytrope_cold_pressure(
         rho,
         segment.k,
         segment.gamma
      );
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eos_thermal_entropy_proxy(
   amrex::Real rho,
   amrex::Real pressure,
   EOSParams const& eos
) noexcept
{
   if constexpr (EOSMethod == EOSRuntime) {
      if (eos.method == EOSPolytrope) {
         return 0.0;
      }
      const amrex::Real adiabatic_index = eos.method == EOSIdealGas
         ? eos.gamma
         : eos.thermal_gamma;
      const amrex::Real base_pressure = eos.method == EOSIdealGas
         ? 0.0
         : eos_cold_pressure<EOSRuntime>(rho, eos);
      return (pressure - base_pressure) / std::pow(
         amrex::max(rho, amrex::Real{1.0e-300}), adiabatic_index
      );
   } else if constexpr (EOSMethod == EOSPolytrope) {
      static_cast<void>(rho);
      static_cast<void>(pressure);
      static_cast<void>(eos);
      return 0.0;
   } else if constexpr (
      EOSMethod == EOSHybridPolytrope || EOSMethod == EOSCompose
   ) {
      return (pressure - eos_cold_pressure<EOSMethod>(rho, eos)) / std::pow(
         amrex::max(rho, amrex::Real{1.0e-300}), eos.thermal_gamma
      );
   } else {
      return pressure / std::pow(
         amrex::max(rho, amrex::Real{1.0e-300}), eos.gamma
      );
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eos_cold_energy_density(amrex::Real rho, EOSParams const& eos) noexcept
{
   if constexpr (EOSMethod == EOSRuntime) {
      if (eos.method == EOSCompose) {
         const TabulatedColdState cold = tabulated_cold_from_density(
            rho, eos.table
         );
         return cold.energy_density - cold.density;
      }
      if (eos.method == EOSPolytrope || eos.method == EOSHybridPolytrope) {
         return eos_cold_energy_density<EOSPolytrope>(rho, eos);
      }
      return 0.0;
   } else if constexpr (EOSMethod == EOSCompose) {
      const TabulatedColdState cold = tabulated_cold_from_density(rho, eos.table);
      return cold.energy_density - cold.density;
   } else if constexpr (EOSMethod == EOSIdealGas) {
      static_cast<void>(rho);
      static_cast<void>(eos);
      return 0.0;
   } else {
      const PolytropeSegment segment = polytrope_segment(rho, eos);
      return generated::polytrope_cold_energy_density(
         rho,
         segment.k,
         segment.gamma,
         segment.energy_offset
      );
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eos_internal_energy_density_from_pressure(
   amrex::Real rho,
   amrex::Real pressure,
   EOSParams const& eos
) noexcept
{
   if constexpr (EOSMethod == EOSRuntime) {
      if (eos.method == EOSPolytrope) {
         return eos_cold_energy_density<EOSRuntime>(rho, eos);
      }
      if (eos.method == EOSHybridPolytrope || eos.method == EOSCompose) {
         const amrex::Real cold_energy = eos_cold_energy_density<EOSRuntime>(
            rho, eos
         );
         const amrex::Real cold_pressure = eos_cold_pressure<EOSRuntime>(rho, eos);
         return cold_energy
            + (pressure - cold_pressure) / (eos.thermal_gamma - 1.0);
      }
      return generated::ideal_gas_internal_energy_density_from_pressure(
         pressure, eos.gamma
      );
   } else if constexpr (EOSMethod == EOSCompose) {
      const amrex::Real cold_energy = eos_cold_energy_density<EOSCompose>(rho, eos);
      const amrex::Real cold_pressure = eos_cold_pressure<EOSCompose>(rho, eos);
      return cold_energy
         + (pressure - cold_pressure) / (eos.thermal_gamma - 1.0);
   } else if constexpr (EOSMethod == EOSPolytrope) {
      static_cast<void>(pressure);
      return eos_cold_energy_density<EOSMethod>(rho, eos);
   } else if constexpr (EOSMethod == EOSHybridPolytrope) {
      const PolytropeSegment segment = polytrope_segment(rho, eos);
      return generated::hybrid_polytrope_internal_energy_density_from_pressure(
         rho,
         pressure,
         segment.k,
         segment.gamma,
         eos.thermal_gamma,
         segment.energy_offset
      );
   } else {
      static_cast<void>(rho);
      return generated::ideal_gas_internal_energy_density_from_pressure(
         pressure,
         eos.gamma
      );
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eos_pressure_from_density_energy(
   amrex::Real rho,
   amrex::Real internal_energy_density,
   EOSParams const& eos
) noexcept
{
   if constexpr (EOSMethod == EOSRuntime) {
      if (eos.method == EOSPolytrope) {
         return eos_cold_pressure<EOSRuntime>(rho, eos);
      }
      if (eos.method == EOSHybridPolytrope || eos.method == EOSCompose) {
         const amrex::Real cold_energy = eos_cold_energy_density<EOSRuntime>(
            rho, eos
         );
         const amrex::Real cold_pressure = eos_cold_pressure<EOSRuntime>(rho, eos);
         return cold_pressure + (eos.thermal_gamma - 1.0)
            * (internal_energy_density - cold_energy);
      }
      return generated::ideal_gas_pressure_from_density_energy(
         internal_energy_density, eos.gamma
      );
   } else if constexpr (EOSMethod == EOSCompose) {
      const amrex::Real cold_energy = eos_cold_energy_density<EOSCompose>(rho, eos);
      const amrex::Real cold_pressure = eos_cold_pressure<EOSCompose>(rho, eos);
      return cold_pressure + (eos.thermal_gamma - 1.0)
         * (internal_energy_density - cold_energy);
   } else if constexpr (EOSMethod == EOSPolytrope) {
      static_cast<void>(internal_energy_density);
      return eos_cold_pressure<EOSMethod>(rho, eos);
   } else if constexpr (EOSMethod == EOSHybridPolytrope) {
      const PolytropeSegment segment = polytrope_segment(rho, eos);
      return generated::hybrid_polytrope_pressure_from_density_energy(
         rho,
         internal_energy_density,
         segment.k,
         segment.gamma,
         eos.thermal_gamma,
         segment.energy_offset
      );
   } else {
      return generated::ideal_gas_pressure_from_density_energy(
         internal_energy_density,
         eos.gamma
      );
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
EOSThermodynamicState eos_project_to_floors(
   amrex::Real rho,
   amrex::Real pressure,
   EOSParams const& eos
) noexcept
{
   const amrex::Real floored_rho = amrex::max(rho, eos.rho_floor);
   const amrex::Real floored_pressure = amrex::max(
      amrex::max(pressure, eos.pressure_floor),
      eos_cold_pressure<EOSMethod>(floored_rho, eos)
   );
   const amrex::Real internal_energy_density =
      eos_internal_energy_density_from_pressure<EOSMethod>(
         floored_rho, floored_pressure, eos
      );
   return {
      floored_rho,
      internal_energy_density / floored_rho,
      eos_pressure_from_density_energy<EOSMethod>(
         floored_rho, internal_energy_density, eos
      ),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eos_pressure_from_density_energy_runtime(
   int eos_method,
   amrex::Real rho,
   amrex::Real internal_energy_density,
   EOSParams const& eos
) noexcept
{
   static_cast<void>(eos_method);
   return eos_pressure_from_density_energy<EOSRuntime>(
      rho, internal_energy_density, eos
   );
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eos_sound_speed_from_primitive(amrex::Real rho, amrex::Real pressure, EOSParams const& eos) noexcept
{
   if constexpr (EOSMethod == EOSRuntime) {
      if (eos.method == EOSCompose) {
         const TabulatedColdState cold = tabulated_cold_from_density(rho, eos.table);
         const amrex::Real sound_speed_squared =
            cold.specific_enthalpy * cold.sound_speed_squared
            + eos.thermal_gamma * (pressure - cold.pressure) / rho;
         return std::sqrt(amrex::max(amrex::Real{0.0}, sound_speed_squared));
      }
      if (eos.method == EOSPolytrope) {
         return eos_sound_speed_from_primitive<EOSPolytrope>(rho, pressure, eos);
      }
      if (eos.method == EOSHybridPolytrope) {
         return eos_sound_speed_from_primitive<EOSHybridPolytrope>(rho, pressure, eos);
      }
      return eos_sound_speed_from_primitive<EOSIdealGas>(rho, pressure, eos);
   } else if constexpr (EOSMethod == EOSCompose) {
      const TabulatedColdState cold = tabulated_cold_from_density(rho, eos.table);
      const amrex::Real sound_speed_squared =
         cold.specific_enthalpy * cold.sound_speed_squared
         + eos.thermal_gamma * (pressure - cold.pressure) / rho;
      return std::sqrt(amrex::max(amrex::Real{0.0}, sound_speed_squared));
   } else if constexpr (EOSMethod == EOSPolytrope) {
      const PolytropeSegment segment = polytrope_segment(rho, eos);
      return std::sqrt(generated::polytrope_sound_speed_squared(
         rho,
         segment.k,
         segment.gamma
      ));
   } else if constexpr (EOSMethod == EOSHybridPolytrope) {
      const PolytropeSegment segment = polytrope_segment(rho, eos);
      return std::sqrt(generated::hybrid_polytrope_sound_speed_squared(
         rho,
         pressure,
         segment.k,
         segment.gamma,
         eos.thermal_gamma
      ));
   } else {
      return std::sqrt(generated::ideal_gas_sound_speed_squared(
         rho,
         pressure,
         eos.gamma
      ));
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real relativistic_specific_internal_energy_from_pressure(
   amrex::Real rho,
   amrex::Real pressure,
   EOSParams const& eos
) noexcept
{
   return eos_internal_energy_density_from_pressure<EOSMethod>(rho, pressure, eos) / rho;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real relativistic_compose_pressure(
   amrex::Real rho,
   amrex::Real specific_internal_energy,
   const TabulatedColdState& cold,
   EOSParams const& eos
) noexcept
{
   const amrex::Real cold_internal_energy_density =
      cold.energy_density - rho;
   return cold.pressure + (eos.thermal_gamma - 1.0) * (
      rho * specific_internal_energy - cold_internal_energy_density
   );
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real relativistic_pressure(
   amrex::Real rho,
   amrex::Real specific_internal_energy,
   EOSParams const& eos
) noexcept
{
   if constexpr (EOSMethod == EOSRuntime) {
      return eos_pressure_from_density_energy<EOSRuntime>(
         rho, rho * specific_internal_energy, eos
      );
   } else if constexpr (EOSMethod == EOSCompose) {
      const TabulatedColdState cold = tabulated_cold_from_density(
         rho, eos.table
      );
      return relativistic_compose_pressure(
         rho, specific_internal_energy, cold, eos
      );
   } else if constexpr (EOSMethod == EOSPolytrope) {
      const PolytropeSegment segment = polytrope_segment(rho, eos);
      return generated::relativistic_polytrope_pressure(
         rho,
         segment.k,
         segment.gamma
      );
   } else if constexpr (EOSMethod == EOSHybridPolytrope) {
      const PolytropeSegment segment = polytrope_segment(rho, eos);
      return generated::relativistic_hybrid_polytrope_pressure(
         rho,
         specific_internal_energy,
         segment.k,
         segment.gamma,
         eos.thermal_gamma,
         segment.energy_offset
      );
   } else {
      return generated::relativistic_ideal_gas_pressure(
         rho, specific_internal_energy, eos.gamma
      );
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real relativistic_specific_enthalpy(
   amrex::Real rho,
   amrex::Real specific_internal_energy,
   EOSParams const& eos
) noexcept
{
   return 1.0 + specific_internal_energy
      + relativistic_pressure<EOSMethod>(rho, specific_internal_energy, eos) / rho;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real relativistic_sound_speed_squared(
   amrex::Real rho,
   amrex::Real specific_internal_energy,
   EOSParams const& eos
) noexcept
{
   if constexpr (EOSMethod == EOSRuntime) {
      if (eos.method == EOSCompose) {
         return relativistic_sound_speed_squared<EOSCompose>(
            rho, specific_internal_energy, eos
         );
      }
      if (eos.method == EOSPolytrope) {
         return relativistic_sound_speed_squared<EOSPolytrope>(
            rho, specific_internal_energy, eos
         );
      }
      if (eos.method == EOSHybridPolytrope) {
         return relativistic_sound_speed_squared<EOSHybridPolytrope>(
            rho, specific_internal_energy, eos
         );
      }
      return relativistic_sound_speed_squared<EOSIdealGas>(
         rho, specific_internal_energy, eos
      );
   } else if constexpr (EOSMethod == EOSCompose) {
      const TabulatedColdState cold = tabulated_cold_from_density(rho, eos.table);
      const amrex::Real pressure = relativistic_compose_pressure(
         rho, specific_internal_energy, cold, eos
      );
      const amrex::Real thermal_energy = specific_internal_energy
         - (cold.energy_density / rho - 1.0);
      const amrex::Real chi = cold.specific_enthalpy
         * cold.sound_speed_squared
         + (eos.thermal_gamma - 1.0)
            * (thermal_energy - cold.pressure / rho);
      const amrex::Real kappa = (eos.thermal_gamma - 1.0) * rho;
      const amrex::Real enthalpy = 1.0 + specific_internal_energy
         + pressure / rho;
      return (chi + pressure * kappa / (rho * rho)) / enthalpy;
   } else if constexpr (EOSMethod == EOSPolytrope) {
      const PolytropeSegment segment = polytrope_segment(rho, eos);
      return generated::relativistic_polytrope_sound_speed_squared(
         rho,
         specific_internal_energy,
         segment.k,
         segment.gamma
      );
   } else if constexpr (EOSMethod == EOSHybridPolytrope) {
      const PolytropeSegment segment = polytrope_segment(rho, eos);
      return generated::relativistic_hybrid_polytrope_sound_speed_squared(
         rho,
         specific_internal_energy,
         segment.k,
         segment.gamma,
         eos.thermal_gamma,
         segment.energy_offset
      );
   } else {
      return generated::relativistic_ideal_gas_sound_speed_squared(
         specific_internal_energy, eos.gamma
      );
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RelativisticCharacteristicState relativistic_characteristic_state(
   amrex::Real rho,
   amrex::Real specific_internal_energy,
   EOSParams const& eos
) noexcept
{
   if constexpr (EOSMethod == EOSRuntime) {
      if (eos.method == EOSCompose) {
         return relativistic_characteristic_state<EOSCompose>(
            rho, specific_internal_energy, eos
         );
      }
      if (eos.method == EOSPolytrope) {
         return relativistic_characteristic_state<EOSPolytrope>(
            rho, specific_internal_energy, eos
         );
      }
      if (eos.method == EOSHybridPolytrope) {
         return relativistic_characteristic_state<EOSHybridPolytrope>(
            rho, specific_internal_energy, eos
         );
      }
      return relativistic_characteristic_state<EOSIdealGas>(
         rho, specific_internal_energy, eos
      );
   } else if constexpr (EOSMethod == EOSCompose) {
      const TabulatedColdState cold = tabulated_cold_from_density(rho, eos.table);
      const amrex::Real pressure = relativistic_compose_pressure(
         rho, specific_internal_energy, cold, eos
      );
      const amrex::Real thermal_energy = specific_internal_energy
         - (cold.energy_density / rho - 1.0);
      const amrex::Real chi = cold.specific_enthalpy
         * cold.sound_speed_squared
         + (eos.thermal_gamma - 1.0)
            * (thermal_energy - cold.pressure / rho);
      const amrex::Real kappa = (eos.thermal_gamma - 1.0) * rho;
      const amrex::Real specific_enthalpy =
         1.0 + specific_internal_energy + pressure / rho;
      return {
         specific_enthalpy,
         (chi + pressure * kappa / (rho * rho)) / specific_enthalpy,
      };
   } else if constexpr (EOSMethod == EOSPolytrope) {
      const PolytropeSegment segment = polytrope_segment(rho, eos);
      return {
         generated::relativistic_polytrope_specific_enthalpy(
            rho, specific_internal_energy, segment.k, segment.gamma
         ),
         generated::relativistic_polytrope_sound_speed_squared(
            rho, specific_internal_energy, segment.k, segment.gamma
         ),
      };
   } else if constexpr (EOSMethod == EOSHybridPolytrope) {
      const PolytropeSegment segment = polytrope_segment(rho, eos);
      return {
         generated::relativistic_hybrid_polytrope_specific_enthalpy(
            rho,
            specific_internal_energy,
            segment.k,
            segment.gamma,
            eos.thermal_gamma,
            segment.energy_offset
         ),
         generated::relativistic_hybrid_polytrope_sound_speed_squared(
            rho,
            specific_internal_energy,
            segment.k,
            segment.gamma,
            eos.thermal_gamma,
            segment.energy_offset
         ),
      };
   } else {
      static_cast<void>(rho);
      return {
         generated::relativistic_ideal_gas_specific_enthalpy(
            specific_internal_energy, eos.gamma
         ),
         generated::relativistic_ideal_gas_sound_speed_squared(
            specific_internal_energy, eos.gamma
         ),
      };
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real total_hydro_energy(
   amrex::Real rho,
   StateVector3 const& velocity,
   amrex::Real pressure,
   EOSParams const& eos
) noexcept
{
   return eos_internal_energy_density_from_pressure<EOSMethod>(rho, pressure, eos)
      + kinetic_energy_density(rho, velocity);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real total_hydro_energy(
   amrex::Real rho,
   amrex::Real vx,
   amrex::Real vy,
   amrex::Real vz,
   amrex::Real pressure,
   EOSParams const& eos
) noexcept
{
   return total_hydro_energy<EOSMethod>(rho, StateVector3{vx, vy, vz}, pressure, eos);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real magnetic_energy_density(amrex::Real bx, amrex::Real by, amrex::Real bz) noexcept
{
   return generated::magnetic_energy_density_from_components(bx, by, bz);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real magnetic_energy_density(StateVector3 const& magnetic) noexcept
{
   return 0.5 * squared_norm(magnetic);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real magnetic_pressure_from_state(amrex::Array4<amrex::Real const> const& state, int i, int j, int k) noexcept
{
   return magnetic_energy_density(load_magnetic_vector3(state, i, j, k));
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real total_mhd_energy(
   amrex::Real rho,
   amrex::Real vx,
   amrex::Real vy,
   amrex::Real vz,
   amrex::Real pressure,
   amrex::Real bx,
   amrex::Real by,
   amrex::Real bz,
   EOSParams const& eos
) noexcept
{
   return total_mhd_energy<EOSMethod>(
      rho,
      StateVector3{vx, vy, vz},
      pressure,
      StateVector3{bx, by, bz},
      eos
   );
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real total_mhd_energy(
   amrex::Real rho,
   StateVector3 const& velocity,
   amrex::Real pressure,
   StateVector3 const& magnetic,
   EOSParams const& eos
) noexcept
{
   return total_hydro_energy<EOSMethod>(rho, velocity, pressure, eos) + magnetic_energy_density(magnetic);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pressure_from_state(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   EOSParams const& eos
) noexcept
{
   const amrex::Real rho = state(i, j, k, RHO);
   const StateVector3 velocity = velocity_from_momentum(
      load_momentum_vector3(state, i, j, k), rho
   );
   const amrex::Real internal = state(i, j, k, ENERGY) - kinetic_energy_density(rho, velocity);
   return eos_pressure_from_density_energy<EOSMethod>(rho, internal, eos);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pressure_from_conserved(HydroState const& state, EOSParams const& eos) noexcept
{
   const StateVector3 velocity = load_velocity_vector3(state);
   const amrex::Real internal = state.energy - kinetic_energy_density(state.rho, velocity);
   return eos_pressure_from_density_energy<EOSMethod>(state.rho, internal, eos);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pressure_from_conserved(MHDState const& state, EOSParams const& eos) noexcept
{
   const StateVector3 momentum = load_momentum_vector3(state);
   const StateVector3 magnetic = load_magnetic_vector3(state);
   const StateVector3 velocity = velocity_from_momentum(momentum, state.rho);
   const amrex::Real internal = state.energy
      - kinetic_energy_density(state.rho, velocity)
      - magnetic_energy_density(magnetic);
   return eos_pressure_from_density_energy<EOSMethod>(state.rho, internal, eos);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pressure_from_array(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   bool mhd
) noexcept
{
   if (mhd) {
      return pressure_from_conserved<EOSMethod>(load_mhd_state(state, i, j, k), eos);
   }
   return pressure_from_state<EOSMethod>(state, i, j, k, eos);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitive primitive_from_conserved(HydroState const& state, EOSParams const& eos) noexcept
{
   const StateVector3 velocity = load_velocity_vector3(state);
   return {
      state.rho,
      velocity.x,
      velocity.y,
      velocity.z,
      pressure_from_conserved<EOSMethod>(state, eos),
   };
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDPrimitive primitive_from_conserved(MHDState const& state, EOSParams const& eos) noexcept
{
   const StateVector3 velocity = load_velocity_vector3(state);
   return {
      state.rho,
      velocity.x,
      velocity.y,
      velocity.z,
      pressure_from_conserved<EOSMethod>(state, eos),
      state.b1,
      state.b2,
      state.b3,
   };
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState conserved_from_primitive(HydroPrimitive const& primitive, EOSParams const& eos) noexcept
{
   return {
      primitive.rho,
      primitive.rho * primitive.v1,
      primitive.rho * primitive.v2,
      primitive.rho * primitive.v3,
      total_hydro_energy<EOSMethod>(primitive.rho, primitive.v1, primitive.v2, primitive.v3, primitive.pressure, eos),
   };
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDState conserved_from_primitive(MHDPrimitive const& primitive, EOSParams const& eos) noexcept
{
   return {
      primitive.rho,
      primitive.rho * primitive.v1,
      primitive.rho * primitive.v2,
      primitive.rho * primitive.v3,
      total_mhd_energy<EOSMethod>(
         primitive.rho,
         primitive.v1,
         primitive.v2,
         primitive.v3,
         primitive.pressure,
         primitive.b1,
         primitive.b2,
         primitive.b3,
         eos
      ),
      primitive.b1,
      primitive.b2,
      primitive.b3,
   };
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sound_speed_from_primitive(amrex::Real rho, amrex::Real pressure, EOSParams const& eos) noexcept
{
   return eos_sound_speed_from_primitive<EOSMethod>(rho, pressure, eos);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mhd_fast_speed_from_components(
   amrex::Real rho,
   amrex::Real pressure,
   amrex::Real bn,
   amrex::Real bt1,
   amrex::Real bt2,
   EOSParams const& eos
) noexcept
{
   const amrex::Real a = sound_speed_from_primitive<EOSMethod>(rho, pressure, eos);
   const amrex::Real a2 = a * a;
   const amrex::Real inv_rho = 1.0 / rho;
   const amrex::Real b2 = (bn * bn + bt1 * bt1 + bt2 * bt2) * inv_rho;
   const amrex::Real bn2 = bn * bn * inv_rho;
   const amrex::Real discriminant =
      amrex::max((a2 + b2) * (a2 + b2) - amrex::Real(4.0) * a2 * bn2, amrex::Real(0.0));
   const amrex::Real cf2 = amrex::Real(0.5) * (a2 + b2 + std::sqrt(discriminant));
   return std::sqrt(amrex::max(cf2, amrex::Real(0.0)));
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mhd_fast_speed_from_primitive(MHDPrimitive const& primitive, int direction, EOSParams const& eos) noexcept
{
   amrex::Real bn = primitive.b1;
   amrex::Real bt1 = primitive.b2;
   amrex::Real bt2 = primitive.b3;
   if (direction == 1) {
      bn = primitive.b2;
      bt1 = primitive.b1;
      bt2 = primitive.b3;
   } else if (direction == 2) {
      bn = primitive.b3;
      bt1 = primitive.b1;
      bt2 = primitive.b2;
   }
   return mhd_fast_speed_from_components<EOSMethod>(primitive.rho, primitive.pressure, bn, bt1, bt2, eos);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void reset_barotropic_hydro_energy(HydroState& state, EOSParams const& eos) noexcept
{
   if (eos_is_barotropic<EOSMethod>(eos)) {
      const StateVector3 velocity = load_velocity_vector3(state);
      const amrex::Real pressure = eos_cold_pressure<EOSMethod>(state.rho, eos);
      state.energy = total_hydro_energy<EOSMethod>(
         state.rho,
         velocity.x,
         velocity.y,
         velocity.z,
         pressure,
         eos
      );
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void reset_barotropic_mhd_energy(MHDState& state, EOSParams const& eos) noexcept
{
   if (eos_is_barotropic<EOSMethod>(eos)) {
      const StateVector3 velocity = load_velocity_vector3(state);
      const amrex::Real pressure = eos_cold_pressure<EOSMethod>(state.rho, eos);
      state.energy = total_mhd_energy<EOSMethod>(
         state.rho,
         velocity,
         pressure,
         StateVector3{state.b1, state.b2, state.b3},
         eos
      );
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real total_hydro_energy_runtime(
   int eos_method,
   amrex::Real rho,
   amrex::Real vx,
   amrex::Real vy,
   amrex::Real vz,
   amrex::Real pressure,
   EOSParams const& eos
) noexcept
{
   static_cast<void>(eos_method);
   return total_hydro_energy<EOSRuntime>(
      rho, vx, vy, vz, pressure, eos
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real total_mhd_energy_runtime(
   int eos_method,
   amrex::Real rho,
   amrex::Real vx,
   amrex::Real vy,
   amrex::Real vz,
   amrex::Real pressure,
   amrex::Real bx,
   amrex::Real by,
   amrex::Real bz,
   EOSParams const& eos
) noexcept
{
   static_cast<void>(eos_method);
   return total_mhd_energy<EOSRuntime>(
      rho, vx, vy, vz, pressure, bx, by, bz, eos
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pressure_from_array_runtime(
   int eos_method,
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   bool mhd
) noexcept
{
   static_cast<void>(eos_method);
   return pressure_from_array<EOSRuntime>(state, i, j, k, eos, mhd);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sound_speed_from_primitive_runtime(
   int eos_method,
   amrex::Real rho,
   amrex::Real pressure,
   EOSParams const& eos
) noexcept
{
   static_cast<void>(eos_method);
   return sound_speed_from_primitive<EOSRuntime>(rho, pressure, eos);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real fluid_sound_speed_from_primitive_runtime(
   PhysicalSystem system,
   int eos_method,
   amrex::Real rho,
   amrex::Real pressure,
   EOSParams const& eos
) noexcept
{
   if (physical_system_is_relativistic(system)) {
      const amrex::Real eps = relativistic_specific_internal_energy_from_pressure<
         EOSRuntime
      >(rho, pressure, eos);
      return std::sqrt(relativistic_sound_speed_squared<EOSRuntime>(rho, eps, eos));
   }
   return sound_speed_from_primitive_runtime(eos_method, rho, pressure, eos);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real base_pressure_runtime(
   int eos_method,
   amrex::Real rho,
   amrex::Real pressure,
   EOSParams const& eos
) noexcept
{
   if (eos_method == EOSPolytrope) {
      return eos_cold_pressure<EOSRuntime>(rho, eos);
   }
   return pressure;
}

} // namespace pyfluid_amrex
