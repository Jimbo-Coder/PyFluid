#pragma once

#include <AMReX_Array4.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "pyfluid_amrex/kernels_generated/eos.hpp"
#include "pyfluid_amrex/kernels_generated/vector_ops.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

struct EOSParams {
   amrex::Real gamma = 5.0 / 3.0;
   amrex::Real polytropic_k = 1.0;
   amrex::Real polytropic_gamma = 5.0 / 3.0;
   amrex::Real thermal_gamma = 5.0 / 3.0;
   amrex::Real rho_floor = 1.0e-12;
   amrex::Real pressure_floor = 1.0e-12;
};

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
constexpr bool eos_is_barotropic() noexcept
{
   return EOSMethod == EOSPolytrope;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eos_cold_pressure(amrex::Real rho, EOSParams const& eos) noexcept
{
   if constexpr (EOSMethod == EOSIdealGas) {
      static_cast<void>(rho);
      static_cast<void>(eos);
      return 0.0;
   } else {
      return generated::polytrope_cold_pressure(
         rho,
         eos.polytropic_k,
         eos.polytropic_gamma
      );
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eos_cold_energy_density(amrex::Real rho, EOSParams const& eos) noexcept
{
   if constexpr (EOSMethod == EOSIdealGas) {
      static_cast<void>(rho);
      static_cast<void>(eos);
      return 0.0;
   } else {
      return generated::polytrope_cold_energy_density(
         rho,
         eos.polytropic_k,
         eos.polytropic_gamma
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
   if constexpr (EOSMethod == EOSPolytrope) {
      static_cast<void>(pressure);
      return eos_cold_energy_density<EOSMethod>(rho, eos);
   } else if constexpr (EOSMethod == EOSHybridPolytrope) {
      return generated::hybrid_polytrope_internal_energy_density_from_pressure(
         rho,
         pressure,
         eos.polytropic_k,
         eos.polytropic_gamma,
         eos.thermal_gamma
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
   if constexpr (EOSMethod == EOSPolytrope) {
      static_cast<void>(internal_energy_density);
      return eos_cold_pressure<EOSMethod>(rho, eos);
   } else if constexpr (EOSMethod == EOSHybridPolytrope) {
      return generated::hybrid_polytrope_pressure_from_density_energy(
         rho,
         internal_energy_density,
         eos.polytropic_k,
         eos.polytropic_gamma,
         eos.thermal_gamma
      );
   } else {
      return generated::ideal_gas_pressure_from_density_energy(
         internal_energy_density,
         eos.gamma
      );
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eos_pressure_from_density_energy_runtime(
   int eos_method,
   amrex::Real rho,
   amrex::Real internal_energy_density,
   EOSParams const& eos
) noexcept
{
   if (eos_method == EOSPolytrope) {
      return eos_pressure_from_density_energy<EOSPolytrope>(rho, internal_energy_density, eos);
   }
   if (eos_method == EOSHybridPolytrope) {
      return eos_pressure_from_density_energy<EOSHybridPolytrope>(rho, internal_energy_density, eos);
   }
   return eos_pressure_from_density_energy<EOSIdealGas>(rho, internal_energy_density, eos);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eos_sound_speed_from_primitive(amrex::Real rho, amrex::Real pressure, EOSParams const& eos) noexcept
{
   if constexpr (EOSMethod == EOSPolytrope) {
      return std::sqrt(generated::polytrope_sound_speed_squared(
         rho,
         eos.polytropic_k,
         eos.polytropic_gamma
      ));
   } else if constexpr (EOSMethod == EOSHybridPolytrope) {
      return std::sqrt(generated::hybrid_polytrope_sound_speed_squared(
         rho,
         pressure,
         eos.polytropic_k,
         eos.polytropic_gamma,
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
   const amrex::Real discriminant = amrex::max((a2 + b2) * (a2 + b2) - 4.0 * a2 * bn2, 0.0);
   const amrex::Real cf2 = 0.5 * (a2 + b2 + std::sqrt(discriminant));
   return std::sqrt(amrex::max(cf2, 0.0));
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
   if constexpr (eos_is_barotropic<EOSMethod>()) {
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
   if constexpr (eos_is_barotropic<EOSMethod>()) {
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
   if (eos_method == EOSPolytrope) {
      return total_hydro_energy<EOSPolytrope>(rho, vx, vy, vz, pressure, eos);
   }
   if (eos_method == EOSHybridPolytrope) {
      return total_hydro_energy<EOSHybridPolytrope>(rho, vx, vy, vz, pressure, eos);
   }
   return total_hydro_energy<EOSIdealGas>(rho, vx, vy, vz, pressure, eos);
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
   if (eos_method == EOSPolytrope) {
      return total_mhd_energy<EOSPolytrope>(rho, vx, vy, vz, pressure, bx, by, bz, eos);
   }
   if (eos_method == EOSHybridPolytrope) {
      return total_mhd_energy<EOSHybridPolytrope>(rho, vx, vy, vz, pressure, bx, by, bz, eos);
   }
   return total_mhd_energy<EOSIdealGas>(rho, vx, vy, vz, pressure, bx, by, bz, eos);
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
   if (eos_method == EOSPolytrope) {
      return pressure_from_array<EOSPolytrope>(state, i, j, k, eos, mhd);
   }
   if (eos_method == EOSHybridPolytrope) {
      return pressure_from_array<EOSHybridPolytrope>(state, i, j, k, eos, mhd);
   }
   return pressure_from_array<EOSIdealGas>(state, i, j, k, eos, mhd);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sound_speed_from_primitive_runtime(
   int eos_method,
   amrex::Real rho,
   amrex::Real pressure,
   EOSParams const& eos
) noexcept
{
   if (eos_method == EOSPolytrope) {
      return sound_speed_from_primitive<EOSPolytrope>(rho, pressure, eos);
   }
   if (eos_method == EOSHybridPolytrope) {
      return sound_speed_from_primitive<EOSHybridPolytrope>(rho, pressure, eos);
   }
   return sound_speed_from_primitive<EOSIdealGas>(rho, pressure, eos);
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
      return eos_cold_pressure<EOSPolytrope>(rho, eos);
   }
   return pressure;
}

} // namespace pyfluid_amrex
