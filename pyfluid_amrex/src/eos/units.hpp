#pragma once

#include <AMReX_REAL.H>

namespace pyfluid_amrex::eos::units {

using namespace amrex::literals;

// Geometric code units have G = c = M_sun = 1. Keep intermediate values
// representable in FP32; c^4 overflows and its reciprocal is subnormal.
inline constexpr amrex::Real gravitational_constant_cgs = 6.67430e-8_rt;
inline constexpr amrex::Real speed_of_light_cgs = 2.99792458e10_rt;
inline constexpr amrex::Real solar_mass_parameter_cgs = 1.3271244e26_rt;
inline constexpr amrex::Real speed_of_light_squared_cgs =
   speed_of_light_cgs * speed_of_light_cgs;
inline constexpr amrex::Real solar_mass_length_cgs =
   solar_mass_parameter_cgs / speed_of_light_squared_cgs;
inline constexpr amrex::Real density_cgs_to_code =
   gravitational_constant_cgs * solar_mass_length_cgs * solar_mass_length_cgs
   / speed_of_light_squared_cgs;

constexpr amrex::Real density_from_cgs(amrex::Real density) noexcept
{
   return density * density_cgs_to_code;
}

constexpr amrex::Real pressure_from_cgs(amrex::Real pressure) noexcept
{
   return (pressure / speed_of_light_squared_cgs) * density_cgs_to_code;
}

inline constexpr amrex::Real mev_per_fm3_to_code =
   pressure_from_cgs(1.602176634e33_rt);

} // namespace pyfluid_amrex::eos::units
