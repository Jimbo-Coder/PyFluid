#pragma once

#include <AMReX_REAL.H>

#include <array>

namespace pyfluid_amrex {

inline constexpr int MaxPolytropeSegments = 8;

struct TabulatedEOSView {
   int size = 0;
   amrex::Real const* log_density = nullptr;
   amrex::Real const* log_pressure = nullptr;
   amrex::Real const* log_specific_enthalpy = nullptr;
   amrex::Real low_density_gamma = 0.0;
   amrex::Real high_density_sound_speed_squared = 0.0;
};

struct EOSParams {
   int method = 0;
   amrex::Real gamma = 5.0 / 3.0;
   amrex::Real thermal_gamma = 5.0 / 3.0;
   amrex::Real rho_floor = 1.0e-12;
   amrex::Real pressure_floor = 1.0e-12;
   int polytrope_segments = 1;
   std::array<amrex::Real, MaxPolytropeSegments> segment_k{{1.0}};
   std::array<amrex::Real, MaxPolytropeSegments> segment_gamma{{5.0 / 3.0}};
   std::array<amrex::Real, MaxPolytropeSegments> segment_energy_offset{};
   std::array<amrex::Real, MaxPolytropeSegments - 1> segment_density_break{};
   TabulatedEOSView table{};
};

struct EOSThermodynamicState {
   amrex::Real rho;
   amrex::Real specific_internal_energy;
   amrex::Real pressure;
};

struct RelativisticCharacteristicState {
   amrex::Real specific_enthalpy = 1.0;
   amrex::Real sound_speed_squared = 0.0;
};

} // namespace pyfluid_amrex
