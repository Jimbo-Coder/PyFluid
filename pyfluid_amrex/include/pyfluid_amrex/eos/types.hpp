#pragma once

#include <AMReX_REAL.H>

#include <array>

namespace pyfluid_amrex {

inline constexpr int MaxPolytropeSegments = 8;

struct EOSParams {
   amrex::Real gamma = 5.0 / 3.0;
   amrex::Real thermal_gamma = 5.0 / 3.0;
   amrex::Real rho_floor = 1.0e-12;
   amrex::Real pressure_floor = 1.0e-12;
   int polytrope_segments = 1;
   std::array<amrex::Real, MaxPolytropeSegments> segment_k{{1.0}};
   std::array<amrex::Real, MaxPolytropeSegments> segment_gamma{{5.0 / 3.0}};
   std::array<amrex::Real, MaxPolytropeSegments> segment_energy_offset{};
   std::array<amrex::Real, MaxPolytropeSegments - 1> segment_density_break{};
};

} // namespace pyfluid_amrex
