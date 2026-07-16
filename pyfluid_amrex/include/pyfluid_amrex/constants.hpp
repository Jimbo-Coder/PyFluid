#pragma once

#include <AMReX_REAL.H>

namespace pyfluid_amrex::constants {

inline constexpr amrex::Real pi =
   static_cast<amrex::Real>(3.141592653589793238462643383279502884L);
inline constexpr amrex::Real half_pi = 0.5 * pi;
inline constexpr amrex::Real two_pi = 2.0 * pi;
inline constexpr amrex::Real four_pi = 4.0 * pi;
inline constexpr amrex::Real radians_per_degree = pi / 180.0;

} // namespace pyfluid_amrex::constants
