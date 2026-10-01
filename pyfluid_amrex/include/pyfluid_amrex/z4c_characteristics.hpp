#pragma once

#include <AMReX_Algorithm.H>
#include "pyfluid_amrex/z4c_state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

// Principal speeds relative to Eulerian observers. These follow the actual
// generated lapse/shift equations, not gauge-name-based coordinate constants.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real z4c_lapse_speed_squared(Z4cState const& state, Z4cLapseGauge gauge) noexcept
{
   if (gauge == Z4cLapseGauge::Frozen) return 0.0_rt;
   if (gauge == Z4cLapseGauge::Harmonic) return 1.0_rt;
   if (gauge == Z4cLapseGauge::Trumpet) return (1.0_rt - state.lapse) / state.lapse;
   return 2.0_rt / state.lapse; // one-plus-log and its lower-order slow-start source
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real z4c_shift_speed_squared(Z4cState const& state,
   Z4cEvolutionParameters const& parameters, Z4cShiftGauge gauge) noexcept
{
   return gauge == Z4cShiftGauge::Frozen ? 0.0_rt : parameters.shift_mu / state.chi;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real z4c_maximum_speed_squared(Z4cState const& state,
   Z4cEvolutionParameters const& parameters, Z4cLapseGauge lapse_gauge,
   Z4cShiftGauge shift_gauge) noexcept
{
   return amrex::max(1.0_rt, amrex::max(z4c_lapse_speed_squared(state, lapse_gauge),
      4.0_rt / 3.0_rt * z4c_shift_speed_squared(state, parameters, shift_gauge)));
}

} // namespace pyfluid_amrex
