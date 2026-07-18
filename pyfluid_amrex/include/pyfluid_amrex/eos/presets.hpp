#pragma once

#include <AMReX_REAL.H>

#include <array>
#include <string_view>
#include <vector>

#include "pyfluid_amrex/eos/types.hpp"

namespace pyfluid_amrex::eos {

enum class PiecewisePolytropePreset {
   custom,
   sly,
   apr4,
   h4,
   ms1,
};

inline constexpr std::array<std::string_view, 5> PiecewisePolytropePresetNames{{
   "custom",
   "sly",
   "apr4",
   "h4",
   "ms1",
}};

struct ReadPiecewisePolytropeFit {
   std::string_view name;
   amrex::Real log10_pressure1_cgs;
   std::array<amrex::Real, 3> core_gamma;
};

PiecewisePolytropePreset piecewise_polytrope_preset(std::string_view name);
ReadPiecewisePolytropeFit read_piecewise_polytrope_fit(
   PiecewisePolytropePreset preset
);

EOSParams make_piecewise_polytrope(
   amrex::Real initial_k,
   amrex::Real initial_gamma,
   std::vector<amrex::Real> const& density_breaks,
   std::vector<amrex::Real> const& following_gammas,
   amrex::Real thermal_gamma,
   amrex::Real rho_floor,
   amrex::Real pressure_floor
);

EOSParams make_read_piecewise_polytrope(
   PiecewisePolytropePreset preset,
   amrex::Real thermal_gamma,
   amrex::Real rho_floor,
   amrex::Real pressure_floor
);

} // namespace pyfluid_amrex::eos
