#pragma once

#include <AMReX_REAL.H>

#include <vector>

#include "pyfluid_amrex/multipoles.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct RadiationDiagnostics {
   amrex::Real radius = 0.0_rt;
   amrex::Real matter_energy_flux = 0.0_rt;
   amrex::Real psi4_l2 = 0.0_rt;
   std::vector<MultipoleMode> psi4_modes;
};

} // namespace pyfluid_amrex
