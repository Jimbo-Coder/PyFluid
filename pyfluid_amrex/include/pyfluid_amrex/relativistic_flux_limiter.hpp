#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <array>

#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"

namespace pyfluid_amrex {

void apply_rhd_admissibility_flux_limiter(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab& rhs,
   amrex::MultiFab const& base_conserved,
   amrex::MultiFab const& conserved,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   int eos_method,
   amrex::Geometry const& geometry,
   FixedSpacetimeParameters const& fixed_spacetime,
   amrex::MultiFab const* cell_spacetime,
   amrex::MultiFab const* candidate_cell_spacetime,
   EOSThermodynamicState const& atmosphere,
   amrex::Real time,
   amrex::Real base_weight,
   amrex::Real conserved_weight,
   amrex::Real increment_scale
);

} // namespace pyfluid_amrex
