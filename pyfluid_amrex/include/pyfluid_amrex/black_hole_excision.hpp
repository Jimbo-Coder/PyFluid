#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <array>

#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/excision_geometry.hpp"
#include "pyfluid_amrex/fixed_spacetime_types.hpp"
#include "pyfluid_amrex/physical_system.hpp"

namespace pyfluid_amrex {

void enforce_black_hole_excision_fluxes(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   amrex::iMultiFab const& mask,
   PhysicalSystem physical_system,
   EOSParams const& eos,
   amrex::Geometry const& geometry,
   FixedSpacetimeParameters const& fixed_spacetime,
   amrex::MultiFab const* cell_spacetime,
   amrex::Real time,
   SphericalExcisionRegion const& excision_region = {}
);

void zero_black_hole_excision_rhs(
   amrex::MultiFab& rhs,
   amrex::iMultiFab const& mask
);

} // namespace pyfluid_amrex
