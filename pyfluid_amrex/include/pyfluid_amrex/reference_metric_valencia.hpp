#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <array>

#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/fixed_spacetime_types.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"

namespace pyfluid_amrex {

void compute_rhd_reference_metric_rhs(
   amrex::MultiFab& rhs,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   amrex::Geometry const& geometry,
   int coordinate,
   MeshAxes const& axes,
   EOSParams const& eos,
   FixedSpacetimeParameters const& fixed_spacetime,
   amrex::Real time,
   amrex::MultiFab const* cell_spacetime = nullptr
);

} // namespace pyfluid_amrex
