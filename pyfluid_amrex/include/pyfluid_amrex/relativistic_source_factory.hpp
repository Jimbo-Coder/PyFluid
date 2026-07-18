#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/physical_system.hpp"

namespace pyfluid_amrex {

struct RelativisticSourceLaunch {
   amrex::MultiFab& rhs;
   amrex::MultiFab const& primitive;
   amrex::Geometry const& geometry;
   EOSParams eos;
   PhysicalSystem physical_system = PhysicalSystem::RHD;
   int eos_method = EOSIdealGas;
   amrex::MultiFab const* cell_spacetime = nullptr;
   FixedSpacetimeParameters fixed_spacetime{};
   amrex::Real time = 0.0;
};

void add_relativistic_sources(RelativisticSourceLaunch const& launch);

} // namespace pyfluid_amrex
