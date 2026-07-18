#pragma once

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/physical_system.hpp"

namespace pyfluid_amrex {

struct PrimitiveCacheLaunch {
   amrex::MultiFab& primitive;
   amrex::MultiFab const& conserved;
   EOSParams eos;
   PhysicalSystem physical_system = PhysicalSystem::HD;
   int eos_method = EOSIdealGas;
   int num_components = 0;
   amrex::MultiFab const* cell_spacetime = nullptr;
   amrex::iMultiFab const* needed_mask = nullptr;
   amrex::Vector<int> const* needed_kind = nullptr;
   amrex::Geometry const* geometry = nullptr;
   FixedSpacetimeParameters fixed_spacetime{};
   amrex::Real time = 0.0;
};

void fill_primitive_cache(PrimitiveCacheLaunch const& launch);

void apply_relativistic_atmosphere(
   amrex::MultiFab& conserved,
   amrex::MultiFab const& cell_spacetime,
   PhysicalSystem physical_system,
   int eos_method,
   EOSParams const& eos,
   int ghost_cells
);

} // namespace pyfluid_amrex
