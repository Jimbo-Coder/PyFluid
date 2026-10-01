#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/fixed_spacetime_types.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/excision_geometry.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct RelativisticSourceLaunch {
   amrex::MultiFab& rhs;
   amrex::MultiFab const& primitive;
   amrex::Geometry const& geometry;
   EOSParams eos;
   PhysicalSystem physical_system = PhysicalSystem::RHD;
   amrex::MultiFab const* cell_spacetime = nullptr;
   int mesh_coordinate = MeshCartesian;
   MeshAxes mesh_axes{};
   FixedSpacetimeParameters fixed_spacetime{};
   amrex::Real time = 0.0_rt;
   SphericalExcisionRegion excision_region{};
};

void add_relativistic_sources(RelativisticSourceLaunch const& launch);

} // namespace pyfluid_amrex
