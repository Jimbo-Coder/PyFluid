#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/black_hole_matter.hpp"
#include "pyfluid_amrex/fixed_spacetime_types.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/excision_geometry.hpp"
#include "pyfluid_amrex/relativistic_exterior.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct FaceMagneticField;

struct PrimitiveCacheLaunch {
   amrex::MultiFab& primitive;
   amrex::MultiFab const& conserved;
   EOSParams eos;
   PhysicalSystem physical_system = PhysicalSystem::HD;
   int num_components = 0;
   amrex::MultiFab const* cell_spacetime = nullptr;
   amrex::iMultiFab const* needed_mask = nullptr;
   amrex::Vector<int> const* needed_kind = nullptr;
   amrex::Geometry const* geometry = nullptr;
   int mesh_coordinate = MeshCartesian;
   MeshAxes mesh_axes{};
   FixedSpacetimeParameters fixed_spacetime{};
   amrex::Real time = 0.0_rt;
   FaceMagneticField const* canonical_face_b = nullptr;
   amrex::iMultiFab const* canonical_face_b_cell_mask = nullptr;
   ExcisionGridView excision{};
   RelativisticExteriorState exterior{};
};

void fill_primitive_cache(PrimitiveCacheLaunch const& launch);

void apply_relativistic_exterior(
   amrex::MultiFab& conserved,
   amrex::MultiFab const& cell_spacetime,
   PhysicalSystem physical_system,
   EOSParams const& eos,
   RelativisticExteriorState const& exterior,
   int ghost_cells,
   amrex::Long* c2p_fallback_count = nullptr,
   amrex::Long* atmosphere_reset_count = nullptr,
   amrex::iMultiFab const* black_hole_mask = nullptr,
   BlackHoleMatterParameters black_hole_matter = {},
   amrex::Long* interior_bound_count = nullptr,
   amrex::Long* interior_reset_count = nullptr
);

} // namespace pyfluid_amrex
