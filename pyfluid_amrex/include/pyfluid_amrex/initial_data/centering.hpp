#pragma once

#include <AMReX_BCRec.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include "pyfluid_amrex/initial_data/elliptic.hpp"

namespace pyfluid_amrex {

namespace detail {

void fill_initial_data_vertex_ghosts(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& fields,
   amrex::Vector<amrex::BCRec> const& boundary_conditions
);

} // namespace detail

void materialize_initial_data_vertices(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& vertex_fields,
   amrex::Vector<amrex::MultiFab const*> const& source_fields,
   amrex::Vector<amrex::BCRec> const& boundary_conditions,
   int transfer_order
);

} // namespace pyfluid_amrex
