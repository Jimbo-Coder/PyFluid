#pragma once

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include "pyfluid_amrex/config.hpp"

namespace pyfluid_amrex {

void validate_fccz4_domain(amrex::Geometry const& geometry, RunConfig const& config);

void compute_fccz4_rhs(amrex::MultiFab const& state, amrex::MultiFab const& matter,
                       amrex::MultiFab& rhs, amrex::Geometry const& geometry,
                       RunConfig const& config, amrex::Real time);
void complete_fccz4_connections(amrex::MultiFab& state, amrex::Geometry const& geometry,
                                RunConfig const& config);
void rotate_fccz4_initial_state(amrex::MultiFab& state, amrex::Geometry const& geometry,
                                RunConfig const& config);

} // namespace pyfluid_amrex
