#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <array>

namespace pyfluid_amrex {

struct PatchGridSystem;
struct PatchLayout;
struct RunConfig;

struct SignalSpeedLaunch {
   amrex::MultiFab& frequency;
   amrex::MultiFab const& state;
   RunConfig const& config;
   int num_components = 0;
   int eos_method = 0;
   int diffusion_method = 0;
};

amrex::Real compute_adaptive_dt(
   SignalSpeedLaunch const& launch,
   amrex::Geometry const& geometry,
   amrex::Real time,
   amrex::MultiFab const* cell_spacetime
);

amrex::Real compute_spacetime_dt(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& cell_spacetime,
   amrex::Geometry const& geometry,
   amrex::Real cfl,
   amrex::Real gauge_speed
);

amrex::Real compute_mapped_hydro_adaptive_dt(
   SignalSpeedLaunch const& launch,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout
);

amrex::Real compute_cutcell_adaptive_dt(
   SignalSpeedLaunch const& launch,
   amrex::iMultiFab const& solid_mask,
   amrex::MultiFab const& volume_fraction,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& apertures,
   amrex::Geometry const& geometry,
   amrex::Vector<int> const* solid_valid_kind
);

} // namespace pyfluid_amrex
