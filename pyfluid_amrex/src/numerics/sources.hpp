#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <array>

namespace pyfluid_amrex {

using namespace amrex::literals;

struct RunConfig;

struct SourceStageContext {
   RunConfig const* cfg = nullptr;
   amrex::Geometry const* geom = nullptr;
   amrex::Real time = 0.0_rt;

   // Geometry masks are optional source-domain restrictions. They do not
   // change the GF contract; they only tell sources which valid fluid cells
   // receive source terms.
   amrex::iMultiFab const* solid_mask = nullptr;
   amrex::Vector<int> const* solid_valid_kind = nullptr;

   // Gravity is an ordinary source from the RHS point of view. The solver
   // prepares phi/grad_phi at the current RK/subcycling stage time, then passes
   // the cached gradient here. Conservative Athena-style energy coupling also
   // receives the stage mass fluxes from the hyperbolic update.
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const* gravity_gradient = nullptr;
   std::array<amrex::Real, AMREX_SPACEDIM> const* gravity_acceleration_offset = nullptr;
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const* mass_fluxes = nullptr;
   bool suppress_athena_gravity_energy_source = false;
};

bool sources_enabled(RunConfig const& cfg);

void apply_sources(
   amrex::MultiFab& rhs,
   amrex::MultiFab const& state,
   SourceStageContext const& context
);

} // namespace pyfluid_amrex
