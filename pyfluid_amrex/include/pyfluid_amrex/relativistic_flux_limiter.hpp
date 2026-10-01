#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <array>

#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/fixed_spacetime_types.hpp"
#include "pyfluid_amrex/excision_geometry.hpp"
#include "pyfluid_amrex/relativistic_diode.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"

namespace pyfluid_amrex {

// Return the global count of cells whose high-order candidate was rejected.
// Each cell is counted once per invocation, including mask-closure additions.
amrex::Long apply_rhd_admissibility_flux_limiter(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab& rhs,
   amrex::MultiFab const& base_conserved,
   amrex::MultiFab const& conserved,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   int eos_method,
   amrex::Geometry const& geometry,
   FixedSpacetimeParameters const& fixed_spacetime,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::MultiFab const* cell_spacetime,
   amrex::MultiFab const* candidate_cell_spacetime,
   EOSThermodynamicState const& atmosphere,
   amrex::Real time,
   amrex::Real conserved_weight,
   amrex::Real increment_scale,
   amrex::iMultiFab const* black_hole_excision_mask = nullptr,
   SphericalExcisionRegion const& excision_region = {},
   RelativisticDiodeBoundary const& diode = {}
);

void apply_rmhd_candidate_flux_limiter(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab& rhs,
   amrex::MultiFab const& base_conserved,
   amrex::MultiFab const& conserved,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   int riemann_method,
   amrex::Geometry const& geometry,
   FixedSpacetimeParameters const& fixed_spacetime,
   amrex::MultiFab const* cell_spacetime,
   amrex::MultiFab const* candidate_cell_spacetime,
   EOSThermodynamicState const& atmosphere,
   amrex::Real time,
   amrex::Real conserved_weight,
   amrex::Real increment_scale,
   bool try_pwl_first,
   bool enforce_dmp,
   amrex::Real dmp_factor,
   amrex::Long* pwl_correction_count,
   amrex::Long* pwc_correction_count,
   amrex::Long* hll_fallback_count,
   amrex::iMultiFab const* black_hole_excision_mask = nullptr,
   SphericalExcisionRegion const& excision_region = {},
   RelativisticDiodeBoundary const& diode = {}
);

} // namespace pyfluid_amrex
