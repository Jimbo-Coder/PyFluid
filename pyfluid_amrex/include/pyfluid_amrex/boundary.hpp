#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/boundary_data.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct PatchDomainBoxes;

void validate_origin_boundary_geometry(BoundaryData const& bc, amrex::Geometry const& geom);
int amrex_state_bc_type(BoundaryKind kind, int component, int dir, bool mhd);
int amrex_face_b_bc_type(BoundaryKind kind, int face_dir, int boundary_dir);

template <int NComp>
void apply_state_boundary(amrex::MultiFab& state, amrex::Geometry const& geom,
                          BoundaryData const& bc, amrex::Real time = 0.0_rt);

// Keep the registered hydro/MHD layouts in one compiled launch owner.
extern template void apply_state_boundary<NVAR_HYDRO>(
   amrex::MultiFab&, amrex::Geometry const&, BoundaryData const&, amrex::Real);
extern template void apply_state_boundary<NVAR_HYDRO + 1>(
   amrex::MultiFab&, amrex::Geometry const&, BoundaryData const&, amrex::Real);
extern template void apply_state_boundary<NVAR_MHD>(
   amrex::MultiFab&, amrex::Geometry const&, BoundaryData const&, amrex::Real);
extern template void apply_state_boundary<NVAR_MHD + 1>(
   amrex::MultiFab&, amrex::Geometry const&, BoundaryData const&, amrex::Real);

// The hierarchy owner supplies this same bounded donor band through FillPatchNLevels.
amrex::Box radial_origin_source_box(amrex::Geometry const& geom, BoundaryData const& bc,
                                    amrex::IntVect const& num_ghosts);
void fill_radial_origin_from_source(amrex::MultiFab& state, amrex::MultiFab const& source_values,
                                   amrex::Geometry const& geom, BoundaryData const& bc);
void apply_radial_antipodal_origin_boundary(amrex::MultiFab& state, amrex::Geometry const& geom,
                                            BoundaryData const& bc);

template <int NComp>
inline void apply_state_boundary(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
   const BoundaryData bc = make_boundary_data(cfg);
   apply_state_boundary<NComp>(state, geom, bc);
   apply_radial_antipodal_origin_boundary(state, geom, bc);
}

inline void apply_hydro_boundary(amrex::MultiFab& state, amrex::Geometry const& geom, BoundaryData const& bc)
{
   apply_state_boundary<NVAR_HYDRO>(state, geom, bc);
   apply_radial_antipodal_origin_boundary(state, geom, bc);
}

inline void apply_mhd_boundary(amrex::MultiFab& state, amrex::Geometry const& geom, BoundaryData const& bc)
{
   apply_state_boundary<NVAR_MHD>(state, geom, bc);
   apply_radial_antipodal_origin_boundary(state, geom, bc);
}

inline void apply_hydro_boundary(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   apply_state_boundary<NVAR_HYDRO>(state, geom, cfg);
}

inline void apply_mhd_boundary(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   apply_state_boundary<NVAR_MHD>(state, geom, cfg);
}

void apply_patch_union_reflecting_boundary(amrex::MultiFab& state,
                                           amrex::Geometry const& geom,
                                           PatchDomainBoxes const& patch_domain);

void apply_configured_state_boundary(amrex::MultiFab& state, amrex::Geometry const& geom,
                                      BoundaryData const& bc, amrex::Real time = 0.0_rt);
void apply_configured_state_boundary(amrex::MultiFab& state, amrex::Geometry const& geom,
                                      RunConfig const& cfg, amrex::Real time = 0.0_rt);

} // namespace pyfluid_amrex
