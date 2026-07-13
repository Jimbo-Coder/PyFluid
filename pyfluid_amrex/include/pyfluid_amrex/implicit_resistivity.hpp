#pragma once

#include "pyfluid_amrex/config.hpp"

namespace amrex {
class Geometry;
class MultiFab;
}

namespace pyfluid_amrex {

struct EdgeElectricField;
struct FaceMagneticField;

#if (AMREX_SPACEDIM >= 2)
DiffusionStats advance_ct_resistive_magnetic_field_implicit(
   FaceMagneticField& face_b,
   EdgeElectricField& edge_e,
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   amrex::Real dt,
   RunConfig const& cfg
);
#endif

} // namespace pyfluid_amrex
