#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <memory>

#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/eos/types.hpp"

namespace pyfluid_amrex {

struct RunConfig;

struct PressurePoloidalMagneticSeedParameters {
   amrex::Real pressure_cutoff_fraction = 0.04;
   amrex::Real exponent = 2.0;
   amrex::Real maximum_comoving_field = 0.0;
   std::array<amrex::Real, 3> axis{{0.0, 0.0, 1.0}};
   std::array<amrex::Real, 3> center{{0.0, 0.0, 0.0}};
};

struct PressurePoloidalMagneticSeedSummary {
   amrex::Real maximum_pressure = 0.0;
   amrex::Real unit_maximum_comoving_field = 0.0;
   amrex::Real normalization = 0.0;
   amrex::Real maximum_comoving_field = 0.0;
   amrex::Real divergence_linf = 0.0;
};

// Transient staggered work only. Persistent magnetic ownership remains the
// face-centered CT field declared by schema/fields.toml and written by the
// physical-ID container.
struct PressurePoloidalMagneticSeedWorkspace {
   std::unique_ptr<amrex::MultiFab> pressure;
   std::unique_ptr<amrex::MultiFab> comoving_field_squared;
   std::unique_ptr<amrex::MultiFab> divergence;
   EdgeElectricField vector_potential;
};

PressurePoloidalMagneticSeedSummary apply_pressure_poloidal_magnetic_seed(
   FaceMagneticField& face_magnetic,
   amrex::MultiFab& primitive,
   amrex::MultiFab const& cell_spacetime,
   amrex::Geometry const& geometry,
   RunConfig const& config,
   EOSParams const& eos,
   PressurePoloidalMagneticSeedParameters const& parameters,
   PressurePoloidalMagneticSeedWorkspace& workspace
);

} // namespace pyfluid_amrex
