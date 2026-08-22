#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <memory>

#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/initial_data/elliptic.hpp"
#include "pyfluid_amrex/initial_data/rotating_star.hpp"

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

// Convert the canonical densitized CT face field sqrt(gamma) B^i to the
// physical Eulerian B^i stored in the reusable primitive ID field.
void sync_rmhd_primitive_b_from_densitized_faces(
   amrex::MultiFab& primitive,
   FaceMagneticField const& face_magnetic,
   amrex::MultiFab const& cell_spacetime
);

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

struct PoloidalGradShafranovParameters {
   amrex::Real poloidal_magnetization = 0.0;
   amrex::Real nonlinear_poloidal = 0.0;
   amrex::Real relaxation = 0.5;
   std::array<amrex::Real, 3> axis{{0.0, 0.0, 1.0}};
   std::array<amrex::Real, 3> center{{0.0, 0.0, 0.0}};
};

struct PoloidalGradShafranovSummary {
   RotatingStarUpdateSummary rotating_star{};
   amrex::Real relative_potential_update = 0.0;
   amrex::Real linear_residual = 0.0;
   amrex::Real equation_relative_residual = 0.0;
   amrex::Real bernoulli_residual = 0.0;
   amrex::Real maximum_comoving_field = 0.0;
   amrex::Real divergence_linf = 0.0;
};

// Solver-private Grad-Shafranov unknowns.  Atilde is the Cartesian vector
// A_phi e_phi/varpi; flux_function stores A_phi.  Neither is a persistent GF:
// the canonical restart/evolution owner is the densitized CT face field.
struct PoloidalGradShafranovWorkspace {
   std::array<std::unique_ptr<amrex::MultiFab>, 3> vector_potential;
   std::array<std::unique_ptr<amrex::MultiFab>, 3> candidate;
   std::array<std::unique_ptr<amrex::MultiFab>, 3> rhs;
   std::unique_ptr<amrex::MultiFab> flux_function;
   std::unique_ptr<amrex::MultiFab> metric_potential;
   std::unique_ptr<amrex::MultiFab> equation_residual;
   std::unique_ptr<amrex::MultiFab> bernoulli_residual;
   std::unique_ptr<amrex::MultiFab> comoving_field_squared;
   std::unique_ptr<amrex::MultiFab> divergence;
   EdgeElectricField edge_vector_potential;
};

PoloidalGradShafranovSummary update_poloidal_grad_shafranov_equilibrium(
   InitialDataEllipticHierarchy const& hierarchy,
   FaceMagneticField& face_magnetic,
   amrex::MultiFab& primitive,
   amrex::MultiFab const& cell_spacetime,
   RunConfig const& config,
   int eos_method,
   EOSParams const& eos,
   RotatingStarParameters const& rotating_star,
   PoloidalGradShafranovParameters const& parameters,
   PoloidalGradShafranovWorkspace& workspace,
   bool update_fluid
);

} // namespace pyfluid_amrex
