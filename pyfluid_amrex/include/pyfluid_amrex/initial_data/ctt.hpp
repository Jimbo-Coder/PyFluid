#pragma once

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include "pyfluid_amrex/initial_data/elliptic.hpp"

namespace pyfluid_amrex {

struct ConformalTraceFreeTensor {
   amrex::Real a11 = 0.0;
   amrex::Real a12 = 0.0;
   amrex::Real a13 = 0.0;
   amrex::Real a22 = 0.0;
   amrex::Real a23 = 0.0;
   amrex::Real a33 = 0.0;
   amrex::Real squared = 0.0;
};

enum ConformalTraceFreeComponent : int {
   CONFORMAL_A11 = 0,
   CONFORMAL_A12,
   CONFORMAL_A13,
   CONFORMAL_A22,
   CONFORMAL_A23,
   CONFORMAL_A33,
   CONFORMAL_A_SQUARED,
   CONFORMAL_A_NUM_COMPONENTS,
};

struct CttHamiltonianOptions {
   int max_nonlinear_iterations = 100;
   amrex::Real relative_update_tolerance = 1.0e-10;
   amrex::Real relaxation = 0.5;
   InitialDataEllipticOptions linear;
};

struct CttHamiltonianResult {
   int nonlinear_iterations = 0;
   amrex::Real relative_update = 0.0;
   amrex::Real linear_residual = 0.0;
   bool converged = false;
};

struct CttOptions {
   int max_coupled_iterations = 100;
   amrex::Real relative_update_tolerance = 1.0e-10;
   amrex::Real relaxation = 0.5;
   InitialDataEllipticOptions linear;
};

struct CttResult {
   int coupled_iterations = 0;
   amrex::Real relative_update = 0.0;
   amrex::Real hamiltonian_residual = 0.0;
   amrex::Real momentum_residual = 0.0;
   bool converged = false;
};

enum class CttkMeanCurvatureBranch : int {
   negative = -1,
   positive = 1,
};

struct CttkOptions {
   int max_coupled_iterations = 100;
   amrex::Real relative_update_tolerance = 1.0e-10;
   amrex::Real relaxation = 0.5;
   CttkMeanCurvatureBranch mean_curvature_branch =
      CttkMeanCurvatureBranch::negative;
   InitialDataEllipticOptions linear;
};

struct CttkResult {
   int coupled_iterations = 0;
   amrex::Real relative_update = 0.0;
   amrex::Real hamiltonian_residual = 0.0;
   amrex::Real momentum_residual = 0.0;
   amrex::Real minimum_mean_curvature_squared = 0.0;
   bool converged = false;
};

struct XctsOptions {
   int max_coupled_iterations = 100;
   amrex::Real relative_update_tolerance = 1.0e-10;
   amrex::Real relaxation = 0.5;
   InitialDataEllipticOptions linear;
};

struct XctsResult {
   int coupled_iterations = 0;
   amrex::Real relative_update = 0.0;
   amrex::Real hamiltonian_residual = 0.0;
   amrex::Real lapse_residual = 0.0;
   amrex::Real momentum_residual = 0.0;
   bool converged = false;
};

CttHamiltonianResult solve_ctt_time_symmetric_hamiltonian(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   InitialDataEllipticBoundaryConditions const& conformal_correction_boundary,
   CttHamiltonianOptions const& options = {}
);

// Solve conformally flat, maximal CTT data in three spatial dimensions:
// Delta_L W^i = 8 pi psi^10 S^i and
// Delta psi = -2 pi E psi^5 - (L W)^2/(8 psi^7).
CttResult solve_ctt_constraints(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& vector_potential,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& momentum_density,
   InitialDataEllipticBoundaryConditions const& conformal_correction_boundary,
   InitialDataEllipticBoundaryConditions const& vector_potential_boundary,
   CttOptions const& options = {}
);

// Solve pure conformally flat CTTK data in three spatial dimensions. The
// conformal factor is prescribed; its valid and ghost values must represent
// that fixed profile on the complete hierarchy.
CttkResult solve_cttk_constraints(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& mean_curvature,
   amrex::Vector<amrex::MultiFab*> const& vector_potential,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& momentum_density,
   InitialDataEllipticBoundaryConditions const& vector_potential_boundary,
   CttkOptions const& options = {}
);

// Hybrid CTTK fixes K^2 = 24 pi E and solves the remaining conformal-factor
// equation together with the same momentum constraint used by pure CTTK.
CttkResult solve_cttk_hybrid_constraints(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& mean_curvature,
   amrex::Vector<amrex::MultiFab*> const& vector_potential,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& momentum_density,
   InitialDataEllipticBoundaryConditions const& conformal_correction_boundary,
   InitialDataEllipticBoundaryConditions const& vector_potential_boundary,
   CttkOptions const& options = {}
);

// Solve conformally flat, maximal, quasiequilibrium XCTS data in three
// spatial dimensions. The unknowns are psi, alpha*psi, and beta^i; the free
// data use dt gamma_tilde_ij = 0, K = 0, and dt K = 0.
XctsResult solve_xcts_constraints(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& shift,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& momentum_density,
   amrex::Vector<amrex::MultiFab const*> const& stress_trace,
   InitialDataEllipticBoundaryConditions const& conformal_correction_boundary,
   InitialDataEllipticBoundaryConditions const& lapse_correction_boundary,
   InitialDataEllipticBoundaryConditions const& shift_boundary,
   XctsOptions const& options = {}
);

// Solve circular, stationary, axisymmetric XCTS data on a physical uniform
// cell-centred (r,theta) hierarchy. Vector quantities are orthonormal physical
// components; only beta_hat_phi and S_hat_phi are independent. The returned
// conformal extrinsic tensor uses the common ordered physical components
// (r,theta,phi) in ConformalTraceFreeComponent.
XctsResult solve_axisymmetric_spherical_xcts_constraints(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& azimuthal_shift,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& azimuthal_momentum_density,
   amrex::Vector<amrex::MultiFab const*> const& stress_trace,
   InitialDataEllipticBoundaryConditions const& conformal_correction_boundary,
   InitialDataEllipticBoundaryConditions const& lapse_correction_boundary,
   InitialDataEllipticBoundaryConditions const& azimuthal_shift_boundary,
   XctsOptions const& options = {}
);

} // namespace pyfluid_amrex
