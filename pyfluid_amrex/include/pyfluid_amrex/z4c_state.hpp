#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/relativistic_state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

// Pointwise full-value Z4c state. Minkowski space is chi=alpha=1,
// gamma_tilde_ij=delta_ij, with every other component zero.
struct Z4cState {
   amrex::Real chi = 1.0_rt;
   amrex::Real gamma_tilde11 = 1.0_rt;
   amrex::Real gamma_tilde12 = 0.0_rt;
   amrex::Real gamma_tilde13 = 0.0_rt;
   amrex::Real gamma_tilde22 = 1.0_rt;
   amrex::Real gamma_tilde23 = 0.0_rt;
   amrex::Real gamma_tilde33 = 1.0_rt;
   amrex::Real k_hat = 0.0_rt;
   amrex::Real a_tilde11 = 0.0_rt;
   amrex::Real a_tilde12 = 0.0_rt;
   amrex::Real a_tilde13 = 0.0_rt;
   amrex::Real a_tilde22 = 0.0_rt;
   amrex::Real a_tilde23 = 0.0_rt;
   amrex::Real a_tilde33 = 0.0_rt;
   amrex::Real gamma_tilde1 = 0.0_rt;
   amrex::Real gamma_tilde2 = 0.0_rt;
   amrex::Real gamma_tilde3 = 0.0_rt;
   amrex::Real theta = 0.0_rt;
   amrex::Real lapse = 1.0_rt;
   amrex::Real shift1 = 0.0_rt;
   amrex::Real shift2 = 0.0_rt;
   amrex::Real shift3 = 0.0_rt;
};

// Logical derivative direction is the first index. Symmetric tensor
// components use the shared 11,12,13,22,23,33 order on the second index.
struct Z4cSpatialDerivatives {
   amrex::Real d_chi[3] = {};
   amrex::Real advection_chi[3] = {};
   amrex::Real dd_chi[3][3] = {};
   amrex::Real d_gamma_tilde[3][6] = {};
   amrex::Real advection_gamma_tilde[3][6] = {};
   amrex::Real dd_gamma_tilde[3][3][6] = {};
   amrex::Real d_a_tilde[3][6] = {};
   amrex::Real advection_a_tilde[3][6] = {};
   amrex::Real d_k_hat[3] = {};
   amrex::Real advection_k_hat[3] = {};
   amrex::Real d_gamma_tilde_vector[3][3] = {};
   amrex::Real advection_gamma_tilde_vector[3][3] = {};
   amrex::Real d_theta[3] = {};
   amrex::Real advection_theta[3] = {};
   amrex::Real d_lapse[3] = {};
   amrex::Real advection_lapse[3] = {};
   amrex::Real dd_lapse[3][3] = {};
   amrex::Real d_shift[3][3] = {};
   amrex::Real advection_shift[3][3] = {};
   amrex::Real dd_shift[3][3][3] = {};
};

using Z4cMatter = EulerianMatter;

struct Z4cConformalMetric {
   amrex::Real determinant = 1.0_rt;
   amrex::Real inv_gamma_tilde11 = 1.0_rt;
   amrex::Real inv_gamma_tilde12 = 0.0_rt;
   amrex::Real inv_gamma_tilde13 = 0.0_rt;
   amrex::Real inv_gamma_tilde22 = 1.0_rt;
   amrex::Real inv_gamma_tilde23 = 0.0_rt;
   amrex::Real inv_gamma_tilde33 = 1.0_rt;
   amrex::Real gamma11 = 1.0_rt;
   amrex::Real gamma12 = 0.0_rt;
   amrex::Real gamma13 = 0.0_rt;
   amrex::Real gamma22 = 1.0_rt;
   amrex::Real gamma23 = 0.0_rt;
   amrex::Real gamma33 = 1.0_rt;
   amrex::Real inv_gamma11 = 1.0_rt;
   amrex::Real inv_gamma12 = 0.0_rt;
   amrex::Real inv_gamma13 = 0.0_rt;
   amrex::Real inv_gamma22 = 1.0_rt;
   amrex::Real inv_gamma23 = 0.0_rt;
   amrex::Real inv_gamma33 = 1.0_rt;
   amrex::Real sqrt_det_gamma = 1.0_rt;
};

struct Z4cConnections {
   amrex::Real christoffel_tilde1_11 = 0.0_rt;
   amrex::Real christoffel_tilde1_12 = 0.0_rt;
   amrex::Real christoffel_tilde1_13 = 0.0_rt;
   amrex::Real christoffel_tilde1_22 = 0.0_rt;
   amrex::Real christoffel_tilde1_23 = 0.0_rt;
   amrex::Real christoffel_tilde1_33 = 0.0_rt;
   amrex::Real christoffel_tilde2_11 = 0.0_rt;
   amrex::Real christoffel_tilde2_12 = 0.0_rt;
   amrex::Real christoffel_tilde2_13 = 0.0_rt;
   amrex::Real christoffel_tilde2_22 = 0.0_rt;
   amrex::Real christoffel_tilde2_23 = 0.0_rt;
   amrex::Real christoffel_tilde2_33 = 0.0_rt;
   amrex::Real christoffel_tilde3_11 = 0.0_rt;
   amrex::Real christoffel_tilde3_12 = 0.0_rt;
   amrex::Real christoffel_tilde3_13 = 0.0_rt;
   amrex::Real christoffel_tilde3_22 = 0.0_rt;
   amrex::Real christoffel_tilde3_23 = 0.0_rt;
   amrex::Real christoffel_tilde3_33 = 0.0_rt;
   amrex::Real contracted_christoffel_tilde1 = 0.0_rt;
   amrex::Real contracted_christoffel_tilde2 = 0.0_rt;
   amrex::Real contracted_christoffel_tilde3 = 0.0_rt;
   amrex::Real christoffel1_11 = 0.0_rt;
   amrex::Real christoffel1_12 = 0.0_rt;
   amrex::Real christoffel1_13 = 0.0_rt;
   amrex::Real christoffel1_22 = 0.0_rt;
   amrex::Real christoffel1_23 = 0.0_rt;
   amrex::Real christoffel1_33 = 0.0_rt;
   amrex::Real christoffel2_11 = 0.0_rt;
   amrex::Real christoffel2_12 = 0.0_rt;
   amrex::Real christoffel2_13 = 0.0_rt;
   amrex::Real christoffel2_22 = 0.0_rt;
   amrex::Real christoffel2_23 = 0.0_rt;
   amrex::Real christoffel2_33 = 0.0_rt;
   amrex::Real christoffel3_11 = 0.0_rt;
   amrex::Real christoffel3_12 = 0.0_rt;
   amrex::Real christoffel3_13 = 0.0_rt;
   amrex::Real christoffel3_22 = 0.0_rt;
   amrex::Real christoffel3_23 = 0.0_rt;
   amrex::Real christoffel3_33 = 0.0_rt;
};

struct Z4cDiagnosticGeometry {
   amrex::Real ricci11 = 0.0_rt;
   amrex::Real ricci12 = 0.0_rt;
   amrex::Real ricci13 = 0.0_rt;
   amrex::Real ricci22 = 0.0_rt;
   amrex::Real ricci23 = 0.0_rt;
   amrex::Real ricci33 = 0.0_rt;
   amrex::Real ricci_scalar = 0.0_rt;
};

struct Z4cConstraints {
   amrex::Real theta = 0.0_rt;
   amrex::Real z_tilde1 = 0.0_rt;
   amrex::Real z_tilde2 = 0.0_rt;
   amrex::Real z_tilde3 = 0.0_rt;
   amrex::Real hamiltonian = 0.0_rt;
   amrex::Real momentum1 = 0.0_rt;
   amrex::Real momentum2 = 0.0_rt;
   amrex::Real momentum3 = 0.0_rt;
   amrex::Real log_conformal_determinant = 0.0_rt;
   amrex::Real trace_a_tilde = 0.0_rt;
};

struct Z4cEvolutionGeometry {
   amrex::Real ricci11 = 0.0_rt;
   amrex::Real ricci12 = 0.0_rt;
   amrex::Real ricci13 = 0.0_rt;
   amrex::Real ricci22 = 0.0_rt;
   amrex::Real ricci23 = 0.0_rt;
   amrex::Real ricci33 = 0.0_rt;
   amrex::Real ricci_scalar = 0.0_rt;
};

struct Z4cGeometricRhs {
   amrex::Real chi = 0.0_rt;
   amrex::Real gamma_tilde11 = 0.0_rt;
   amrex::Real gamma_tilde12 = 0.0_rt;
   amrex::Real gamma_tilde13 = 0.0_rt;
   amrex::Real gamma_tilde22 = 0.0_rt;
   amrex::Real gamma_tilde23 = 0.0_rt;
   amrex::Real gamma_tilde33 = 0.0_rt;
   amrex::Real k_hat = 0.0_rt;
   amrex::Real a_tilde11 = 0.0_rt;
   amrex::Real a_tilde12 = 0.0_rt;
   amrex::Real a_tilde13 = 0.0_rt;
   amrex::Real a_tilde22 = 0.0_rt;
   amrex::Real a_tilde23 = 0.0_rt;
   amrex::Real a_tilde33 = 0.0_rt;
   amrex::Real gamma_tilde1 = 0.0_rt;
   amrex::Real gamma_tilde2 = 0.0_rt;
   amrex::Real gamma_tilde3 = 0.0_rt;
   amrex::Real theta = 0.0_rt;
};

struct Z4cShiftRhs {
   amrex::Real shift1 = 0.0_rt;
   amrex::Real shift2 = 0.0_rt;
   amrex::Real shift3 = 0.0_rt;
};

enum class Z4cLapseGauge : int {
   Frozen,
   Harmonic,
   OnePlusLog,
   Trumpet,
   SlowStart,
};

enum class Z4cShiftGauge : int {
   Frozen,
   GammaDriver,
};

struct Z4cEvolutionParameters {
   amrex::Real kappa1 = 0.02_rt;
   amrex::Real kappa2 = 0.0_rt;
   amrex::Real slow_start_strength = 0.0_rt;
   amrex::Real slow_start_width = 1.0_rt;
   amrex::Real shift_mu = 1.0_rt;
   amrex::Real shift_eta = 0.0_rt;
};

struct Z4cRhs {
   Z4cGeometricRhs geometry;
   amrex::Real lapse = 0.0_rt;
   Z4cShiftRhs shift;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr Z4cMatter vacuum_z4c_matter() noexcept
{
   return {};
}

} // namespace pyfluid_amrex
