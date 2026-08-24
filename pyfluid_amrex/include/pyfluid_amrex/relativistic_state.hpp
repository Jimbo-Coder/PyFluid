#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace pyfluid_amrex {

struct RHDPrimitive {
   amrex::Real rho = 0.0;
   amrex::Real eps = 0.0;
   amrex::Real pressure = 0.0;
   amrex::Real v1 = 0.0;
   amrex::Real v2 = 0.0;
   amrex::Real v3 = 0.0;
};

struct RHDConserved {
   amrex::Real dens = 0.0;
   amrex::Real mom1 = 0.0;
   amrex::Real mom2 = 0.0;
   amrex::Real mom3 = 0.0;
   amrex::Real tau = 0.0;
};

struct RHDRecoveryState {
   amrex::Real dens = 0.0;
   amrex::Real mom1 = 0.0;
   amrex::Real mom2 = 0.0;
   amrex::Real mom3 = 0.0;
   amrex::Real tau = 0.0;
   amrex::Real momentum_squared = 0.0;
};

struct RHDWaveSpeeds {
   amrex::Real minus = 0.0;
   amrex::Real plus = 0.0;
};

struct RMHDPrimitive {
   amrex::Real rho = 0.0;
   amrex::Real eps = 0.0;
   amrex::Real pressure = 0.0;
   amrex::Real v1 = 0.0;
   amrex::Real v2 = 0.0;
   amrex::Real v3 = 0.0;
   amrex::Real b1 = 0.0;
   amrex::Real b2 = 0.0;
   amrex::Real b3 = 0.0;
};

struct RMHDConserved {
   amrex::Real dens = 0.0;
   amrex::Real mom1 = 0.0;
   amrex::Real mom2 = 0.0;
   amrex::Real mom3 = 0.0;
   amrex::Real tau = 0.0;
   amrex::Real b1 = 0.0;
   amrex::Real b2 = 0.0;
   amrex::Real b3 = 0.0;
};

struct RMHDRecoveryState {
   amrex::Real dens = 0.0;
   amrex::Real mom1 = 0.0;
   amrex::Real mom2 = 0.0;
   amrex::Real mom3 = 0.0;
   amrex::Real tau = 0.0;
   amrex::Real b1 = 0.0;
   amrex::Real b2 = 0.0;
   amrex::Real b3 = 0.0;
   amrex::Real momentum_squared = 0.0;
   amrex::Real magnetic_squared = 0.0;
   amrex::Real magnetic_momentum = 0.0;
};

// EoS-independent trial state produced by the generated RMHD recovery
// kinematics.  Thermodynamic closure remains owned by EOSParams/EOSRuntime.
struct RMHDRecoveryKinematics {
   amrex::Real rho = 0.0;
   amrex::Real eps = 0.0;
   amrex::Real lorentz_factor = 1.0;
   amrex::Real v1 = 0.0;
   amrex::Real v2 = 0.0;
   amrex::Real v3 = 0.0;
   amrex::Real b1 = 0.0;
   amrex::Real b2 = 0.0;
   amrex::Real b3 = 0.0;
};

struct RMHDRePrimAndKinematics {
   amrex::Real rho = 0.0;
   amrex::Real eps = 0.0;
   amrex::Real lorentz_factor = 1.0;
   amrex::Real fluid_momentum_squared = 0.0;
   amrex::Real fluid_energy = 0.0;
   amrex::Real v1 = 0.0;
   amrex::Real v2 = 0.0;
   amrex::Real v3 = 0.0;
   amrex::Real b1 = 0.0;
   amrex::Real b2 = 0.0;
   amrex::Real b3 = 0.0;
};

struct RMHDWaveSpeeds {
   amrex::Real minus = 0.0;
   amrex::Real plus = 0.0;
};

struct StressEnergyTensor {
   amrex::Real t00 = 0.0;
   amrex::Real t01 = 0.0;
   amrex::Real t02 = 0.0;
   amrex::Real t03 = 0.0;
   amrex::Real t11 = 0.0;
   amrex::Real t12 = 0.0;
   amrex::Real t13 = 0.0;
   amrex::Real t22 = 0.0;
   amrex::Real t23 = 0.0;
   amrex::Real t33 = 0.0;
};

struct EulerianMatter {
   amrex::Real rho = 0.0;
   amrex::Real momentum1 = 0.0;
   amrex::Real momentum2 = 0.0;
   amrex::Real momentum3 = 0.0;
   amrex::Real stress11 = 0.0;
   amrex::Real stress12 = 0.0;
   amrex::Real stress13 = 0.0;
   amrex::Real stress22 = 0.0;
   amrex::Real stress23 = 0.0;
   amrex::Real stress33 = 0.0;
};

struct RelativisticFluidDiagnosticIntegrands {
   amrex::Real lorentz_factor = 1.0;
};

struct RMHDDiagnosticIntegrands {
   amrex::Real electromagnetic_energy = 0.0;
   amrex::Real b_squared = 0.0;
   amrex::Real enthalpy_magnetization = 0.0;
   amrex::Real plasma_beta = 0.0;
   amrex::Real lorentz_factor = 1.0;
};

enum EulerianMatterComponent : int {
   EULERIAN_MATTER_RHO = 0,
   EULERIAN_MATTER_MOMENTUM1,
   EULERIAN_MATTER_MOMENTUM2,
   EULERIAN_MATTER_MOMENTUM3,
   EULERIAN_MATTER_STRESS11,
   EULERIAN_MATTER_STRESS12,
   EULERIAN_MATTER_STRESS13,
   EULERIAN_MATTER_STRESS22,
   EULERIAN_MATTER_STRESS23,
   EULERIAN_MATTER_STRESS33,
   EULERIAN_MATTER_NUM_COMPONENTS,
};

struct SpacetimeMetric {
   amrex::Real lapse = 1.0;
   amrex::Real shift1 = 0.0;
   amrex::Real shift2 = 0.0;
   amrex::Real shift3 = 0.0;
   amrex::Real gamma11 = 1.0;
   amrex::Real gamma12 = 0.0;
   amrex::Real gamma13 = 0.0;
   amrex::Real gamma22 = 1.0;
   amrex::Real gamma23 = 0.0;
   amrex::Real gamma33 = 1.0;
   amrex::Real inv_gamma11 = 1.0;
   amrex::Real inv_gamma12 = 0.0;
   amrex::Real inv_gamma13 = 0.0;
   amrex::Real inv_gamma22 = 1.0;
   amrex::Real inv_gamma23 = 0.0;
   amrex::Real inv_gamma33 = 1.0;
   amrex::Real sqrt_det_gamma = 1.0;
};

struct SpacetimeMetricDerivatives {
   amrex::Real d1_lapse = 0.0;
   amrex::Real d2_lapse = 0.0;
   amrex::Real d3_lapse = 0.0;
   amrex::Real d1_shift1 = 0.0;
   amrex::Real d2_shift1 = 0.0;
   amrex::Real d3_shift1 = 0.0;
   amrex::Real d1_shift2 = 0.0;
   amrex::Real d2_shift2 = 0.0;
   amrex::Real d3_shift2 = 0.0;
   amrex::Real d1_shift3 = 0.0;
   amrex::Real d2_shift3 = 0.0;
   amrex::Real d3_shift3 = 0.0;
   amrex::Real d1_gamma11 = 0.0;
   amrex::Real d1_gamma12 = 0.0;
   amrex::Real d1_gamma13 = 0.0;
   amrex::Real d1_gamma22 = 0.0;
   amrex::Real d1_gamma23 = 0.0;
   amrex::Real d1_gamma33 = 0.0;
   amrex::Real d2_gamma11 = 0.0;
   amrex::Real d2_gamma12 = 0.0;
   amrex::Real d2_gamma13 = 0.0;
   amrex::Real d2_gamma22 = 0.0;
   amrex::Real d2_gamma23 = 0.0;
   amrex::Real d2_gamma33 = 0.0;
   amrex::Real d3_gamma11 = 0.0;
   amrex::Real d3_gamma12 = 0.0;
   amrex::Real d3_gamma13 = 0.0;
   amrex::Real d3_gamma22 = 0.0;
   amrex::Real d3_gamma23 = 0.0;
   amrex::Real d3_gamma33 = 0.0;
   amrex::Real extrinsic11 = 0.0;
   amrex::Real extrinsic12 = 0.0;
   amrex::Real extrinsic13 = 0.0;
   amrex::Real extrinsic22 = 0.0;
   amrex::Real extrinsic23 = 0.0;
   amrex::Real extrinsic33 = 0.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real spatial_vector_norm_squared(
   amrex::Real vector1,
   amrex::Real vector2,
   amrex::Real vector3,
   const SpacetimeMetric& metric
) noexcept
{
   return vector1 * (
      metric.gamma11 * vector1 + metric.gamma12 * vector2
         + metric.gamma13 * vector3
   ) + vector2 * (
      metric.gamma12 * vector1 + metric.gamma22 * vector2
         + metric.gamma23 * vector3
   ) + vector3 * (
      metric.gamma13 * vector1 + metric.gamma23 * vector2
         + metric.gamma33 * vector3
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr SpacetimeMetric minkowski_spacetime_metric() noexcept
{
   return {};
}

} // namespace pyfluid_amrex
