#pragma once

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"
#include "pyfluid_amrex/state.hpp"

#include <cmath>
#include <limits>

namespace pyfluid_amrex {

using namespace amrex::literals;

struct SpatialFourVelocity {
   amrex::Real z1 = 0.0_rt;
   amrex::Real z2 = 0.0_rt;
   amrex::Real z3 = 0.0_rt;
};

struct EulerianVelocity {
   amrex::Real v1 = 0.0_rt;
   amrex::Real v2 = 0.0_rt;
   amrex::Real v3 = 0.0_rt;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SpatialFourVelocity spatial_four_velocity_from_eulerian(
   amrex::Real v1,
   amrex::Real v2,
   amrex::Real v3,
   SpacetimeMetric const& metric
) noexcept
{
   const amrex::Real velocity_squared = spatial_vector_norm_squared(
      v1, v2, v3, metric
   );
   const amrex::Real lorentz_factor = 1.0_rt / std::sqrt(1.0_rt - velocity_squared);
   return {
      lorentz_factor * v1,
      lorentz_factor * v2,
      lorentz_factor * v3,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
EulerianVelocity eulerian_velocity_from_spatial_four_velocity(
   SpatialFourVelocity const& spatial_four_velocity,
   SpacetimeMetric const& metric
) noexcept
{
   const amrex::Real lorentz_factor = std::sqrt(
      1.0_rt + spatial_vector_norm_squared(
         spatial_four_velocity.z1,
         spatial_four_velocity.z2,
         spatial_four_velocity.z3,
         metric
      )
   );
   return {
      spatial_four_velocity.z1 / lorentz_factor,
      spatial_four_velocity.z2 / lorentz_factor,
      spatial_four_velocity.z3 / lorentz_factor,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool primitive_is_atmosphere(
   amrex::Real rho,
   amrex::Real pressure,
   EOSParams const& eos
) noexcept
{
   constexpr amrex::Real tolerance =
      1024.0_rt * std::numeric_limits<amrex::Real>::epsilon();
   return rho <= eos.rho_floor * (1.0_rt + tolerance)
      || pressure <= eos.pressure_floor * (1.0_rt + tolerance);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDPrimitive rhd_primitive_at(
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& eos
) noexcept
{
   const amrex::Real rho = primitive(i, j, k, RHO);
   const amrex::Real pressure = primitive(i, j, k, ENERGY);
   return {
      rho,
      relativistic_specific_internal_energy_from_pressure<EOSMethod>(rho, pressure, eos),
      pressure,
      primitive(i, j, k, MOM1),
      primitive(i, j, k, MOM2),
      primitive(i, j, k, MOM3),
   };
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDPrimitive rmhd_primitive_at(
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& eos
) noexcept
{
   const amrex::Real rho = primitive(i, j, k, RHO);
   const amrex::Real pressure = primitive(i, j, k, ENERGY);
   return {
      rho,
      relativistic_specific_internal_energy_from_pressure<EOSMethod>(
         rho, pressure, eos
      ),
      pressure,
      primitive(i, j, k, MOM1),
      primitive(i, j, k, MOM2),
      primitive(i, j, k, MOM3),
      primitive(i, j, k, B1),
      primitive(i, j, k, B2),
      primitive(i, j, k, B3),
   };
}

// Runtime GRMHD evolution currently owns an ideal-gas C2P implementation.
// Keep its existing zero-branch call sites concise while allowing physical-ID
// construction to request the actual cold/hybrid/tabulated EoS explicitly.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDPrimitive rmhd_primitive_at(
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& eos
) noexcept
{
   return rmhd_primitive_at<EOSIdealGas>(primitive, i, j, k, eos);
}

} // namespace pyfluid_amrex
