#pragma once

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"
#include "pyfluid_amrex/state.hpp"

#include <limits>

namespace pyfluid_amrex {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool primitive_is_atmosphere(
   amrex::Real rho,
   amrex::Real pressure,
   EOSParams const& eos
) noexcept
{
   constexpr amrex::Real tolerance =
      1024.0 * std::numeric_limits<amrex::Real>::epsilon();
   return rho <= eos.rho_floor * (1.0 + tolerance)
      || pressure <= eos.pressure_floor * (1.0 + tolerance);
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
      pressure / ((eos.gamma - 1.0) * rho),
      pressure,
      primitive(i, j, k, MOM1),
      primitive(i, j, k, MOM2),
      primitive(i, j, k, MOM3),
      primitive(i, j, k, B1),
      primitive(i, j, k, B2),
      primitive(i, j, k, B3),
   };
}

} // namespace pyfluid_amrex
