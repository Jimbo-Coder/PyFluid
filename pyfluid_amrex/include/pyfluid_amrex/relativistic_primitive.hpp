#pragma once

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

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
