#pragma once

#include "pyfluid_amrex/kernels_generated/z4c/evolution.hpp"
#include "pyfluid_amrex/z4c_derivatives.hpp"

namespace pyfluid_amrex {

template <
   int Order,
   int Dim,
   typename Array,
   typename SpacingArray
>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cRhs evaluate_z4c_rhs_at(
   Array const& field,
   int i,
   int j,
   int k,
   SpacingArray const& inv_spacing,
   Z4cMatter const& matter,
   amrex::Real time,
   Z4cEvolutionParameters const& parameters,
   Z4cLapseGauge lapse_gauge,
   Z4cShiftGauge shift_gauge
) noexcept
{
   const Z4cState state = load_z4c_state(field, i, j, k);
   const Z4cSpatialDerivatives derivatives =
      gather_z4c_spatial_derivatives<Order, Dim>(
         field, i, j, k, inv_spacing, state
      );
   return generated::evaluate_z4c_rhs(
      state, derivatives, matter, time, parameters, lapse_gauge, shift_gauge
   );
}

template <
   int Order,
   int Dim,
   typename Array,
   typename SpacingArray
>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cRhs evaluate_vacuum_z4c_rhs_at(
   Array const& field,
   int i,
   int j,
   int k,
   SpacingArray const& inv_spacing,
   amrex::Real time,
   Z4cEvolutionParameters const& parameters,
   Z4cLapseGauge lapse_gauge,
   Z4cShiftGauge shift_gauge
) noexcept
{
   return evaluate_z4c_rhs_at<Order, Dim>(
      field,
      i,
      j,
      k,
      inv_spacing,
      vacuum_z4c_matter(),
      time,
      parameters,
      lapse_gauge,
      shift_gauge
   );
}

template <typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_z4c_rhs(
   Array const& rhs,
   int i,
   int j,
   int k,
   Z4cRhs const& z4c_rhs
) noexcept
{
   Z4cGeometricRhs const& geometric_rhs = z4c_rhs.geometry;
   rhs(i, j, k, Z4C_CHI) = geometric_rhs.chi;
   rhs(i, j, k, Z4C_GAMMA_TILDE11) = geometric_rhs.gamma_tilde11;
   rhs(i, j, k, Z4C_GAMMA_TILDE12) = geometric_rhs.gamma_tilde12;
   rhs(i, j, k, Z4C_GAMMA_TILDE13) = geometric_rhs.gamma_tilde13;
   rhs(i, j, k, Z4C_GAMMA_TILDE22) = geometric_rhs.gamma_tilde22;
   rhs(i, j, k, Z4C_GAMMA_TILDE23) = geometric_rhs.gamma_tilde23;
   rhs(i, j, k, Z4C_GAMMA_TILDE33) = geometric_rhs.gamma_tilde33;
   rhs(i, j, k, Z4C_K_HAT) = geometric_rhs.k_hat;
   rhs(i, j, k, Z4C_A_TILDE11) = geometric_rhs.a_tilde11;
   rhs(i, j, k, Z4C_A_TILDE12) = geometric_rhs.a_tilde12;
   rhs(i, j, k, Z4C_A_TILDE13) = geometric_rhs.a_tilde13;
   rhs(i, j, k, Z4C_A_TILDE22) = geometric_rhs.a_tilde22;
   rhs(i, j, k, Z4C_A_TILDE23) = geometric_rhs.a_tilde23;
   rhs(i, j, k, Z4C_A_TILDE33) = geometric_rhs.a_tilde33;
   rhs(i, j, k, Z4C_GAMMA_TILDE1) = geometric_rhs.gamma_tilde1;
   rhs(i, j, k, Z4C_GAMMA_TILDE2) = geometric_rhs.gamma_tilde2;
   rhs(i, j, k, Z4C_GAMMA_TILDE3) = geometric_rhs.gamma_tilde3;
   rhs(i, j, k, Z4C_THETA) = geometric_rhs.theta;
   rhs(i, j, k, Z4C_LAPSE) = z4c_rhs.lapse;
   rhs(i, j, k, Z4C_SHIFT1) = z4c_rhs.shift.shift1;
   rhs(i, j, k, Z4C_SHIFT2) = z4c_rhs.shift.shift2;
   rhs(i, j, k, Z4C_SHIFT3) = z4c_rhs.shift.shift3;
}

} // namespace pyfluid_amrex
