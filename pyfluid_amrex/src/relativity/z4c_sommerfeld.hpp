#pragma once

#include <cmath>
#include "pyfluid_amrex/z4c_characteristics.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

// Hilditch et al. 1212.2901, Eqs. (26)--(30): approximate outgoing waves,
// NOT constraint-preserving BCs. Radius is the asymptotic coordinate radius;
// radial direction is normalized with the evolved physical metric. AthenaK
// supplies the independent second-order boundary-derivative reference.
template <typename Array, typename RhsArray>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void z4c_sommerfeld_rhs_at_point(Array const& state,
   RhsArray const& rhs, int i, int j, int k,
   amrex::GpuArray<amrex::Real, 3> const& position,
   amrex::GpuArray<amrex::Real, 3> const& inv_spacing,
   Z4cEvolutionParameters const& parameters, Z4cLapseGauge lapse_gauge,
   Z4cShiftGauge shift_gauge) noexcept
{
   const Z4cState local = load_z4c_state(state, i, j, k);
   const auto metric = generated::z4c_spacetime_metric(local);
   const amrex::Real radius = std::sqrt(position[0] * position[0]
      + position[1] * position[1] + position[2] * position[2]);
   AMREX_ALWAYS_ASSERT(radius > 0.0_rt);
   const amrex::Real inverse_radius = 1.0_rt / radius;
   amrex::GpuArray<amrex::Real, 3> covariant{{
      position[0] * inverse_radius, position[1] * inverse_radius,
      position[2] * inverse_radius}};
   amrex::GpuArray<amrex::Real, 3> normal{{
      metric.inv_gamma11 * covariant[0] + metric.inv_gamma12 * covariant[1] + metric.inv_gamma13 * covariant[2],
      metric.inv_gamma12 * covariant[0] + metric.inv_gamma22 * covariant[1] + metric.inv_gamma23 * covariant[2],
      metric.inv_gamma13 * covariant[0] + metric.inv_gamma23 * covariant[1] + metric.inv_gamma33 * covariant[2]}};
   const amrex::Real normalization = 1.0_rt / std::sqrt(covariant[0] * normal[0]
      + covariant[1] * normal[1] + covariant[2] * normal[2]);
   for (int direction = 0; direction < 3; ++direction) {
      normal[direction] *= normalization;
      covariant[direction] *= normalization;
   }
   const amrex::GpuArray<amrex::Real, 3> shift{{metric.shift1, metric.shift2, metric.shift3}};
   const auto gradient = [=](int component) noexcept {
      amrex::GpuArray<amrex::Real, 3> result{{
         generated::centered_first<2, 0>(state, i, j, k, component, inv_spacing[0]), 0.0_rt, 0.0_rt}};
#if (AMREX_SPACEDIM >= 2)
      result[1] = generated::centered_first<2, 1>(state, i, j, k, component, inv_spacing[1]);
#endif
#if (AMREX_SPACEDIM == 3)
      result[2] = generated::centered_first<2, 2>(state, i, j, k, component, inv_spacing[2]);
#endif
      return result;
   };
   const auto outgoing = [=](int component, amrex::Real speed) noexcept {
      const auto derivative = gradient(component);
      amrex::Real value = -speed * state(i, j, k, component) * inverse_radius;
      for (int direction = 0; direction < 3; ++direction) {
         value += (shift[direction] - speed * normal[direction]) * derivative[direction];
      }
      return value;
   };
   // Frozen lapse has no incoming lapse-gauge characteristic to prescribe.
   if (lapse_gauge != Z4cLapseGauge::Frozen) {
      const amrex::Real lapse_speed_squared = z4c_lapse_speed_squared(local, lapse_gauge);
      AMREX_ALWAYS_ASSERT(lapse_speed_squared >= 0.0_rt);
      rhs(i, j, k, Z4C_K_HAT) = outgoing(Z4C_K_HAT,
         metric.lapse * std::sqrt(lapse_speed_squared));
   }
   rhs(i, j, k, Z4C_THETA) = outgoing(Z4C_THETA, metric.lapse);
   for (int component : z4c_a_tilde_components()) {
      rhs(i, j, k, component) = outgoing(component, metric.lapse);
   }
   // With frozen shift, retain the bulk Gamma equation (no shift-gauge BC).
   if (shift_gauge != Z4cShiftGauge::Frozen) {
      const amrex::Real speed = metric.lapse * std::sqrt(
         z4c_shift_speed_squared(local, parameters, shift_gauge));
      amrex::Real longitudinal = 0.0_rt;
      for (int axis = 0; axis < 3; ++axis) {
         const int component = z4c_gamma_tilde_vector_components()[axis];
         const auto derivative = gradient(component);
         amrex::Real radial = state(i, j, k, component) * inverse_radius;
         for (int direction = 0; direction < 3; ++direction) radial += normal[direction] * derivative[direction];
         longitudinal += covariant[axis] * radial;
         rhs(i, j, k, component) = outgoing(component, speed);
      }
      for (int axis = 0; axis < 3; ++axis) {
         rhs(i, j, k, z4c_gamma_tilde_vector_components()[axis]) -=
            (2.0_rt / std::sqrt(3.0_rt) - 1.0_rt) * speed * normal[axis] * longitudinal;
      }
   }
}

// Alcubierre et al. gr-qc/0206072 Sec. VI / ETK NewRad: retain the
// non-radiative part of the interior MOL update rather than interpreting
// the entire stationary/Coulomb background as an outgoing 1/r wave.
// This is still an approximate radiative BC, not a Z4c CPBC.
template <typename Array, typename RhsArray>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void z4c_radiative_rhs_at_point(Array const& state, RhsArray const& rhs,
   int i, int j, int k, amrex::GpuArray<int, 3> const& donor,
   amrex::GpuArray<amrex::Real, 3> const& position,
   amrex::GpuArray<amrex::Real, 3> const& inv_spacing,
   Z4cEvolutionParameters const& parameters, Z4cLapseGauge lapse_gauge,
   Z4cShiftGauge shift_gauge, int nonradiative_power) noexcept
{
   if (nonradiative_power < 0) {
      z4c_sommerfeld_rhs_at_point(state, rhs, i, j, k, position,
         inv_spacing, parameters, lapse_gauge, shift_gauge);
      return;
   }
   const int point[3]{i, j, k};
   auto donor_position = position;
   amrex::Real radius_squared = 0.0_rt, donor_radius_squared = 0.0_rt;
   for (int direction = 0; direction < 3; ++direction) {
      if (direction < AMREX_SPACEDIM) {
         donor_position[direction] += (donor[direction] - point[direction])
            / inv_spacing[direction];
      }
      radius_squared += position[direction] * position[direction];
      donor_radius_squared += donor_position[direction] * donor_position[direction];
   }
   AMREX_ALWAYS_ASSERT(donor_radius_squared > 0.0_rt && radius_squared > 0.0_rt);
   const amrex::Real scale = std::pow(donor_radius_squared / radius_squared,
      0.5_rt * nonradiative_power);
   // Initialize from the bulk RHS. Components not owned by the radiative
   // operator then have exactly zero correction, including frozen gauges.
   amrex::Real donor_outgoing[Z4C_NUM_COMPONENTS];
   for (int component = 0; component < Z4C_NUM_COMPONENTS; ++component) {
      donor_outgoing[component] = rhs(donor[0], donor[1], donor[2], component);
   }
   // Inherit this pointwise function's host/device execution space.
   const auto scratch = [&](int, int, int, int component) noexcept -> amrex::Real& {
      return donor_outgoing[component];
   };
   z4c_sommerfeld_rhs_at_point(state, scratch, donor[0], donor[1], donor[2],
      donor_position, inv_spacing, parameters, lapse_gauge, shift_gauge);
   z4c_sommerfeld_rhs_at_point(state, rhs, i, j, k, position,
      inv_spacing, parameters, lapse_gauge, shift_gauge);
   for (int component = 0; component < Z4C_NUM_COMPONENTS; ++component) {
      rhs(i, j, k, component) += scale
         * (rhs(donor[0], donor[1], donor[2], component) - donor_outgoing[component]);
   }
}

} // namespace pyfluid_amrex
