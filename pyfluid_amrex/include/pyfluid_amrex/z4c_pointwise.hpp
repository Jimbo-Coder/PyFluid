#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/kernels_generated/differential_ops.hpp"
#include "pyfluid_amrex/kernels_generated/z4c.hpp"
#include "pyfluid_amrex/spacetime_fields.hpp"
#include "pyfluid_amrex/z4c_state.hpp"

namespace pyfluid_amrex {

template <typename ComponentReader>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cState load_z4c_state_components(ComponentReader const& component) noexcept
{
   return {
      component(Z4C_CHI),
      component(Z4C_GAMMA_TILDE11),
      component(Z4C_GAMMA_TILDE12),
      component(Z4C_GAMMA_TILDE13),
      component(Z4C_GAMMA_TILDE22),
      component(Z4C_GAMMA_TILDE23),
      component(Z4C_GAMMA_TILDE33),
      component(Z4C_K_HAT),
      component(Z4C_A_TILDE11),
      component(Z4C_A_TILDE12),
      component(Z4C_A_TILDE13),
      component(Z4C_A_TILDE22),
      component(Z4C_A_TILDE23),
      component(Z4C_A_TILDE33),
      component(Z4C_GAMMA_TILDE1),
      component(Z4C_GAMMA_TILDE2),
      component(Z4C_GAMMA_TILDE3),
      component(Z4C_THETA),
      component(Z4C_LAPSE),
      component(Z4C_SHIFT1),
      component(Z4C_SHIFT2),
      component(Z4C_SHIFT3),
   };
}

template <typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cState load_z4c_state(Array const& state, int i, int j, int k) noexcept
{
   struct Reader {
      Array const& state;
      int i;
      int j;
      int k;

      AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
      amrex::Real operator()(int component) const noexcept
      {
         return state(i, j, k, component);
      }
   };
   return load_z4c_state_components(Reader{state, i, j, k});
}

template <int Order, int Direction, typename Array>
struct Z4cCellToFaceReader {
   Array const& state;
   int i;
   int j;
   int k;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real operator()(int component) const noexcept
   {
      return generated::cell_to_face_ordered<Order, Direction>(
         state, i, j, k, component
      );
   }
};

template <int Order, int Direction, typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SpacetimeMetric evolved_spacetime_metric_at_face(
   Array const& cell_spacetime,
   int i,
   int j,
   int k
) noexcept
{
   const Z4cState state = load_z4c_state_components(
      Z4cCellToFaceReader<Order, Direction, Array>{cell_spacetime, i, j, k}
   );
   return generated::z4c_spacetime_metric(state);
}

template <typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_z4c_state(
   Array const& state,
   int i,
   int j,
   int k,
   Z4cState const& point
) noexcept
{
   state(i, j, k, Z4C_CHI) = point.chi;
   state(i, j, k, Z4C_GAMMA_TILDE11) = point.gamma_tilde11;
   state(i, j, k, Z4C_GAMMA_TILDE12) = point.gamma_tilde12;
   state(i, j, k, Z4C_GAMMA_TILDE13) = point.gamma_tilde13;
   state(i, j, k, Z4C_GAMMA_TILDE22) = point.gamma_tilde22;
   state(i, j, k, Z4C_GAMMA_TILDE23) = point.gamma_tilde23;
   state(i, j, k, Z4C_GAMMA_TILDE33) = point.gamma_tilde33;
   state(i, j, k, Z4C_K_HAT) = point.k_hat;
   state(i, j, k, Z4C_A_TILDE11) = point.a_tilde11;
   state(i, j, k, Z4C_A_TILDE12) = point.a_tilde12;
   state(i, j, k, Z4C_A_TILDE13) = point.a_tilde13;
   state(i, j, k, Z4C_A_TILDE22) = point.a_tilde22;
   state(i, j, k, Z4C_A_TILDE23) = point.a_tilde23;
   state(i, j, k, Z4C_A_TILDE33) = point.a_tilde33;
   state(i, j, k, Z4C_GAMMA_TILDE1) = point.gamma_tilde1;
   state(i, j, k, Z4C_GAMMA_TILDE2) = point.gamma_tilde2;
   state(i, j, k, Z4C_GAMMA_TILDE3) = point.gamma_tilde3;
   state(i, j, k, Z4C_THETA) = point.theta;
   state(i, j, k, Z4C_LAPSE) = point.lapse;
   state(i, j, k, Z4C_SHIFT1) = point.shift1;
   state(i, j, k, Z4C_SHIFT2) = point.shift2;
   state(i, j, k, Z4C_SHIFT3) = point.shift3;
}

template <typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void project_z4c_state_at(Array const& state, int i, int j, int k) noexcept
{
   store_z4c_state(
      state,
      i,
      j,
      k,
      generated::z4c_project_algebraic_constraints(
         load_z4c_state(state, i, j, k)
      )
   );
}

template <int Order, int Direction, typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gather_z4c_directional_derivatives(
   Z4cSpatialDerivatives& derivatives,
   Array const& field,
   int i,
   int j,
   int k,
   amrex::Real inv_spacing,
   Z4cState const& state
) noexcept
{
   static_assert(Order == 2 || Order == 4 || Order == 6 || Order == 8);
   static_assert(Direction >= 0 && Direction < 3);
   const amrex::Real shift_speed = Direction == 0
      ? state.shift1
      : Direction == 1 ? state.shift2 : state.shift3;
   const amrex::Real inv_spacing_squared = inv_spacing * inv_spacing;

   derivatives.d_chi[Direction] = generated::centered_first<Order, Direction>(
      field, i, j, k, Z4C_CHI, inv_spacing
   );
   derivatives.advection_chi[Direction] = generated::upwind_first<Order, Direction>(
      field, i, j, k, Z4C_CHI, shift_speed, inv_spacing
   );
   derivatives.dd_chi[Direction][Direction] = generated::centered_second<Order, Direction>(
      field, i, j, k, Z4C_CHI, inv_spacing_squared
   );
   for (int component = 0; component < 6; ++component) {
      derivatives.d_gamma_tilde[Direction][component] =
         generated::centered_first<Order, Direction>(
            field, i, j, k, Z4C_GAMMA_TILDE_COMPONENTS[component], inv_spacing
         );
      derivatives.advection_gamma_tilde[Direction][component] =
         generated::upwind_first<Order, Direction>(
            field,
            i,
            j,
            k,
            Z4C_GAMMA_TILDE_COMPONENTS[component],
            shift_speed,
            inv_spacing
         );
      derivatives.dd_gamma_tilde[Direction][Direction][component] =
         generated::centered_second<Order, Direction>(
            field,
            i,
            j,
            k,
            Z4C_GAMMA_TILDE_COMPONENTS[component],
            inv_spacing_squared
         );
      derivatives.d_a_tilde[Direction][component] =
         generated::centered_first<Order, Direction>(
            field, i, j, k, Z4C_A_TILDE_COMPONENTS[component], inv_spacing
         );
      derivatives.advection_a_tilde[Direction][component] =
         generated::upwind_first<Order, Direction>(
            field,
            i,
            j,
            k,
            Z4C_A_TILDE_COMPONENTS[component],
            shift_speed,
            inv_spacing
         );
   }

   derivatives.d_k_hat[Direction] = generated::centered_first<Order, Direction>(
      field, i, j, k, Z4C_K_HAT, inv_spacing
   );
   derivatives.advection_k_hat[Direction] = generated::upwind_first<Order, Direction>(
      field, i, j, k, Z4C_K_HAT, shift_speed, inv_spacing
   );
   derivatives.d_theta[Direction] = generated::centered_first<Order, Direction>(
      field, i, j, k, Z4C_THETA, inv_spacing
   );
   derivatives.advection_theta[Direction] = generated::upwind_first<Order, Direction>(
      field, i, j, k, Z4C_THETA, shift_speed, inv_spacing
   );
   derivatives.d_lapse[Direction] = generated::centered_first<Order, Direction>(
      field, i, j, k, Z4C_LAPSE, inv_spacing
   );
   derivatives.advection_lapse[Direction] = generated::upwind_first<Order, Direction>(
      field, i, j, k, Z4C_LAPSE, shift_speed, inv_spacing
   );
   derivatives.dd_lapse[Direction][Direction] =
      generated::centered_second<Order, Direction>(
         field, i, j, k, Z4C_LAPSE, inv_spacing_squared
      );
   for (int component = 0; component < 3; ++component) {
      derivatives.d_gamma_tilde_vector[Direction][component] =
         generated::centered_first<Order, Direction>(
            field,
            i,
            j,
            k,
            Z4C_GAMMA_TILDE_VECTOR_COMPONENTS[component],
            inv_spacing
         );
      derivatives.advection_gamma_tilde_vector[Direction][component] =
         generated::upwind_first<Order, Direction>(
            field,
            i,
            j,
            k,
            Z4C_GAMMA_TILDE_VECTOR_COMPONENTS[component],
            shift_speed,
            inv_spacing
         );
      derivatives.d_shift[Direction][component] =
         generated::centered_first<Order, Direction>(
            field, i, j, k, Z4C_SHIFT_COMPONENTS[component], inv_spacing
         );
      derivatives.advection_shift[Direction][component] =
         generated::upwind_first<Order, Direction>(
            field,
            i,
            j,
            k,
            Z4C_SHIFT_COMPONENTS[component],
            shift_speed,
            inv_spacing
         );
      derivatives.dd_shift[Direction][Direction][component] =
         generated::centered_second<Order, Direction>(
            field,
            i,
            j,
            k,
            Z4C_SHIFT_COMPONENTS[component],
            inv_spacing_squared
         );
   }
}

template <int Order, int Direction1, int Direction2, typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gather_z4c_mixed_derivatives(
   Z4cSpatialDerivatives& derivatives,
   Array const& field,
   int i,
   int j,
   int k,
   amrex::Real inv_spacing_product
) noexcept
{
   static_assert(Direction1 >= 0 && Direction1 < Direction2 && Direction2 < 3);
   const amrex::Real dd_chi = generated::centered_mixed<Order, Direction1, Direction2>(
      field, i, j, k, Z4C_CHI, inv_spacing_product
   );
   derivatives.dd_chi[Direction1][Direction2] = dd_chi;
   derivatives.dd_chi[Direction2][Direction1] = dd_chi;
   const amrex::Real dd_lapse = generated::centered_mixed<Order, Direction1, Direction2>(
      field, i, j, k, Z4C_LAPSE, inv_spacing_product
   );
   derivatives.dd_lapse[Direction1][Direction2] = dd_lapse;
   derivatives.dd_lapse[Direction2][Direction1] = dd_lapse;
   for (int component = 0; component < 6; ++component) {
      const amrex::Real value = generated::centered_mixed<Order, Direction1, Direction2>(
         field,
         i,
         j,
         k,
         Z4C_GAMMA_TILDE_COMPONENTS[component],
         inv_spacing_product
      );
      derivatives.dd_gamma_tilde[Direction1][Direction2][component] = value;
      derivatives.dd_gamma_tilde[Direction2][Direction1][component] = value;
   }
   for (int component = 0; component < 3; ++component) {
      const amrex::Real value = generated::centered_mixed<Order, Direction1, Direction2>(
         field,
         i,
         j,
         k,
         Z4C_SHIFT_COMPONENTS[component],
         inv_spacing_product
      );
      derivatives.dd_shift[Direction1][Direction2][component] = value;
      derivatives.dd_shift[Direction2][Direction1][component] = value;
   }
}

template <int Order, int Dim, typename Array, typename SpacingArray>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cSpatialDerivatives gather_z4c_spatial_derivatives(
   Array const& field,
   int i,
   int j,
   int k,
   SpacingArray const& inv_spacing,
   Z4cState const& state
) noexcept
{
   static_assert(Dim >= 1 && Dim <= 3);
   Z4cSpatialDerivatives derivatives{};
   gather_z4c_directional_derivatives<Order, 0>(
      derivatives, field, i, j, k, inv_spacing[0], state
   );
   if constexpr (Dim >= 2) {
      gather_z4c_directional_derivatives<Order, 1>(
         derivatives, field, i, j, k, inv_spacing[1], state
      );
      gather_z4c_mixed_derivatives<Order, 0, 1>(
         derivatives, field, i, j, k, inv_spacing[0] * inv_spacing[1]
      );
   }
   if constexpr (Dim == 3) {
      gather_z4c_directional_derivatives<Order, 2>(
         derivatives, field, i, j, k, inv_spacing[2], state
      );
      gather_z4c_mixed_derivatives<Order, 0, 2>(
         derivatives, field, i, j, k, inv_spacing[0] * inv_spacing[2]
      );
      gather_z4c_mixed_derivatives<Order, 1, 2>(
         derivatives, field, i, j, k, inv_spacing[1] * inv_spacing[2]
      );
   }
   return derivatives;
}

template <int Order, int Dim, typename Array, typename SpacingArray>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cConstraints evaluate_z4c_constraints_at(
   Array const& field,
   int i,
   int j,
   int k,
   SpacingArray const& inv_spacing,
   Z4cMatter const& matter
) noexcept
{
   const Z4cState state = load_z4c_state(field, i, j, k);
   const Z4cSpatialDerivatives derivatives =
      gather_z4c_spatial_derivatives<Order, Dim>(
         field, i, j, k, inv_spacing, state
      );
   return generated::evaluate_z4c_constraints(state, derivatives, matter);
}

template <int Order, int Dim, typename Array, typename SpacingArray>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cConstraints evaluate_vacuum_z4c_constraints_at(
   Array const& field,
   int i,
   int j,
   int k,
   SpacingArray const& inv_spacing
) noexcept
{
   return evaluate_z4c_constraints_at<Order, Dim>(
      field, i, j, k, inv_spacing, vacuum_z4c_matter()
   );
}

template <
   int Order,
   int Dim,
   Z4cLapseGauge LapseGauge,
   Z4cShiftGauge ShiftGauge,
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
   Z4cEvolutionParameters const& parameters
) noexcept
{
   const Z4cState state = load_z4c_state(field, i, j, k);
   const Z4cSpatialDerivatives derivatives =
      gather_z4c_spatial_derivatives<Order, Dim>(
         field, i, j, k, inv_spacing, state
      );
   return generated::evaluate_z4c_rhs<LapseGauge, ShiftGauge>(
      state, derivatives, matter, time, parameters
   );
}

template <
   int Order,
   int Dim,
   Z4cLapseGauge LapseGauge,
   Z4cShiftGauge ShiftGauge,
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
   Z4cEvolutionParameters const& parameters
) noexcept
{
   return evaluate_z4c_rhs_at<Order, Dim, LapseGauge, ShiftGauge>(
      field,
      i,
      j,
      k,
      inv_spacing,
      vacuum_z4c_matter(),
      time,
      parameters
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
