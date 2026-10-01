#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/kernels_generated/differential_ops.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {

// Constraints depend on spatial geometry and curvature, but not on gauge
// derivatives or transport. Keep both consumers on the same stencil assembly.
enum class Z4cDerivativePurpose { Evolution, Constraints };

template <int Order>
struct CenteredZ4cDerivativeOperator {
   template <int Direction, typename Array>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real first(Array const& field, int i, int j, int k, int component,
                    amrex::Real scale) const noexcept
   {
      return generated::centered_first<Order, Direction>(
         field, i, j, k, component, scale);
   }

   template <int Direction, typename Array>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real second(Array const& field, int i, int j, int k, int component,
                     amrex::Real scale) const noexcept
   {
      return generated::centered_second<Order, Direction>(
         field, i, j, k, component, scale);
   }

   template <int Direction, typename Array>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real advection(Array const& field, int i, int j, int k, int component,
                        amrex::Real speed, amrex::Real scale) const noexcept
   {
      // Z4c has +beta*d_i u on the RHS, hence transport velocity -beta.
      // The generic upwind stencil accepts the transport (LHS) velocity.
      return generated::upwind_first<Order, Direction>(
         field, i, j, k, component, -speed, scale);
   }

   template <int Direction1, int Direction2, typename Array>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real mixed(Array const& field, int i, int j, int k, int component,
                    amrex::Real scale) const noexcept
   {
      return generated::centered_mixed<Order, Direction1, Direction2>(
         field, i, j, k, component, scale);
   }
};

// Connection geometry needs only first derivatives of chi and the conformal
// metric. Share this assembly with the full evolution gather so initialization
// never reads shift, curvature, or the connection components it is writing.
template <int Order, int Direction, typename Array,
          typename DerivativeOperator = CenteredZ4cDerivativeOperator<Order>>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gather_z4c_metric_directional_derivatives(
   Z4cSpatialDerivatives& derivatives, Array const& field,
   int i, int j, int k, amrex::Real inv_spacing,
   DerivativeOperator const& derivative_operator = {}) noexcept
{
   static_assert(is_supported_spatial_order(Order));
   static_assert(Direction >= 0 && Direction < 3);
   derivatives.d_chi[Direction] = derivative_operator.template first<Direction>(
      field, i, j, k, Z4C_CHI, inv_spacing);
   for (int component = 0; component < 6; ++component) {
      derivatives.d_gamma_tilde[Direction][component] =
         derivative_operator.template first<Direction>(
            field, i, j, k, z4c_gamma_tilde_components()[component], inv_spacing);
   }
}

template <int Order, int Dim, typename Array, typename SpacingArray,
          typename DerivativeOperator = CenteredZ4cDerivativeOperator<Order>>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cSpatialDerivatives gather_z4c_metric_derivatives(
   Array const& field, int i, int j, int k, SpacingArray const& inv_spacing,
   DerivativeOperator const& derivative_operator = {}) noexcept
{
   static_assert(Dim >= 1 && Dim <= 3);
   Z4cSpatialDerivatives derivatives{};
   gather_z4c_metric_directional_derivatives<Order, 0>(
      derivatives, field, i, j, k, inv_spacing[0], derivative_operator);
   if constexpr (Dim >= 2) {
      gather_z4c_metric_directional_derivatives<Order, 1>(
         derivatives, field, i, j, k, inv_spacing[1], derivative_operator);
   }
   if constexpr (Dim == 3) {
      gather_z4c_metric_directional_derivatives<Order, 2>(
         derivatives, field, i, j, k, inv_spacing[2], derivative_operator);
   }
   return derivatives;
}

template <int Order, int Direction, typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gather_z4c_geometry_directional_derivatives(
   Z4cSpatialDerivatives& derivatives,
   Array const& field,
   int i,
   int j,
   int k,
   int coordinate_index,
   int domain_lower,
   int domain_upper,
   amrex::Real inv_spacing
) noexcept
{
   static_assert(is_supported_spatial_order(Order));
   static_assert(Direction >= 0 && Direction < 3);
   derivatives.d_chi[Direction] = generated::domain_first<Order, Direction>(
      field,
      i,
      j,
      k,
      Z4C_CHI,
      coordinate_index,
      domain_lower,
      domain_upper,
      inv_spacing
   );
   derivatives.d_lapse[Direction] = generated::domain_first<Order, Direction>(
      field,
      i,
      j,
      k,
      Z4C_LAPSE,
      coordinate_index,
      domain_lower,
      domain_upper,
      inv_spacing
   );
   for (int component = 0; component < 6; ++component) {
      derivatives.d_gamma_tilde[Direction][component] =
         generated::domain_first<Order, Direction>(
            field,
            i,
            j,
            k,
            z4c_gamma_tilde_components()[component],
            coordinate_index,
            domain_lower,
            domain_upper,
            inv_spacing
         );
   }
}

template <int Order, int Dim, typename Array, typename SpacingArray, typename IndexArray>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cSpatialDerivatives gather_z4c_geometry_derivatives(
   Array const& field,
   int i,
   int j,
   int k,
   SpacingArray const& inv_spacing,
   IndexArray const& domain_lower,
   IndexArray const& domain_upper
) noexcept
{
   static_assert(Dim >= 1 && Dim <= 3);
   Z4cSpatialDerivatives derivatives{};
   gather_z4c_geometry_directional_derivatives<Order, 0>(
      derivatives,
      field,
      i,
      j,
      k,
      i,
      domain_lower[0],
      domain_upper[0],
      inv_spacing[0]
   );
   if constexpr (Dim >= 2) {
      gather_z4c_geometry_directional_derivatives<Order, 1>(
         derivatives,
         field,
         i,
         j,
         k,
         j,
         domain_lower[1],
         domain_upper[1],
         inv_spacing[1]
      );
   }
   if constexpr (Dim == 3) {
      gather_z4c_geometry_directional_derivatives<Order, 2>(
         derivatives,
         field,
         i,
         j,
         k,
         k,
         domain_lower[2],
         domain_upper[2],
         inv_spacing[2]
      );
   }
   return derivatives;
}

template <int Order, int Direction,
          Z4cDerivativePurpose Purpose = Z4cDerivativePurpose::Evolution, typename Array,
          typename DerivativeOperator = CenteredZ4cDerivativeOperator<Order>>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gather_z4c_directional_derivatives(
   Z4cSpatialDerivatives& derivatives,
   Array const& field,
   int i,
   int j,
   int k,
   amrex::Real inv_spacing,
   Z4cState const& state,
   DerivativeOperator const& derivative_operator = {}
) noexcept
{
   static_assert(is_supported_spatial_order(Order));
   static_assert(Direction >= 0 && Direction < 3);
   const amrex::Real inv_spacing_squared = inv_spacing * inv_spacing;

   gather_z4c_metric_directional_derivatives<Order, Direction>(
      derivatives, field, i, j, k, inv_spacing, derivative_operator);
   derivatives.dd_chi[Direction][Direction] = derivative_operator.template second<Direction>(
      field, i, j, k, Z4C_CHI, inv_spacing_squared
   );
   for (int component = 0; component < 6; ++component) {
      derivatives.dd_gamma_tilde[Direction][Direction][component] =
         derivative_operator.template second<Direction>(
            field,
            i,
            j,
            k,
            z4c_gamma_tilde_components()[component],
            inv_spacing_squared
         );
      derivatives.d_a_tilde[Direction][component] =
         derivative_operator.template first<Direction>(
            field, i, j, k, z4c_a_tilde_components()[component], inv_spacing
         );
   }

   derivatives.d_k_hat[Direction] = derivative_operator.template first<Direction>(
      field, i, j, k, Z4C_K_HAT, inv_spacing
   );
   derivatives.d_theta[Direction] = derivative_operator.template first<Direction>(
      field, i, j, k, Z4C_THETA, inv_spacing
   );

   if constexpr (Purpose == Z4cDerivativePurpose::Evolution) {
      const amrex::Real shift_speed = Direction == 0
         ? state.shift1
         : Direction == 1 ? state.shift2 : state.shift3;
      derivatives.advection_chi[Direction] = derivative_operator.template advection<Direction>(
         field, i, j, k, Z4C_CHI, shift_speed, inv_spacing
      );
      derivatives.advection_k_hat[Direction] = derivative_operator.template advection<Direction>(
         field, i, j, k, Z4C_K_HAT, shift_speed, inv_spacing
      );
      for (int component = 0; component < 6; ++component) {
         derivatives.advection_gamma_tilde[Direction][component] =
            derivative_operator.template advection<Direction>(
               field, i, j, k, z4c_gamma_tilde_components()[component], shift_speed, inv_spacing);
         derivatives.advection_a_tilde[Direction][component] =
            derivative_operator.template advection<Direction>(
               field, i, j, k, z4c_a_tilde_components()[component], shift_speed, inv_spacing);
      }
      derivatives.advection_theta[Direction] = derivative_operator.template advection<Direction>(
         field, i, j, k, Z4C_THETA, shift_speed, inv_spacing
      );
      derivatives.d_lapse[Direction] = derivative_operator.template first<Direction>(
         field, i, j, k, Z4C_LAPSE, inv_spacing
      );
      derivatives.advection_lapse[Direction] = derivative_operator.template advection<Direction>(
         field, i, j, k, Z4C_LAPSE, shift_speed, inv_spacing
      );
      derivatives.dd_lapse[Direction][Direction] =
         derivative_operator.template second<Direction>(
            field, i, j, k, Z4C_LAPSE, inv_spacing_squared
         );
      for (int component = 0; component < 3; ++component) {
         derivatives.d_gamma_tilde_vector[Direction][component] =
            derivative_operator.template first<Direction>(
               field,
               i,
               j,
               k,
               z4c_gamma_tilde_vector_components()[component],
               inv_spacing
            );
         derivatives.advection_gamma_tilde_vector[Direction][component] =
            derivative_operator.template advection<Direction>(
               field,
               i,
               j,
               k,
               z4c_gamma_tilde_vector_components()[component],
               shift_speed,
               inv_spacing
            );
         derivatives.d_shift[Direction][component] =
            derivative_operator.template first<Direction>(
               field, i, j, k, z4c_shift_components()[component], inv_spacing
            );
         derivatives.advection_shift[Direction][component] =
            derivative_operator.template advection<Direction>(
               field,
               i,
               j,
               k,
               z4c_shift_components()[component],
               shift_speed,
               inv_spacing
            );
         derivatives.dd_shift[Direction][Direction][component] =
            derivative_operator.template second<Direction>(
               field,
               i,
               j,
               k,
               z4c_shift_components()[component],
               inv_spacing_squared
            );
      }
   }
}

template <int Order, int Direction1, int Direction2,
          Z4cDerivativePurpose Purpose = Z4cDerivativePurpose::Evolution, typename Array,
          typename DerivativeOperator = CenteredZ4cDerivativeOperator<Order>>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gather_z4c_mixed_derivatives(
   Z4cSpatialDerivatives& derivatives,
   Array const& field,
   int i,
   int j,
   int k,
   amrex::Real inv_spacing_product,
   DerivativeOperator const& derivative_operator = {}
) noexcept
{
   static_assert(Direction1 >= 0 && Direction1 < Direction2 && Direction2 < 3);
   const amrex::Real dd_chi = derivative_operator.template mixed<Direction1, Direction2>(
      field, i, j, k, Z4C_CHI, inv_spacing_product
   );
   derivatives.dd_chi[Direction1][Direction2] = dd_chi;
   derivatives.dd_chi[Direction2][Direction1] = dd_chi;
   for (int component = 0; component < 6; ++component) {
      const amrex::Real value = derivative_operator.template mixed<Direction1, Direction2>(
         field,
         i,
         j,
         k,
         z4c_gamma_tilde_components()[component],
         inv_spacing_product
      );
      derivatives.dd_gamma_tilde[Direction1][Direction2][component] = value;
      derivatives.dd_gamma_tilde[Direction2][Direction1][component] = value;
   }
   if constexpr (Purpose == Z4cDerivativePurpose::Evolution) {
      const amrex::Real dd_lapse = derivative_operator.template mixed<Direction1, Direction2>(
         field, i, j, k, Z4C_LAPSE, inv_spacing_product
      );
      derivatives.dd_lapse[Direction1][Direction2] = dd_lapse;
      derivatives.dd_lapse[Direction2][Direction1] = dd_lapse;
      for (int component = 0; component < 3; ++component) {
         const amrex::Real value = derivative_operator.template mixed<Direction1, Direction2>(
            field,
            i,
            j,
            k,
            z4c_shift_components()[component],
            inv_spacing_product
         );
         derivatives.dd_shift[Direction1][Direction2][component] = value;
         derivatives.dd_shift[Direction2][Direction1][component] = value;
      }
   }
}

template <int Order, int Dim,
          Z4cDerivativePurpose Purpose = Z4cDerivativePurpose::Evolution,
          typename Array, typename SpacingArray,
          typename DerivativeOperator = CenteredZ4cDerivativeOperator<Order>>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cSpatialDerivatives gather_z4c_spatial_derivatives(
   Array const& field,
   int i,
   int j,
   int k,
   SpacingArray const& inv_spacing,
   Z4cState const& state,
   DerivativeOperator const& derivative_operator = {}
) noexcept
{
   static_assert(Dim >= 1 && Dim <= 3);
   Z4cSpatialDerivatives derivatives{};
   gather_z4c_directional_derivatives<Order, 0, Purpose>(
      derivatives, field, i, j, k, inv_spacing[0], state, derivative_operator
   );
   if constexpr (Dim >= 2) {
      gather_z4c_directional_derivatives<Order, 1, Purpose>(
         derivatives, field, i, j, k, inv_spacing[1], state, derivative_operator
      );
      gather_z4c_mixed_derivatives<Order, 0, 1, Purpose>(
         derivatives, field, i, j, k, inv_spacing[0] * inv_spacing[1], derivative_operator
      );
   }
   if constexpr (Dim == 3) {
      gather_z4c_directional_derivatives<Order, 2, Purpose>(
         derivatives, field, i, j, k, inv_spacing[2], state, derivative_operator
      );
      gather_z4c_mixed_derivatives<Order, 0, 2, Purpose>(
         derivatives, field, i, j, k, inv_spacing[0] * inv_spacing[2], derivative_operator
      );
      gather_z4c_mixed_derivatives<Order, 1, 2, Purpose>(
         derivatives, field, i, j, k, inv_spacing[1] * inv_spacing[2], derivative_operator
      );
   }
   return derivatives;
}

} // namespace pyfluid_amrex
