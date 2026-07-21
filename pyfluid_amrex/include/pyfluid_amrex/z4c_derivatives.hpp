#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/kernels_generated/differential_ops.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {

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
   static_assert(Order == 2 || Order == 4 || Order == 6 || Order == 8);
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
            Z4C_GAMMA_TILDE_COMPONENTS[component],
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

} // namespace pyfluid_amrex
