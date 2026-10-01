#pragma once

#include <AMReX_Array4.H>

#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/flux.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void black_hole_excision_left_cell(
   int i,
   int j,
   int k,
   int& il,
   int& jl,
   int& kl
) noexcept
{
   il = i;
   jl = j;
   kl = k;
   if constexpr (Direction == 0) {
      --il;
   }
#if (AMREX_SPACEDIM >= 2)
   else if constexpr (Direction == 1) {
      --jl;
   }
#endif
#if (AMREX_SPACEDIM == 3)
   else {
      --kl;
   }
#endif
}

template <int Direction, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDConserved black_hole_excision_rhd_flux(
   amrex::Array4<amrex::Real const> const& primitive,
   amrex::Array4<int const> const& mask,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   MetricAccessor const& metric,
   bool& replaced
) noexcept
{
   int il;
   int jl;
   int kl;
   black_hole_excision_left_cell<Direction>(i, j, k, il, jl, kl);
   const bool left_excised = mask(il, jl, kl, 0) != 0;
   const bool right_excised = mask(i, j, k, 0) != 0;
   replaced = left_excised || right_excised;
   if (!replaced || (left_excised && right_excised)) {
      return {};
   }
   const int ie = left_excised ? i : il;
   const int je = left_excised ? j : jl;
   const int ke = left_excised ? k : kl;
   RHDPrimitive exterior = rhd_primitive_at<EOSRuntime>(
      primitive, ie, je, ke, eos
   );
   const SpacetimeMetric face_metric = metric.template face<Direction>(i, j, k);
   const EulerianVelocity face_velocity =
      eulerian_velocity_from_spatial_four_velocity(
         spatial_four_velocity_from_eulerian(
            exterior.v1,
            exterior.v2,
            exterior.v3,
            metric.cell(ie, je, ke)
         ),
         face_metric
      );
   exterior.v1 = face_velocity.v1;
   exterior.v2 = face_velocity.v2;
   exterior.v3 = face_velocity.v3;
   RHDConserved flux = generated::rhd_flux<Direction>(exterior, face_metric);
   const bool enters_active_domain =
      left_excised ? flux.dens > 0.0_rt : flux.dens < 0.0_rt;
   return enters_active_domain ? RHDConserved{} : flux;
}

template <int Direction, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDConserved black_hole_excision_rmhd_flux(
   amrex::Array4<amrex::Real const> const& primitive,
   amrex::Array4<int const> const& mask,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   MetricAccessor const& metric,
   bool& replaced
) noexcept
{
   int il;
   int jl;
   int kl;
   black_hole_excision_left_cell<Direction>(i, j, k, il, jl, kl);
   const bool left_excised = mask(il, jl, kl, 0) != 0;
   const bool right_excised = mask(i, j, k, 0) != 0;
   replaced = left_excised || right_excised;
   if (!replaced || (left_excised && right_excised)) {
      return {};
   }
   const int ie = left_excised ? i : il;
   const int je = left_excised ? j : jl;
   const int ke = left_excised ? k : kl;
   RMHDPrimitive exterior = rmhd_primitive_at<EOSRuntime>(
      primitive, ie, je, ke, eos
   );
   const SpacetimeMetric face_metric = metric.template face<Direction>(i, j, k);
   const EulerianVelocity face_velocity =
      eulerian_velocity_from_spatial_four_velocity(
         spatial_four_velocity_from_eulerian(
            exterior.v1,
            exterior.v2,
            exterior.v3,
            metric.cell(ie, je, ke)
         ),
         face_metric
      );
   exterior.v1 = face_velocity.v1;
   exterior.v2 = face_velocity.v2;
   exterior.v3 = face_velocity.v3;
   RMHDConserved flux = generated::rmhd_flux<Direction>(exterior, face_metric);
   const bool enters_active_domain =
      left_excised ? flux.dens > 0.0_rt : flux.dens < 0.0_rt;
   return enters_active_domain ? RMHDConserved{} : flux;
}

} // namespace pyfluid_amrex
