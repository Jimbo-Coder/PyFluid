#pragma once

#include "relativity/rmhd_correction_flux.hpp"
#include "relativity/relativistic_limiter_index.hpp"
#include "relativity/relativistic_limiter_metric.hpp"
#include "relativity/rmhd_system.hpp"
#include "pyfluid_amrex/rmhd_riemann.hpp"
#include "relativity/black_hole_excision_kernels.hpp"

#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuLaunch.H>

namespace pyfluid_amrex::limiter_detail {

template <int RiemannMethod, int Direction, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRiemannFluxResult rmhd_correction_flux(
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   MetricAccessor const& metric,
   EOSThermodynamicState const& atmosphere,
   amrex::Array4<int const> const& excision_mask,
   bool use_excision_mask,
   bool use_pwl
) noexcept
{
   if (use_excision_mask) {
      bool replaced = false;
      const RMHDConserved replacement =
         black_hole_excision_rmhd_flux<Direction>(
            primitive, excision_mask, i, j, k, eos, metric, replaced
         );
      if (replaced) {
         return {replacement, false};
      }
   }
   int il;
   int jl;
   int kl;
   left_cell<Direction>(i, j, k, il, jl, kl);
   // Both correction orders solve the same Riemann problem after their
   // face states are selected. Keep the expensive fan as one call site.
   const RMHDPrimitiveFaceStates face = use_pwl
      ? rmhd_primitive_face_states<ReconPWL, Direction>(
           primitive, il, jl, kl, i, j, k, eos, metric, atmosphere, atmosphere
        )
      : rmhd_primitive_face_states<ReconPWC, Direction>(
           primitive, il, jl, kl, i, j, k, eos, metric, atmosphere, atmosphere
        );
   const SpacetimeMetric face_metric = metric.template face<Direction>(i, j, k);
   if constexpr (RiemannMethod == RiemannHLLC) {
      return rmhd_hllc_flux<Direction>(face.left, face.right, eos, face_metric);
   } else if constexpr (RiemannMethod == RiemannHLLD) {
      return rmhd_hlld_flux<Direction>(face.left, face.right, eos, face_metric);
   } else if constexpr (RiemannMethod == RiemannLLF) {
      return {rmhd_llf_flux<Direction>(face.left, face.right, eos, face_metric), false};
   } else {
      return {rmhd_hll_flux<Direction>(face.left, face.right, eos, face_metric), false};
   }
}

template <int RiemannMethod, int Direction, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDConserved selected_rmhd_flux(
   int correction_kind,
   RMHDConserved const& high,
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   MetricAccessor const& metric,
   EOSThermodynamicState const& atmosphere,
   amrex::Array4<int const> const& excision_mask,
   bool use_excision_mask,
   amrex::Long* hll_fallback_count = nullptr
) noexcept
{
   if (correction_kind == 0) {
      return high;
   }
   const RMHDRiemannFluxResult result = rmhd_correction_flux<RiemannMethod, Direction>(
      primitive, i, j, k, eos, metric, atmosphere,
      excision_mask, use_excision_mask, correction_kind == 1
   );
   if (hll_fallback_count != nullptr && result.used_hll) {
      amrex::HostDevice::Atomic::Add(
         hll_fallback_count, static_cast<amrex::Long>(1)
      );
   }
   return result.flux;
}

// Both correction passes use the same reconstructed face states and fan.
// Select the launch mode uniformly; keep different Riemann methods in their
// own kernels so cheap solvers retain their independent resource footprint.
template <int RiemannMethod, int Direction, typename MetricAccessor>
void launch_rmhd_correction_direction(
   RmhdCorrectionFluxContext const& context,
   MetricAccessor const& metric,
   bool pwl_only
)
{
   const auto box = context.box;
   const auto output = context.output;
   const auto high_flux = context.high_flux;
   const auto primitive = context.primitive;
   const auto correction = context.correction;
   const auto eos = context.eos;
   const auto atmosphere = context.atmosphere;
   const auto excision_mask = context.excision_mask;
   const auto use_excision_mask = context.use_excision_mask;
   const auto hll_fallback_count = context.hll_fallback_count;
   amrex::ParallelFor(context.face_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      int il, jl, kl;
      left_cell<Direction>(i, j, k, il, jl, kl);
      int kind;
      RMHDConserved high{};
      if (pwl_only) {
         // Classification belongs to this tile until mask synchronization.
         const bool left_troubled = box.contains(amrex::IntVect(AMREX_D_DECL(il, jl, kl)))
            && correction(il, jl, kl, 0) == 2;
         const bool right_troubled = box.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))
            && correction(i, j, k, 0) == 2;
         if (!left_troubled && !right_troubled) return;
         kind = 1;
      } else {
         kind = amrex::max(correction(il, jl, kl, 0), correction(i, j, k, 0));
         high = load_rmhd_conserved(high_flux, i, j, k);
      }
      store_rmhd_conserved(output, i, j, k,
         selected_rmhd_flux<RiemannMethod, Direction>(
            kind, high, primitive, i, j, k, eos, metric, atmosphere,
            excision_mask, use_excision_mask, hll_fallback_count
         ));
   });
}

template <int RiemannMethod, typename MetricAccessor>
void launch_rmhd_correction_fluxes(
   int direction,
   RmhdCorrectionFluxContext const& context,
   MetricAccessor const& metric,
   bool pwl_only
)
{
   switch (direction) {
   case 0:
      launch_rmhd_correction_direction<RiemannMethod, 0>(context, metric, pwl_only);
      return;
#if AMREX_SPACEDIM >= 2
   case 1:
      launch_rmhd_correction_direction<RiemannMethod, 1>(context, metric, pwl_only);
      return;
#endif
#if AMREX_SPACEDIM == 3
   case 2:
      launch_rmhd_correction_direction<RiemannMethod, 2>(context, metric, pwl_only);
      return;
#endif
   default:
      amrex::Abort("invalid RMHD correction flux direction");
   }
}

} // namespace pyfluid_amrex::limiter_detail
