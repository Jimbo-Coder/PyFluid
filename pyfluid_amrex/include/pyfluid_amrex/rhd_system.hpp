#pragma once

#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic_fluid.hpp"
#include "pyfluid_amrex/kernels_generated/riemann_ops.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/reconstruction.hpp"
#include "pyfluid_amrex/z4c_pointwise.hpp"
#include "pyfluid_amrex/state.hpp"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include <limits>

namespace pyfluid_amrex {

struct RHDPrimitiveFaceStates {
   RHDPrimitive left;
   RHDPrimitive right;
};

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDPrimitive rhd_primitive_at(
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   const EOSParams& eos
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
amrex::Real causal_limiter_theta(
   const RHDPrimitive& face,
   const RHDPrimitive& center,
   const SpacetimeMetric& metric
) noexcept
{
   constexpr amrex::Real velocity_limit_squared =
      1.0 - 128.0 * std::numeric_limits<amrex::Real>::epsilon();
   const amrex::Real face_velocity_squared = spatial_vector_norm_squared(
      face.v1, face.v2, face.v3, metric
   );
   if (amrex::Math::isfinite(face_velocity_squared)
       && face_velocity_squared < velocity_limit_squared) {
      return 1.0;
   }
   const amrex::Real dv1 = face.v1 - center.v1;
   const amrex::Real dv2 = face.v2 - center.v2;
   const amrex::Real dv3 = face.v3 - center.v3;
   const amrex::Real quadratic = spatial_vector_norm_squared(dv1, dv2, dv3, metric);
   const amrex::Real center_delta = center.v1 * (
      metric.gamma11 * dv1 + metric.gamma12 * dv2 + metric.gamma13 * dv3
   ) + center.v2 * (
      metric.gamma12 * dv1 + metric.gamma22 * dv2 + metric.gamma23 * dv3
   ) + center.v3 * (
      metric.gamma13 * dv1 + metric.gamma23 * dv2 + metric.gamma33 * dv3
   );
   const amrex::Real linear = 2.0 * center_delta;
   const amrex::Real constant = spatial_vector_norm_squared(
      center.v1, center.v2, center.v3, metric
   ) - velocity_limit_squared;
   if (!(quadratic > 0.0) || !amrex::Math::isfinite(quadratic)
       || !amrex::Math::isfinite(linear) || !amrex::Math::isfinite(constant)) {
      return 0.0;
   }
   const amrex::Real discriminant = linear * linear - 4.0 * quadratic * constant;
   if (!(discriminant >= 0.0) || !amrex::Math::isfinite(discriminant)) {
      return 0.0;
   }
   return amrex::max(
      0.0,
      amrex::min(1.0, (-linear + std::sqrt(discriminant)) / (2.0 * quadratic))
   );
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDPrimitive admissibility_limit_rhd_primitive(
   RHDPrimitive face,
   const RHDPrimitive& center,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   amrex::Real theta = positivity_limiter_theta(face.rho, center.rho, eos.rho_floor);
   theta = amrex::min(
      theta,
      positivity_limiter_theta(face.pressure, center.pressure, eos.pressure_floor)
   );
   theta = amrex::min(theta, causal_limiter_theta(face, center, metric));
   face.rho = center.rho + theta * (face.rho - center.rho);
   face.pressure = center.pressure + theta * (face.pressure - center.pressure);
   face.v1 = center.v1 + theta * (face.v1 - center.v1);
   face.v2 = center.v2 + theta * (face.v2 - center.v2);
   face.v3 = center.v3 + theta * (face.v3 - center.v3);
   face.eps = relativistic_specific_internal_energy_from_pressure<EOSMethod>(
      face.rho, face.pressure, eos
   );
   return face;
}

template <int ReconMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDPrimitiveFaceStates rhd_primitive_face_states(
   amrex::Array4<amrex::Real const> const& primitive,
   int il,
   int jl,
   int kl,
   int ir,
   int jr,
   int kr,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const ScalarFaceValues rho = primitive_component_face_values<ReconMethod, Direction, RHO>(
      primitive, il, jl, kl, ir, jr, kr
   );
   const ScalarFaceValues v1 = primitive_component_face_values<ReconMethod, Direction, MOM1>(
      primitive, il, jl, kl, ir, jr, kr
   );
   const ScalarFaceValues v2 = primitive_component_face_values<ReconMethod, Direction, MOM2>(
      primitive, il, jl, kl, ir, jr, kr
   );
   const ScalarFaceValues v3 = primitive_component_face_values<ReconMethod, Direction, MOM3>(
      primitive, il, jl, kl, ir, jr, kr
   );
   const ScalarFaceValues pressure = primitive_component_face_values<ReconMethod, Direction, ENERGY>(
      primitive, il, jl, kl, ir, jr, kr
   );
   RHDPrimitiveFaceStates face{
      {rho.left, 0.0, pressure.left, v1.left, v2.left, v3.left},
      {rho.right, 0.0, pressure.right, v1.right, v2.right, v3.right},
   };
   face.left = admissibility_limit_rhd_primitive<EOSMethod>(
      face.left,
      rhd_primitive_at<EOSMethod>(primitive, il, jl, kl, eos),
      eos,
      metric
   );
   face.right = admissibility_limit_rhd_primitive<EOSMethod>(
      face.right,
      rhd_primitive_at<EOSMethod>(primitive, ir, jr, kr, eos),
      eos,
      metric
   );
   return face;
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDConserved rhd_hll_flux(
   const RHDPrimitive& primitive_l,
   const RHDPrimitive& primitive_r,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const RHDConserved state_l = generated::rhd_primitive_to_conserved(primitive_l, metric);
   const RHDConserved state_r = generated::rhd_primitive_to_conserved(primitive_r, metric);
   const RHDConserved flux_l = generated::rhd_flux<Direction>(primitive_l, metric);
   const RHDConserved flux_r = generated::rhd_flux<Direction>(primitive_r, metric);
   const RHDWaveSpeeds speed_l = generated::rhd_wave_speeds<Direction>(
      primitive_l, metric,
      relativistic_sound_speed_squared<EOSMethod>(primitive_l.rho, primitive_l.eps, eos)
   );
   const RHDWaveSpeeds speed_r = generated::rhd_wave_speeds<Direction>(
      primitive_r, metric,
      relativistic_sound_speed_squared<EOSMethod>(primitive_r.rho, primitive_r.eps, eos)
   );
   const amrex::Real speed_minus = amrex::min(0.0, amrex::min(speed_l.minus, speed_r.minus));
   const amrex::Real speed_plus = amrex::max(0.0, amrex::max(speed_l.plus, speed_r.plus));
   if (speed_minus >= 0.0) {
      return flux_l;
   }
   if (speed_plus <= 0.0) {
      return flux_r;
   }
   return generated::rhd_hll_flux(
      flux_l, flux_r, state_l, state_r, speed_minus, speed_plus
   );
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDConserved rhd_llf_flux(
   const RHDPrimitive& primitive_l,
   const RHDPrimitive& primitive_r,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const RHDConserved state_l = generated::rhd_primitive_to_conserved(primitive_l, metric);
   const RHDConserved state_r = generated::rhd_primitive_to_conserved(primitive_r, metric);
   const RHDConserved flux_l = generated::rhd_flux<Direction>(primitive_l, metric);
   const RHDConserved flux_r = generated::rhd_flux<Direction>(primitive_r, metric);
   const RHDWaveSpeeds speed_l = generated::rhd_wave_speeds<Direction>(
      primitive_l, metric,
      relativistic_sound_speed_squared<EOSMethod>(primitive_l.rho, primitive_l.eps, eos)
   );
   const RHDWaveSpeeds speed_r = generated::rhd_wave_speeds<Direction>(
      primitive_r, metric,
      relativistic_sound_speed_squared<EOSMethod>(primitive_r.rho, primitive_r.eps, eos)
   );
   const amrex::Real wavespeed = amrex::max(
      amrex::max(amrex::Math::abs(speed_l.minus), amrex::Math::abs(speed_l.plus)),
      amrex::max(amrex::Math::abs(speed_r.minus), amrex::Math::abs(speed_r.plus))
   );
   return generated::rhd_llf_flux(flux_l, flux_r, state_l, state_r, wavespeed);
}

template <int ReconMethod, int RiemannMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDConserved rhd_riemann_flux(
   amrex::Array4<amrex::Real const> const& primitive,
   int il,
   int jl,
   int kl,
   int ir,
   int jr,
   int kr,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const RHDPrimitiveFaceStates face = rhd_primitive_face_states<ReconMethod, Direction, EOSMethod>(
      primitive, il, jl, kl, ir, jr, kr, eos, metric
   );
   if constexpr (RiemannMethod == RiemannLLF) {
      return rhd_llf_flux<Direction, EOSMethod>(face.left, face.right, eos, metric);
   } else if constexpr (RiemannMethod == RiemannHLL || RiemannMethod == RiemannHLLE) {
      return rhd_hll_flux<Direction, EOSMethod>(face.left, face.right, eos, metric);
   } else {
      AMREX_ASSERT_WITH_MESSAGE(false, "unsupported RHD Riemann solver reached a device kernel");
      return {};
   }
}

template <int ReconMethod, int RiemannMethod, int Direction, int EOSMethod>
inline void fill_rhd_flux_direction(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   const amrex::MultiFab& primitive,
   const EOSParams& eos,
   const amrex::Geometry& geometry,
   const FixedSpacetimeParameters& spacetime,
   amrex::Real time
)
{
   amrex::MultiFab& flux = *fluxes[Direction];
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto cached = primitive.const_array(mfi);
      const auto face_flux = flux.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         int il = i;
         int jl = j;
         int kl = k;
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
         const FixedSpacetimeGeometry face_geometry =
            fixed_spacetime_geometry_at_face<Direction, AMREX_SPACEDIM>(
               spacetime, lower, spacing, i, j, k, time
            );
         const RHDConserved value = rhd_riemann_flux<ReconMethod, RiemannMethod, Direction, EOSMethod>(
            cached, il, jl, kl, i, j, k, eos, face_geometry.metric
         );
         store_rhd_conserved(face_flux, i, j, k, value);
      });
   }
}

template <int ReconMethod, int RiemannMethod, int Direction, int EOSMethod>
inline void fill_rhd_flux_direction(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   const amrex::MultiFab& primitive,
   const EOSParams& eos,
   const amrex::MultiFab& cell_spacetime
)
{
   amrex::MultiFab& flux = *fluxes[Direction];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto cached = primitive.const_array(mfi);
      const auto spacetime = cell_spacetime.const_array(mfi);
      const auto face_flux = flux.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         int il = i;
         int jl = j;
         int kl = k;
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
         const SpacetimeMetric metric = evolved_spacetime_metric_at_face<
            4,
            Direction
         >(spacetime, i, j, k);
         const RHDConserved value = rhd_riemann_flux<
            ReconMethod,
            RiemannMethod,
            Direction,
            EOSMethod
         >(cached, il, jl, kl, i, j, k, eos, metric);
         store_rhd_conserved(face_flux, i, j, k, value);
      });
   }
}

template <int ReconMethod, int Direction, int EOSMethod>
inline void fill_rhd_flux_direction_runtime(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   const amrex::MultiFab& primitive,
   const EOSParams& eos,
   int riemann_method,
   const amrex::Geometry& geometry,
   const FixedSpacetimeParameters& spacetime,
   amrex::Real time
)
{
   switch (riemann_method) {
   case RiemannLLF:
      fill_rhd_flux_direction<ReconMethod, RiemannLLF, Direction, EOSMethod>(
         fluxes, primitive, eos, geometry, spacetime, time
      );
      return;
   case RiemannHLL:
      fill_rhd_flux_direction<ReconMethod, RiemannHLL, Direction, EOSMethod>(
         fluxes, primitive, eos, geometry, spacetime, time
      );
      return;
   case RiemannHLLE:
      fill_rhd_flux_direction<ReconMethod, RiemannHLLE, Direction, EOSMethod>(
         fluxes, primitive, eos, geometry, spacetime, time
      );
      return;
   default:
      amrex::Abort("RHD supports LLF, HLL, and HLLE Riemann solvers");
   }
}

template <int ReconMethod, int Direction, int EOSMethod>
inline void fill_rhd_flux_direction_runtime(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   const amrex::MultiFab& primitive,
   const EOSParams& eos,
   int riemann_method,
   const amrex::MultiFab& cell_spacetime
)
{
   switch (riemann_method) {
   case RiemannLLF:
      fill_rhd_flux_direction<ReconMethod, RiemannLLF, Direction, EOSMethod>(
         fluxes, primitive, eos, cell_spacetime
      );
      return;
   case RiemannHLL:
      fill_rhd_flux_direction<ReconMethod, RiemannHLL, Direction, EOSMethod>(
         fluxes, primitive, eos, cell_spacetime
      );
      return;
   case RiemannHLLE:
      fill_rhd_flux_direction<ReconMethod, RiemannHLLE, Direction, EOSMethod>(
         fluxes, primitive, eos, cell_spacetime
      );
      return;
   default:
      amrex::Abort("RHD supports LLF, HLL, and HLLE Riemann solvers");
   }
}

} // namespace pyfluid_amrex
