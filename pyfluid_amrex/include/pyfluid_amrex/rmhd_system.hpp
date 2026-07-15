#pragma once

#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic_fluid.hpp"
#include "pyfluid_amrex/kernels_generated/riemann_ops.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/reconstruction.hpp"
#include "pyfluid_amrex/rhd_system.hpp"
#include "pyfluid_amrex/state.hpp"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

namespace pyfluid_amrex {

struct RMHDPrimitiveFaceStates {
   RMHDPrimitive left;
   RMHDPrimitive right;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDPrimitive rmhd_primitive_at(
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDPrimitive admissibility_limit_rmhd_primitive(
   RMHDPrimitive face,
   const RMHDPrimitive& center,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const RHDPrimitive limited = admissibility_limit_rhd_primitive(
      {face.rho, face.eps, face.pressure, face.v1, face.v2, face.v3},
      {center.rho, center.eps, center.pressure, center.v1, center.v2, center.v3},
      eos,
      metric
   );
   face.rho = limited.rho;
   face.eps = limited.eps;
   face.pressure = limited.pressure;
   face.v1 = limited.v1;
   face.v2 = limited.v2;
   face.v3 = limited.v3;
   return face;
}

template <int ReconMethod, int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDPrimitiveFaceStates rmhd_primitive_face_states(
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
   const ScalarFaceValues b1 = primitive_component_face_values<ReconMethod, Direction, B1>(
      primitive, il, jl, kl, ir, jr, kr
   );
   const ScalarFaceValues b2 = primitive_component_face_values<ReconMethod, Direction, B2>(
      primitive, il, jl, kl, ir, jr, kr
   );
   const ScalarFaceValues b3 = primitive_component_face_values<ReconMethod, Direction, B3>(
      primitive, il, jl, kl, ir, jr, kr
   );
   RMHDPrimitiveFaceStates face{
      {
         rho.left,
         0.0,
         pressure.left,
         v1.left,
         v2.left,
         v3.left,
         b1.left,
         b2.left,
         b3.left,
      },
      {
         rho.right,
         0.0,
         pressure.right,
         v1.right,
         v2.right,
         v3.right,
         b1.right,
         b2.right,
         b3.right,
      },
   };
   face.left = admissibility_limit_rmhd_primitive(
      face.left,
      rmhd_primitive_at(primitive, il, jl, kl, eos),
      eos,
      metric
   );
   face.right = admissibility_limit_rmhd_primitive(
      face.right,
      rmhd_primitive_at(primitive, ir, jr, kr, eos),
      eos,
      metric
   );
   return face;
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDConserved rmhd_hll_flux(
   const RMHDPrimitive& primitive_l,
   const RMHDPrimitive& primitive_r,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const RMHDConserved state_l = generated::rmhd_primitive_to_conserved(
      primitive_l, metric
   );
   const RMHDConserved state_r = generated::rmhd_primitive_to_conserved(
      primitive_r, metric
   );
   const RMHDConserved flux_l = generated::rmhd_flux<Direction>(primitive_l, metric);
   const RMHDConserved flux_r = generated::rmhd_flux<Direction>(primitive_r, metric);
   const RMHDWaveSpeeds speed_l = generated::srmhd_wave_speeds<Direction>(
      primitive_l, metric, eos.gamma
   );
   const RMHDWaveSpeeds speed_r = generated::srmhd_wave_speeds<Direction>(
      primitive_r, metric, eos.gamma
   );
   const amrex::Real speed_minus = amrex::min(
      0.0, amrex::min(speed_l.minus, speed_r.minus)
   );
   const amrex::Real speed_plus = amrex::max(
      0.0, amrex::max(speed_l.plus, speed_r.plus)
   );
   if (speed_minus >= 0.0) {
      return flux_l;
   }
   if (speed_plus <= 0.0) {
      return flux_r;
   }
   return generated::rmhd_hll_flux(
      flux_l, flux_r, state_l, state_r, speed_minus, speed_plus
   );
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDConserved rmhd_llf_flux(
   const RMHDPrimitive& primitive_l,
   const RMHDPrimitive& primitive_r,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const RMHDConserved state_l = generated::rmhd_primitive_to_conserved(
      primitive_l, metric
   );
   const RMHDConserved state_r = generated::rmhd_primitive_to_conserved(
      primitive_r, metric
   );
   const RMHDConserved flux_l = generated::rmhd_flux<Direction>(primitive_l, metric);
   const RMHDConserved flux_r = generated::rmhd_flux<Direction>(primitive_r, metric);
   const RMHDWaveSpeeds speed_l = generated::srmhd_wave_speeds<Direction>(
      primitive_l, metric, eos.gamma
   );
   const RMHDWaveSpeeds speed_r = generated::srmhd_wave_speeds<Direction>(
      primitive_r, metric, eos.gamma
   );
   const amrex::Real wavespeed = amrex::max(
      amrex::max(amrex::Math::abs(speed_l.minus), amrex::Math::abs(speed_l.plus)),
      amrex::max(amrex::Math::abs(speed_r.minus), amrex::Math::abs(speed_r.plus))
   );
   return generated::rmhd_llf_flux(
      flux_l, flux_r, state_l, state_r, wavespeed
   );
}

template <int ReconMethod, int RiemannMethod, int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDConserved rmhd_riemann_flux(
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
   const RMHDPrimitiveFaceStates face =
      rmhd_primitive_face_states<ReconMethod, Direction>(
         primitive, il, jl, kl, ir, jr, kr, eos, metric
      );
   if constexpr (RiemannMethod == RiemannLLF) {
      return rmhd_llf_flux<Direction>(face.left, face.right, eos, metric);
   } else if constexpr (RiemannMethod == RiemannHLL
                        || RiemannMethod == RiemannHLLE) {
      return rmhd_hll_flux<Direction>(face.left, face.right, eos, metric);
   } else {
      AMREX_ASSERT_WITH_MESSAGE(
         false,
         "unsupported RMHD Riemann solver reached a device kernel"
      );
      return {};
   }
}

template <int ReconMethod, int RiemannMethod, int Direction>
inline void fill_rmhd_flux_direction(
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
         const RMHDConserved value =
            rmhd_riemann_flux<ReconMethod, RiemannMethod, Direction>(
               cached, il, jl, kl, i, j, k, eos, face_geometry.metric
            );
         store_rmhd_conserved(face_flux, i, j, k, value);
      });
   }
}

template <int ReconMethod, int Direction>
inline void fill_rmhd_flux_direction_runtime(
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
      fill_rmhd_flux_direction<ReconMethod, RiemannLLF, Direction>(
         fluxes, primitive, eos, geometry, spacetime, time
      );
      return;
   case RiemannHLL:
      fill_rmhd_flux_direction<ReconMethod, RiemannHLL, Direction>(
         fluxes, primitive, eos, geometry, spacetime, time
      );
      return;
   case RiemannHLLE:
      fill_rmhd_flux_direction<ReconMethod, RiemannHLLE, Direction>(
         fluxes, primitive, eos, geometry, spacetime, time
      );
      return;
   default:
      amrex::Abort("RMHD supports LLF, HLL, and HLLE Riemann solvers");
   }
}

} // namespace pyfluid_amrex
