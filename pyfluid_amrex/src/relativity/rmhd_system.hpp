#pragma once

#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/flux.hpp"
#include "pyfluid_amrex/kernels_generated/riemann_ops.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/reconstruction/scalar.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "relativity/rhd_system.hpp"
#include "pyfluid_amrex/state.hpp"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

namespace pyfluid_amrex {

using namespace amrex::literals;

struct RMHDPrimitiveFaceStates {
   RMHDPrimitive left;
   RMHDPrimitive right;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDPrimitive admissibility_limit_rmhd_primitive(
   RMHDPrimitive face,
   const RMHDPrimitive& center,
   const EOSParams& eos,
   const SpacetimeMetric& metric,
   amrex::Real density_floor,
   amrex::Real pressure_floor
) noexcept
{
   const RHDPrimitive limited = admissibility_limit_rhd_primitive<EOSRuntime>(
      {face.rho, face.eps, face.pressure, face.v1, face.v2, face.v3},
      {center.rho, center.eps, center.pressure, center.v1, center.v2, center.v3},
      eos,
      metric,
      density_floor,
      pressure_floor
   );
   face.rho = limited.rho;
   face.eps = limited.eps;
   face.pressure = limited.pressure;
   face.v1 = limited.v1;
   face.v2 = limited.v2;
   face.v3 = limited.v3;
   return face;
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDWaveSpeeds rmhd_wave_speeds(
   const RMHDPrimitive& primitive,
   const SpacetimeMetric& metric,
   const EOSParams& eos
) noexcept
{
   const RelativisticCharacteristicState thermodynamics =
      relativistic_characteristic_state<EOSRuntime>(
         primitive.rho, primitive.eps, eos
      );
   return generated::srmhd_wave_speeds<Direction>(
      primitive,
      metric,
      amrex::max(
         amrex::Real{0.0_rt}, thermodynamics.sound_speed_squared
      ),
      thermodynamics.specific_enthalpy
   );
}

template <int ReconMethod, int Direction, typename MetricAccessor>
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
   MetricAccessor const& metric,
   EOSThermodynamicState const& atmosphere_l,
   EOSThermodynamicState const& atmosphere_r
) noexcept
{
   RMHDPrimitive center_l = rmhd_primitive_at(
      primitive, il, jl, kl, eos
   );
   RMHDPrimitive center_r = rmhd_primitive_at(
      primitive, ir, jr, kr, eos
   );
   const SpacetimeMetric face_metric = metric.template face<Direction>(
      ir, jr, kr
   );
   const EulerianVelocity center_v_l =
      eulerian_velocity_from_spatial_four_velocity(
         spatial_four_velocity_from_eulerian(
            center_l.v1, center_l.v2, center_l.v3,
            metric.cell(il, jl, kl)
         ),
         face_metric
      );
   const EulerianVelocity center_v_r =
      eulerian_velocity_from_spatial_four_velocity(
         spatial_four_velocity_from_eulerian(
            center_r.v1, center_r.v2, center_r.v3,
            metric.cell(ir, jr, kr)
         ),
         face_metric
      );
   center_l.v1 = center_v_l.v1;
   center_l.v2 = center_v_l.v2;
   center_l.v3 = center_v_l.v3;
   center_r.v1 = center_v_r.v1;
   center_r.v2 = center_v_r.v2;
   center_r.v3 = center_v_r.v3;
   if (apply_relativistic_diode_face<Direction>(
          center_l, center_r, face_metric, metric.diode, ir, jr, kr)) {
      return {center_l, center_r};
   }
   if (!relativistic_reconstruction_stencil_is_active<ReconMethod, Direction>(
          metric, il, jl, kl, ir, jr, kr)
       || center_l.rho <= atmosphere_l.rho
       || center_l.pressure <= atmosphere_l.pressure
       || center_r.rho <= atmosphere_r.rho
       || center_r.pressure <= atmosphere_r.pressure) {
      return {center_l, center_r};
   }
   const ScalarFaceValues rho = primitive_component_face_values<ReconMethod, Direction, RHO>(
      primitive, il, jl, kl, ir, jr, kr
   );
   const auto z = spatial_four_velocity_face_values<ReconMethod, Direction>(
      primitive, il, jl, kl, ir, jr, kr, metric
   );
   const EulerianVelocity velocity_l =
      eulerian_velocity_from_spatial_four_velocity(
         {z[0].left, z[1].left, z[2].left}, face_metric
      );
   const EulerianVelocity velocity_r =
      eulerian_velocity_from_spatial_four_velocity(
         {z[0].right, z[1].right, z[2].right}, face_metric
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
         0.0_rt,
         pressure.left,
         velocity_l.v1,
         velocity_l.v2,
         velocity_l.v3,
         b1.left,
         b2.left,
         b3.left,
      },
      {
         rho.right,
         0.0_rt,
         pressure.right,
         velocity_r.v1,
         velocity_r.v2,
         velocity_r.v3,
         b1.right,
         b2.right,
         b3.right,
      },
   };
   face.left = admissibility_limit_rmhd_primitive(
      face.left,
      center_l,
      eos,
      face_metric,
      atmosphere_l.rho,
      atmosphere_l.pressure
   );
   face.right = admissibility_limit_rmhd_primitive(
      face.right,
      center_r,
      eos,
      face_metric,
      atmosphere_r.rho,
      atmosphere_r.pressure
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
   const RMHDWaveSpeeds speed_l = rmhd_wave_speeds<Direction>(
      primitive_l, metric, eos
   );
   const RMHDWaveSpeeds speed_r = rmhd_wave_speeds<Direction>(
      primitive_r, metric, eos
   );
   const amrex::Real speed_minus = amrex::min(
      0.0_rt, amrex::min(speed_l.minus, speed_r.minus)
   );
   const amrex::Real speed_plus = amrex::max(
      0.0_rt, amrex::max(speed_l.plus, speed_r.plus)
   );
   if (speed_minus >= 0.0_rt) {
      return flux_l;
   }
   if (speed_plus <= 0.0_rt) {
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
   const RMHDWaveSpeeds speed_l = rmhd_wave_speeds<Direction>(
      primitive_l, metric, eos
   );
   const RMHDWaveSpeeds speed_r = rmhd_wave_speeds<Direction>(
      primitive_r, metric, eos
   );
   const amrex::Real wavespeed = amrex::max(
      amrex::max(amrex::Math::abs(speed_l.minus), amrex::Math::abs(speed_l.plus)),
      amrex::max(amrex::Math::abs(speed_r.minus), amrex::Math::abs(speed_r.plus))
   );
   return generated::rmhd_llf_flux(
      flux_l, flux_r, state_l, state_r, wavespeed
   );
}

template <int ReconMethod, int RiemannMethod, int Direction, typename MetricAccessor>
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
   MetricAccessor const& metric,
   EOSThermodynamicState const& atmosphere_l,
   EOSThermodynamicState const& atmosphere_r
) noexcept
{
   const RMHDPrimitiveFaceStates face =
      rmhd_primitive_face_states<ReconMethod, Direction>(
         primitive, il, jl, kl, ir, jr, kr, eos, metric,
         atmosphere_l, atmosphere_r
      );
   const SpacetimeMetric face_metric = metric.template face<Direction>(
      ir, jr, kr
   );
   if constexpr (RiemannMethod == RiemannLLF) {
      return rmhd_llf_flux<Direction>(face.left, face.right, eos, face_metric);
   } else if constexpr (RiemannMethod == RiemannHLL
                        || RiemannMethod == RiemannHLLE) {
      return rmhd_hll_flux<Direction>(face.left, face.right, eos, face_metric);
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
   EOSThermodynamicState const& atmosphere,
   amrex::Real time,
   RelativisticDiodeBoundary const& diode = {}
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
         const RHDFixedMetricAccessor metric{
            spacetime,
            lower,
            spacing,
            MeshCartesian,
            mesh_default_axes_layout(MeshCartesian),
            time,
            {},
            diode
         };
         const RMHDConserved value =
            rmhd_riemann_flux<ReconMethod, RiemannMethod, Direction>(
               cached,
               il,
               jl,
               kl,
               i,
               j,
               k,
               eos,
               metric,
               atmosphere,
               atmosphere
            );
         store_rmhd_conserved(face_flux, i, j, k, value);
      });
   }
}

template <int ReconMethod, int RiemannMethod, int Direction>
inline void fill_rmhd_flux_direction(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   const amrex::MultiFab& primitive,
   const EOSParams& eos,
   const amrex::MultiFab& cell_spacetime,
   EOSThermodynamicState const& atmosphere,
   ExcisionGridView const& excision = {},
   RelativisticDiodeBoundary const& diode = {}
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
         const RHDStoredMetricAccessor metric{spacetime, excision, diode};
         const RMHDConserved value = rmhd_riemann_flux<
            ReconMethod,
            RiemannMethod,
            Direction
         >(
            cached,
            il,
            jl,
            kl,
            i,
            j,
            k,
            eos,
            metric,
            atmosphere,
            atmosphere
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
   EOSThermodynamicState const& atmosphere,
   amrex::Real time,
   RelativisticDiodeBoundary const& diode = {}
)
{
   switch (riemann_method) {
   case RiemannLLF:
      fill_rmhd_flux_direction<ReconMethod, RiemannLLF, Direction>(
         fluxes, primitive, eos, geometry, spacetime,
         atmosphere, time, diode
      );
      return;
   case RiemannHLL:
      fill_rmhd_flux_direction<ReconMethod, RiemannHLL, Direction>(
         fluxes, primitive, eos, geometry, spacetime,
         atmosphere, time, diode
      );
      return;
   case RiemannHLLE:
      fill_rmhd_flux_direction<ReconMethod, RiemannHLLE, Direction>(
         fluxes, primitive, eos, geometry, spacetime,
         atmosphere, time, diode
      );
      return;
   default:
      amrex::Abort("invalid basic RMHD Riemann solver dispatch");
   }
}

template <int ReconMethod, int Direction>
inline void fill_rmhd_flux_direction_runtime(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   const amrex::MultiFab& primitive,
   const EOSParams& eos,
   int riemann_method,
   const amrex::MultiFab& cell_spacetime,
   EOSThermodynamicState const& atmosphere,
   ExcisionGridView const& excision = {},
   RelativisticDiodeBoundary const& diode = {}
)
{
   switch (riemann_method) {
   case RiemannLLF:
      fill_rmhd_flux_direction<ReconMethod, RiemannLLF, Direction>(
         fluxes, primitive, eos, cell_spacetime,
         atmosphere, excision, diode
      );
      return;
   case RiemannHLL:
      fill_rmhd_flux_direction<ReconMethod, RiemannHLL, Direction>(
         fluxes, primitive, eos, cell_spacetime,
         atmosphere, excision, diode
      );
      return;
   case RiemannHLLE:
      fill_rmhd_flux_direction<ReconMethod, RiemannHLLE, Direction>(
         fluxes, primitive, eos, cell_spacetime,
         atmosphere, excision, diode
      );
      return;
   default:
      amrex::Abort("invalid basic RMHD Riemann solver dispatch");
   }
}

} // namespace pyfluid_amrex
