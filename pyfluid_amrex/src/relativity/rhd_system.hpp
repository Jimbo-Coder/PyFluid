#pragma once

#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/flux.hpp"
#include "pyfluid_amrex/kernels_generated/riemann_ops.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/reconstruction/scalar.hpp"
#include "pyfluid_amrex/reference_metric_spacetime.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "pyfluid_amrex/relativistic_diode.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"
#include "pyfluid_amrex/state.hpp"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include <limits>

namespace pyfluid_amrex {

using namespace amrex::literals;

struct RHDPrimitiveFaceStates {
   RHDPrimitive left;
   RHDPrimitive right;
};

struct RHDStoredMetricAccessor {
   amrex::Array4<amrex::Real const> spacetime;
   ExcisionGridView excision{};
   RelativisticDiodeBoundary diode{};

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric cell(int i, int j, int k) const noexcept
   {
      return generated::z4c_spacetime_metric(
         load_z4c_state(spacetime, i, j, k)
      );
   }

   template <int Direction>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric face(int i, int j, int k) const noexcept
   {
      return stored_spacetime_metric_at_face<4, Direction>(
         spacetime, i, j, k, excision
      );
   }
};

struct RHDFixedMetricAccessor {
   FixedSpacetimeParameters spacetime;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lower;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> spacing;
   int coordinate;
   MeshAxes axes;
   amrex::Real time;
   ExcisionGridView excision{};
   RelativisticDiodeBoundary diode{};

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric cell(int i, int j, int k) const noexcept
   {
      if (coordinate == MeshCartesian) {
         return fixed_spacetime_geometry_at_cell<AMREX_SPACEDIM>(
            spacetime, lower, spacing, i, j, k, time
         ).metric;
      }
      return reference_metric_fixed_spacetime_geometry_at_cell(
         spacetime, coordinate, axes, lower, spacing, i, j, k, time
      ).metric;
   }

   template <int Direction>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric face(int i, int j, int k) const noexcept
   {
      if (coordinate == MeshCartesian) {
         return fixed_spacetime_geometry_at_face<Direction, AMREX_SPACEDIM>(
            spacetime, lower, spacing, i, j, k, time
         ).metric;
      }
      return reference_metric_fixed_spacetime_geometry_at_face<Direction>(
         spacetime, coordinate, axes, lower, spacing, i, j, k, time
      ).metric;
   }
};

template <int ReconMethod, int Direction, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool relativistic_reconstruction_stencil_is_active(
   MetricAccessor const& metric, int il, int jl, int kl, int ir, int jr, int kr
) noexcept
{
   if (!metric.excision.region.enabled()) return true;
   constexpr int radius = reconstruction_stencil_radius(ReconMethod);
   auto first = metric.excision.position(il, jl, kl);
   auto last = metric.excision.position(ir, jr, kr);
   first[Direction] -= radius * metric.excision.spacing[Direction];
   last[Direction] += radius * metric.excision.spacing[Direction];
   return !metric.excision.region.intersects(first, last);
}

template <int Direction, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SpatialFourVelocity spatial_four_velocity_neighbor(
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   int offset,
   MetricAccessor const& metric
) noexcept
{
   if constexpr (Direction == 0) {
      i += offset;
   }
#if (AMREX_SPACEDIM >= 2)
   else if constexpr (Direction == 1) {
      j += offset;
   }
#endif
#if (AMREX_SPACEDIM == 3)
   else {
      k += offset;
   }
#endif
   return spatial_four_velocity_from_eulerian(
      primitive(i, j, k, MOM1),
      primitive(i, j, k, MOM2),
      primitive(i, j, k, MOM3),
      metric.cell(i, j, k)
   );
}

template <int ReconMethod, int Direction, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::array<ScalarFaceValues, 3> spatial_four_velocity_face_values(
   amrex::Array4<amrex::Real const> const& primitive,
   int il,
   int jl,
   int kl,
   int ir,
   int jr,
   int kr,
   MetricAccessor const& metric
) noexcept
{
   if constexpr (ReconMethod == ReconPWL) {
      const SpatialFourVelocity zm = spatial_four_velocity_neighbor<Direction>(
         primitive, il, jl, kl, -1, metric
      );
      const SpatialFourVelocity zl = spatial_four_velocity_neighbor<Direction>(
         primitive, il, jl, kl, 0, metric
      );
      const SpatialFourVelocity zr = spatial_four_velocity_neighbor<Direction>(
         primitive, ir, jr, kr, 0, metric
      );
      const SpatialFourVelocity zp = spatial_four_velocity_neighbor<Direction>(
         primitive, ir, jr, kr, 1, metric
      );
      return {{
         {zl.z1 + 0.5_rt * mc_slope(zm.z1, zl.z1, zr.z1),
          zr.z1 - 0.5_rt * mc_slope(zl.z1, zr.z1, zp.z1)},
         {zl.z2 + 0.5_rt * mc_slope(zm.z2, zl.z2, zr.z2),
          zr.z2 - 0.5_rt * mc_slope(zl.z2, zr.z2, zp.z2)},
         {zl.z3 + 0.5_rt * mc_slope(zm.z3, zl.z3, zr.z3),
          zr.z3 - 0.5_rt * mc_slope(zl.z3, zr.z3, zp.z3)},
      }};
   } else if constexpr (is_five_point_reconstruction(ReconMethod)) {
      const SpatialFourVelocity zmm = spatial_four_velocity_neighbor<Direction>(
         primitive, il, jl, kl, -2, metric
      );
      const SpatialFourVelocity zm = spatial_four_velocity_neighbor<Direction>(
         primitive, il, jl, kl, -1, metric
      );
      const SpatialFourVelocity z0 = spatial_four_velocity_neighbor<Direction>(
         primitive, il, jl, kl, 0, metric
      );
      const SpatialFourVelocity zp = spatial_four_velocity_neighbor<Direction>(
         primitive, il, jl, kl, 1, metric
      );
      const SpatialFourVelocity zpp = spatial_four_velocity_neighbor<Direction>(
         primitive, il, jl, kl, 2, metric
      );
      const SpatialFourVelocity zppp = spatial_four_velocity_neighbor<Direction>(
         primitive, il, jl, kl, 3, metric
      );
      return {{
         {reconstruct_left<ReconMethod>(zmm.z1, zm.z1, z0.z1, zp.z1, zpp.z1),
          reconstruct_right<ReconMethod>(zm.z1, z0.z1, zp.z1, zpp.z1, zppp.z1)},
         {reconstruct_left<ReconMethod>(zmm.z2, zm.z2, z0.z2, zp.z2, zpp.z2),
          reconstruct_right<ReconMethod>(zm.z2, z0.z2, zp.z2, zpp.z2, zppp.z2)},
         {reconstruct_left<ReconMethod>(zmm.z3, zm.z3, z0.z3, zp.z3, zpp.z3),
          reconstruct_right<ReconMethod>(zm.z3, z0.z3, zp.z3, zpp.z3, zppp.z3)},
      }};
   } else {
      const SpatialFourVelocity zl = spatial_four_velocity_neighbor<Direction>(
         primitive, il, jl, kl, 0, metric
      );
      const SpatialFourVelocity zr = spatial_four_velocity_neighbor<Direction>(
         primitive, ir, jr, kr, 0, metric
      );
      return {{{zl.z1, zr.z1}, {zl.z2, zr.z2}, {zl.z3, zr.z3}}};
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real causal_limiter_theta(
   const RHDPrimitive& face,
   const RHDPrimitive& center,
   const SpacetimeMetric& metric
) noexcept
{
   constexpr amrex::Real velocity_limit_squared =
      1.0_rt - 128.0_rt * std::numeric_limits<amrex::Real>::epsilon();
   const amrex::Real face_velocity_squared = spatial_vector_norm_squared(
      face.v1, face.v2, face.v3, metric
   );
   if (amrex::Math::isfinite(face_velocity_squared)
       && face_velocity_squared < velocity_limit_squared) {
      return 1.0_rt;
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
   const amrex::Real linear = 2.0_rt * center_delta;
   const amrex::Real constant = spatial_vector_norm_squared(
      center.v1, center.v2, center.v3, metric
   ) - velocity_limit_squared;
   if (!(quadratic > 0.0_rt) || !amrex::Math::isfinite(quadratic)
       || !amrex::Math::isfinite(linear) || !amrex::Math::isfinite(constant)) {
      return 0.0_rt;
   }
   const amrex::Real discriminant = linear * linear - 4.0_rt * quadratic * constant;
   if (!(discriminant >= 0.0_rt) || !amrex::Math::isfinite(discriminant)) {
      return 0.0_rt;
   }
   return amrex::max(
      0.0_rt,
      amrex::min(1.0_rt, (-linear + std::sqrt(discriminant)) / (2.0_rt * quadratic))
   );
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDPrimitive admissibility_limit_rhd_primitive(
   RHDPrimitive face,
   const RHDPrimitive& center,
   const EOSParams& eos,
   const SpacetimeMetric& metric,
   amrex::Real density_floor,
   amrex::Real pressure_floor
) noexcept
{
   if (!amrex::Math::isfinite(face.rho)
       || !amrex::Math::isfinite(face.pressure)
       || !amrex::Math::isfinite(face.v1)
       || !amrex::Math::isfinite(face.v2)
       || !amrex::Math::isfinite(face.v3)) {
      return center;
   }
   amrex::Real theta = positivity_limiter_theta(face.rho, center.rho, density_floor);
   theta = amrex::min(
      theta,
      positivity_limiter_theta(face.pressure, center.pressure, pressure_floor)
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
   const amrex::Real velocity_squared = spatial_vector_norm_squared(
      face.v1, face.v2, face.v3, metric
   );
   const amrex::Real sound_speed_squared =
      relativistic_sound_speed_squared<EOSMethod>(
         face.rho, face.pressure, eos
      );
   if (!amrex::Math::isfinite(face.eps)
       || !amrex::Math::isfinite(velocity_squared)
       || !amrex::Math::isfinite(sound_speed_squared)
       || face.eps < 0.0_rt
       || !(velocity_squared < 1.0_rt)
       || sound_speed_squared < 0.0_rt
       || sound_speed_squared > 1.0_rt) {
      return center;
   }
   return face;
}

template <int ReconMethod, int Direction, int EOSMethod, typename MetricAccessor>
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
   MetricAccessor const& metric,
   EOSThermodynamicState const& atmosphere_l,
   EOSThermodynamicState const& atmosphere_r
) noexcept
{
   RHDPrimitive center_l = rhd_primitive_at<EOSMethod>(
      primitive, il, jl, kl, eos
   );
   RHDPrimitive center_r = rhd_primitive_at<EOSMethod>(
      primitive, ir, jr, kr, eos
   );
   const SpacetimeMetric face_metric = metric.template face<Direction>(
      ir, jr, kr
   );
   const SpatialFourVelocity center_z_l = spatial_four_velocity_from_eulerian(
      center_l.v1, center_l.v2, center_l.v3, metric.cell(il, jl, kl)
   );
   const SpatialFourVelocity center_z_r = spatial_four_velocity_from_eulerian(
      center_r.v1, center_r.v2, center_r.v3, metric.cell(ir, jr, kr)
   );
   const EulerianVelocity center_v_l =
      eulerian_velocity_from_spatial_four_velocity(center_z_l, face_metric);
   const EulerianVelocity center_v_r =
      eulerian_velocity_from_spatial_four_velocity(center_z_r, face_metric);
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
   RHDPrimitiveFaceStates face{
      {rho.left, 0.0_rt, pressure.left, velocity_l.v1, velocity_l.v2, velocity_l.v3},
      {rho.right, 0.0_rt, pressure.right, velocity_r.v1, velocity_r.v2, velocity_r.v3},
   };
   face.left = admissibility_limit_rhd_primitive<EOSMethod>(
      face.left,
      center_l,
      eos,
      face_metric,
      atmosphere_l.rho,
      atmosphere_l.pressure
   );
   face.right = admissibility_limit_rhd_primitive<EOSMethod>(
      face.right,
      center_r,
      eos,
      face_metric,
      atmosphere_r.rho,
      atmosphere_r.pressure
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
   const amrex::Real speed_minus = amrex::min(0.0_rt, amrex::min(speed_l.minus, speed_r.minus));
   const amrex::Real speed_plus = amrex::max(0.0_rt, amrex::max(speed_l.plus, speed_r.plus));
   if (speed_minus >= 0.0_rt) {
      return flux_l;
   }
   if (speed_plus <= 0.0_rt) {
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

template <int ReconMethod, int RiemannMethod, int Direction, int EOSMethod, typename MetricAccessor>
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
   MetricAccessor const& metric,
   EOSThermodynamicState const& atmosphere_l,
   EOSThermodynamicState const& atmosphere_r
) noexcept
{
   const RHDPrimitiveFaceStates face = rhd_primitive_face_states<ReconMethod, Direction, EOSMethod>(
      primitive, il, jl, kl, ir, jr, kr, eos, metric, atmosphere_l, atmosphere_r
   );
   const SpacetimeMetric face_metric = metric.template face<Direction>(
      ir, jr, kr
   );
   if constexpr (RiemannMethod == RiemannLLF) {
      return rhd_llf_flux<Direction, EOSMethod>(face.left, face.right, eos, face_metric);
   } else if constexpr (RiemannMethod == RiemannHLL || RiemannMethod == RiemannHLLE) {
      return rhd_hll_flux<Direction, EOSMethod>(face.left, face.right, eos, face_metric);
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
   EOSThermodynamicState const& atmosphere,
   amrex::Real time,
   int coordinate,
   MeshAxes const& axes,
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
            spacetime, lower, spacing, coordinate, axes, time, {}, diode
         };
         const RHDConserved value = rhd_riemann_flux<ReconMethod, RiemannMethod, Direction, EOSMethod>(
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
         store_rhd_conserved(face_flux, i, j, k, value);
      });
   }
}

template <int ReconMethod, int RiemannMethod, int Direction, int EOSMethod>
inline void fill_rhd_flux_direction(
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
         const RHDConserved value = rhd_riemann_flux<
            ReconMethod,
            RiemannMethod,
            Direction,
            EOSMethod
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
   EOSThermodynamicState const& atmosphere,
   amrex::Real time,
   int coordinate,
   MeshAxes const& axes,
   RelativisticDiodeBoundary const& diode = {}
)
{
   switch (riemann_method) {
   case RiemannLLF:
      fill_rhd_flux_direction<ReconMethod, RiemannLLF, Direction, EOSMethod>(
         fluxes, primitive, eos, geometry, spacetime,
         atmosphere, time, coordinate, axes, diode
      );
      return;
   case RiemannHLL:
      fill_rhd_flux_direction<ReconMethod, RiemannHLL, Direction, EOSMethod>(
         fluxes, primitive, eos, geometry, spacetime,
         atmosphere, time, coordinate, axes, diode
      );
      return;
   case RiemannHLLE:
      fill_rhd_flux_direction<ReconMethod, RiemannHLLE, Direction, EOSMethod>(
         fluxes, primitive, eos, geometry, spacetime,
         atmosphere, time, coordinate, axes, diode
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
   const amrex::MultiFab& cell_spacetime,
   EOSThermodynamicState const& atmosphere,
   ExcisionGridView const& excision = {},
   RelativisticDiodeBoundary const& diode = {}
)
{
   switch (riemann_method) {
   case RiemannLLF:
      fill_rhd_flux_direction<ReconMethod, RiemannLLF, Direction, EOSMethod>(
         fluxes, primitive, eos, cell_spacetime,
         atmosphere, excision, diode
      );
      return;
   case RiemannHLL:
      fill_rhd_flux_direction<ReconMethod, RiemannHLL, Direction, EOSMethod>(
         fluxes, primitive, eos, cell_spacetime,
         atmosphere, excision, diode
      );
      return;
   case RiemannHLLE:
      fill_rhd_flux_direction<ReconMethod, RiemannHLLE, Direction, EOSMethod>(
         fluxes, primitive, eos, cell_spacetime,
         atmosphere, excision, diode
      );
      return;
   default:
      amrex::Abort("RHD supports LLF, HLL, and HLLE Riemann solvers");
   }
}

} // namespace pyfluid_amrex
