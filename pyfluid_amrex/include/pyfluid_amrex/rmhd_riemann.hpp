#pragma once

#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/flux.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"

#include <AMReX_Algorithm.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace pyfluid_amrex {

// The Mignone--Bodo HLLC and Mignone--Ugliano--Bodo HLLD algorithms are
// special-relativistic, face-normal solvers.  This header owns that pointwise
// mathematics plus the tetrad transformation needed by GRMHD.  Host dispatch,
// reconstruction, AMReX iteration, and candidate correction remain outside.

struct SRMHDState {
   amrex::Real dens = 0.0;
   amrex::Real momn = 0.0;
   amrex::Real momt1 = 0.0;
   amrex::Real momt2 = 0.0;
   amrex::Real energy = 0.0;
   amrex::Real bn = 0.0;
   amrex::Real bt1 = 0.0;
   amrex::Real bt2 = 0.0;
};

struct SRMHDRiemannSample {
   SRMHDState state{};
   SRMHDState flux{};
   amrex::Real contact_speed = 0.0;
   amrex::Real total_pressure = 0.0;
   bool used_hll = false;
};

struct RMHDFrameVector {
   amrex::Real x = 0.0;
   amrex::Real y = 0.0;
   amrex::Real z = 0.0;
};

struct RMHDFaceFrame {
   RMHDFrameVector basis_n{};
   RMHDFrameVector basis_t1{};
   RMHDFrameVector basis_t2{};
   RMHDFrameVector coframe_n{};
   RMHDFrameVector coframe_t1{};
   RMHDFrameVector coframe_t2{};
   amrex::Real normal_coordinate_scale = 1.0;
   amrex::Real interface_speed = 0.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rmhd_frame_component(const RMHDFrameVector& vector, int component) noexcept
{
   if (component == 0) {
      return vector.x;
   }
   if (component == 1) {
      return vector.y;
   }
   return vector.z;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDFrameVector rmhd_frame_axis(int component) noexcept
{
   if (component == 0) {
      return {1.0, 0.0, 0.0};
   }
   if (component == 1) {
      return {0.0, 1.0, 0.0};
   }
   return {0.0, 0.0, 1.0};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDFrameVector rmhd_frame_add_scaled(
   const RMHDFrameVector& left,
   amrex::Real scale,
   const RMHDFrameVector& right
) noexcept
{
   return {
      left.x + scale * right.x,
      left.y + scale * right.y,
      left.z + scale * right.z,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDFrameVector rmhd_lower_spatial_vector(
   const RMHDFrameVector& vector,
   const SpacetimeMetric& metric
) noexcept
{
   return {
      metric.gamma11 * vector.x + metric.gamma12 * vector.y
         + metric.gamma13 * vector.z,
      metric.gamma12 * vector.x + metric.gamma22 * vector.y
         + metric.gamma23 * vector.z,
      metric.gamma13 * vector.x + metric.gamma23 * vector.y
         + metric.gamma33 * vector.z,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rmhd_frame_dot(
   const RMHDFrameVector& left,
   const RMHDFrameVector& right
) noexcept
{
   return left.x * right.x + left.y * right.y + left.z * right.z;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rmhd_spatial_dot(
   const RMHDFrameVector& left,
   const RMHDFrameVector& right,
   const SpacetimeMetric& metric
) noexcept
{
   return rmhd_frame_dot(left, rmhd_lower_spatial_vector(right, metric));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rmhd_inverse_metric_diagonal(
   const SpacetimeMetric& metric,
   int direction
) noexcept
{
   if (direction == 0) {
      return metric.inv_gamma11;
   }
   if (direction == 1) {
      return metric.inv_gamma22;
   }
   return metric.inv_gamma33;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDFrameVector rmhd_inverse_metric_row(
   const SpacetimeMetric& metric,
   int direction
) noexcept
{
   if (direction == 0) {
      return {metric.inv_gamma11, metric.inv_gamma12, metric.inv_gamma13};
   }
   if (direction == 1) {
      return {metric.inv_gamma12, metric.inv_gamma22, metric.inv_gamma23};
   }
   return {metric.inv_gamma13, metric.inv_gamma23, metric.inv_gamma33};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rmhd_shift_component(
   const SpacetimeMetric& metric,
   int component
) noexcept
{
   if (component == 0) {
      return metric.shift1;
   }
   if (component == 1) {
      return metric.shift2;
   }
   return metric.shift3;
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDFaceFrame rmhd_face_frame(const SpacetimeMetric& metric) noexcept
{
   static_assert(Direction >= 0 && Direction < 3);
   const int tangent1_direction = (Direction + 1) % 3;
   const int tangent2_direction = (Direction + 2) % 3;
   const amrex::Real inverse_normal_norm = std::sqrt(
      rmhd_inverse_metric_diagonal(metric, Direction)
   );
   const RMHDFrameVector normal = rmhd_frame_add_scaled(
      {},
      1.0 / inverse_normal_norm,
      rmhd_inverse_metric_row(metric, Direction)
   );
   const RMHDFrameVector tangent1_axis = rmhd_frame_axis(tangent1_direction);
   const amrex::Real tangent1_norm = std::sqrt(
      rmhd_spatial_dot(tangent1_axis, tangent1_axis, metric)
   );
   const RMHDFrameVector tangent1 = rmhd_frame_add_scaled(
      {}, 1.0 / tangent1_norm, tangent1_axis
   );
   const RMHDFrameVector tangent2_axis = rmhd_frame_axis(tangent2_direction);
   const amrex::Real tangent_projection = rmhd_spatial_dot(
      tangent2_axis, tangent1, metric
   );
   const RMHDFrameVector tangent2_unnormalized = rmhd_frame_add_scaled(
      tangent2_axis, -tangent_projection, tangent1
   );
   const amrex::Real tangent2_norm = std::sqrt(
      rmhd_spatial_dot(tangent2_unnormalized, tangent2_unnormalized, metric)
   );
   const RMHDFrameVector tangent2 = rmhd_frame_add_scaled(
      {}, 1.0 / tangent2_norm, tangent2_unnormalized
   );
   return {
      normal,
      tangent1,
      tangent2,
      rmhd_lower_spatial_vector(normal, metric),
      rmhd_lower_spatial_vector(tangent1, metric),
      rmhd_lower_spatial_vector(tangent2, metric),
      inverse_normal_norm,
      rmhd_shift_component(metric, Direction)
         / (metric.lapse * inverse_normal_norm),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDPrimitive rmhd_primitive_to_face_frame(
   const RMHDPrimitive& primitive,
   const RMHDFaceFrame& frame
) noexcept
{
   const RMHDFrameVector velocity{primitive.v1, primitive.v2, primitive.v3};
   const RMHDFrameVector magnetic{primitive.b1, primitive.b2, primitive.b3};
   return {
      primitive.rho,
      primitive.eps,
      primitive.pressure,
      rmhd_frame_dot(frame.coframe_n, velocity),
      rmhd_frame_dot(frame.coframe_t1, velocity),
      rmhd_frame_dot(frame.coframe_t2, velocity),
      rmhd_frame_dot(frame.coframe_n, magnetic),
      rmhd_frame_dot(frame.coframe_t1, magnetic),
      rmhd_frame_dot(frame.coframe_t2, magnetic),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDState srmhd_state_from_conserved(const RMHDConserved& conserved) noexcept
{
   return {
      conserved.dens,
      conserved.mom1,
      conserved.mom2,
      conserved.mom3,
      conserved.tau + conserved.dens,
      conserved.b1,
      conserved.b2,
      conserved.b3,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDState srmhd_state_from_primitive(const RMHDPrimitive& primitive) noexcept
{
   return srmhd_state_from_conserved(
      generated::rmhd_primitive_to_conserved(
         primitive, minkowski_spacetime_metric()
      )
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDState srmhd_flux_from_primitive(const RMHDPrimitive& primitive) noexcept
{
   return srmhd_state_from_conserved(
      generated::rmhd_flux<0>(primitive, minkowski_spacetime_metric())
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDState srmhd_state_add_scaled(
   const SRMHDState& state,
   amrex::Real scale,
   const SRMHDState& delta
) noexcept
{
   return {
      state.dens + scale * delta.dens,
      state.momn + scale * delta.momn,
      state.momt1 + scale * delta.momt1,
      state.momt2 + scale * delta.momt2,
      state.energy + scale * delta.energy,
      state.bn + scale * delta.bn,
      state.bt1 + scale * delta.bt1,
      state.bt2 + scale * delta.bt2,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDState srmhd_state_difference(
   const SRMHDState& left,
   const SRMHDState& right
) noexcept
{
   return {
      left.dens - right.dens,
      left.momn - right.momn,
      left.momt1 - right.momt1,
      left.momt2 - right.momt2,
      left.energy - right.energy,
      left.bn - right.bn,
      left.bt1 - right.bt1,
      left.bt2 - right.bt2,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDState srmhd_wave_flux(
   const SRMHDState& flux,
   amrex::Real wave_speed,
   const SRMHDState& state_after,
   const SRMHDState& state_before
) noexcept
{
   return srmhd_state_add_scaled(
      flux, wave_speed, srmhd_state_difference(state_after, state_before)
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool finite_srmhd_state(const SRMHDState& state) noexcept
{
   return amrex::Math::isfinite(state.dens)
      && amrex::Math::isfinite(state.momn)
      && amrex::Math::isfinite(state.momt1)
      && amrex::Math::isfinite(state.momt2)
      && amrex::Math::isfinite(state.energy)
      && amrex::Math::isfinite(state.bn)
      && amrex::Math::isfinite(state.bt1)
      && amrex::Math::isfinite(state.bt2);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDState srmhd_hll_state(
   const SRMHDState& state_l,
   const SRMHDState& state_r,
   const SRMHDState& flux_l,
   const SRMHDState& flux_r,
   amrex::Real speed_l,
   amrex::Real speed_r
) noexcept
{
   const amrex::Real inverse_width = 1.0 / (speed_r - speed_l);
   SRMHDState result = srmhd_state_add_scaled(
      state_l, speed_r * inverse_width,
      srmhd_state_difference(state_r, state_l)
   );
   result = srmhd_state_add_scaled(
      result, inverse_width, srmhd_state_difference(flux_l, flux_r)
   );
   return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDState srmhd_hll_flux(
   const SRMHDState& state_l,
   const SRMHDState& state_r,
   const SRMHDState& flux_l,
   const SRMHDState& flux_r,
   amrex::Real speed_l,
   amrex::Real speed_r
) noexcept
{
   const amrex::Real inverse_width = 1.0 / (speed_r - speed_l);
   SRMHDState result = srmhd_state_add_scaled(
      flux_l, -speed_l / (speed_r - speed_l),
      srmhd_state_difference(flux_r, flux_l)
   );
   result = srmhd_state_add_scaled(
      result,
      speed_l * speed_r * inverse_width,
      srmhd_state_difference(state_r, state_l)
   );
   return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDRiemannSample srmhd_hll_sample(
   const SRMHDState& state_l,
   const SRMHDState& state_r,
   const SRMHDState& flux_l,
   const SRMHDState& flux_r,
   amrex::Real speed_l,
   amrex::Real speed_r,
   amrex::Real sample_speed
) noexcept
{
   if (sample_speed <= speed_l) {
      return {state_l, flux_l, 0.0, 0.0, true};
   }
   if (sample_speed >= speed_r) {
      return {state_r, flux_r, 0.0, 0.0, true};
   }
   return {
      srmhd_hll_state(state_l, state_r, flux_l, flux_r, speed_l, speed_r),
      srmhd_hll_flux(state_l, state_r, flux_l, flux_r, speed_l, speed_r),
      0.0,
      0.0,
      true,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srmhd_minus_quadratic_root(
   amrex::Real quadratic,
   amrex::Real linear,
   amrex::Real constant
) noexcept
{
   const amrex::Real scale = amrex::max(
      1.0,
      amrex::max(
         amrex::Math::abs(quadratic),
         amrex::max(amrex::Math::abs(linear), amrex::Math::abs(constant))
      )
   );
   if (amrex::Math::abs(quadratic) <= 1.0e-14 * scale) {
      return amrex::Math::abs(linear) > 1.0e-14 * scale
         ? -constant / linear
         : 0.0;
   }
   const amrex::Real discriminant = linear * linear
      - 4.0 * quadratic * constant;
   if (!(discriminant >= 0.0) || !amrex::Math::isfinite(discriminant)) {
      return 2.0;
   }
   return (-linear - std::sqrt(discriminant)) / (2.0 * quadratic);
}

struct SRMHDHLLCStates {
   SRMHDState state_l{};
   SRMHDState state_r{};
   SRMHDState flux_l{};
   SRMHDState flux_r{};
   amrex::Real contact_speed = 0.0;
   amrex::Real total_pressure = 0.0;
   bool valid = false;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDHLLCStates srmhd_hllc_intermediate_states(
   const RMHDPrimitive& primitive_l,
   const RMHDPrimitive& primitive_r,
   const SRMHDState& state_l,
   const SRMHDState& state_r,
   const SRMHDState& flux_l,
   const SRMHDState& flux_r,
   amrex::Real speed_l,
   amrex::Real speed_r
) noexcept
{
   SRMHDHLLCStates result{};
   const SRMHDState hll_state = srmhd_hll_state(
      state_l, state_r, flux_l, flux_r, speed_l, speed_r
   );
   const SRMHDState hll_flux = srmhd_hll_flux(
      state_l, state_r, flux_l, flux_r, speed_l, speed_r
   );
   const amrex::Real magnetic_scale = amrex::max(
      1.0,
      amrex::max(
         amrex::Math::abs(hll_state.bn),
         amrex::max(amrex::Math::abs(hll_state.bt1), amrex::Math::abs(hll_state.bt2))
      )
   );
   const bool zero_normal_field = amrex::Math::abs(hll_state.bn)
      <= 1.0e-12 * magnetic_scale;
   amrex::Real quadratic = hll_flux.energy;
   amrex::Real linear = -hll_flux.momn - hll_state.energy;
   amrex::Real constant = hll_state.momn;
   if (!zero_normal_field) {
      const amrex::Real magnetic_flux_dot = hll_state.bt1 * hll_flux.bt1
         + hll_state.bt2 * hll_flux.bt2;
      quadratic -= magnetic_flux_dot;
      linear += hll_state.bt1 * hll_state.bt1
         + hll_state.bt2 * hll_state.bt2
         + hll_flux.bt1 * hll_flux.bt1
         + hll_flux.bt2 * hll_flux.bt2;
      constant -= magnetic_flux_dot;
   }
   const amrex::Real contact_speed = srmhd_minus_quadratic_root(
      quadratic, linear, constant
   );
   if (!amrex::Math::isfinite(contact_speed)
       || !(contact_speed > speed_l && contact_speed < speed_r)
       || !(amrex::Math::abs(contact_speed) < 1.0)) {
      return result;
   }

   amrex::Real velocity_t1 = 0.0;
   amrex::Real velocity_t2 = 0.0;
   amrex::Real total_pressure = hll_flux.momn
      - hll_flux.energy * contact_speed;
   amrex::Real magnetic_t1_l = 0.0;
   amrex::Real magnetic_t2_l = 0.0;
   amrex::Real magnetic_t1_r = 0.0;
   amrex::Real magnetic_t2_r = 0.0;
   if (!zero_normal_field) {
      velocity_t1 = (
         hll_state.bt1 * contact_speed - hll_flux.bt1
      ) / hll_state.bn;
      velocity_t2 = (
         hll_state.bt2 * contact_speed - hll_flux.bt2
      ) / hll_state.bn;
      const amrex::Real velocity_squared = contact_speed * contact_speed
         + velocity_t1 * velocity_t1 + velocity_t2 * velocity_t2;
      if (!(velocity_squared < 1.0) || !amrex::Math::isfinite(velocity_squared)) {
         return result;
      }
      const amrex::Real velocity_dot_magnetic = contact_speed * hll_state.bn
         + velocity_t1 * hll_state.bt1 + velocity_t2 * hll_state.bt2;
      total_pressure = hll_flux.momn
         - (hll_flux.energy - hll_state.bn * velocity_dot_magnetic)
            * contact_speed
         + hll_state.bn * hll_state.bn * (1.0 - velocity_squared);
      magnetic_t1_l = hll_state.bt1;
      magnetic_t2_l = hll_state.bt2;
      magnetic_t1_r = hll_state.bt1;
      magnetic_t2_r = hll_state.bt2;
   } else {
      const amrex::Real denom_l = speed_l - contact_speed;
      const amrex::Real denom_r = speed_r - contact_speed;
      magnetic_t1_l = (speed_l - primitive_l.v1) * primitive_l.b2 / denom_l;
      magnetic_t2_l = (speed_l - primitive_l.v1) * primitive_l.b3 / denom_l;
      magnetic_t1_r = (speed_r - primitive_r.v1) * primitive_r.b2 / denom_r;
      magnetic_t2_r = (speed_r - primitive_r.v1) * primitive_r.b3 / denom_r;
   }
   if (!amrex::Math::isfinite(total_pressure) || !(total_pressure > 0.0)) {
      return result;
   }

   const auto build_state = [=] AMREX_GPU_HOST_DEVICE(
      const RMHDPrimitive& primitive,
      const SRMHDState& state,
      const SRMHDState& flux,
      amrex::Real wave_speed,
      amrex::Real magnetic_t1,
      amrex::Real magnetic_t2
   ) noexcept -> SRMHDState {
      const amrex::Real denominator = wave_speed - contact_speed;
      SRMHDState star{};
      star.dens = (wave_speed - primitive.v1) * state.dens / denominator;
      star.bn = hll_state.bn;
      star.bt1 = magnetic_t1;
      star.bt2 = magnetic_t2;
      if (zero_normal_field) {
         star.momt1 = (wave_speed - primitive.v1) * state.momt1 / denominator;
         star.momt2 = (wave_speed - primitive.v1) * state.momt2 / denominator;
         star.energy = (
            wave_speed * state.energy - state.momn
            + total_pressure * contact_speed
         ) / denominator;
         star.momn = (star.energy + total_pressure) * contact_speed;
      } else {
         const amrex::Real velocity_squared = contact_speed * contact_speed
            + velocity_t1 * velocity_t1 + velocity_t2 * velocity_t2;
         const amrex::Real velocity_dot_magnetic = contact_speed * star.bn
            + velocity_t1 * star.bt1 + velocity_t2 * star.bt2;
         star.momt1 = (
            -star.bn * (
               star.bt1 * (1.0 - velocity_squared)
               + velocity_dot_magnetic * velocity_t1
            )
            + wave_speed * state.momt1 - flux.momt1
         ) / denominator;
         star.momt2 = (
            -star.bn * (
               star.bt2 * (1.0 - velocity_squared)
               + velocity_dot_magnetic * velocity_t2
            )
            + wave_speed * state.momt2 - flux.momt2
         ) / denominator;
         star.energy = (
            wave_speed * state.energy - state.momn
            + total_pressure * contact_speed - velocity_dot_magnetic * star.bn
         ) / denominator;
         star.momn = (star.energy + total_pressure) * contact_speed
            - velocity_dot_magnetic * star.bn;
      }
      return star;
   };

   result.state_l = build_state(
      primitive_l, state_l, flux_l, speed_l, magnetic_t1_l, magnetic_t2_l
   );
   result.state_r = build_state(
      primitive_r, state_r, flux_r, speed_r, magnetic_t1_r, magnetic_t2_r
   );
   result.flux_l = srmhd_wave_flux(
      flux_l, speed_l, result.state_l, state_l
   );
   result.flux_r = srmhd_wave_flux(
      flux_r, speed_r, result.state_r, state_r
   );
   result.contact_speed = contact_speed;
   result.total_pressure = total_pressure;
   result.valid = finite_srmhd_state(result.state_l)
      && finite_srmhd_state(result.state_r)
      && finite_srmhd_state(result.flux_l)
      && finite_srmhd_state(result.flux_r)
      && result.state_l.dens > 0.0
      && result.state_r.dens > 0.0
      && result.state_l.energy > 0.0
      && result.state_r.energy > 0.0;
   return result;
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE inline
SRMHDRiemannSample srmhd_hllc_sample(
   RMHDPrimitive primitive_l,
   RMHDPrimitive primitive_r,
   amrex::Real gamma,
   amrex::Real sample_speed
) noexcept
{
   const amrex::Real shared_bn = 0.5 * (primitive_l.b1 + primitive_r.b1);
   primitive_l.b1 = shared_bn;
   primitive_r.b1 = shared_bn;
   const SRMHDState state_l = srmhd_state_from_primitive(primitive_l);
   const SRMHDState state_r = srmhd_state_from_primitive(primitive_r);
   const SRMHDState flux_l = srmhd_flux_from_primitive(primitive_l);
   const SRMHDState flux_r = srmhd_flux_from_primitive(primitive_r);
   if (primitive_l.rho == primitive_r.rho
       && primitive_l.eps == primitive_r.eps
       && primitive_l.pressure == primitive_r.pressure
       && primitive_l.v1 == primitive_r.v1
       && primitive_l.v2 == primitive_r.v2
       && primitive_l.v3 == primitive_r.v3
       && primitive_l.b1 == primitive_r.b1
       && primitive_l.b2 == primitive_r.b2
       && primitive_l.b3 == primitive_r.b3) {
      return {
         state_l,
         flux_l,
         primitive_l.v1,
         primitive_l.pressure + 0.5
            * generated::rmhd_comoving_magnetic_squared(
               primitive_l, minkowski_spacetime_metric()
            ),
         false,
      };
   }
   const RMHDWaveSpeeds waves_l = generated::srmhd_wave_speeds<0>(
      primitive_l, minkowski_spacetime_metric(), gamma
   );
   const RMHDWaveSpeeds waves_r = generated::srmhd_wave_speeds<0>(
      primitive_r, minkowski_spacetime_metric(), gamma
   );
   const amrex::Real speed_l = amrex::min(waves_l.minus, waves_r.minus);
   const amrex::Real speed_r = amrex::max(waves_l.plus, waves_r.plus);
   if (sample_speed <= speed_l) {
      return {state_l, flux_l, 0.0, 0.0, false};
   }
   if (sample_speed >= speed_r) {
      return {state_r, flux_r, 0.0, 0.0, false};
   }
   const SRMHDHLLCStates intermediate = srmhd_hllc_intermediate_states(
      primitive_l, primitive_r, state_l, state_r, flux_l, flux_r,
      speed_l, speed_r
   );
   if (!intermediate.valid) {
      return srmhd_hll_sample(
         state_l, state_r, flux_l, flux_r, speed_l, speed_r, sample_speed
      );
   }
   if (sample_speed <= intermediate.contact_speed) {
      return {
         intermediate.state_l,
         intermediate.flux_l,
         intermediate.contact_speed,
         intermediate.total_pressure,
         false,
      };
   }
   return {
      intermediate.state_r,
      intermediate.flux_r,
      intermediate.contact_speed,
      intermediate.total_pressure,
      false,
   };
}

struct SRMHDHLLDVector {
   amrex::Real n = 0.0;
   amrex::Real t1 = 0.0;
   amrex::Real t2 = 0.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srmhd_hlld_dot(
   const SRMHDHLLDVector& left,
   const SRMHDHLLDVector& right
) noexcept
{
   return left.n * right.n + left.t1 * right.t1 + left.t2 * right.t2;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool finite_srmhd_hlld_vector(const SRMHDHLLDVector& vector) noexcept
{
   return amrex::Math::isfinite(vector.n)
      && amrex::Math::isfinite(vector.t1)
      && amrex::Math::isfinite(vector.t2);
}

struct SRMHDHLLDFastState {
   SRMHDState state{};
   SRMHDState flux{};
   SRMHDState jump{};
   SRMHDHLLDVector velocity{};
   SRMHDHLLDVector magnetic{};
   SRMHDHLLDVector k{};
   amrex::Real total_enthalpy = 0.0;
   amrex::Real eta = 0.0;
   amrex::Real alfven_speed = 0.0;
   amrex::Real outer_speed = 0.0;
   bool valid = false;
};

struct SRMHDHLLDEvaluation {
   SRMHDHLLDFastState fast_l{};
   SRMHDHLLDFastState fast_r{};
   SRMHDHLLDVector contact_magnetic{};
   SRMHDHLLDVector contact_velocity_l{};
   SRMHDHLLDVector contact_velocity_r{};
   amrex::Real residual = 0.0;
   bool valid = false;
};

struct SRMHDHLLDSolution {
   SRMHDHLLDEvaluation evaluation{};
   amrex::Real total_pressure = 0.0;
   bool valid = false;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDState srmhd_hlld_jump(
   const SRMHDState& state,
   const SRMHDState& flux,
   amrex::Real wave_speed
) noexcept
{
   return srmhd_state_add_scaled(
      srmhd_state_add_scaled({}, wave_speed, state), -1.0, flux
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDHLLDFastState srmhd_hlld_fast_state(
   const SRMHDState& input_state,
   const SRMHDState& input_flux,
   amrex::Real outer_speed,
   amrex::Real normal_magnetic,
   amrex::Real total_pressure,
   amrex::Real eta_side
) noexcept
{
   SRMHDHLLDFastState result{};
   result.outer_speed = outer_speed;
   result.jump = srmhd_hlld_jump(input_state, input_flux, outer_speed);
   if (!(total_pressure > 0.0) || !amrex::Math::isfinite(total_pressure)) {
      return result;
   }

   const amrex::Real a = result.jump.momn
      - outer_speed * result.jump.energy
      + total_pressure * (1.0 - outer_speed * outer_speed);
   const amrex::Real g = result.jump.bt1 * result.jump.bt1
      + result.jump.bt2 * result.jump.bt2;
   const amrex::Real c = result.jump.momt1 * result.jump.bt1
      + result.jump.momt2 * result.jump.bt2;
   const amrex::Real q = -a - g
      + normal_magnetic * normal_magnetic
         * (1.0 - outer_speed * outer_speed);
   const amrex::Real x = normal_magnetic
         * (a * outer_speed * normal_magnetic + c)
      - (a + g) * (outer_speed * total_pressure + result.jump.energy);
   const amrex::Real algebra_scale = amrex::max(
      1.0,
      amrex::max(
         amrex::Math::abs(a) + g,
         amrex::Math::abs(outer_speed * total_pressure + result.jump.energy)
      )
   );
   if (!amrex::Math::isfinite(x)
       || amrex::Math::abs(x) <= 1.0e-14 * algebra_scale * algebra_scale) {
      return result;
   }

   result.velocity.n = (
      normal_magnetic * (a * normal_magnetic + outer_speed * c)
      - (a + g) * (total_pressure + result.jump.momn)
   ) / x;
   const amrex::Real transverse_common = c + normal_magnetic
      * (outer_speed * result.jump.momn - result.jump.energy);
   result.velocity.t1 = (
      q * result.jump.momt1 + result.jump.bt1 * transverse_common
   ) / x;
   result.velocity.t2 = (
      q * result.jump.momt2 + result.jump.bt2 * transverse_common
   ) / x;
   const amrex::Real velocity_squared = srmhd_hlld_dot(
      result.velocity, result.velocity
   );
   const amrex::Real fast_denominator = outer_speed - result.velocity.n;
   if (!finite_srmhd_hlld_vector(result.velocity)
       || !(velocity_squared < 1.0)
       || amrex::Math::abs(fast_denominator) <= 1.0e-14) {
      return result;
   }

   result.magnetic = {
      normal_magnetic,
      (result.jump.bt1 - normal_magnetic * result.velocity.t1)
         / fast_denominator,
      (result.jump.bt2 - normal_magnetic * result.velocity.t2)
         / fast_denominator,
   };
   const SRMHDHLLDVector jump_momentum{
      result.jump.momn, result.jump.momt1, result.jump.momt2
   };
   result.total_enthalpy = total_pressure
      + (result.jump.energy - srmhd_hlld_dot(result.velocity, jump_momentum))
         / fast_denominator;
   if (!finite_srmhd_hlld_vector(result.magnetic)
       || !amrex::Math::isfinite(result.total_enthalpy)
       || !(result.total_enthalpy > total_pressure)) {
      return result;
   }

   result.state.dens = result.jump.dens / fast_denominator;
   result.state.bn = normal_magnetic;
   result.state.bt1 = result.magnetic.t1;
   result.state.bt2 = result.magnetic.t2;
   const amrex::Real velocity_dot_magnetic = srmhd_hlld_dot(
      result.velocity, result.magnetic
   );
   result.state.energy = (
      result.jump.energy + total_pressure * result.velocity.n
      - velocity_dot_magnetic * normal_magnetic
   ) / fast_denominator;
   result.state.momn = (result.state.energy + total_pressure)
         * result.velocity.n
      - velocity_dot_magnetic * result.magnetic.n;
   result.state.momt1 = (result.state.energy + total_pressure)
         * result.velocity.t1
      - velocity_dot_magnetic * result.magnetic.t1;
   result.state.momt2 = (result.state.energy + total_pressure)
         * result.velocity.t2
      - velocity_dot_magnetic * result.magnetic.t2;
   result.flux = srmhd_wave_flux(
      input_flux, outer_speed, result.state, input_state
   );

   const amrex::Real normal_sign = normal_magnetic < 0.0 ? -1.0 : 1.0;
   result.eta = eta_side * normal_sign * std::sqrt(result.total_enthalpy);
   const amrex::Real k_denominator = outer_speed * total_pressure
      + result.jump.energy + normal_magnetic * result.eta;
   if (!amrex::Math::isfinite(k_denominator)
       || amrex::Math::abs(k_denominator) <= 1.0e-14 * algebra_scale) {
      return result;
   }
   result.k = {
      (result.jump.momn + total_pressure
         + result.jump.bn * result.eta) / k_denominator,
      (result.jump.momt1 + result.jump.bt1 * result.eta) / k_denominator,
      (result.jump.momt2 + result.jump.bt2 * result.eta) / k_denominator,
   };
   result.alfven_speed = result.k.n;
   result.valid = finite_srmhd_state(result.state)
      && finite_srmhd_state(result.flux)
      && finite_srmhd_hlld_vector(result.k)
      && result.state.dens > 0.0
      && result.state.energy > 0.0
      && amrex::Math::abs(result.alfven_speed) < 1.0;
   return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDHLLDVector srmhd_hlld_contact_velocity(
   const SRMHDHLLDVector& k,
   const SRMHDHLLDVector& magnetic,
   amrex::Real eta
) noexcept
{
   const amrex::Real denominator = eta - srmhd_hlld_dot(k, magnetic);
   const amrex::Real factor = (1.0 - srmhd_hlld_dot(k, k)) / denominator;
   return {
      k.n - magnetic.n * factor,
      k.t1 - magnetic.t1 * factor,
      k.t2 - magnetic.t2 * factor,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDHLLDEvaluation srmhd_hlld_evaluate(
   const SRMHDState& state_l,
   const SRMHDState& state_r,
   const SRMHDState& flux_l,
   const SRMHDState& flux_r,
   amrex::Real speed_l,
   amrex::Real speed_r,
   amrex::Real normal_magnetic,
   amrex::Real total_pressure
) noexcept
{
   SRMHDHLLDEvaluation result{};
   result.fast_l = srmhd_hlld_fast_state(
      state_l, flux_l, speed_l, normal_magnetic, total_pressure, -1.0
   );
   result.fast_r = srmhd_hlld_fast_state(
      state_r, flux_r, speed_r, normal_magnetic, total_pressure, 1.0
   );
   if (!result.fast_l.valid || !result.fast_r.valid) {
      return result;
   }
   const amrex::Real delta_k = result.fast_r.k.n - result.fast_l.k.n;
   if (!amrex::Math::isfinite(delta_k)
       || amrex::Math::abs(delta_k) <= 1.0e-14) {
      return result;
   }
   result.contact_magnetic = {
      normal_magnetic,
      (
         result.fast_r.magnetic.t1 * (
            result.fast_r.alfven_speed - result.fast_r.velocity.n
         ) + normal_magnetic * result.fast_r.velocity.t1
         - result.fast_l.magnetic.t1 * (
            result.fast_l.alfven_speed - result.fast_l.velocity.n
         ) - normal_magnetic * result.fast_l.velocity.t1
      ) / delta_k,
      (
         result.fast_r.magnetic.t2 * (
            result.fast_r.alfven_speed - result.fast_r.velocity.n
         ) + normal_magnetic * result.fast_r.velocity.t2
         - result.fast_l.magnetic.t2 * (
            result.fast_l.alfven_speed - result.fast_l.velocity.n
         ) - normal_magnetic * result.fast_l.velocity.t2
      ) / delta_k,
   };
   const amrex::Real denominator_l = result.fast_l.eta
      - srmhd_hlld_dot(result.fast_l.k, result.contact_magnetic);
   const amrex::Real denominator_r = result.fast_r.eta
      - srmhd_hlld_dot(result.fast_r.k, result.contact_magnetic);
   const amrex::Real denominator_scale = amrex::max(
      1.0,
      amrex::max(amrex::Math::abs(result.fast_l.eta),
                 amrex::Math::abs(result.fast_r.eta))
   );
   if (!finite_srmhd_hlld_vector(result.contact_magnetic)
       || amrex::Math::abs(denominator_l) <= 1.0e-14 * denominator_scale
       || amrex::Math::abs(denominator_r) <= 1.0e-14 * denominator_scale) {
      return result;
   }
   result.contact_velocity_l = srmhd_hlld_contact_velocity(
      result.fast_l.k, result.contact_magnetic, result.fast_l.eta
   );
   result.contact_velocity_r = srmhd_hlld_contact_velocity(
      result.fast_r.k, result.contact_magnetic, result.fast_r.eta
   );
   const amrex::Real velocity_squared_l = srmhd_hlld_dot(
      result.contact_velocity_l, result.contact_velocity_l
   );
   const amrex::Real velocity_squared_r = srmhd_hlld_dot(
      result.contact_velocity_r, result.contact_velocity_r
   );
   result.residual = result.contact_velocity_l.n
      - result.contact_velocity_r.n;
   result.valid = finite_srmhd_hlld_vector(result.contact_velocity_l)
      && finite_srmhd_hlld_vector(result.contact_velocity_r)
      && amrex::Math::isfinite(result.residual)
      && velocity_squared_l < 1.0
      && velocity_squared_r < 1.0;
   return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool srmhd_hlld_physical_solution(
   const SRMHDHLLDEvaluation& evaluation,
   amrex::Real speed_l,
   amrex::Real speed_r,
   amrex::Real total_pressure
) noexcept
{
   if (!evaluation.valid || !(total_pressure > 0.0)
       || amrex::Math::abs(evaluation.residual) > 1.0e-7) {
      return false;
   }
   const amrex::Real contact_speed = 0.5 * (
      evaluation.contact_velocity_l.n + evaluation.contact_velocity_r.n
   );
   return evaluation.fast_l.total_enthalpy > total_pressure
      && evaluation.fast_r.total_enthalpy > total_pressure
      && evaluation.fast_l.velocity.n > speed_l
      && evaluation.fast_r.velocity.n < speed_r
      && evaluation.fast_l.alfven_speed > speed_l
      && evaluation.fast_r.alfven_speed < speed_r
      && contact_speed > evaluation.fast_l.alfven_speed
      && contact_speed < evaluation.fast_r.alfven_speed
      && evaluation.contact_velocity_l.n
         > evaluation.fast_l.alfven_speed
      && evaluation.contact_velocity_r.n
         < evaluation.fast_r.alfven_speed;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srmhd_total_pressure(const RMHDPrimitive& primitive) noexcept
{
   return primitive.pressure + 0.5
      * generated::rmhd_comoving_magnetic_squared(
         primitive, minkowski_spacetime_metric()
      );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real srmhd_hlld_zero_normal_pressure(
   const SRMHDState& hll_state,
   const SRMHDState& hll_flux
) noexcept
{
   const amrex::Real linear = hll_state.energy - hll_flux.momn;
   const amrex::Real constant = hll_state.momn * hll_flux.energy
      - hll_flux.momn * hll_state.energy;
   const amrex::Real discriminant = linear * linear - 4.0 * constant;
   if (!(discriminant >= 0.0) || !amrex::Math::isfinite(discriminant)) {
      return -1.0;
   }
   return 0.5 * (-linear + std::sqrt(discriminant));
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE inline
SRMHDHLLDSolution srmhd_hlld_pressure_solution(
   const RMHDPrimitive& primitive_l,
   const RMHDPrimitive& primitive_r,
   const SRMHDState& state_l,
   const SRMHDState& state_r,
   const SRMHDState& flux_l,
   const SRMHDState& flux_r,
   amrex::Real speed_l,
   amrex::Real speed_r,
   amrex::Real normal_magnetic
) noexcept
{
   SRMHDHLLDSolution solution{};
   const SRMHDState hll_state = srmhd_hll_state(
      state_l, state_r, flux_l, flux_r, speed_l, speed_r
   );
   const SRMHDState hll_flux = srmhd_hll_flux(
      state_l, state_r, flux_l, flux_r, speed_l, speed_r
   );
   const SRMHDHLLCStates hllc = srmhd_hllc_intermediate_states(
      primitive_l, primitive_r, state_l, state_r, flux_l, flux_r,
      speed_l, speed_r
   );
   const amrex::Real side_pressure_scale = amrex::max(
      srmhd_total_pressure(primitive_l), srmhd_total_pressure(primitive_r)
   );
   amrex::Real pressure_guess = hllc.valid
      ? hllc.total_pressure
      : 0.5 * (
         srmhd_total_pressure(primitive_l) + srmhd_total_pressure(primitive_r)
      );
   const amrex::Real zero_normal_pressure = srmhd_hlld_zero_normal_pressure(
      hll_state, hll_flux
   );
   if (zero_normal_pressure > 0.0
       && normal_magnetic * normal_magnetic
          < 0.1 * amrex::max(pressure_guess, 1.0e-30)) {
      pressure_guess = zero_normal_pressure;
   }
   const amrex::Real pressure_scale = amrex::max(
      1.0e-30, amrex::max(side_pressure_scale, pressure_guess)
   );
   const amrex::Real pressure_floor = 1.0e-12 * pressure_scale;
   pressure_guess = amrex::max(pressure_floor, pressure_guess);

   SRMHDHLLDEvaluation current = srmhd_hlld_evaluate(
      state_l, state_r, flux_l, flux_r, speed_l, speed_r,
      normal_magnetic, pressure_guess
   );
   if (srmhd_hlld_physical_solution(
          current, speed_l, speed_r, pressure_guess
       )) {
      return {current, pressure_guess, true};
   }
   amrex::Real previous_pressure = amrex::max(
      pressure_floor, pressure_guess * (1.0 - 1.0e-4)
   );
   SRMHDHLLDEvaluation previous = srmhd_hlld_evaluate(
      state_l, state_r, flux_l, flux_r, speed_l, speed_r,
      normal_magnetic, previous_pressure
   );
   for (int iteration = 0; iteration < 20; ++iteration) {
      if (!current.valid || !previous.valid) {
         break;
      }
      const amrex::Real residual_difference = current.residual
         - previous.residual;
      const amrex::Real residual_scale = amrex::max(
         1.0, amrex::max(amrex::Math::abs(current.residual),
                         amrex::Math::abs(previous.residual))
      );
      if (amrex::Math::abs(residual_difference)
          <= 1.0e-14 * residual_scale) {
         break;
      }
      amrex::Real next_pressure = pressure_guess
         - current.residual * (pressure_guess - previous_pressure)
            / residual_difference;
      const amrex::Real lower_step = amrex::max(
         pressure_floor, 0.2 * pressure_guess
      );
      const amrex::Real upper_step = 5.0 * pressure_guess;
      next_pressure = amrex::max(
         lower_step, amrex::min(upper_step, next_pressure)
      );
      previous_pressure = pressure_guess;
      previous = current;
      pressure_guess = next_pressure;
      current = srmhd_hlld_evaluate(
         state_l, state_r, flux_l, flux_r, speed_l, speed_r,
         normal_magnetic, pressure_guess
      );
      if (srmhd_hlld_physical_solution(
             current, speed_l, speed_r, pressure_guess
          )) {
         return {current, pressure_guess, true};
      }
   }

   // The paper prescribes HLL when the nonlinear state is inadmissible.  A
   // logarithmic bracket is attempted first so that a poor secant seed does
   // not turn an otherwise valid HLLD fan into that robust limit.
   amrex::Real scan_pressure = amrex::max(
      pressure_floor, pressure_guess / 256.0
   );
   SRMHDHLLDEvaluation scan_previous{};
   amrex::Real scan_previous_pressure = scan_pressure;
   bool have_previous = false;
   for (int scan = 0; scan < 20; ++scan) {
      const SRMHDHLLDEvaluation scan_current = srmhd_hlld_evaluate(
         state_l, state_r, flux_l, flux_r, speed_l, speed_r,
         normal_magnetic, scan_pressure
      );
      if (srmhd_hlld_physical_solution(
             scan_current, speed_l, speed_r, scan_pressure
          )) {
         return {scan_current, scan_pressure, true};
      }
      if (scan_current.valid && have_previous
          && scan_current.residual * scan_previous.residual <= 0.0) {
         amrex::Real bracket_l = scan_previous_pressure;
         amrex::Real bracket_r = scan_pressure;
         SRMHDHLLDEvaluation evaluation_l = scan_previous;
         SRMHDHLLDEvaluation evaluation_r = scan_current;
         for (int iteration = 0; iteration < 40; ++iteration) {
            const amrex::Real midpoint = 0.5 * (bracket_l + bracket_r);
            const SRMHDHLLDEvaluation evaluation_mid = srmhd_hlld_evaluate(
               state_l, state_r, flux_l, flux_r, speed_l, speed_r,
               normal_magnetic, midpoint
            );
            if (!evaluation_mid.valid) {
               break;
            }
            if (srmhd_hlld_physical_solution(
                   evaluation_mid, speed_l, speed_r, midpoint
                )) {
               return {evaluation_mid, midpoint, true};
            }
            if (evaluation_l.residual * evaluation_mid.residual <= 0.0) {
               bracket_r = midpoint;
               evaluation_r = evaluation_mid;
            } else {
               bracket_l = midpoint;
               evaluation_l = evaluation_mid;
            }
            if (amrex::Math::abs(bracket_r - bracket_l)
                <= 1.0e-10 * amrex::max(pressure_scale, midpoint)) {
               const SRMHDHLLDEvaluation& candidate =
                  amrex::Math::abs(evaluation_l.residual)
                     < amrex::Math::abs(evaluation_r.residual)
                  ? evaluation_l : evaluation_r;
               const amrex::Real candidate_pressure =
                  amrex::Math::abs(evaluation_l.residual)
                     < amrex::Math::abs(evaluation_r.residual)
                  ? bracket_l : bracket_r;
               if (srmhd_hlld_physical_solution(
                      candidate, speed_l, speed_r, candidate_pressure
                   )) {
                  return {candidate, candidate_pressure, true};
               }
               break;
            }
         }
      }
      if (scan_current.valid) {
         scan_previous = scan_current;
         scan_previous_pressure = scan_pressure;
         have_previous = true;
      }
      scan_pressure *= 2.0;
   }
   return solution;
}

struct SRMHDHLLDContactState {
   SRMHDState state{};
   SRMHDState flux{};
   bool valid = false;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SRMHDHLLDContactState srmhd_hlld_contact_state(
   const SRMHDHLLDFastState& fast,
   const SRMHDHLLDVector& contact_velocity,
   const SRMHDHLLDVector& contact_magnetic,
   amrex::Real total_pressure
) noexcept
{
   SRMHDHLLDContactState result{};
   const amrex::Real denominator = fast.alfven_speed
      - contact_velocity.n;
   if (amrex::Math::abs(denominator) <= 1.0e-14) {
      return result;
   }
   result.state.dens = fast.state.dens
      * (fast.alfven_speed - fast.velocity.n) / denominator;
   result.state.bn = contact_magnetic.n;
   result.state.bt1 = contact_magnetic.t1;
   result.state.bt2 = contact_magnetic.t2;
   const amrex::Real velocity_dot_magnetic = srmhd_hlld_dot(
      contact_velocity, contact_magnetic
   );
   result.state.energy = (
      fast.alfven_speed * fast.state.energy - fast.state.momn
      + total_pressure * contact_velocity.n
      - velocity_dot_magnetic * contact_magnetic.n
   ) / denominator;
   result.state.momn = (result.state.energy + total_pressure)
         * contact_velocity.n
      - velocity_dot_magnetic * contact_magnetic.n;
   result.state.momt1 = (result.state.energy + total_pressure)
         * contact_velocity.t1
      - velocity_dot_magnetic * contact_magnetic.t1;
   result.state.momt2 = (result.state.energy + total_pressure)
         * contact_velocity.t2
      - velocity_dot_magnetic * contact_magnetic.t2;
   result.flux = srmhd_wave_flux(
      fast.flux, fast.alfven_speed, result.state, fast.state
   );
   result.valid = finite_srmhd_state(result.state)
      && finite_srmhd_state(result.flux)
      && result.state.dens > 0.0
      && result.state.energy > 0.0;
   return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool equal_srmhd_primitive(
   const RMHDPrimitive& left,
   const RMHDPrimitive& right
) noexcept
{
   return left.rho == right.rho
      && left.eps == right.eps
      && left.pressure == right.pressure
      && left.v1 == right.v1
      && left.v2 == right.v2
      && left.v3 == right.v3
      && left.b1 == right.b1
      && left.b2 == right.b2
      && left.b3 == right.b3;
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE inline
SRMHDRiemannSample srmhd_hlld_sample(
   RMHDPrimitive primitive_l,
   RMHDPrimitive primitive_r,
   amrex::Real gamma,
   amrex::Real sample_speed
) noexcept
{
   const amrex::Real shared_bn = 0.5 * (primitive_l.b1 + primitive_r.b1);
   primitive_l.b1 = shared_bn;
   primitive_r.b1 = shared_bn;
   const SRMHDState state_l = srmhd_state_from_primitive(primitive_l);
   const SRMHDState state_r = srmhd_state_from_primitive(primitive_r);
   const SRMHDState flux_l = srmhd_flux_from_primitive(primitive_l);
   const SRMHDState flux_r = srmhd_flux_from_primitive(primitive_r);
   if (equal_srmhd_primitive(primitive_l, primitive_r)) {
      return {state_l, flux_l, primitive_l.v1,
              srmhd_total_pressure(primitive_l), false};
   }
   const RMHDWaveSpeeds waves_l = generated::srmhd_wave_speeds<0>(
      primitive_l, minkowski_spacetime_metric(), gamma
   );
   const RMHDWaveSpeeds waves_r = generated::srmhd_wave_speeds<0>(
      primitive_r, minkowski_spacetime_metric(), gamma
   );
   const amrex::Real speed_l = amrex::min(waves_l.minus, waves_r.minus);
   const amrex::Real speed_r = amrex::max(waves_l.plus, waves_r.plus);
   if (sample_speed <= speed_l) {
      return {state_l, flux_l, 0.0, 0.0, false};
   }
   if (sample_speed >= speed_r) {
      return {state_r, flux_r, 0.0, 0.0, false};
   }
   const amrex::Real magnetic_scale = amrex::max(
      1.0,
      amrex::max(
         amrex::Math::abs(shared_bn),
         amrex::max(
            amrex::max(amrex::Math::abs(primitive_l.b2),
                       amrex::Math::abs(primitive_l.b3)),
            amrex::max(amrex::Math::abs(primitive_r.b2),
                       amrex::Math::abs(primitive_r.b3))
         )
      )
   );
   if (amrex::Math::abs(shared_bn) <= 1.0e-12 * magnetic_scale) {
      return srmhd_hllc_sample(
         primitive_l, primitive_r, gamma, sample_speed
      );
   }

   const SRMHDHLLDSolution solution = srmhd_hlld_pressure_solution(
      primitive_l, primitive_r, state_l, state_r, flux_l, flux_r,
      speed_l, speed_r, shared_bn
   );
   if (!solution.valid) {
      return srmhd_hll_sample(
         state_l, state_r, flux_l, flux_r, speed_l, speed_r, sample_speed
      );
   }
   const SRMHDHLLDEvaluation& fan = solution.evaluation;
   const SRMHDHLLDContactState contact_l = srmhd_hlld_contact_state(
      fan.fast_l, fan.contact_velocity_l, fan.contact_magnetic,
      solution.total_pressure
   );
   const SRMHDHLLDContactState contact_r = srmhd_hlld_contact_state(
      fan.fast_r, fan.contact_velocity_r, fan.contact_magnetic,
      solution.total_pressure
   );
   if (!contact_l.valid || !contact_r.valid) {
      return srmhd_hll_sample(
         state_l, state_r, flux_l, flux_r, speed_l, speed_r, sample_speed
      );
   }
   const amrex::Real contact_speed = 0.5 * (
      fan.contact_velocity_l.n + fan.contact_velocity_r.n
   );
   if (sample_speed <= fan.fast_l.alfven_speed) {
      return {fan.fast_l.state, fan.fast_l.flux, contact_speed,
              solution.total_pressure, false};
   }
   if (sample_speed <= contact_speed) {
      return {contact_l.state, contact_l.flux, contact_speed,
              solution.total_pressure, false};
   }
   if (sample_speed <= fan.fast_r.alfven_speed) {
      return {contact_r.state, contact_r.flux, contact_speed,
              solution.total_pressure, false};
   }
   return {fan.fast_r.state, fan.fast_r.flux, contact_speed,
           solution.total_pressure, false};
}

struct RMHDRiemannFluxResult {
   RMHDConserved flux{};
   bool used_hll = false;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDFrameVector rmhd_frame_linear_combination(
   const RMHDFaceFrame& frame,
   amrex::Real normal,
   amrex::Real tangent1,
   amrex::Real tangent2
) noexcept
{
   return {
      frame.basis_n.x * normal + frame.basis_t1.x * tangent1
         + frame.basis_t2.x * tangent2,
      frame.basis_n.y * normal + frame.basis_t1.y * tangent1
         + frame.basis_t2.y * tangent2,
      frame.basis_n.z * normal + frame.basis_t1.z * tangent1
         + frame.basis_t2.z * tangent2,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rmhd_coframe_linear_combination(
   const RMHDFaceFrame& frame,
   int component,
   amrex::Real normal,
   amrex::Real tangent1,
   amrex::Real tangent2
) noexcept
{
   return rmhd_frame_component(frame.coframe_n, component) * normal
      + rmhd_frame_component(frame.coframe_t1, component) * tangent1
      + rmhd_frame_component(frame.coframe_t2, component) * tangent2;
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rmhd_face_magnetic_flux_component(
   int component,
   const SRMHDRiemannSample& sample,
   const RMHDFaceFrame& frame,
   const SpacetimeMetric& metric,
   const RMHDFrameVector& magnetic_state,
   const RMHDFrameVector& magnetic_flux
) noexcept
{
   if (component == Direction) {
      return 0.0;
   }
   const amrex::Real normal_scale = frame.normal_coordinate_scale;
   const amrex::Real normal_shift = rmhd_shift_component(metric, Direction);
   return metric.sqrt_det_gamma * (
      metric.lapse * normal_scale
         * rmhd_frame_component(magnetic_flux, component)
      - normal_shift * rmhd_frame_component(magnetic_state, component)
      + rmhd_shift_component(metric, component) * normal_scale
         * sample.state.bn
   );
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDConserved rmhd_face_sample_to_coordinate_flux(
   const SRMHDRiemannSample& sample,
   const RMHDFaceFrame& frame,
   const SpacetimeMetric& metric
) noexcept
{
   static_assert(Direction >= 0 && Direction < 3);
   const amrex::Real normal_scale = frame.normal_coordinate_scale;
   const amrex::Real normal_shift = rmhd_shift_component(metric, Direction);
   const amrex::Real lapse_normal_scale = metric.lapse * normal_scale;
   const amrex::Real volume = metric.sqrt_det_gamma;
   RMHDConserved result{};
   result.dens = volume * (
      lapse_normal_scale * sample.flux.dens
      - normal_shift * sample.state.dens
   );
   result.mom1 = volume * (
      lapse_normal_scale * rmhd_coframe_linear_combination(
         frame, 0, sample.flux.momn, sample.flux.momt1,
         sample.flux.momt2
      )
      - normal_shift * rmhd_coframe_linear_combination(
         frame, 0, sample.state.momn, sample.state.momt1,
         sample.state.momt2
      )
   );
   result.mom2 = volume * (
      lapse_normal_scale * rmhd_coframe_linear_combination(
         frame, 1, sample.flux.momn, sample.flux.momt1,
         sample.flux.momt2
      )
      - normal_shift * rmhd_coframe_linear_combination(
         frame, 1, sample.state.momn, sample.state.momt1,
         sample.state.momt2
      )
   );
   result.mom3 = volume * (
      lapse_normal_scale * rmhd_coframe_linear_combination(
         frame, 2, sample.flux.momn, sample.flux.momt1,
         sample.flux.momt2
      )
      - normal_shift * rmhd_coframe_linear_combination(
         frame, 2, sample.state.momn, sample.state.momt1,
         sample.state.momt2
      )
   );
   result.tau = volume * (
      lapse_normal_scale * (sample.flux.energy - sample.flux.dens)
      - normal_shift * (sample.state.energy - sample.state.dens)
   );

   const RMHDFrameVector magnetic_state = rmhd_frame_linear_combination(
      frame, sample.state.bn, sample.state.bt1, sample.state.bt2
   );
   const RMHDFrameVector magnetic_flux = rmhd_frame_linear_combination(
      frame, sample.flux.bn, sample.flux.bt1, sample.flux.bt2
   );
   result.b1 = rmhd_face_magnetic_flux_component<Direction>(
      0, sample, frame, metric, magnetic_state, magnetic_flux
   );
   result.b2 = rmhd_face_magnetic_flux_component<Direction>(
      1, sample, frame, metric, magnetic_state, magnetic_flux
   );
   result.b3 = rmhd_face_magnetic_flux_component<Direction>(
      2, sample, frame, metric, magnetic_state, magnetic_flux
   );
   return result;
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE inline
RMHDRiemannFluxResult rmhd_hllc_flux(
   const RMHDPrimitive& primitive_l,
   const RMHDPrimitive& primitive_r,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const RMHDFaceFrame frame = rmhd_face_frame<Direction>(metric);
   const RMHDPrimitive local_l = rmhd_primitive_to_face_frame(
      primitive_l, frame
   );
   const RMHDPrimitive local_r = rmhd_primitive_to_face_frame(
      primitive_r, frame
   );
   const SRMHDRiemannSample sample = srmhd_hllc_sample(
      local_l, local_r, eos.gamma, frame.interface_speed
   );
   return {
      rmhd_face_sample_to_coordinate_flux<Direction>(sample, frame, metric),
      sample.used_hll,
   };
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE inline
RMHDRiemannFluxResult rmhd_hlld_flux(
   const RMHDPrimitive& primitive_l,
   const RMHDPrimitive& primitive_r,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const RMHDFaceFrame frame = rmhd_face_frame<Direction>(metric);
   const RMHDPrimitive local_l = rmhd_primitive_to_face_frame(
      primitive_l, frame
   );
   const RMHDPrimitive local_r = rmhd_primitive_to_face_frame(
      primitive_r, frame
   );
   const SRMHDRiemannSample sample = srmhd_hlld_sample(
      local_l, local_r, eos.gamma, frame.interface_speed
   );
   return {
      rmhd_face_sample_to_coordinate_flux<Direction>(sample, frame, metric),
      sample.used_hll,
   };
}

} // namespace pyfluid_amrex
