#include "pyfluid_amrex/relativistic_flux_limiter.hpp"

#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>

#include <limits>

#include "pyfluid_amrex/build_options.hpp"
#include "pyfluid_amrex/black_hole_excision.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/flux.hpp"
#include "pyfluid_amrex/kernels_generated/riemann_ops.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "pyfluid_amrex/reference_metric_spacetime.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/time_integration.hpp"
#include "relativity/rmhd_system.hpp"
#include "relativity/relativistic_limiter_index.hpp"
#include "relativity/relativistic_limiter_metric.hpp"
#include "relativity/rmhd_correction_flux.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"
#include "pyfluid_amrex/spacetime_excision.hpp"
#include "relativity/black_hole_excision_kernels.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;
namespace {

using limiter_detail::left_cell;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDConserved stage_candidate(
   RHDConserved const& base,
   RHDConserved const& input,
   RHDConserved const& rhs,
   amrex::Real input_weight,
   amrex::Real increment_scale
) noexcept
{
   return {
      affine_state_update(base.dens, input.dens, input_weight, increment_scale * rhs.dens),
      affine_state_update(base.mom1, input.mom1, input_weight, increment_scale * rhs.mom1),
      affine_state_update(base.mom2, input.mom2, input_weight, increment_scale * rhs.mom2),
      affine_state_update(base.mom3, input.mom3, input_weight, increment_scale * rhs.mom3),
      affine_state_update(base.tau, input.tau, input_weight, increment_scale * rhs.tau),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDConserved add_scaled(
   RHDConserved state,
   RHDConserved const& increment,
   amrex::Real scale
) noexcept
{
   state.dens += scale * increment.dens;
   state.mom1 += scale * increment.mom1;
   state.mom2 += scale * increment.mom2;
   state.mom3 += scale * increment.mom3;
   state.tau += scale * increment.tau;
   return state;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDConserved difference(
   RHDConserved const& left,
   RHDConserved const& right
) noexcept
{
   return {
      left.dens - right.dens,
      left.mom1 - right.mom1,
      left.mom2 - right.mom2,
      left.mom3 - right.mom3,
      left.tau - right.tau,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDConserved blend(
   RHDConserved const& low_order,
   RHDConserved const& high_order,
   amrex::Real theta
) noexcept
{
   return add_scaled(low_order, difference(high_order, low_order), theta);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real admissibility_floor(
   RHDConserved const& state,
   SpacetimeMetric const& metric,
   amrex::Real thermodynamic_floor
) noexcept
{
   const RHDRecoveryState recovery = generated::rhd_recovery_state(state, metric);
   const amrex::Real energy = recovery.dens + recovery.tau;
   const amrex::Real momentum = std::sqrt(amrex::max(
      amrex::Real(0.0_rt), recovery.momentum_squared
   ));
   const amrex::Real scale = amrex::max(
      amrex::Math::abs(energy),
      amrex::max(amrex::Math::abs(recovery.dens), momentum)
   );
   const amrex::Real roundoff_floor =
      256.0_rt * std::numeric_limits<amrex::Real>::epsilon()
      * amrex::max(scale, std::numeric_limits<amrex::Real>::min());
   return amrex::max(roundoff_floor, thermodynamic_floor);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real admissibility_theta(
   RHDConserved const& low_order,
   RHDConserved const& high_order,
   SpacetimeMetric const& metric,
   amrex::Real density_floor,
   amrex::Real thermodynamic_floor
) noexcept
{
   const RHDRecoveryState high = generated::rhd_recovery_state(high_order, metric);
   const amrex::Real target = admissibility_floor(
      low_order, metric, thermodynamic_floor
   );
   return high.dens >= density_floor
      && rhd_conserved_admissibility_margin(high_order, metric) >= target
      ? 1.0_rt
      : 0.0_rt;
}

// Host factories supply named types to the shared CUDA launch templates.
struct StoredMetricFactory {
   amrex::MultiFab const& spacetime;
   amrex::MultiFab const& candidate_spacetime;
   ExcisionGridView excision;
   RelativisticDiodeBoundary diode;

   StoredLimiterMetricAccessor operator()(amrex::MFIter const& mfi) const
   {
      return {spacetime.const_array(mfi), candidate_spacetime.const_array(mfi),
              excision, diode};
   }
};

struct FixedMetricFactory {
   FixedLimiterMetricAccessor metric;

   FixedLimiterMetricAccessor operator()(amrex::MFIter const&) const
   {
      return metric;
   }
};

template <int Direction, int EOSMethod, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDConserved low_order_flux(
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   MetricAccessor const& metric,
   amrex::Array4<int const> const& excision_mask,
   bool use_excision_mask
) noexcept
{
   if (use_excision_mask) {
      bool replaced = false;
      const RHDConserved replacement = black_hole_excision_rhd_flux<Direction>(
         primitive, excision_mask, i, j, k, eos, metric, replaced
      );
      if (replaced) {
         return replacement;
      }
   }
   int il;
   int jl;
   int kl;
   left_cell<Direction>(i, j, k, il, jl, kl);
   RHDPrimitive left = rhd_primitive_at<EOSMethod>(
      primitive, il, jl, kl, eos
   );
   RHDPrimitive right = rhd_primitive_at<EOSMethod>(
      primitive, i, j, k, eos
   );
   const SpacetimeMetric face_metric = metric.template face<Direction>(
      i, j, k
   );
   const EulerianVelocity left_velocity =
      eulerian_velocity_from_spatial_four_velocity(
         spatial_four_velocity_from_eulerian(
            left.v1, left.v2, left.v3, metric.cell(il, jl, kl)
         ),
         face_metric
      );
   const EulerianVelocity right_velocity =
      eulerian_velocity_from_spatial_four_velocity(
         spatial_four_velocity_from_eulerian(
            right.v1, right.v2, right.v3, metric.cell(i, j, k)
         ),
         face_metric
      );
   left.v1 = left_velocity.v1;
   left.v2 = left_velocity.v2;
   left.v3 = left_velocity.v3;
   right.v1 = right_velocity.v1;
   right.v2 = right_velocity.v2;
   right.v3 = right_velocity.v3;
   if (apply_relativistic_diode_face<Direction>(
          left, right, face_metric, metric.diode, i, j, k)) {
      return generated::rhd_flux<Direction>(left, face_metric);
   }
   const RHDConserved state_l = generated::rhd_primitive_to_conserved(
      left, face_metric
   );
   const RHDConserved state_r = generated::rhd_primitive_to_conserved(
      right, face_metric
   );
   const RHDConserved flux_l = generated::rhd_flux<Direction>(left, face_metric);
   const RHDConserved flux_r = generated::rhd_flux<Direction>(right, face_metric);
   const RHDWaveSpeeds speed_l = generated::rhd_wave_speeds<Direction>(
      left,
      face_metric,
      relativistic_sound_speed_squared<EOSMethod>(left.rho, left.eps, eos)
   );
   const RHDWaveSpeeds speed_r = generated::rhd_wave_speeds<Direction>(
      right,
      face_metric,
      relativistic_sound_speed_squared<EOSMethod>(right.rho, right.eps, eos)
   );
   // Use the same physical spectral-radius contract as timestep control.
   // Forcing a unit lower bound makes a subluminal problem's low-order
   // fallback violate its own CFL condition even when the accepted global
   // timestep satisfies pyfluid.cfl.
   amrex::Real speed = 0.0_rt;
   speed = amrex::max(speed, amrex::Math::abs(speed_l.minus));
   speed = amrex::max(speed, amrex::Math::abs(speed_l.plus));
   speed = amrex::max(speed, amrex::Math::abs(speed_r.minus));
   speed = amrex::max(speed, amrex::Math::abs(speed_r.plus));
   return generated::rhd_llf_flux(
      flux_l, flux_r, state_l, state_r, speed
   );
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void high_face(int i, int j, int k, int& ih, int& jh, int& kh) noexcept
{
   ih = i;
   jh = j;
   kh = k;
   if constexpr (Direction == 0) {
      ++ih;
   } else if constexpr (Direction == 1) {
      ++jh;
   } else {
      ++kh;
   }
}

struct FaceCorrectionWeights {
   amrex::Real lower;
   amrex::Real upper;
};

struct FaceCorrectionGeometry {
   int coordinate;
   MeshAxes axes;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lower;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> spacing;

   template <int Direction>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   FaceCorrectionWeights weights(int i, int j, int k, amrex::Real scale) const noexcept
   {
      if (coordinate == MeshCartesian) {
         const amrex::Real value = scale / spacing[Direction];
         return {value, value};
      }
      int ih, jh, kh;
      high_face<Direction>(i, j, k, ih, jh, kh);
      const amrex::Real inverse_volume = scale
         / mesh_cell_volume_scale(coordinate, axes, lower, spacing, i, j, k);
      return {
         inverse_volume * mesh_face_area_scale(Direction, coordinate, axes, lower, spacing, i, j, k),
         inverse_volume * mesh_face_area_scale(Direction, coordinate, axes, lower, spacing, ih, jh, kh),
      };
   }
};

template <int Direction, int EOSMethod, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void add_low_order_candidate_correction(
   RHDConserved& candidate,
   amrex::Array4<amrex::Real const> const& high_flux,
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   MetricAccessor const& metric,
   FaceCorrectionWeights const& weights,
   amrex::Array4<int const> const& excision_mask,
   bool use_excision_mask
) noexcept
{
   int ih;
   int jh;
   int kh;
   high_face<Direction>(i, j, k, ih, jh, kh);
   const RHDConserved high_hi = load_rhd_conserved(
      high_flux, ih, jh, kh
   );
   const RHDConserved high_lo = load_rhd_conserved(high_flux, i, j, k);
   const RHDConserved low_hi = low_order_flux<Direction, EOSMethod>(
      primitive, ih, jh, kh, eos, metric, excision_mask, use_excision_mask
   );
   const RHDConserved low_lo = low_order_flux<Direction, EOSMethod>(
      primitive, i, j, k, eos, metric, excision_mask, use_excision_mask
   );
   candidate = add_scaled(candidate, difference(high_hi, low_hi), weights.upper);
   candidate = add_scaled(candidate, difference(high_lo, low_lo), -weights.lower);
}

template <int Direction, int EOSMethod, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void add_limited_rhs_correction(
   RHDConserved& rhs,
   amrex::Array4<amrex::Real const> const& high_flux,
   amrex::Array4<amrex::Real const> const& primitive,
   amrex::Array4<amrex::Real const> const& theta,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   MetricAccessor const& metric,
   FaceCorrectionWeights const& weights,
   amrex::Array4<int const> const& excision_mask,
   bool use_excision_mask
) noexcept
{
   int ih;
   int jh;
   int kh;
   high_face<Direction>(i, j, k, ih, jh, kh);
   int il;
   int jl;
   int kl;
   left_cell<Direction>(i, j, k, il, jl, kl);
   const amrex::Real theta_hi = amrex::min(
      theta(i, j, k, 0), theta(ih, jh, kh, 0)
   );
   const amrex::Real theta_lo = amrex::min(
      theta(il, jl, kl, 0), theta(i, j, k, 0)
   );
   const RHDConserved high_hi = load_rhd_conserved(
      high_flux, ih, jh, kh
   );
   const RHDConserved high_lo = load_rhd_conserved(high_flux, i, j, k);
   const RHDConserved low_minus_high_hi = difference(
      low_order_flux<Direction, EOSMethod>(
         primitive, ih, jh, kh, eos, metric,
         excision_mask, use_excision_mask
      ),
      high_hi
   );
   const RHDConserved delta_hi = add_scaled(
      low_minus_high_hi, low_minus_high_hi, -theta_hi
   );
   const RHDConserved low_minus_high_lo = difference(
      low_order_flux<Direction, EOSMethod>(
         primitive, i, j, k, eos, metric,
         excision_mask, use_excision_mask
      ),
      high_lo
   );
   const RHDConserved delta_lo = add_scaled(
      low_minus_high_lo, low_minus_high_lo, -theta_lo
   );
   if (weights.lower == weights.upper) {
      // Retain Cartesian arithmetic as well as its fast geometry path.
      rhs = add_scaled(rhs, difference(delta_hi, delta_lo), -weights.upper);
   } else {
      rhs = add_scaled(rhs, delta_hi, -weights.upper);
      rhs = add_scaled(rhs, delta_lo, weights.lower);
   }
}

template <int Direction, int EOSMethod, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void blend_face_flux(
   amrex::Array4<amrex::Real> const& high_flux,
   amrex::Array4<amrex::Real const> const& primitive,
   amrex::Array4<amrex::Real const> const& theta,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   MetricAccessor const& metric,
   amrex::Array4<int const> const& excision_mask,
   bool use_excision_mask
) noexcept
{
   int il;
   int jl;
   int kl;
   left_cell<Direction>(i, j, k, il, jl, kl);
   const amrex::Real face_theta = amrex::min(
      theta(il, jl, kl, 0), theta(i, j, k, 0)
   );
   const RHDConserved high = load_rhd_conserved(high_flux, i, j, k);
   const RHDConserved low = low_order_flux<Direction, EOSMethod>(
      primitive, i, j, k, eos, metric, excision_mask, use_excision_mask
   );
   store_rhd_conserved(high_flux, i, j, k, blend(low, high, face_theta));
}

template <int EOSMethod, typename MetricAccessorFactory>
amrex::Long apply_limiter_impl(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab& rhs,
   amrex::MultiFab const& base_conserved,
   amrex::MultiFab const& conserved,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   amrex::Geometry const& geometry,
   int coordinate,
   MeshAxes const& axes,
   EOSThermodynamicState const& atmosphere,
   amrex::Real conserved_weight,
   amrex::Real increment_scale,
   amrex::iMultiFab const* black_hole_excision_mask,
   MetricAccessorFactory&& metric_accessor_factory
)
{
   amrex::MultiFab theta(
      rhs.boxArray(), rhs.DistributionMap(), 1, 1
   );
   theta.setVal(1.0_rt);
   const FaceCorrectionGeometry correction_geometry{coordinate, axes,
      geometry.ProbLoArray(), geometry.CellSizeArray()};
   amrex::MultiFab theta_next(rhs.boxArray(), rhs.DistributionMap(), 1, 1);
   amrex::Gpu::DeviceScalar<amrex::Long> newly_troubled_count{0};
   // Replacing a shared face can invalidate its other cell. Recheck the
   // concrete mixed-flux candidate until the binary mask stops growing.
   // A zero cell already uses low-order flux on every face and is terminal.
   amrex::Long limited_cells = 0;
   while (true) {
      theta.FillBoundary(geometry.periodicity());
      amrex::MultiFab::Copy(theta_next, theta, 0, 0, 1, 0);
      const amrex::Long zero = 0;
#ifdef AMREX_USE_GPU
      amrex::Gpu::htod_memcpy(newly_troubled_count.dataPtr(), &zero, sizeof(zero));
#else
      *newly_troubled_count.dataPtr() = zero;
#endif
      auto* changed_count = newly_troubled_count.dataPtr();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& box = mfi.tilebox();
         const auto rhs_values = rhs.const_array(mfi);
         const auto base_values = base_conserved.const_array(mfi);
         const auto state_values = conserved.const_array(mfi);
         const auto primitive_values = primitive.const_array(mfi);
         const auto flux0 = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
         const auto flux1 = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
         const auto flux2 = fluxes[2]->const_array(mfi);
#endif
         const auto theta_values = theta.const_array(mfi);
         const auto next_values = theta_next.array(mfi);
         const auto metric = metric_accessor_factory(mfi);
         amrex::Array4<int const> excision_mask{};
         if (black_hole_excision_mask != nullptr) {
            excision_mask = black_hole_excision_mask->const_array(mfi);
         }
         const bool use_excision_mask = black_hole_excision_mask != nullptr;
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (theta_values(i, j, k, 0) == 0.0_rt) {
               return;
            }
            RHDConserved high_candidate = stage_candidate(
               load_rhd_conserved(base_values, i, j, k),
               load_rhd_conserved(state_values, i, j, k),
               load_rhd_conserved(rhs_values, i, j, k),
               conserved_weight, increment_scale
            );
            RHDConserved low_candidate = high_candidate;
            add_low_order_candidate_correction<0, EOSMethod>(
               low_candidate, flux0, primitive_values, i, j, k, eos, metric,
               correction_geometry.weights<0>(i, j, k, increment_scale), excision_mask, use_excision_mask
            );
#if (AMREX_SPACEDIM >= 2)
            add_low_order_candidate_correction<1, EOSMethod>(
               low_candidate, flux1, primitive_values, i, j, k, eos, metric,
               correction_geometry.weights<1>(i, j, k, increment_scale), excision_mask, use_excision_mask
            );
#endif
#if (AMREX_SPACEDIM == 3)
            add_low_order_candidate_correction<2, EOSMethod>(
               low_candidate, flux2, primitive_values, i, j, k, eos, metric,
               correction_geometry.weights<2>(i, j, k, increment_scale), excision_mask, use_excision_mask
            );
#endif
            RHDConserved mixed_candidate = high_candidate;
            add_limited_rhs_correction<0, EOSMethod>(
               mixed_candidate, flux0, primitive_values, theta_values, i, j, k,
               eos, metric, correction_geometry.weights<0>(i, j, k, increment_scale), excision_mask, use_excision_mask
            );
#if (AMREX_SPACEDIM >= 2)
            add_limited_rhs_correction<1, EOSMethod>(
               mixed_candidate, flux1, primitive_values, theta_values, i, j, k,
               eos, metric, correction_geometry.weights<1>(i, j, k, increment_scale), excision_mask, use_excision_mask
            );
#endif
#if (AMREX_SPACEDIM == 3)
            add_limited_rhs_correction<2, EOSMethod>(
               mixed_candidate, flux2, primitive_values, theta_values, i, j, k,
               eos, metric, correction_geometry.weights<2>(i, j, k, increment_scale), excision_mask, use_excision_mask
            );
#endif
            next_values(i, j, k, 0) = admissibility_theta(
               low_candidate,
               mixed_candidate,
               metric.candidate_cell(i, j, k),
               atmosphere.rho,
               atmosphere.pressure
            );
            if (next_values(i, j, k, 0) == 0.0_rt) {
               amrex::HostDevice::Atomic::Add(changed_count, amrex::Long(1));
            }
         });
      }
      amrex::Long changed = newly_troubled_count.dataValue();
      amrex::ParallelDescriptor::ReduceLongSum(changed);
      limited_cells += changed;
      amrex::MultiFab::Copy(theta, theta_next, 0, 0, 1, 0);
      if (changed == 0) {
         break;
      }
   }
   theta.FillBoundary(geometry.periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto rhs_values = rhs.array(mfi);
      const auto primitive_values = primitive.const_array(mfi);
      const auto theta_values = theta.const_array(mfi);
      const auto flux0 = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto flux1 = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto flux2 = fluxes[2]->const_array(mfi);
#endif
      const auto metric = metric_accessor_factory(mfi);
      amrex::Array4<int const> excision_mask{};
      if (black_hole_excision_mask != nullptr) {
         excision_mask = black_hole_excision_mask->const_array(mfi);
      }
      const bool use_excision_mask = black_hole_excision_mask != nullptr;
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         RHDConserved value = load_rhd_conserved(rhs_values, i, j, k);
         add_limited_rhs_correction<0, EOSMethod>(
            value, flux0, primitive_values, theta_values, i, j, k, eos,
            metric, correction_geometry.weights<0>(i, j, k, 1.0_rt), excision_mask, use_excision_mask
         );
#if (AMREX_SPACEDIM >= 2)
         add_limited_rhs_correction<1, EOSMethod>(
            value, flux1, primitive_values, theta_values, i, j, k, eos,
            metric, correction_geometry.weights<1>(i, j, k, 1.0_rt), excision_mask, use_excision_mask
         );
#endif
#if (AMREX_SPACEDIM == 3)
         add_limited_rhs_correction<2, EOSMethod>(
            value, flux2, primitive_values, theta_values, i, j, k, eos,
            metric, correction_geometry.weights<2>(i, j, k, 1.0_rt), excision_mask, use_excision_mask
         );
#endif
         store_rhd_conserved(rhs_values, i, j, k, value);
      });
   }

   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(
              *fluxes[direction], amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box& box = mfi.tilebox();
         const auto flux_values = fluxes[direction]->array(mfi);
         const auto primitive_values = primitive.const_array(mfi);
         const auto theta_values = theta.const_array(mfi);
         const auto metric = metric_accessor_factory(mfi);
         amrex::Array4<int const> excision_mask{};
         if (black_hole_excision_mask != nullptr) {
            excision_mask = black_hole_excision_mask->const_array(mfi);
         }
         const bool use_excision_mask = black_hole_excision_mask != nullptr;
         if (direction == 0) {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               blend_face_flux<0, EOSMethod>(
                  flux_values, primitive_values, theta_values, i, j, k,
                  eos, metric, excision_mask, use_excision_mask
               );
            });
         }
#if (AMREX_SPACEDIM >= 2)
         else if (direction == 1) {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               blend_face_flux<1, EOSMethod>(
                  flux_values, primitive_values, theta_values, i, j, k,
                  eos, metric, excision_mask, use_excision_mask
               );
            });
         }
#endif
#if (AMREX_SPACEDIM == 3)
         else {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               blend_face_flux<2, EOSMethod>(
                  flux_values, primitive_values, theta_values, i, j, k,
                  eos, metric, excision_mask, use_excision_mask
               );
            });
         }
#endif
      }
   }
   return limited_cells;
}

template <int EOSMethod>
amrex::Long apply_limiter(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab& rhs,
   amrex::MultiFab const& base_conserved,
   amrex::MultiFab const& conserved,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   amrex::Geometry const& geometry,
   FixedSpacetimeParameters const& fixed_spacetime,
   int coordinate,
   MeshAxes const& axes,
   amrex::MultiFab const* cell_spacetime,
   amrex::MultiFab const* candidate_cell_spacetime,
   EOSThermodynamicState const& atmosphere,
   amrex::Real time,
   amrex::Real conserved_weight,
   amrex::Real increment_scale,
   amrex::iMultiFab const* black_hole_excision_mask,
   SphericalExcisionRegion const& excision_region,
   RelativisticDiodeBoundary const& diode
)
{
   if (cell_spacetime != nullptr) {
      AMREX_ALWAYS_ASSERT(candidate_cell_spacetime != nullptr);
      return apply_limiter_impl<EOSMethod>(
         fluxes, rhs, base_conserved, conserved, primitive, eos, geometry, coordinate, axes,
         atmosphere, conserved_weight, increment_scale,
         black_hole_excision_mask,
         StoredMetricFactory{
            *cell_spacetime, *candidate_cell_spacetime,
            excision_grid_view(excision_region, geometry, primitive.ixType()), diode
         }
      );
   } else {
      const auto lower = geometry.ProbLoArray();
      const auto spacing = geometry.CellSizeArray();
      return apply_limiter_impl<EOSMethod>(
         fluxes, rhs, base_conserved, conserved, primitive, eos, geometry, coordinate, axes,
         atmosphere, conserved_weight, increment_scale,
         black_hole_excision_mask,
         FixedMetricFactory{{
            fixed_spacetime, lower, spacing, coordinate, axes, time, {}, diode
         }}
      );
   }
}

} // namespace

amrex::Long apply_rhd_admissibility_flux_limiter(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab& rhs,
   amrex::MultiFab const& base_conserved,
   amrex::MultiFab const& conserved,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   int eos_method,
   amrex::Geometry const& geometry,
   FixedSpacetimeParameters const& fixed_spacetime,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::MultiFab const* cell_spacetime,
   amrex::MultiFab const* candidate_cell_spacetime,
   EOSThermodynamicState const& atmosphere,
   amrex::Real time,
   amrex::Real conserved_weight,
   amrex::Real increment_scale,
   amrex::iMultiFab const* black_hole_excision_mask,
   SphericalExcisionRegion const& excision_region,
   RelativisticDiodeBoundary const& diode
)
{
   if (!(increment_scale > 0.0_rt)) {
      return 0;
   }
   static_cast<void>(eos_method);
   return apply_limiter<EOSRuntime>(
      fluxes, rhs, base_conserved, conserved, primitive, eos, geometry,
      fixed_spacetime, mesh_coordinate, mesh_axes,
      cell_spacetime, candidate_cell_spacetime, atmosphere,
      time, conserved_weight, increment_scale,
      black_hole_excision_mask, excision_region, diode
   );
}

#if PYFLUID_AMREX_ENABLE_MHD
namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDConserved stage_candidate(
   RMHDConserved const& base,
   RMHDConserved const& input,
   RMHDConserved const& rhs,
   amrex::Real input_weight,
   amrex::Real increment_scale
) noexcept
{
   return {
      affine_state_update(base.dens, input.dens, input_weight, increment_scale * rhs.dens),
      affine_state_update(base.mom1, input.mom1, input_weight, increment_scale * rhs.mom1),
      affine_state_update(base.mom2, input.mom2, input_weight, increment_scale * rhs.mom2),
      affine_state_update(base.mom3, input.mom3, input_weight, increment_scale * rhs.mom3),
      affine_state_update(base.tau, input.tau, input_weight, increment_scale * rhs.tau),
      affine_state_update(base.b1, input.b1, input_weight, increment_scale * rhs.b1),
      affine_state_update(base.b2, input.b2, input_weight, increment_scale * rhs.b2),
      affine_state_update(base.b3, input.b3, input_weight, increment_scale * rhs.b3),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDConserved add_scaled(
   RMHDConserved state,
   RMHDConserved const& increment,
   amrex::Real scale
) noexcept
{
   state.dens += scale * increment.dens;
   state.mom1 += scale * increment.mom1;
   state.mom2 += scale * increment.mom2;
   state.mom3 += scale * increment.mom3;
   state.tau += scale * increment.tau;
   state.b1 += scale * increment.b1;
   state.b2 += scale * increment.b2;
   state.b3 += scale * increment.b3;
   return state;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDConserved difference(
   RMHDConserved const& left,
   RMHDConserved const& right
) noexcept
{
   return {
      left.dens - right.dens,
      left.mom1 - right.mom1,
      left.mom2 - right.mom2,
      left.mom3 - right.mom3,
      left.tau - right.tau,
      left.b1 - right.b1,
      left.b2 - right.b2,
      left.b3 - right.b3,
   };
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void add_rmhd_candidate_correction(
   RMHDConserved& candidate,
   amrex::Array4<amrex::Real const> const& high_flux,
   amrex::Array4<amrex::Real const> const& corrected_flux,
   int i, int j, int k,
   amrex::Real dt_over_dx
) noexcept
{
   int ih, jh, kh;
   high_face<Direction>(i, j, k, ih, jh, kh);
   candidate = add_scaled(candidate,
      difference(load_rmhd_conserved(high_flux, ih, jh, kh),
                 load_rmhd_conserved(corrected_flux, ih, jh, kh)), dt_over_dx);
   candidate = add_scaled(candidate,
      difference(load_rmhd_conserved(high_flux, i, j, k),
                 load_rmhd_conserved(corrected_flux, i, j, k)), -dt_over_dx);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool rmhd_candidate_recovers(
   RMHDConserved const& candidate,
   SpacetimeMetric const& metric,
   EOSParams const& eos,
   EOSThermodynamicState const& atmosphere
) noexcept
{
   const RMHDRecoveryResult recovered = recover_srmhd(
      candidate, metric, eos
   );
   return recovered.succeeded()
      && recovered.primitive.rho >= atmosphere.rho
      && recovered.primitive.pressure >= atmosphere.pressure;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool rmhd_candidate_violates_dmp(
   RMHDConserved const& candidate,
   amrex::Array4<amrex::Real const> const& conserved,
   int i,
   int j,
   int k,
   bool enforce_dmp,
   amrex::Real factor,
   amrex::Array4<int const> const& excision_mask,
   bool use_excision_mask
) noexcept
{
   if (!enforce_dmp || (use_excision_mask && excision_mask(i, j, k, 0) != 0)) {
      return false;
   }
   amrex::Real dens_min = std::numeric_limits<amrex::Real>::max();
   amrex::Real dens_max = -std::numeric_limits<amrex::Real>::max();
   amrex::Real tau_min = std::numeric_limits<amrex::Real>::max();
   amrex::Real tau_max = -std::numeric_limits<amrex::Real>::max();
#if (AMREX_SPACEDIM == 1)
   constexpr int j_radius = 0;
   constexpr int k_radius = 0;
#elif (AMREX_SPACEDIM == 2)
   constexpr int j_radius = 1;
   constexpr int k_radius = 0;
#else
   constexpr int j_radius = 1;
   constexpr int k_radius = 1;
#endif
   for (int kk = k - k_radius; kk <= k + k_radius; ++kk) {
      for (int jj = j - j_radius; jj <= j + j_radius; ++jj) {
         for (int ii = i - 1; ii <= i + 1; ++ii) {
            if (use_excision_mask && excision_mask(ii, jj, kk, 0) != 0) {
               continue;
            }
            const RMHDConserved neighbor = load_rmhd_conserved(
               conserved, ii, jj, kk
            );
            dens_min = amrex::min(dens_min, neighbor.dens);
            dens_max = amrex::max(dens_max, neighbor.dens);
            tau_min = amrex::min(tau_min, neighbor.tau);
            tau_max = amrex::max(tau_max, neighbor.tau);
         }
      }
   }
   return candidate.dens < dens_min / factor
      || candidate.dens > factor * dens_max
      || candidate.tau < tau_min / factor
      || candidate.tau > factor * tau_max;
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void add_selected_rmhd_rhs_correction(
   RMHDConserved& rhs,
   amrex::Array4<amrex::Real const> const& high_flux,
   amrex::Array4<amrex::Real const> const& selected_flux,
   int i, int j, int k,
   amrex::Real inverse_dx
) noexcept
{
   int ih, jh, kh;
   high_face<Direction>(i, j, k, ih, jh, kh);
   rhs = add_scaled(rhs,
      difference(
         difference(load_rmhd_conserved(selected_flux, ih, jh, kh),
                    load_rmhd_conserved(high_flux, ih, jh, kh)),
         difference(load_rmhd_conserved(selected_flux, i, j, k),
                    load_rmhd_conserved(high_flux, i, j, k))
      ), -inverse_dx);
}

template <int Direction, typename MetricAccessor>
void add_pwl_rmhd_candidate_direction(
   int riemann_method,
   amrex::Box const& box,
   amrex::Array4<amrex::Real const> const& high_flux,
   amrex::Array4<amrex::Real const> const& primitive_values,
   amrex::Array4<int const> const& correction_values,
   amrex::Array4<amrex::Real> const& candidate_values,
   EOSParams const& eos,
   MetricAccessor const& metric,
   EOSThermodynamicState const& atmosphere,
   amrex::Array4<int const> const& excision_mask,
   bool use_excision_mask,
   amrex::Real correction_scale
)
{
   const amrex::Box face_box = amrex::surroundingNodes(box, Direction);
   amrex::FArrayBox face_storage(face_box, NVAR_MHD, amrex::The_Async_Arena());
   const auto corrected_flux = face_storage.array();
   fill_rmhd_correction_fluxes(
      riemann_method, Direction,
      {box, face_box, corrected_flux, {}, primitive_values, correction_values,
       eos, atmosphere, excision_mask, use_excision_mask}, metric, true
   );
   amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      if (correction_values(i, j, k, 0) != 2) return;
      RMHDConserved candidate = load_rmhd_conserved(candidate_values, i, j, k);
      add_rmhd_candidate_correction<Direction>(
         candidate, high_flux, corrected_flux, i, j, k,
         correction_scale
      );
      store_rmhd_conserved(candidate_values, i, j, k, candidate);
   });
}

template <int Direction, typename MetricAccessor>
void add_selected_rmhd_candidate_direction(
   int riemann_method,
   amrex::Box const& box,
   amrex::Array4<amrex::Real const> const& high_flux,
   amrex::Array4<amrex::Real const> const& primitive_values,
   amrex::Array4<int const> const& current_values,
   amrex::Array4<amrex::Real> const& candidate_values,
   EOSParams const& eos,
   MetricAccessor const& metric,
   EOSThermodynamicState const& atmosphere,
   amrex::Array4<int const> const& excision_mask,
   bool use_excision_mask,
   amrex::Real correction_scale
)
{
   const amrex::Box face_box = amrex::surroundingNodes(box, Direction);
   amrex::FArrayBox face_storage(face_box, NVAR_MHD, amrex::The_Async_Arena());
   const auto selected_flux = face_storage.array();
   fill_rmhd_correction_fluxes(
      riemann_method, Direction,
      {box, face_box, selected_flux, high_flux, primitive_values, current_values,
       eos, atmosphere, excision_mask, use_excision_mask}, metric, false
   );
   amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      if ((use_excision_mask && excision_mask(i, j, k, 0) != 0)
          || current_values(i, j, k, 0) == 2) return;
      RMHDConserved candidate = load_rmhd_conserved(candidate_values, i, j, k);
      add_selected_rmhd_rhs_correction<Direction>(
         candidate, high_flux, selected_flux, i, j, k,
         correction_scale
      );
      store_rmhd_conserved(candidate_values, i, j, k, candidate);
   });
}

template <int Direction, typename MetricAccessor>
void correct_selected_rmhd_rhs_direction(
   int riemann_method,
   amrex::Box const& box,
   amrex::Array4<amrex::Real const> const& high_flux,
   amrex::Array4<amrex::Real const> const& primitive_values,
   amrex::Array4<int const> const& correction_values,
   amrex::Array4<amrex::Real> const& rhs_values,
   EOSParams const& eos,
   MetricAccessor const& metric,
   EOSThermodynamicState const& atmosphere,
   amrex::Array4<int const> const& excision_mask,
   bool use_excision_mask,
   amrex::Real correction_scale
)
{
   const amrex::Box face_box = amrex::surroundingNodes(box, Direction);
   amrex::FArrayBox face_storage(face_box, NVAR_MHD, amrex::The_Async_Arena());
   const auto selected_flux = face_storage.array();
   fill_rmhd_correction_fluxes(
      riemann_method, Direction,
      {box, face_box, selected_flux, high_flux, primitive_values, correction_values,
       eos, atmosphere, excision_mask, use_excision_mask}, metric, false
   );
   amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      if (use_excision_mask && excision_mask(i, j, k, 0) != 0) {
         store_rmhd_conserved(rhs_values, i, j, k, RMHDConserved{});
         return;
      }
      RMHDConserved value = load_rmhd_conserved(rhs_values, i, j, k);
      add_selected_rmhd_rhs_correction<Direction>(
         value, high_flux, selected_flux, i, j, k, correction_scale
      );
      store_rmhd_conserved(rhs_values, i, j, k, value);
   });
}

template <typename MetricAccessorFactory>
void apply_rmhd_limiter_impl(
   int riemann_method,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab& rhs,
   amrex::MultiFab const& base_conserved,
   amrex::MultiFab const& conserved,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   amrex::Geometry const& geometry,
   EOSThermodynamicState const& atmosphere,
   amrex::Real conserved_weight,
   amrex::Real increment_scale,
   bool try_pwl_first,
   bool enforce_dmp,
   amrex::Real dmp_factor,
   amrex::Long* pwl_correction_count,
   amrex::Long* pwc_correction_count,
   amrex::Long* hll_fallback_count,
   amrex::iMultiFab const* black_hole_excision_mask,
   MetricAccessorFactory&& metric_accessor_factory
)
{
   amrex::iMultiFab correction(
      rhs.boxArray(), rhs.DistributionMap(), 1, 1
   );
   correction.setVal(0);
   const auto spacing = geometry.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto rhs_values = rhs.const_array(mfi);
      const auto base_values = base_conserved.const_array(mfi);
      const auto state_values = conserved.const_array(mfi);
      const auto primitive_values = primitive.const_array(mfi);
      amrex::FArrayBox candidate_storage(box, NVAR_MHD, amrex::The_Async_Arena());
      const auto candidate_values = candidate_storage.array();
      const auto correction_values = correction.array(mfi);
      const auto metric = metric_accessor_factory(mfi);
      amrex::Array4<int const> excision_mask;
      if (black_hole_excision_mask != nullptr) {
         excision_mask = black_hole_excision_mask->const_array(mfi);
      }
      const bool use_excision_mask = black_hole_excision_mask != nullptr;
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (use_excision_mask && excision_mask(i, j, k, 0) != 0) {
            return;
         }
         const RMHDConserved high_candidate = stage_candidate(
            load_rmhd_conserved(base_values, i, j, k),
            load_rmhd_conserved(state_values, i, j, k),
            load_rmhd_conserved(rhs_values, i, j, k),
            conserved_weight, increment_scale
         );
         store_rmhd_conserved(candidate_values, i, j, k, high_candidate);
         const bool high_bad = rmhd_candidate_violates_dmp(
            high_candidate, state_values, i, j, k, enforce_dmp, dmp_factor,
            excision_mask, use_excision_mask
         ) || !rmhd_candidate_recovers(
            high_candidate, metric.candidate_cell(i, j, k), eos, atmosphere
         );
         correction_values(i, j, k, 0) = high_bad ? 2 : 0;
      });
      if (!try_pwl_first) continue;

      // Build the optional PWL candidate one direction at a time. Recovery
      // stays in its own launch; neither kernel contains the other's physics.
      auto add_pwl_direction = [&]<int Direction>() {
         add_pwl_rmhd_candidate_direction<Direction>(
            riemann_method,
            box, fluxes[Direction]->const_array(mfi), primitive_values,
            correction_values, candidate_values, eos, metric, atmosphere,
            excision_mask, use_excision_mask, increment_scale / spacing[Direction]
         );
      };
      add_pwl_direction.template operator()<0>();
#if (AMREX_SPACEDIM >= 2)
      add_pwl_direction.template operator()<1>();
#endif
#if (AMREX_SPACEDIM == 3)
      add_pwl_direction.template operator()<2>();
#endif
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (correction_values(i, j, k, 0) != 2) return;
         const RMHDConserved candidate = load_rmhd_conserved(candidate_values, i, j, k);
         const bool pwl_bad = rmhd_candidate_violates_dmp(
            candidate, state_values, i, j, k, enforce_dmp, dmp_factor,
            excision_mask, use_excision_mask
         ) || !rmhd_candidate_recovers(
            candidate, metric.candidate_cell(i, j, k), eos, atmosphere
         );
         if (!pwl_bad) correction_values(i, j, k, 0) = 1;
      });
   }

   amrex::iMultiFab correction_next(
      rhs.boxArray(), rhs.DistributionMap(), 1, 1
   );
   amrex::iMultiFab* current_correction = &correction;
   amrex::iMultiFab* next_correction = &correction_next;
   amrex::Gpu::DeviceScalar<amrex::Long> newly_troubled_count{0};
   // A corrected face is shared by two cells. Grow the mask monotonically
   // until every nonterminal affected candidate recovers. PWC is the terminal
   // conservative candidate; if even it does not recover, the normal accepted-
   // state atmosphere projection records the reset instead of adding another
   // hidden guard fallback.
   while (true) {
      current_correction->FillBoundary(geometry.periodicity());
      amrex::iMultiFab::Copy(
         *next_correction, *current_correction, 0, 0, 1, 0
      );
      const amrex::Long zero = 0;
#ifdef AMREX_USE_GPU
      amrex::Gpu::htod_memcpy(
         newly_troubled_count.dataPtr(), &zero, sizeof(zero)
      );
#else
      *newly_troubled_count.dataPtr() = zero;
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(
              rhs, amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box& box = mfi.tilebox();
         const auto rhs_values = rhs.const_array(mfi);
         const auto base_values = base_conserved.const_array(mfi);
         const auto state_values = conserved.const_array(mfi);
         const auto primitive_values = primitive.const_array(mfi);
         const auto current_values = current_correction->const_array(mfi);
         const auto next_values = next_correction->array(mfi);
         amrex::FArrayBox candidate_storage(box, NVAR_MHD, amrex::The_Async_Arena());
         const auto candidate_values = candidate_storage.array();
         const auto metric = metric_accessor_factory(mfi);
         amrex::Array4<int const> excision_mask;
         if (black_hole_excision_mask != nullptr) {
            excision_mask = black_hole_excision_mask->const_array(mfi);
         }
         const bool use_excision_mask = black_hole_excision_mask != nullptr;
         amrex::Long* newly_troubled = newly_troubled_count.dataPtr();
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if ((use_excision_mask && excision_mask(i, j, k, 0) != 0)
                || current_values(i, j, k, 0) == 2) return;
            store_rmhd_conserved(candidate_values, i, j, k, stage_candidate(
               load_rmhd_conserved(base_values, i, j, k),
               load_rmhd_conserved(state_values, i, j, k),
               load_rmhd_conserved(rhs_values, i, j, k),
               conserved_weight, increment_scale
            ));
         });
         auto add_candidate_direction = [&]<int Direction>() {
            add_selected_rmhd_candidate_direction<Direction>(
               riemann_method,
               box, fluxes[Direction]->const_array(mfi), primitive_values,
               current_values, candidate_values, eos, metric, atmosphere,
               excision_mask, use_excision_mask, increment_scale / spacing[Direction]
            );
         };
         add_candidate_direction.template operator()<0>();
#if (AMREX_SPACEDIM >= 2)
         add_candidate_direction.template operator()<1>();
#endif
#if (AMREX_SPACEDIM == 3)
         add_candidate_direction.template operator()<2>();
#endif
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if ((use_excision_mask && excision_mask(i, j, k, 0) != 0)
                || current_values(i, j, k, 0) == 2) return;
            const RMHDConserved candidate = load_rmhd_conserved(candidate_values, i, j, k);
            if (!rmhd_candidate_recovers(candidate, metric.candidate_cell(i, j, k), eos, atmosphere)) {
               next_values(i, j, k, 0) = 2;
               amrex::HostDevice::Atomic::Add(newly_troubled, static_cast<amrex::Long>(1));
            }
         });
      }
      amrex::Long newly_troubled =
         newly_troubled_count.dataValue();
      amrex::ParallelDescriptor::ReduceLongSum(newly_troubled);
      std::swap(current_correction, next_correction);
      if (newly_troubled == 0) {
         break;
      }
   }
   if (current_correction != &correction) {
      amrex::iMultiFab::Copy(
         correction, *current_correction, 0, 0, 1, 0
      );
   }
   correction.FillBoundary(geometry.periodicity());

   if (
      pwl_correction_count != nullptr || pwc_correction_count != nullptr
   ) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(
              correction, amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box& box = mfi.tilebox();
         const auto correction_values = correction.const_array(mfi);
         amrex::Array4<int const> excision_mask;
         if (black_hole_excision_mask != nullptr) {
            excision_mask = black_hole_excision_mask->const_array(mfi);
         }
         const bool use_excision_mask = black_hole_excision_mask != nullptr;
         amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if (use_excision_mask && excision_mask(i, j, k, 0) != 0) {
                  return;
               }
               const int kind = correction_values(i, j, k, 0);
               if (kind == 1 && pwl_correction_count != nullptr) {
                  amrex::HostDevice::Atomic::Add(
                     pwl_correction_count, static_cast<amrex::Long>(1)
                  );
               } else if (kind == 2 && pwc_correction_count != nullptr) {
                  amrex::HostDevice::Atomic::Add(
                     pwc_correction_count, static_cast<amrex::Long>(1)
                  );
               }
            }
         );
      }
   }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto rhs_values = rhs.array(mfi);
      const auto primitive_values = primitive.const_array(mfi);
      const auto correction_values = correction.const_array(mfi);
      const auto metric = metric_accessor_factory(mfi);
      amrex::Array4<int const> excision_mask;
      if (black_hole_excision_mask != nullptr) {
         excision_mask = black_hole_excision_mask->const_array(mfi);
      }
      const bool use_excision_mask = black_hole_excision_mask != nullptr;
      auto correct_rhs_direction = [&]<int Direction>() {
         correct_selected_rmhd_rhs_direction<Direction>(
            riemann_method,
            box, fluxes[Direction]->const_array(mfi), primitive_values,
            correction_values, rhs_values, eos, metric, atmosphere,
            excision_mask, use_excision_mask, 1.0_rt / spacing[Direction]
         );
      };
      correct_rhs_direction.template operator()<0>();
#if (AMREX_SPACEDIM >= 2)
      correct_rhs_direction.template operator()<1>();
#endif
#if (AMREX_SPACEDIM == 3)
      correct_rhs_direction.template operator()<2>();
#endif
   }

   auto replace_direction_fluxes = [&]<int Direction>() {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(*fluxes[Direction], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const auto flux_values = fluxes[Direction]->array(mfi);
         const auto primitive_values = primitive.const_array(mfi);
         const auto correction_values = correction.const_array(mfi);
         const auto metric = metric_accessor_factory(mfi);
         amrex::Array4<int const> excision_mask;
         if (black_hole_excision_mask != nullptr) {
            excision_mask = black_hole_excision_mask->const_array(mfi);
         }
         fill_rmhd_correction_fluxes(
            riemann_method, Direction,
            {mfi.tilebox(), mfi.tilebox(), flux_values, flux_values,
             primitive_values, correction_values, eos, atmosphere, excision_mask,
             black_hole_excision_mask != nullptr, hll_fallback_count}, metric, false
         );
      }
   };
   replace_direction_fluxes.template operator()<0>();
#if (AMREX_SPACEDIM >= 2)
   replace_direction_fluxes.template operator()<1>();
#endif
#if (AMREX_SPACEDIM == 3)
   replace_direction_fluxes.template operator()<2>();
#endif
}

void apply_rmhd_limiter(
   int riemann_method,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab& rhs,
   amrex::MultiFab const& base_conserved,
   amrex::MultiFab const& conserved,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   amrex::Geometry const& geometry,
   FixedSpacetimeParameters const& fixed_spacetime,
   amrex::MultiFab const* cell_spacetime,
   amrex::MultiFab const* candidate_cell_spacetime,
   EOSThermodynamicState const& atmosphere,
   amrex::Real time,
   amrex::Real conserved_weight,
   amrex::Real increment_scale,
   bool try_pwl_first,
   bool enforce_dmp,
   amrex::Real dmp_factor,
   amrex::Long* pwl_correction_count,
   amrex::Long* pwc_correction_count,
   amrex::Long* hll_fallback_count,
   amrex::iMultiFab const* black_hole_excision_mask,
   SphericalExcisionRegion const& excision_region,
   RelativisticDiodeBoundary const& diode
)
{
   auto apply = [&](auto&& metric_factory) {
      apply_rmhd_limiter_impl(
         riemann_method,
         fluxes, rhs, base_conserved, conserved, primitive, eos, geometry,
         atmosphere, conserved_weight, increment_scale,
         try_pwl_first, enforce_dmp, dmp_factor, pwl_correction_count,
         pwc_correction_count, hll_fallback_count,
         black_hole_excision_mask, metric_factory
      );
   };
   if (cell_spacetime != nullptr) {
      AMREX_ALWAYS_ASSERT(candidate_cell_spacetime != nullptr);
      apply(StoredMetricFactory{
         *cell_spacetime, *candidate_cell_spacetime,
         excision_grid_view(excision_region, geometry, primitive.ixType()), diode
      });
   } else {
      apply(FixedMetricFactory{{
         fixed_spacetime, geometry.ProbLoArray(), geometry.CellSizeArray(),
         MeshCartesian, mesh_default_axes_layout(MeshCartesian), time, {}, diode
      }});
   }
}

} // namespace

#endif // PYFLUID_AMREX_ENABLE_MHD

void apply_rmhd_candidate_flux_limiter(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab& rhs,
   amrex::MultiFab const& base_conserved,
   amrex::MultiFab const& conserved,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   int riemann_method,
   amrex::Geometry const& geometry,
   FixedSpacetimeParameters const& fixed_spacetime,
   amrex::MultiFab const* cell_spacetime,
   amrex::MultiFab const* candidate_cell_spacetime,
   EOSThermodynamicState const& atmosphere,
   amrex::Real time,
   amrex::Real conserved_weight,
   amrex::Real increment_scale,
   bool try_pwl_first,
   bool enforce_dmp,
   amrex::Real dmp_factor,
   amrex::Long* pwl_correction_count,
   amrex::Long* pwc_correction_count,
   amrex::Long* hll_fallback_count,
   amrex::iMultiFab const* black_hole_excision_mask,
   SphericalExcisionRegion const& excision_region,
   RelativisticDiodeBoundary const& diode
)
{
   if (!(increment_scale > 0.0_rt)) {
      return;
   }
#if PYFLUID_AMREX_ENABLE_MHD
   apply_rmhd_limiter(
      riemann_method,
      fluxes, rhs, base_conserved, conserved, primitive, eos, geometry,
      fixed_spacetime, cell_spacetime, candidate_cell_spacetime, atmosphere,
      time, conserved_weight, increment_scale, try_pwl_first,
      enforce_dmp, dmp_factor, pwl_correction_count, pwc_correction_count,
      hll_fallback_count, black_hole_excision_mask, excision_region, diode
   );
#else
   amrex::Abort("RMHD candidate correction is disabled in this build");
#endif
}

} // namespace pyfluid_amrex
