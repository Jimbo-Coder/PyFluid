#include "pyfluid_amrex/relativistic_flux_limiter.hpp"

#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>

#include <limits>

#include "pyfluid_amrex/build_options.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/flux.hpp"
#include "pyfluid_amrex/kernels_generated/riemann_ops.hpp"
#include "pyfluid_amrex/relativistic_boundary.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_riemann.hpp"
#include "pyfluid_amrex/state.hpp"
#include "relativity/rmhd_system.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {
namespace {

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
      amrex::Real(0.0), recovery.momentum_squared
   ));
   const amrex::Real scale = amrex::max(
      amrex::Math::abs(energy),
      amrex::max(amrex::Math::abs(recovery.dens), momentum)
   );
   const amrex::Real roundoff_floor =
      256.0 * std::numeric_limits<amrex::Real>::epsilon()
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
      ? 1.0
      : 0.0;
}

struct StoredMetricAccessor {
   amrex::Array4<amrex::Real const> spacetime;
   amrex::Array4<amrex::Real const> candidate_spacetime;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric cell(int i, int j, int k) const noexcept
   {
      return generated::z4c_spacetime_metric(
         load_z4c_state(spacetime, i, j, k)
      );
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric candidate_cell(int i, int j, int k) const noexcept
   {
      return generated::z4c_spacetime_metric(
         load_z4c_state(candidate_spacetime, i, j, k)
      );
   }

   template <int Direction>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric face(int i, int j, int k) const noexcept
   {
      return stored_spacetime_metric_at_face<4, Direction>(
         spacetime, i, j, k
      );
   }
};

struct FixedMetricAccessor {
   FixedSpacetimeParameters spacetime;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lower;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> spacing;
   amrex::Real time;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric cell(int i, int j, int k) const noexcept
   {
      return fixed_spacetime_geometry_at_cell<AMREX_SPACEDIM>(
         spacetime, lower, spacing, i, j, k, time
      ).metric;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric candidate_cell(int i, int j, int k) const noexcept
   {
      return cell(i, j, k);
   }

   template <int Direction>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric face(int i, int j, int k) const noexcept
   {
      return fixed_spacetime_geometry_at_face<Direction, AMREX_SPACEDIM>(
         spacetime, lower, spacing, i, j, k, time
      ).metric;
   }
};

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void left_cell(int i, int j, int k, int& il, int& jl, int& kl) noexcept
{
   il = i;
   jl = j;
   kl = k;
   if constexpr (Direction == 0) {
      --il;
   } else if constexpr (Direction == 1) {
      --jl;
   } else {
      --kl;
   }
}

template <int Direction, int EOSMethod, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDConserved low_order_flux(
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   MetricAccessor const& metric
) noexcept
{
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
   amrex::Real speed = 1.0;
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
   amrex::Real dt_over_dx
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
      primitive, ih, jh, kh, eos, metric
   );
   const RHDConserved low_lo = low_order_flux<Direction, EOSMethod>(
      primitive, i, j, k, eos, metric
   );
   candidate = add_scaled(candidate, difference(high_hi, low_hi), dt_over_dx);
   candidate = add_scaled(candidate, difference(high_lo, low_lo), -dt_over_dx);
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
   amrex::Real inverse_dx
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
         primitive, ih, jh, kh, eos, metric
      ),
      high_hi
   );
   const RHDConserved delta_hi = add_scaled(
      low_minus_high_hi, low_minus_high_hi, -theta_hi
   );
   const RHDConserved low_minus_high_lo = difference(
      low_order_flux<Direction, EOSMethod>(
         primitive, i, j, k, eos, metric
      ),
      high_lo
   );
   const RHDConserved delta_lo = add_scaled(
      low_minus_high_lo, low_minus_high_lo, -theta_lo
   );
   rhs = add_scaled(rhs, difference(delta_hi, delta_lo), -inverse_dx);
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
   MetricAccessor const& metric
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
      primitive, i, j, k, eos, metric
   );
   store_rhd_conserved(high_flux, i, j, k, blend(low, high, face_theta));
}

template <int EOSMethod, typename MetricAccessorFactory>
void apply_limiter_impl(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab& rhs,
   amrex::MultiFab const& base_conserved,
   amrex::MultiFab const& conserved,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   amrex::Geometry const& geometry,
   EOSThermodynamicState const& atmosphere,
   amrex::Real base_weight,
   amrex::Real conserved_weight,
   amrex::Real increment_scale,
   MetricAccessorFactory&& metric_accessor_factory
)
{
   amrex::MultiFab theta(
      rhs.boxArray(), rhs.DistributionMap(), 1, 1
   );
   theta.setVal(1.0);
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
      const auto flux0 = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto flux1 = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto flux2 = fluxes[2]->const_array(mfi);
#endif
      const auto theta_values = theta.array(mfi);
      const auto metric = metric_accessor_factory(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         RHDConserved high_candidate{};
         high_candidate = add_scaled(
            high_candidate,
            load_rhd_conserved(base_values, i, j, k),
            base_weight
         );
         high_candidate = add_scaled(
            high_candidate,
            load_rhd_conserved(state_values, i, j, k),
            conserved_weight
         );
         high_candidate = add_scaled(
            high_candidate,
            load_rhd_conserved(rhs_values, i, j, k),
            increment_scale
         );
         RHDConserved low_candidate = high_candidate;
         add_low_order_candidate_correction<0, EOSMethod>(
            low_candidate, flux0, primitive_values, i, j, k, eos, metric,
            increment_scale / spacing[0]
         );
#if (AMREX_SPACEDIM >= 2)
         add_low_order_candidate_correction<1, EOSMethod>(
            low_candidate, flux1, primitive_values, i, j, k, eos, metric,
            increment_scale / spacing[1]
         );
#endif
#if (AMREX_SPACEDIM == 3)
         add_low_order_candidate_correction<2, EOSMethod>(
            low_candidate, flux2, primitive_values, i, j, k, eos, metric,
            increment_scale / spacing[2]
         );
#endif
         theta_values(i, j, k, 0) = admissibility_theta(
            low_candidate,
            high_candidate,
            metric.candidate_cell(i, j, k),
            atmosphere.rho,
            atmosphere.pressure
         );
      });
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
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         RHDConserved value = load_rhd_conserved(rhs_values, i, j, k);
         add_limited_rhs_correction<0, EOSMethod>(
            value, flux0, primitive_values, theta_values, i, j, k, eos,
            metric, 1.0 / spacing[0]
         );
#if (AMREX_SPACEDIM >= 2)
         add_limited_rhs_correction<1, EOSMethod>(
            value, flux1, primitive_values, theta_values, i, j, k, eos,
            metric, 1.0 / spacing[1]
         );
#endif
#if (AMREX_SPACEDIM == 3)
         add_limited_rhs_correction<2, EOSMethod>(
            value, flux2, primitive_values, theta_values, i, j, k, eos,
            metric, 1.0 / spacing[2]
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
         if (direction == 0) {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               blend_face_flux<0, EOSMethod>(
                  flux_values, primitive_values, theta_values, i, j, k,
                  eos, metric
               );
            });
         }
#if (AMREX_SPACEDIM >= 2)
         else if (direction == 1) {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               blend_face_flux<1, EOSMethod>(
                  flux_values, primitive_values, theta_values, i, j, k,
                  eos, metric
               );
            });
         }
#endif
#if (AMREX_SPACEDIM == 3)
         else {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               blend_face_flux<2, EOSMethod>(
                  flux_values, primitive_values, theta_values, i, j, k,
                  eos, metric
               );
            });
         }
#endif
      }
   }
}

template <int EOSMethod>
void apply_limiter(
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
   amrex::Real base_weight,
   amrex::Real conserved_weight,
   amrex::Real increment_scale
)
{
   if (cell_spacetime != nullptr) {
      AMREX_ALWAYS_ASSERT(candidate_cell_spacetime != nullptr);
      apply_limiter_impl<EOSMethod>(
         fluxes, rhs, base_conserved, conserved, primitive, eos, geometry,
         atmosphere, base_weight, conserved_weight, increment_scale,
         [&](amrex::MFIter const& mfi) {
            return StoredMetricAccessor{
               cell_spacetime->const_array(mfi),
               candidate_cell_spacetime->const_array(mfi)
            };
         }
      );
   } else {
      const auto lower = geometry.ProbLoArray();
      const auto spacing = geometry.CellSizeArray();
      apply_limiter_impl<EOSMethod>(
         fluxes, rhs, base_conserved, conserved, primitive, eos, geometry,
         atmosphere, base_weight, conserved_weight, increment_scale,
         [=](amrex::MFIter const&) {
            return FixedMetricAccessor{
               fixed_spacetime, lower, spacing, time
            };
         }
      );
   }
}

} // namespace

void apply_rhd_admissibility_flux_limiter(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab& rhs,
   amrex::MultiFab const& base_conserved,
   amrex::MultiFab const& conserved,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   int eos_method,
   amrex::Geometry const& geometry,
   FixedSpacetimeParameters const& fixed_spacetime,
   amrex::MultiFab const* cell_spacetime,
   amrex::MultiFab const* candidate_cell_spacetime,
   EOSThermodynamicState const& atmosphere,
   amrex::Real time,
   amrex::Real base_weight,
   amrex::Real conserved_weight,
   amrex::Real increment_scale
)
{
   if (!(increment_scale > 0.0)) {
      return;
   }
   static_cast<void>(eos_method);
   apply_limiter<EOSRuntime>(
      fluxes, rhs, base_conserved, conserved, primitive, eos, geometry,
      fixed_spacetime, cell_spacetime, candidate_cell_spacetime, atmosphere,
      time, base_weight, conserved_weight, increment_scale
   );
}

namespace {

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

template <int ReconMethod, int RiemannMethod, int Direction, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRiemannFluxResult rmhd_correction_flux(
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   MetricAccessor const& metric,
   EOSThermodynamicState const& atmosphere
) noexcept
{
   int il;
   int jl;
   int kl;
   left_cell<Direction>(i, j, k, il, jl, kl);
   if constexpr (RiemannMethod == RiemannHLLC
                 || RiemannMethod == RiemannHLLD) {
      const RMHDPrimitiveFaceStates face =
         rmhd_primitive_face_states<ReconMethod, Direction>(
            primitive, il, jl, kl, i, j, k, eos, metric,
            atmosphere, atmosphere
         );
      const SpacetimeMetric face_metric = metric.template face<Direction>(
         i, j, k
      );
      if constexpr (RiemannMethod == RiemannHLLC) {
         return rmhd_hllc_flux<Direction>(
            face.left, face.right, eos, face_metric
         );
      } else {
         return rmhd_hlld_flux<Direction>(
            face.left, face.right, eos, face_metric
         );
      }
   } else {
      return {
         rmhd_riemann_flux<ReconMethod, RiemannMethod, Direction>(
            primitive, il, jl, kl, i, j, k, eos, metric,
            atmosphere, atmosphere
         ),
         false,
      };
   }
}

template <int ReconMethod, int RiemannMethod, int Direction, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void add_rmhd_candidate_correction(
   RMHDConserved& candidate,
   amrex::Array4<amrex::Real const> const& high_flux,
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   MetricAccessor const& metric,
   EOSThermodynamicState const& atmosphere,
   amrex::Real dt_over_dx
) noexcept
{
   int ih;
   int jh;
   int kh;
   high_face<Direction>(i, j, k, ih, jh, kh);
   const RMHDConserved high_hi = load_rmhd_conserved(
      high_flux, ih, jh, kh
   );
   const RMHDConserved high_lo = load_rmhd_conserved(high_flux, i, j, k);
   const RMHDConserved corrected_hi =
      rmhd_correction_flux<
         ReconMethod,
         RiemannMethod,
         Direction
      >(primitive, ih, jh, kh, eos, metric, atmosphere).flux;
   const RMHDConserved corrected_lo =
      rmhd_correction_flux<
         ReconMethod,
         RiemannMethod,
         Direction
      >(primitive, i, j, k, eos, metric, atmosphere).flux;
   candidate = add_scaled(
      candidate, difference(high_hi, corrected_hi), dt_over_dx
   );
   candidate = add_scaled(
      candidate, difference(high_lo, corrected_lo), -dt_over_dx
   );
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
   amrex::Real factor
) noexcept
{
   if (!enforce_dmp) {
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
   amrex::Long* hll_fallback_count = nullptr
) noexcept
{
   if (correction_kind == 0) {
      return high;
   }
   if (correction_kind == 1) {
      const RMHDRiemannFluxResult result = rmhd_correction_flux<
         ReconPWL,
         RiemannMethod,
         Direction
      >(primitive, i, j, k, eos, metric, atmosphere);
      if (hll_fallback_count != nullptr && result.used_hll) {
         amrex::HostDevice::Atomic::Add(
            hll_fallback_count, static_cast<amrex::Long>(1)
         );
      }
      return result.flux;
   }
   const RMHDRiemannFluxResult result = rmhd_correction_flux<
      ReconPWC,
      RiemannMethod,
      Direction
   >(primitive, i, j, k, eos, metric, atmosphere);
   if (hll_fallback_count != nullptr && result.used_hll) {
      amrex::HostDevice::Atomic::Add(
         hll_fallback_count, static_cast<amrex::Long>(1)
      );
   }
   return result.flux;
}

template <int RiemannMethod, int Direction, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void add_selected_rmhd_rhs_correction(
   RMHDConserved& rhs,
   amrex::Array4<amrex::Real const> const& high_flux,
   amrex::Array4<amrex::Real const> const& primitive,
   amrex::Array4<int const> const& correction,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   MetricAccessor const& metric,
   EOSThermodynamicState const& atmosphere,
   amrex::Real inverse_dx
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
   const int high_kind = amrex::max(
      correction(i, j, k, 0), correction(ih, jh, kh, 0)
   );
   const int low_kind = amrex::max(
      correction(il, jl, kl, 0), correction(i, j, k, 0)
   );
   const RMHDConserved high_hi = load_rmhd_conserved(
      high_flux, ih, jh, kh
   );
   const RMHDConserved high_lo = load_rmhd_conserved(high_flux, i, j, k);
   const RMHDConserved selected_hi = selected_rmhd_flux<
      RiemannMethod,
      Direction
   >(
      high_kind, high_hi, primitive, ih, jh, kh, eos, metric, atmosphere
   );
   const RMHDConserved selected_lo = selected_rmhd_flux<
      RiemannMethod,
      Direction
   >(
      low_kind, high_lo, primitive, i, j, k, eos, metric, atmosphere
   );
   rhs = add_scaled(
      rhs,
      difference(
         difference(selected_hi, high_hi),
         difference(selected_lo, high_lo)
      ),
      -inverse_dx
   );
}

template <int RiemannMethod, typename MetricAccessorFactory>
void apply_rmhd_limiter_impl(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab& rhs,
   amrex::MultiFab const& base_conserved,
   amrex::MultiFab const& conserved,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   amrex::Geometry const& geometry,
   EOSThermodynamicState const& atmosphere,
   amrex::Real base_weight,
   amrex::Real conserved_weight,
   amrex::Real increment_scale,
   bool try_pwl_first,
   bool enforce_dmp,
   amrex::Real dmp_factor,
   amrex::Long* pwl_correction_count,
   amrex::Long* pwc_correction_count,
   amrex::Long* hll_fallback_count,
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
      const auto flux0 = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto flux1 = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto flux2 = fluxes[2]->const_array(mfi);
#endif
      const auto correction_values = correction.array(mfi);
      const auto metric = metric_accessor_factory(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         RMHDConserved high_candidate{};
         high_candidate = add_scaled(
            high_candidate,
            load_rmhd_conserved(base_values, i, j, k),
            base_weight
         );
         high_candidate = add_scaled(
            high_candidate,
            load_rmhd_conserved(state_values, i, j, k),
            conserved_weight
         );
         high_candidate = add_scaled(
            high_candidate,
            load_rmhd_conserved(rhs_values, i, j, k),
            increment_scale
         );
         const SpacetimeMetric candidate_metric =
            metric.candidate_cell(i, j, k);
         const bool high_bad = rmhd_candidate_violates_dmp(
            high_candidate, state_values, i, j, k, enforce_dmp, dmp_factor
         ) || !rmhd_candidate_recovers(
            high_candidate, candidate_metric, eos, atmosphere
         );
         if (!high_bad) {
            return;
         }
         int correction_kind = 2;
         if (try_pwl_first) {
            RMHDConserved pwl_candidate = high_candidate;
            add_rmhd_candidate_correction<
               ReconPWL,
               RiemannMethod,
               0
            >(
               pwl_candidate, flux0, primitive_values, i, j, k, eos, metric,
               atmosphere, increment_scale / spacing[0]
            );
#if (AMREX_SPACEDIM >= 2)
            add_rmhd_candidate_correction<
               ReconPWL,
               RiemannMethod,
               1
            >(
               pwl_candidate, flux1, primitive_values, i, j, k, eos, metric,
               atmosphere, increment_scale / spacing[1]
            );
#endif
#if (AMREX_SPACEDIM == 3)
            add_rmhd_candidate_correction<
               ReconPWL,
               RiemannMethod,
               2
            >(
               pwl_candidate, flux2, primitive_values, i, j, k, eos, metric,
               atmosphere, increment_scale / spacing[2]
            );
#endif
            const bool pwl_bad = rmhd_candidate_violates_dmp(
               pwl_candidate, state_values, i, j, k, enforce_dmp, dmp_factor
            ) || !rmhd_candidate_recovers(
               pwl_candidate, candidate_metric, eos, atmosphere
            );
            if (!pwl_bad) {
               correction_kind = 1;
            }
         }
         correction_values(i, j, k, 0) = correction_kind;
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
         const auto flux0 = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
         const auto flux1 = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
         const auto flux2 = fluxes[2]->const_array(mfi);
#endif
         const auto metric = metric_accessor_factory(mfi);
         amrex::Long* newly_troubled = newly_troubled_count.dataPtr();
         amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if (current_values(i, j, k, 0) == 2) {
                  return;
               }
               RMHDConserved candidate{};
               candidate = add_scaled(
                  candidate,
                  load_rmhd_conserved(base_values, i, j, k),
                  base_weight
               );
               candidate = add_scaled(
                  candidate,
                  load_rmhd_conserved(state_values, i, j, k),
                  conserved_weight
               );
               candidate = add_scaled(
                  candidate,
                  load_rmhd_conserved(rhs_values, i, j, k),
                  increment_scale
               );
               add_selected_rmhd_rhs_correction<RiemannMethod, 0>(
                  candidate, flux0, primitive_values, current_values,
                  i, j, k, eos, metric, atmosphere,
                  increment_scale / spacing[0]
               );
#if (AMREX_SPACEDIM >= 2)
               add_selected_rmhd_rhs_correction<RiemannMethod, 1>(
                  candidate, flux1, primitive_values, current_values,
                  i, j, k, eos, metric, atmosphere,
                  increment_scale / spacing[1]
               );
#endif
#if (AMREX_SPACEDIM == 3)
               add_selected_rmhd_rhs_correction<RiemannMethod, 2>(
                  candidate, flux2, primitive_values, current_values,
                  i, j, k, eos, metric, atmosphere,
                  increment_scale / spacing[2]
               );
#endif
               const SpacetimeMetric candidate_metric =
                  metric.candidate_cell(i, j, k);
               const bool candidate_bad = !rmhd_candidate_recovers(
                  candidate, candidate_metric, eos, atmosphere
               );
               if (candidate_bad) {
                  next_values(i, j, k, 0) = 2;
                  amrex::HostDevice::Atomic::Add(
                     newly_troubled, static_cast<amrex::Long>(1)
                  );
               }
            }
         );
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
         amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
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
      const auto flux0 = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto flux1 = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto flux2 = fluxes[2]->const_array(mfi);
#endif
      const auto metric = metric_accessor_factory(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         RMHDConserved value = load_rmhd_conserved(rhs_values, i, j, k);
         add_selected_rmhd_rhs_correction<RiemannMethod, 0>(
            value, flux0, primitive_values, correction_values, i, j, k, eos,
            metric, atmosphere, 1.0 / spacing[0]
         );
#if (AMREX_SPACEDIM >= 2)
         add_selected_rmhd_rhs_correction<RiemannMethod, 1>(
            value, flux1, primitive_values, correction_values, i, j, k, eos,
            metric, atmosphere, 1.0 / spacing[1]
         );
#endif
#if (AMREX_SPACEDIM == 3)
         add_selected_rmhd_rhs_correction<RiemannMethod, 2>(
            value, flux2, primitive_values, correction_values, i, j, k, eos,
            metric, atmosphere, 1.0 / spacing[2]
         );
#endif
         store_rmhd_conserved(rhs_values, i, j, k, value);
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
         const auto correction_values = correction.const_array(mfi);
         const auto metric = metric_accessor_factory(mfi);
         if (direction == 0) {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               int il;
               int jl;
               int kl;
               left_cell<0>(i, j, k, il, jl, kl);
               const int kind = amrex::max(
                  correction_values(il, jl, kl, 0),
                  correction_values(i, j, k, 0)
               );
               const RMHDConserved high = load_rmhd_conserved(
                  flux_values, i, j, k
               );
               store_rmhd_conserved(
                  flux_values, i, j, k,
                  selected_rmhd_flux<RiemannMethod, 0>(
                     kind, high, primitive_values, i, j, k, eos, metric,
                     atmosphere, hll_fallback_count
                  )
               );
            });
         }
#if (AMREX_SPACEDIM >= 2)
         else if (direction == 1) {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               int il;
               int jl;
               int kl;
               left_cell<1>(i, j, k, il, jl, kl);
               const int kind = amrex::max(
                  correction_values(il, jl, kl, 0),
                  correction_values(i, j, k, 0)
               );
               const RMHDConserved high = load_rmhd_conserved(
                  flux_values, i, j, k
               );
               store_rmhd_conserved(
                  flux_values, i, j, k,
                  selected_rmhd_flux<RiemannMethod, 1>(
                     kind, high, primitive_values, i, j, k, eos, metric,
                     atmosphere, hll_fallback_count
                  )
               );
            });
         }
#endif
#if (AMREX_SPACEDIM == 3)
         else {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               int il;
               int jl;
               int kl;
               left_cell<2>(i, j, k, il, jl, kl);
               const int kind = amrex::max(
                  correction_values(il, jl, kl, 0),
                  correction_values(i, j, k, 0)
               );
               const RMHDConserved high = load_rmhd_conserved(
                  flux_values, i, j, k
               );
               store_rmhd_conserved(
                  flux_values, i, j, k,
                  selected_rmhd_flux<RiemannMethod, 2>(
                     kind, high, primitive_values, i, j, k, eos, metric,
                     atmosphere, hll_fallback_count
                  )
               );
            });
         }
#endif
      }
   }
}

template <int RiemannMethod>
void apply_rmhd_limiter(
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
   amrex::Real base_weight,
   amrex::Real conserved_weight,
   amrex::Real increment_scale,
   bool try_pwl_first,
   bool enforce_dmp,
   amrex::Real dmp_factor,
   amrex::Long* pwl_correction_count,
   amrex::Long* pwc_correction_count,
   amrex::Long* hll_fallback_count
)
{
   auto apply = [&](auto&& metric_factory) {
      apply_rmhd_limiter_impl<RiemannMethod>(
         fluxes, rhs, base_conserved, conserved, primitive, eos, geometry,
         atmosphere, base_weight, conserved_weight, increment_scale,
         try_pwl_first, enforce_dmp, dmp_factor, pwl_correction_count,
         pwc_correction_count, hll_fallback_count, metric_factory
      );
   };
   if (cell_spacetime != nullptr) {
      AMREX_ALWAYS_ASSERT(candidate_cell_spacetime != nullptr);
      apply([&](amrex::MFIter const& mfi) {
         return StoredMetricAccessor{
            cell_spacetime->const_array(mfi),
            candidate_cell_spacetime->const_array(mfi)
         };
      });
   } else {
      const auto lower = geometry.ProbLoArray();
      const auto spacing = geometry.CellSizeArray();
      apply([=](amrex::MFIter const&) {
         return FixedMetricAccessor{
            fixed_spacetime, lower, spacing, time
         };
      });
   }
}

} // namespace

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
   amrex::Real base_weight,
   amrex::Real conserved_weight,
   amrex::Real increment_scale,
   bool try_pwl_first,
   bool enforce_dmp,
   amrex::Real dmp_factor,
   amrex::Long* pwl_correction_count,
   amrex::Long* pwc_correction_count,
   amrex::Long* hll_fallback_count
)
{
   if (!(increment_scale > 0.0)) {
      return;
   }
   if (riemann_method == RiemannLLF) {
      apply_rmhd_limiter<RiemannLLF>(
         fluxes, rhs, base_conserved, conserved, primitive, eos, geometry,
         fixed_spacetime, cell_spacetime, candidate_cell_spacetime, atmosphere,
         time, base_weight, conserved_weight, increment_scale, try_pwl_first,
         enforce_dmp, dmp_factor, pwl_correction_count, pwc_correction_count,
         hll_fallback_count
      );
   } else if (riemann_method == RiemannHLL
              || riemann_method == RiemannHLLE) {
      apply_rmhd_limiter<RiemannHLL>(
         fluxes, rhs, base_conserved, conserved, primitive, eos, geometry,
         fixed_spacetime, cell_spacetime, candidate_cell_spacetime, atmosphere,
         time, base_weight, conserved_weight, increment_scale, try_pwl_first,
         enforce_dmp, dmp_factor, pwl_correction_count, pwc_correction_count,
         hll_fallback_count
      );
   } else if (riemann_method == RiemannHLLC) {
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLLC
      apply_rmhd_limiter<RiemannHLLC>(
         fluxes, rhs, base_conserved, conserved, primitive, eos, geometry,
         fixed_spacetime, cell_spacetime, candidate_cell_spacetime, atmosphere,
         time, base_weight, conserved_weight, increment_scale, try_pwl_first,
         enforce_dmp, dmp_factor, pwl_correction_count, pwc_correction_count,
         hll_fallback_count
      );
#else
      amrex::Abort("RMHD HLLC candidate correction is disabled in this build");
#endif
   } else if (riemann_method == RiemannHLLD) {
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLLD
      apply_rmhd_limiter<RiemannHLLD>(
         fluxes, rhs, base_conserved, conserved, primitive, eos, geometry,
         fixed_spacetime, cell_spacetime, candidate_cell_spacetime, atmosphere,
         time, base_weight, conserved_weight, increment_scale, try_pwl_first,
         enforce_dmp, dmp_factor, pwl_correction_count, pwc_correction_count,
         hll_fallback_count
      );
#else
      amrex::Abort("RMHD HLLD candidate correction is disabled in this build");
#endif
   } else {
      amrex::Abort("invalid RMHD candidate-correction Riemann solver");
   }
}

} // namespace pyfluid_amrex
