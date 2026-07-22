#include "pyfluid_amrex/relativistic_flux_limiter.hpp"

#include <AMReX_GpuLaunch.H>

#include <limits>

#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/flux.hpp"
#include "pyfluid_amrex/kernels_generated/riemann_ops.hpp"
#include "pyfluid_amrex/relativistic_boundary.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/state.hpp"
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
   if (eos_method == EOSPolytrope) {
      apply_limiter<EOSPolytrope>(
         fluxes, rhs, base_conserved, conserved, primitive, eos, geometry, fixed_spacetime,
         cell_spacetime, candidate_cell_spacetime, atmosphere, time,
         base_weight, conserved_weight, increment_scale
      );
   } else if (eos_method == EOSHybridPolytrope) {
      apply_limiter<EOSHybridPolytrope>(
         fluxes, rhs, base_conserved, conserved, primitive, eos, geometry, fixed_spacetime,
         cell_spacetime, candidate_cell_spacetime, atmosphere, time,
         base_weight, conserved_weight, increment_scale
      );
   } else {
      apply_limiter<EOSIdealGas>(
         fluxes, rhs, base_conserved, conserved, primitive, eos, geometry, fixed_spacetime,
         cell_spacetime, candidate_cell_spacetime, atmosphere, time,
         base_weight, conserved_weight, increment_scale
      );
   }
}

} // namespace pyfluid_amrex
