#pragma once

#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

#include "pyfluid_amrex/relativistic_state.hpp"

namespace pyfluid_amrex {

// Outgoing-null expansion of a level surface F(x^i) = 0.  The caller owns
// the surface representation and supplies m_i = partial_i F and
// partial_i partial_j F in Cartesian components.  This pointwise geometry is
// shared by future Newton or hyperbolic-flow surface solvers.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real apparent_horizon_expansion(
   SpacetimeMetric const& metric,
   SpacetimeMetricDerivatives const& derivatives,
   amrex::Real const level_set_gradient[3],
   amrex::Real const level_set_hessian[3][3]
) noexcept
{
   const amrex::Real inverse_metric[3][3]{
      {metric.inv_gamma11, metric.inv_gamma12, metric.inv_gamma13},
      {metric.inv_gamma12, metric.inv_gamma22, metric.inv_gamma23},
      {metric.inv_gamma13, metric.inv_gamma23, metric.inv_gamma33},
   };
   const amrex::Real metric_derivative[3][3][3]{
      {
         {derivatives.d1_gamma11, derivatives.d1_gamma12, derivatives.d1_gamma13},
         {derivatives.d1_gamma12, derivatives.d1_gamma22, derivatives.d1_gamma23},
         {derivatives.d1_gamma13, derivatives.d1_gamma23, derivatives.d1_gamma33},
      },
      {
         {derivatives.d2_gamma11, derivatives.d2_gamma12, derivatives.d2_gamma13},
         {derivatives.d2_gamma12, derivatives.d2_gamma22, derivatives.d2_gamma23},
         {derivatives.d2_gamma13, derivatives.d2_gamma23, derivatives.d2_gamma33},
      },
      {
         {derivatives.d3_gamma11, derivatives.d3_gamma12, derivatives.d3_gamma13},
         {derivatives.d3_gamma12, derivatives.d3_gamma22, derivatives.d3_gamma23},
         {derivatives.d3_gamma13, derivatives.d3_gamma23, derivatives.d3_gamma33},
      },
   };
   const amrex::Real extrinsic_curvature[3][3]{
      {derivatives.extrinsic11, derivatives.extrinsic12, derivatives.extrinsic13},
      {derivatives.extrinsic12, derivatives.extrinsic22, derivatives.extrinsic23},
      {derivatives.extrinsic13, derivatives.extrinsic23, derivatives.extrinsic33},
   };

   amrex::Real normal_squared = 0.0;
   for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
         normal_squared += inverse_metric[first][second]
            * level_set_gradient[first] * level_set_gradient[second];
      }
   }
   // This also rejects NaN because its ordered comparison is false.
   if (!(normal_squared > 0.0)) {
      return std::numeric_limits<amrex::Real>::quiet_NaN();
   }
   const amrex::Real inverse_normal = 1.0 / std::sqrt(normal_squared);
   amrex::Real unit_normal[3]{};
   for (int upper = 0; upper < 3; ++upper) {
      for (int lower = 0; lower < 3; ++lower) {
         unit_normal[upper] += inverse_metric[upper][lower]
            * level_set_gradient[lower] * inverse_normal;
      }
   }

   amrex::Real divergence = 0.0;
   for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
         amrex::Real covariant_hessian = level_set_hessian[first][second];
         for (int upper = 0; upper < 3; ++upper) {
            amrex::Real connection = 0.0;
            for (int contracted = 0; contracted < 3; ++contracted) {
               connection += 0.5 * inverse_metric[upper][contracted] * (
                  metric_derivative[first][contracted][second]
                  + metric_derivative[second][contracted][first]
                  - metric_derivative[contracted][first][second]
               );
            }
            covariant_hessian -= connection * level_set_gradient[upper];
         }
         divergence += (
            inverse_metric[first][second]
            - unit_normal[first] * unit_normal[second]
         ) * covariant_hessian * inverse_normal;
      }
   }

   amrex::Real trace_extrinsic = 0.0;
   amrex::Real normal_extrinsic = 0.0;
   for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
         trace_extrinsic += inverse_metric[first][second]
            * extrinsic_curvature[first][second];
         normal_extrinsic += unit_normal[first] * unit_normal[second]
            * extrinsic_curvature[first][second];
      }
   }
   return divergence + normal_extrinsic - trace_extrinsic;
}

} // namespace pyfluid_amrex
