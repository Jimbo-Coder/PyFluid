#pragma once

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "pyfluid_amrex/fixed_spacetime_types.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"

namespace pyfluid_amrex {

struct FixedSpacetimeGeometry {
   SpacetimeMetric metric;
   SpacetimeMetricDerivatives derivatives;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FixedSpacetimeGeometry fixed_spacetime_geometry(
   const FixedSpacetimeParameters& parameters,
   amrex::Real position1,
   amrex::Real position2,
   amrex::Real position3,
   amrex::Real /* time */
) noexcept
{
   FixedSpacetimeGeometry geometry{};
   if (parameters.metric == FixedSpacetimeMetric::Rindler) {
      geometry.metric.lapse = 1.0
         + parameters.acceleration1 * (position1 - parameters.origin1)
         + parameters.acceleration2 * (position2 - parameters.origin2)
         + parameters.acceleration3 * (position3 - parameters.origin3);
      geometry.derivatives.d1_lapse = parameters.acceleration1;
      geometry.derivatives.d2_lapse = parameters.acceleration2;
      geometry.derivatives.d3_lapse = parameters.acceleration3;
   } else if (parameters.metric == FixedSpacetimeMetric::SchwarzschildIsotropic) {
      const amrex::Real displacement1 = position1 - parameters.origin1;
      const amrex::Real displacement2 = position2 - parameters.origin2;
      const amrex::Real displacement3 = position3 - parameters.origin3;
      const amrex::Real radius_squared = displacement1 * displacement1
         + displacement2 * displacement2 + displacement3 * displacement3;
      const amrex::Real radius = std::sqrt(radius_squared);
      const amrex::Real inverse_radius = 1.0 / radius;
      const amrex::Real inverse_radius_cubed = inverse_radius / radius_squared;
      const amrex::Real compactness = 0.5 * parameters.mass * inverse_radius;
      const amrex::Real conformal_factor = 1.0 + compactness;
      const amrex::Real conformal_squared = conformal_factor * conformal_factor;
      const amrex::Real conformal_cubed = conformal_squared * conformal_factor;
      const amrex::Real conformal_fourth = conformal_squared * conformal_squared;
      geometry.metric.lapse = (1.0 - compactness) / conformal_factor;
      geometry.metric.gamma11 = conformal_fourth;
      geometry.metric.gamma22 = conformal_fourth;
      geometry.metric.gamma33 = conformal_fourth;
      geometry.metric.inv_gamma11 = 1.0 / conformal_fourth;
      geometry.metric.inv_gamma22 = 1.0 / conformal_fourth;
      geometry.metric.inv_gamma33 = 1.0 / conformal_fourth;
      geometry.metric.sqrt_det_gamma = conformal_fourth * conformal_squared;
      const amrex::Real lapse_derivative_scale = parameters.mass
         * inverse_radius_cubed / conformal_squared;
      geometry.derivatives.d1_lapse = lapse_derivative_scale * displacement1;
      geometry.derivatives.d2_lapse = lapse_derivative_scale * displacement2;
      geometry.derivatives.d3_lapse = lapse_derivative_scale * displacement3;
      const amrex::Real metric_derivative_scale = -2.0 * parameters.mass
         * conformal_cubed * inverse_radius_cubed;
      geometry.derivatives.d1_gamma11 = metric_derivative_scale * displacement1;
      geometry.derivatives.d1_gamma22 = metric_derivative_scale * displacement1;
      geometry.derivatives.d1_gamma33 = metric_derivative_scale * displacement1;
      geometry.derivatives.d2_gamma11 = metric_derivative_scale * displacement2;
      geometry.derivatives.d2_gamma22 = metric_derivative_scale * displacement2;
      geometry.derivatives.d2_gamma33 = metric_derivative_scale * displacement2;
      geometry.derivatives.d3_gamma11 = metric_derivative_scale * displacement3;
      geometry.derivatives.d3_gamma22 = metric_derivative_scale * displacement3;
      geometry.derivatives.d3_gamma33 = metric_derivative_scale * displacement3;
   }
   return geometry;
}

constexpr bool has_fixed_spacetime_sources(
   const FixedSpacetimeParameters& parameters
) noexcept
{
   return parameters.metric != FixedSpacetimeMetric::Minkowski;
}

inline void validate_fixed_spacetime_domain(
   const FixedSpacetimeParameters& parameters,
   const amrex::Geometry& geometry
)
{
   if (parameters.metric == FixedSpacetimeMetric::Minkowski) {
      return;
   }
   const auto lower = geometry.ProbLoArray();
   const auto upper = geometry.ProbHiArray();
   if (parameters.metric == FixedSpacetimeMetric::SchwarzschildIsotropic) {
      const auto interval_distance = [](amrex::Real point, amrex::Real lo, amrex::Real hi) {
         return point < lo ? lo - point : (point > hi ? point - hi : 0.0);
      };
      const amrex::Real distance1 = interval_distance(parameters.origin1, lower[0], upper[0]);
      amrex::Real radius_squared = distance1 * distance1;
#if (AMREX_SPACEDIM >= 2)
      const amrex::Real distance2 = interval_distance(parameters.origin2, lower[1], upper[1]);
      radius_squared += distance2 * distance2;
#endif
#if (AMREX_SPACEDIM == 3)
      const amrex::Real distance3 = interval_distance(parameters.origin3, lower[2], upper[2]);
      radius_squared += distance3 * distance3;
#endif
      if (!(parameters.mass > 0.0)
          || !(std::sqrt(radius_squared) > 0.5 * parameters.mass)) {
         amrex::Abort(
            "the isotropic-Schwarzschild domain must remain outside r = M/2"
         );
      }
      return;
   }
   amrex::Real minimum_lapse = 1.0;
   minimum_lapse += parameters.acceleration1
      * ((parameters.acceleration1 >= 0.0 ? lower[0] : upper[0]) - parameters.origin1);
#if (AMREX_SPACEDIM >= 2)
   minimum_lapse += parameters.acceleration2
      * ((parameters.acceleration2 >= 0.0 ? lower[1] : upper[1]) - parameters.origin2);
#endif
#if (AMREX_SPACEDIM == 3)
   minimum_lapse += parameters.acceleration3
      * ((parameters.acceleration3 >= 0.0 ? lower[2] : upper[2]) - parameters.origin3);
#endif
   if (!(minimum_lapse > 0.0)) {
      amrex::Abort("the fixed Rindler lapse must remain positive over the domain");
   }
}

template <int Dimension>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FixedSpacetimeGeometry fixed_spacetime_geometry_at_cell(
   const FixedSpacetimeParameters& parameters,
   const amrex::GpuArray<amrex::Real, Dimension>& lower,
   const amrex::GpuArray<amrex::Real, Dimension>& spacing,
   int i,
   int j,
   int k,
   amrex::Real time
) noexcept
{
   const amrex::Real position1 = lower[0] + (static_cast<amrex::Real>(i) + 0.5) * spacing[0];
   amrex::Real position2 = 0.0;
   amrex::Real position3 = 0.0;
   if constexpr (Dimension >= 2) {
      position2 = lower[1] + (static_cast<amrex::Real>(j) + 0.5) * spacing[1];
   }
   if constexpr (Dimension >= 3) {
      position3 = lower[2] + (static_cast<amrex::Real>(k) + 0.5) * spacing[2];
   }
   return fixed_spacetime_geometry(
      parameters, position1, position2, position3, time
   );
}

template <int Direction, int Dimension>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FixedSpacetimeGeometry fixed_spacetime_geometry_at_face(
   const FixedSpacetimeParameters& parameters,
   const amrex::GpuArray<amrex::Real, Dimension>& lower,
   const amrex::GpuArray<amrex::Real, Dimension>& spacing,
   int i,
   int j,
   int k,
   amrex::Real time
) noexcept
{
   static_assert(Direction >= 0 && Direction < Dimension);
   const amrex::Real position1 = lower[0]
      + (static_cast<amrex::Real>(i) + (Direction == 0 ? 0.0 : 0.5)) * spacing[0];
   amrex::Real position2 = 0.0;
   amrex::Real position3 = 0.0;
   if constexpr (Dimension >= 2) {
      position2 = lower[1]
         + (static_cast<amrex::Real>(j) + (Direction == 1 ? 0.0 : 0.5)) * spacing[1];
   }
   if constexpr (Dimension >= 3) {
      position3 = lower[2]
         + (static_cast<amrex::Real>(k) + (Direction == 2 ? 0.0 : 0.5)) * spacing[2];
   }
   return fixed_spacetime_geometry(
      parameters, position1, position2, position3, time
   );
}

} // namespace pyfluid_amrex
