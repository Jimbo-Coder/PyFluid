#pragma once

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <limits>

#include "pyfluid_amrex/fixed_spacetime_types.hpp"
#include "pyfluid_amrex/physical_geometry.hpp"
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
      const amrex::Real radius_squared = cartesian_radius_squared(
         displacement1, displacement2, displacement3
      );
      const amrex::Real radius = cartesian_radius(
         displacement1, displacement2, displacement3
      );
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
   } else if (parameters.metric == FixedSpacetimeMetric::StaticTrumpet) {
      // Exact stationary Schwarzschild trumpet of Dennison and Baumgarte
      // (2014), expressed directly in Cartesian ADM components. The puncture
      // itself is excluded from sampling by the mesh-alignment contract.
      const amrex::Real displacement[3]{
         position1 - parameters.origin1,
         position2 - parameters.origin2,
         position3 - parameters.origin3,
      };
      const amrex::Real radius_squared =
         displacement[0] * displacement[0]
         + displacement[1] * displacement[1]
         + displacement[2] * displacement[2];
      const amrex::Real radius = std::sqrt(radius_squared);
      const amrex::Real inverse_radius = 1.0 / radius;
      const amrex::Real inverse_radius_squared = 1.0 / radius_squared;
      const amrex::Real radius_plus_mass = radius + parameters.mass;
      const amrex::Real inverse_radius_plus_mass = 1.0 / radius_plus_mass;
      const amrex::Real conformal_squared = radius_plus_mass * inverse_radius;
      const amrex::Real conformal_fourth =
         conformal_squared * conformal_squared;
      const amrex::Real inverse_conformal_fourth = 1.0 / conformal_fourth;

      geometry.metric.lapse = radius * inverse_radius_plus_mass;
      geometry.metric.gamma11 = conformal_fourth;
      geometry.metric.gamma22 = conformal_fourth;
      geometry.metric.gamma33 = conformal_fourth;
      geometry.metric.inv_gamma11 = inverse_conformal_fourth;
      geometry.metric.inv_gamma22 = inverse_conformal_fourth;
      geometry.metric.inv_gamma33 = inverse_conformal_fourth;
      geometry.metric.sqrt_det_gamma =
         conformal_squared * conformal_squared * conformal_squared;

      const amrex::Real shift_scale = parameters.mass
         * inverse_radius_plus_mass * inverse_radius_plus_mass;
      geometry.metric.shift1 = shift_scale * displacement[0];
      geometry.metric.shift2 = shift_scale * displacement[1];
      geometry.metric.shift3 = shift_scale * displacement[2];

      const amrex::Real lapse_derivative_scale = parameters.mass
         * inverse_radius * inverse_radius_plus_mass
         * inverse_radius_plus_mass;
      geometry.derivatives.d1_lapse =
         lapse_derivative_scale * displacement[0];
      geometry.derivatives.d2_lapse =
         lapse_derivative_scale * displacement[1];
      geometry.derivatives.d3_lapse =
         lapse_derivative_scale * displacement[2];

      amrex::Real shift_derivative[3][3]{};
      const amrex::Real shift_radial_scale = -2.0 * parameters.mass
         * inverse_radius * inverse_radius_plus_mass
         * inverse_radius_plus_mass * inverse_radius_plus_mass;
      for (int derivative = 0; derivative < 3; ++derivative) {
         for (int component = 0; component < 3; ++component) {
            shift_derivative[derivative][component] =
               shift_radial_scale * displacement[derivative]
                  * displacement[component]
               + (derivative == component ? shift_scale : 0.0);
         }
      }
      geometry.derivatives.d1_shift1 = shift_derivative[0][0];
      geometry.derivatives.d2_shift1 = shift_derivative[1][0];
      geometry.derivatives.d3_shift1 = shift_derivative[2][0];
      geometry.derivatives.d1_shift2 = shift_derivative[0][1];
      geometry.derivatives.d2_shift2 = shift_derivative[1][1];
      geometry.derivatives.d3_shift2 = shift_derivative[2][1];
      geometry.derivatives.d1_shift3 = shift_derivative[0][2];
      geometry.derivatives.d2_shift3 = shift_derivative[1][2];
      geometry.derivatives.d3_shift3 = shift_derivative[2][2];

      const amrex::Real metric_derivative_scale = -2.0 * parameters.mass
         * radius_plus_mass * inverse_radius_squared * inverse_radius_squared;
      geometry.derivatives.d1_gamma11 =
         metric_derivative_scale * displacement[0];
      geometry.derivatives.d1_gamma22 =
         metric_derivative_scale * displacement[0];
      geometry.derivatives.d1_gamma33 =
         metric_derivative_scale * displacement[0];
      geometry.derivatives.d2_gamma11 =
         metric_derivative_scale * displacement[1];
      geometry.derivatives.d2_gamma22 =
         metric_derivative_scale * displacement[1];
      geometry.derivatives.d2_gamma33 =
         metric_derivative_scale * displacement[1];
      geometry.derivatives.d3_gamma11 =
         metric_derivative_scale * displacement[2];
      geometry.derivatives.d3_gamma22 =
         metric_derivative_scale * displacement[2];
      geometry.derivatives.d3_gamma33 =
         metric_derivative_scale * displacement[2];

      const amrex::Real extrinsic_scale = parameters.mass
         * inverse_radius_squared;
      const amrex::Real unit[3]{
         displacement[0] * inverse_radius,
         displacement[1] * inverse_radius,
         displacement[2] * inverse_radius,
      };
      geometry.derivatives.extrinsic11 = extrinsic_scale
         * (1.0 - 2.0 * unit[0] * unit[0]);
      geometry.derivatives.extrinsic12 = extrinsic_scale
         * (-2.0 * unit[0] * unit[1]);
      geometry.derivatives.extrinsic13 = extrinsic_scale
         * (-2.0 * unit[0] * unit[2]);
      geometry.derivatives.extrinsic22 = extrinsic_scale
         * (1.0 - 2.0 * unit[1] * unit[1]);
      geometry.derivatives.extrinsic23 = extrinsic_scale
         * (-2.0 * unit[1] * unit[2]);
      geometry.derivatives.extrinsic33 = extrinsic_scale
         * (1.0 - 2.0 * unit[2] * unit[2]);
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
#if (AMREX_SPACEDIM >= 2)
      const amrex::Real distance2 = interval_distance(parameters.origin2, lower[1], upper[1]);
#else
      const amrex::Real distance2 = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
      const amrex::Real distance3 = interval_distance(parameters.origin3, lower[2], upper[2]);
#else
      const amrex::Real distance3 = 0.0;
#endif
      if (!(parameters.mass > 0.0)
          || !(cartesian_radius(distance1, distance2, distance3)
               > 0.5 * parameters.mass)) {
         amrex::Abort(
            "the isotropic-Schwarzschild domain must remain outside r = M/2"
         );
      }
      return;
   }
   if (parameters.metric == FixedSpacetimeMetric::KerrSchild) {
      const amrex::Real spin_squared =
         parameters.dimensionless_spin1 * parameters.dimensionless_spin1
         + parameters.dimensionless_spin2 * parameters.dimensionless_spin2
         + parameters.dimensionless_spin3 * parameters.dimensionless_spin3;
      if (!(parameters.mass > 0.0) || !(spin_squared < 1.0)) {
         amrex::Abort(
            "Kerr-Schild fixed spacetime requires positive mass and "
            "dimensionless spin magnitude below one"
         );
      }
      if (parameters.black_hole_interior == BlackHoleInterior::Filled) {
         return;
      }
      const auto interval_distance = [](
         amrex::Real point, amrex::Real lo, amrex::Real hi
      ) {
         return point < lo ? lo - point : (point > hi ? point - hi : 0.0);
      };
      const amrex::Real distance1 = interval_distance(
         parameters.origin1, lower[0], upper[0]
      );
#if (AMREX_SPACEDIM >= 2)
      const amrex::Real distance2 = interval_distance(
         parameters.origin2, lower[1], upper[1]
      );
#else
      const amrex::Real distance2 = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
      const amrex::Real distance3 = interval_distance(
         parameters.origin3, lower[2], upper[2]
      );
#else
      const amrex::Real distance3 = 0.0;
#endif
      const amrex::Real minimum_euclidean_radius_squared =
         distance1 * distance1 + distance2 * distance2
         + distance3 * distance3;
      const amrex::Real spin_length_squared =
         parameters.mass * parameters.mass * spin_squared;
      const amrex::Real horizon_radius = parameters.mass * (
         1.0 + std::sqrt(1.0 - spin_squared)
      );
      if (!(minimum_euclidean_radius_squared - spin_length_squared
            > horizon_radius * horizon_radius)) {
         amrex::Abort(
            "the current Kerr-Schild fixed-spacetime contract requires the "
            "complete domain to remain outside the outer horizon"
         );
      }
      return;
   }
   if (parameters.metric == FixedSpacetimeMetric::StaticTrumpet) {
      if (!(parameters.mass > 0.0)
          || parameters.black_hole_interior != BlackHoleInterior::Puncture) {
         amrex::Abort(
            "static-trumpet spacetime requires positive mass and the "
            "puncture interior contract"
         );
      }
      const auto spacing = geometry.CellSizeArray();
      const amrex::Real origin[3]{
         parameters.origin1, parameters.origin2, parameters.origin3,
      };
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         const amrex::Real face_coordinate =
            (origin[direction] - lower[direction]) / spacing[direction];
         const amrex::Real nearest_face = std::round(face_coordinate);
         const amrex::Real tolerance = 128.0
            * std::numeric_limits<amrex::Real>::epsilon()
            * std::max<amrex::Real>(1.0, std::abs(face_coordinate));
         const amrex::Real coordinate_tolerance = tolerance
            * std::abs(spacing[direction]);
         if (origin[direction] < lower[direction] - coordinate_tolerance
             || origin[direction] > upper[direction] + coordinate_tolerance
             || std::abs(face_coordinate - nearest_face) > tolerance) {
            amrex::Abort(
               "the static-trumpet puncture must lie on a mesh face in "
               "every Cartesian direction"
            );
         }
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
