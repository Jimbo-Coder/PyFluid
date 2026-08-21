#pragma once

#include <algorithm>
#include <limits>

#include "pyfluid_amrex/fixed_spacetime.hpp"

namespace pyfluid_amrex {

// Exact ingoing Cartesian Kerr-Schild 3+1 data. This is an initial-data
// materialization kernel, not a runtime flux-kernel branch: nontrivial fixed
// metrics use the same stored Z4c/cell-metric path as evolved spacetime.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FixedSpacetimeGeometry exact_kerr_schild_initial_data_geometry(
   const FixedSpacetimeParameters& parameters,
   amrex::Real position1,
   amrex::Real position2,
   amrex::Real position3
) noexcept
{
   FixedSpacetimeGeometry geometry{};
   const amrex::Real displacement[3]{
      position1 - parameters.origin1,
      position2 - parameters.origin2,
      position3 - parameters.origin3,
   };
   const amrex::Real spin[3]{
      parameters.mass * parameters.dimensionless_spin1,
      parameters.mass * parameters.dimensionless_spin2,
      parameters.mass * parameters.dimensionless_spin3,
   };
   const amrex::Real spin_squared = spin[0] * spin[0]
      + spin[1] * spin[1] + spin[2] * spin[2];
   const amrex::Real euclidean_radius_squared =
      displacement[0] * displacement[0]
      + displacement[1] * displacement[1]
      + displacement[2] * displacement[2];
   const amrex::Real spin_dot_displacement =
      spin[0] * displacement[0]
      + spin[1] * displacement[1]
      + spin[2] * displacement[2];
   const amrex::Real radial_quadratic = euclidean_radius_squared - spin_squared;
   const amrex::Real radial_discriminant = std::sqrt(
      radial_quadratic * radial_quadratic
      + 4.0 * spin_dot_displacement * spin_dot_displacement
   );
   const amrex::Real radius_squared = 0.5 * (
      radial_quadratic + radial_discriminant
   );
   const amrex::Real radius = std::sqrt(radius_squared);
   const amrex::Real inverse_radius = 1.0 / radius;
   amrex::Real radius_derivative[3]{};
   for (int direction = 0; direction < 3; ++direction) {
      const amrex::Real radius_squared_derivative = displacement[direction]
         + (
              radial_quadratic * displacement[direction]
              + 2.0 * spin_dot_displacement * spin[direction]
           ) / radial_discriminant;
      radius_derivative[direction] = 0.5 * radius_squared_derivative
         * inverse_radius;
   }

   const amrex::Real spin_cross_displacement[3]{
      spin[1] * displacement[2] - spin[2] * displacement[1],
      spin[2] * displacement[0] - spin[0] * displacement[2],
      spin[0] * displacement[1] - spin[1] * displacement[0],
   };
   const amrex::Real denominator = radius_squared + spin_squared;
   amrex::Real null_spatial[3]{};
   amrex::Real null_derivative[3][3]{};
   const amrex::Real minus_cross_derivative[3][3]{
      {0.0, -spin[2], spin[1]},
      {spin[2], 0.0, -spin[0]},
      {-spin[1], spin[0], 0.0},
   };
   amrex::Real null_numerator[3]{};
   for (int component = 0; component < 3; ++component) {
      null_numerator[component] = radius * displacement[component]
         - spin_cross_displacement[component]
         + spin_dot_displacement * spin[component] * inverse_radius;
      null_spatial[component] = null_numerator[component] / denominator;
   }
   for (int direction = 0; direction < 3; ++direction) {
      const amrex::Real denominator_derivative =
         2.0 * radius * radius_derivative[direction];
      for (int component = 0; component < 3; ++component) {
         const amrex::Real numerator_derivative =
            radius_derivative[direction] * displacement[component]
            + (component == direction ? radius : 0.0)
            + minus_cross_derivative[direction][component]
            + spin[component] * (
                 spin[direction] * inverse_radius
                 - spin_dot_displacement * radius_derivative[direction]
                    * inverse_radius * inverse_radius
              );
         null_derivative[direction][component] = (
            numerator_derivative * denominator
            - null_numerator[component] * denominator_derivative
         ) / (denominator * denominator);
      }
   }

   const amrex::Real radius_fourth = radius_squared * radius_squared;
   const amrex::Real profile_denominator = radius_fourth
      + spin_dot_displacement * spin_dot_displacement;
   const amrex::Real profile = parameters.mass * radius * radius_squared
      / profile_denominator;
   amrex::Real profile_derivative[3]{};
   for (int direction = 0; direction < 3; ++direction) {
      const amrex::Real profile_denominator_derivative =
         4.0 * radius * radius_squared * radius_derivative[direction]
         + 2.0 * spin_dot_displacement * spin[direction];
      profile_derivative[direction] = parameters.mass * (
         3.0 * radius_squared * radius_derivative[direction]
            * profile_denominator
         - radius * radius_squared * profile_denominator_derivative
      ) / (profile_denominator * profile_denominator);
   }

   const amrex::Real lapse_denominator = 1.0 + 2.0 * profile;
   const amrex::Real lapse = 1.0 / std::sqrt(lapse_denominator);
   const amrex::Real shift_scale = 2.0 * profile / lapse_denominator;
   amrex::Real spatial_metric[3][3]{};
   amrex::Real inverse_spatial_metric[3][3]{};
   amrex::Real spatial_metric_derivative[3][3][3]{};
   amrex::Real shift_derivative[3][3]{};
   amrex::Real covariant_shift[3]{};
   amrex::Real covariant_shift_derivative[3][3]{};
   for (int first = 0; first < 3; ++first) {
      covariant_shift[first] = 2.0 * profile * null_spatial[first];
      for (int second = 0; second < 3; ++second) {
         const amrex::Real identity = first == second ? 1.0 : 0.0;
         spatial_metric[first][second] = identity
            + 2.0 * profile * null_spatial[first] * null_spatial[second];
         inverse_spatial_metric[first][second] = identity
            - shift_scale * null_spatial[first] * null_spatial[second];
      }
   }
   for (int direction = 0; direction < 3; ++direction) {
      const amrex::Real shift_scale_derivative =
         2.0 * profile_derivative[direction]
         / (lapse_denominator * lapse_denominator);
      for (int component = 0; component < 3; ++component) {
         shift_derivative[direction][component] =
            shift_scale_derivative * null_spatial[component]
            + shift_scale * null_derivative[direction][component];
         covariant_shift_derivative[direction][component] =
            2.0 * profile_derivative[direction] * null_spatial[component]
            + 2.0 * profile * null_derivative[direction][component];
         for (int second = 0; second < 3; ++second) {
            spatial_metric_derivative[direction][component][second] =
               2.0 * profile_derivative[direction]
                  * null_spatial[component] * null_spatial[second]
               + 2.0 * profile * (
                    null_derivative[direction][component]
                       * null_spatial[second]
                    + null_spatial[component]
                       * null_derivative[direction][second]
                 );
         }
      }
   }

   amrex::Real extrinsic_curvature[3][3]{};
   for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
         amrex::Real connection_shift = 0.0;
         for (int upper = 0; upper < 3; ++upper) {
            amrex::Real connection = 0.0;
            for (int contracted = 0; contracted < 3; ++contracted) {
               connection += 0.5
                  * inverse_spatial_metric[upper][contracted]
                  * (
                       spatial_metric_derivative[first][contracted][second]
                       + spatial_metric_derivative[second][contracted][first]
                       - spatial_metric_derivative[contracted][first][second]
                    );
            }
            connection_shift += connection * covariant_shift[upper];
         }
         extrinsic_curvature[first][second] = 0.5 / lapse * (
            covariant_shift_derivative[first][second]
            + covariant_shift_derivative[second][first]
            - 2.0 * connection_shift
         );
      }
   }

   geometry.metric.lapse = lapse;
   geometry.metric.shift1 = shift_scale * null_spatial[0];
   geometry.metric.shift2 = shift_scale * null_spatial[1];
   geometry.metric.shift3 = shift_scale * null_spatial[2];
   geometry.metric.gamma11 = spatial_metric[0][0];
   geometry.metric.gamma12 = spatial_metric[0][1];
   geometry.metric.gamma13 = spatial_metric[0][2];
   geometry.metric.gamma22 = spatial_metric[1][1];
   geometry.metric.gamma23 = spatial_metric[1][2];
   geometry.metric.gamma33 = spatial_metric[2][2];
   geometry.metric.inv_gamma11 = inverse_spatial_metric[0][0];
   geometry.metric.inv_gamma12 = inverse_spatial_metric[0][1];
   geometry.metric.inv_gamma13 = inverse_spatial_metric[0][2];
   geometry.metric.inv_gamma22 = inverse_spatial_metric[1][1];
   geometry.metric.inv_gamma23 = inverse_spatial_metric[1][2];
   geometry.metric.inv_gamma33 = inverse_spatial_metric[2][2];
   geometry.metric.sqrt_det_gamma = std::sqrt(lapse_denominator);
   geometry.derivatives.d1_lapse = -profile_derivative[0]
      * lapse * lapse * lapse;
   geometry.derivatives.d2_lapse = -profile_derivative[1]
      * lapse * lapse * lapse;
   geometry.derivatives.d3_lapse = -profile_derivative[2]
      * lapse * lapse * lapse;
   geometry.derivatives.d1_shift1 = shift_derivative[0][0];
   geometry.derivatives.d2_shift1 = shift_derivative[1][0];
   geometry.derivatives.d3_shift1 = shift_derivative[2][0];
   geometry.derivatives.d1_shift2 = shift_derivative[0][1];
   geometry.derivatives.d2_shift2 = shift_derivative[1][1];
   geometry.derivatives.d3_shift2 = shift_derivative[2][1];
   geometry.derivatives.d1_shift3 = shift_derivative[0][2];
   geometry.derivatives.d2_shift3 = shift_derivative[1][2];
   geometry.derivatives.d3_shift3 = shift_derivative[2][2];
   geometry.derivatives.d1_gamma11 = spatial_metric_derivative[0][0][0];
   geometry.derivatives.d1_gamma12 = spatial_metric_derivative[0][0][1];
   geometry.derivatives.d1_gamma13 = spatial_metric_derivative[0][0][2];
   geometry.derivatives.d1_gamma22 = spatial_metric_derivative[0][1][1];
   geometry.derivatives.d1_gamma23 = spatial_metric_derivative[0][1][2];
   geometry.derivatives.d1_gamma33 = spatial_metric_derivative[0][2][2];
   geometry.derivatives.d2_gamma11 = spatial_metric_derivative[1][0][0];
   geometry.derivatives.d2_gamma12 = spatial_metric_derivative[1][0][1];
   geometry.derivatives.d2_gamma13 = spatial_metric_derivative[1][0][2];
   geometry.derivatives.d2_gamma22 = spatial_metric_derivative[1][1][1];
   geometry.derivatives.d2_gamma23 = spatial_metric_derivative[1][1][2];
   geometry.derivatives.d2_gamma33 = spatial_metric_derivative[1][2][2];
   geometry.derivatives.d3_gamma11 = spatial_metric_derivative[2][0][0];
   geometry.derivatives.d3_gamma12 = spatial_metric_derivative[2][0][1];
   geometry.derivatives.d3_gamma13 = spatial_metric_derivative[2][0][2];
   geometry.derivatives.d3_gamma22 = spatial_metric_derivative[2][1][1];
   geometry.derivatives.d3_gamma23 = spatial_metric_derivative[2][1][2];
   geometry.derivatives.d3_gamma33 = spatial_metric_derivative[2][2][2];
   geometry.derivatives.extrinsic11 = extrinsic_curvature[0][0];
   geometry.derivatives.extrinsic12 = extrinsic_curvature[0][1];
   geometry.derivatives.extrinsic13 = extrinsic_curvature[0][2];
   geometry.derivatives.extrinsic22 = extrinsic_curvature[1][1];
   geometry.derivatives.extrinsic23 = extrinsic_curvature[1][2];
   geometry.derivatives.extrinsic33 = extrinsic_curvature[2][2];
   return geometry;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FixedSpacetimeGeometry kerr_schild_initial_data_geometry(
   const FixedSpacetimeParameters& parameters,
   amrex::Real position1,
   amrex::Real position2,
   amrex::Real position3
) noexcept
{
   const amrex::Real displacement1 = position1 - parameters.origin1;
   const amrex::Real displacement2 = position2 - parameters.origin2;
   const amrex::Real displacement3 = position3 - parameters.origin3;
   const amrex::Real euclidean_radius = std::sqrt(
      displacement1 * displacement1
      + displacement2 * displacement2
      + displacement3 * displacement3
   );
   if (parameters.black_hole_interior != BlackHoleInterior::Filled
       || euclidean_radius >= parameters.black_hole_fill_radius) {
      return exact_kerr_schild_initial_data_geometry(
         parameters, position1, position2, position3
      );
   }

   FixedSpacetimeGeometry geometry{};
   const amrex::Real spin_length = parameters.mass * std::sqrt(
      parameters.dimensionless_spin1 * parameters.dimensionless_spin1
      + parameters.dimensionless_spin2 * parameters.dimensionless_spin2
      + parameters.dimensionless_spin3 * parameters.dimensionless_spin3
   );
   const amrex::Real safe_radius = std::max(
      0.25 * parameters.black_hole_fill_radius,
      spin_length + 256.0 * std::numeric_limits<amrex::Real>::epsilon()
         * parameters.mass
   );
   if (euclidean_radius <= safe_radius) {
      return geometry;
   }
   const amrex::Real transition = (
      euclidean_radius - safe_radius
   ) / (parameters.black_hole_fill_radius - safe_radius);
   const amrex::Real transition_squared = transition * transition;
   const amrex::Real weight = transition_squared * transition
      * (10.0 + transition * (-15.0 + 6.0 * transition));
   const FixedSpacetimeGeometry exact =
      exact_kerr_schild_initial_data_geometry(
         parameters, position1, position2, position3
      );
   geometry.metric.lapse = 1.0 + weight * (exact.metric.lapse - 1.0);
   geometry.metric.shift1 = weight * exact.metric.shift1;
   geometry.metric.shift2 = weight * exact.metric.shift2;
   geometry.metric.shift3 = weight * exact.metric.shift3;
   geometry.metric.gamma11 = 1.0 + weight * (exact.metric.gamma11 - 1.0);
   geometry.metric.gamma12 = weight * exact.metric.gamma12;
   geometry.metric.gamma13 = weight * exact.metric.gamma13;
   geometry.metric.gamma22 = 1.0 + weight * (exact.metric.gamma22 - 1.0);
   geometry.metric.gamma23 = weight * exact.metric.gamma23;
   geometry.metric.gamma33 = 1.0 + weight * (exact.metric.gamma33 - 1.0);

   const amrex::Real determinant =
      geometry.metric.gamma11 * (
         geometry.metric.gamma22 * geometry.metric.gamma33
         - geometry.metric.gamma23 * geometry.metric.gamma23
      )
      - geometry.metric.gamma12 * (
         geometry.metric.gamma12 * geometry.metric.gamma33
         - geometry.metric.gamma13 * geometry.metric.gamma23
      )
      + geometry.metric.gamma13 * (
         geometry.metric.gamma12 * geometry.metric.gamma23
         - geometry.metric.gamma13 * geometry.metric.gamma22
      );
   const amrex::Real inverse_determinant = 1.0 / determinant;
   geometry.metric.inv_gamma11 = (
      geometry.metric.gamma22 * geometry.metric.gamma33
      - geometry.metric.gamma23 * geometry.metric.gamma23
   ) * inverse_determinant;
   geometry.metric.inv_gamma12 = (
      geometry.metric.gamma13 * geometry.metric.gamma23
      - geometry.metric.gamma12 * geometry.metric.gamma33
   ) * inverse_determinant;
   geometry.metric.inv_gamma13 = (
      geometry.metric.gamma12 * geometry.metric.gamma23
      - geometry.metric.gamma13 * geometry.metric.gamma22
   ) * inverse_determinant;
   geometry.metric.inv_gamma22 = (
      geometry.metric.gamma11 * geometry.metric.gamma33
      - geometry.metric.gamma13 * geometry.metric.gamma13
   ) * inverse_determinant;
   geometry.metric.inv_gamma23 = (
      geometry.metric.gamma12 * geometry.metric.gamma13
      - geometry.metric.gamma11 * geometry.metric.gamma23
   ) * inverse_determinant;
   geometry.metric.inv_gamma33 = (
      geometry.metric.gamma11 * geometry.metric.gamma22
      - geometry.metric.gamma12 * geometry.metric.gamma12
   ) * inverse_determinant;
   geometry.metric.sqrt_det_gamma = std::sqrt(determinant);
   geometry.derivatives.extrinsic11 =
      weight * exact.derivatives.extrinsic11;
   geometry.derivatives.extrinsic12 =
      weight * exact.derivatives.extrinsic12;
   geometry.derivatives.extrinsic13 =
      weight * exact.derivatives.extrinsic13;
   geometry.derivatives.extrinsic22 =
      weight * exact.derivatives.extrinsic22;
   geometry.derivatives.extrinsic23 =
      weight * exact.derivatives.extrinsic23;
   geometry.derivatives.extrinsic33 =
      weight * exact.derivatives.extrinsic33;
   return geometry;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FixedSpacetimeGeometry initial_data_spacetime_geometry(
   const FixedSpacetimeParameters& parameters,
   amrex::Real position1,
   amrex::Real position2,
   amrex::Real position3,
   amrex::Real time
) noexcept
{
   if (parameters.metric == FixedSpacetimeMetric::KerrSchild) {
      return kerr_schild_initial_data_geometry(
         parameters, position1, position2, position3
      );
   }
   return fixed_spacetime_geometry(
      parameters, position1, position2, position3, time
   );
}

} // namespace pyfluid_amrex
