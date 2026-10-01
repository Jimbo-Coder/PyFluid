#pragma once

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

#include "pyfluid_amrex/coordinate_locations.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

// Unlike FixedSpacetimeGeometry, the derivative components here are
// reference-covariant derivatives in the local orthonormal mesh basis. This
// is the form needed by reference-metric Valencia source algebra.
struct ReferenceMetricSpacetimeGeometry {
   SpacetimeMetric metric{};
   SpacetimeMetricDerivatives reference_derivatives{};
};

struct ReferenceMetricOrthonormalFrame {
   // mesh_to_cartesian[mesh component][Cartesian component]
   amrex::Real mesh_to_cartesian[3][3]{};
};

inline void validate_reference_metric_fixed_spacetime_domain(
   FixedSpacetimeParameters const& parameters,
   amrex::Geometry const& geometry,
   int coordinate,
   MeshAxes const& axes
)
{
   if (coordinate == MeshCartesian) {
      validate_fixed_spacetime_domain(parameters, geometry);
      return;
   }
   if (parameters.metric == FixedSpacetimeMetric::Minkowski) {
      return;
   }
   if (coordinate != MeshSpherical
       || parameters.metric != FixedSpacetimeMetric::SchwarzschildIsotropic) {
      amrex::Abort(
         "the curved reference-metric fixed-spacetime domain validator "
         "currently supports centered isotropic Schwarzschild on spherical grids"
      );
   }
   if (parameters.origin1 != 0.0_rt || parameters.origin2 != 0.0_rt
       || parameters.origin3 != 0.0_rt) {
      amrex::Abort(
         "spherical isotropic-Schwarzschild reference-metric evolution "
         "requires the black hole at the coordinate origin"
      );
   }
   const int radial_direction = mesh_direction_for_axis(axes, MeshAxisR);
   if (radial_direction < 0) {
      amrex::Abort(
         "spherical isotropic-Schwarzschild reference-metric evolution "
         "requires an active radial axis"
      );
   }
   const auto lower = geometry.ProbLoArray();
   const auto upper = geometry.ProbHiArray();
   if (!(parameters.mass > 0.0_rt)
       || !(lower[radial_direction] > 0.5_rt * parameters.mass)
       || !(upper[radial_direction] > lower[radial_direction])) {
      amrex::Abort(
         "the spherical isotropic-Schwarzschild domain must be a radial "
         "shell strictly outside r = M/2"
      );
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ReferenceMetricOrthonormalFrame reference_metric_orthonormal_frame(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, 3> const& q
) noexcept
{
   ReferenceMetricOrthonormalFrame frame{};
   for (int mesh_component = 0; mesh_component < 3; ++mesh_component) {
      StateVector3 unit{};
      if (mesh_component == 0) {
         unit.x = 1.0_rt;
      } else if (mesh_component == 1) {
         unit.y = 1.0_rt;
      } else {
         unit.z = 1.0_rt;
      }
      const StateVector3 cartesian = mesh_basis_vector_to_cartesian(
         coordinate, axes, q, unit
      );
      frame.mesh_to_cartesian[mesh_component][0] = cartesian.x;
      frame.mesh_to_cartesian[mesh_component][1] = cartesian.y;
      frame.mesh_to_cartesian[mesh_component][2] = cartesian.z;
   }
   return frame;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ReferenceMetricSpacetimeGeometry reference_metric_spacetime_geometry_from_cartesian(
   FixedSpacetimeGeometry const& cartesian_geometry,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, 3> const& q
) noexcept
{
   const ReferenceMetricOrthonormalFrame frame =
      reference_metric_orthonormal_frame(coordinate, axes, q);
   const auto& cartesian_metric = cartesian_geometry.metric;
   const auto& cartesian_derivatives = cartesian_geometry.derivatives;

   const amrex::Real shift_cartesian[3]{
      cartesian_metric.shift1,
      cartesian_metric.shift2,
      cartesian_metric.shift3,
   };
   const amrex::Real gamma_cartesian[3][3]{
      {cartesian_metric.gamma11, cartesian_metric.gamma12, cartesian_metric.gamma13},
      {cartesian_metric.gamma12, cartesian_metric.gamma22, cartesian_metric.gamma23},
      {cartesian_metric.gamma13, cartesian_metric.gamma23, cartesian_metric.gamma33},
   };
   const amrex::Real inverse_gamma_cartesian[3][3]{
      {
         cartesian_metric.inv_gamma11,
         cartesian_metric.inv_gamma12,
         cartesian_metric.inv_gamma13,
      },
      {
         cartesian_metric.inv_gamma12,
         cartesian_metric.inv_gamma22,
         cartesian_metric.inv_gamma23,
      },
      {
         cartesian_metric.inv_gamma13,
         cartesian_metric.inv_gamma23,
         cartesian_metric.inv_gamma33,
      },
   };
   const amrex::Real lapse_gradient_cartesian[3]{
      cartesian_derivatives.d1_lapse,
      cartesian_derivatives.d2_lapse,
      cartesian_derivatives.d3_lapse,
   };
   // First index is the derivative direction and the second is the vector
   // component: d_shift_cartesian[i][j] = partial_i beta^j.
   const amrex::Real shift_derivative_cartesian[3][3]{
      {
         cartesian_derivatives.d1_shift1,
         cartesian_derivatives.d1_shift2,
         cartesian_derivatives.d1_shift3,
      },
      {
         cartesian_derivatives.d2_shift1,
         cartesian_derivatives.d2_shift2,
         cartesian_derivatives.d2_shift3,
      },
      {
         cartesian_derivatives.d3_shift1,
         cartesian_derivatives.d3_shift2,
         cartesian_derivatives.d3_shift3,
      },
   };
   const amrex::Real gamma_derivative_cartesian[3][3][3]{
      {
         {
            cartesian_derivatives.d1_gamma11,
            cartesian_derivatives.d1_gamma12,
            cartesian_derivatives.d1_gamma13,
         },
         {
            cartesian_derivatives.d1_gamma12,
            cartesian_derivatives.d1_gamma22,
            cartesian_derivatives.d1_gamma23,
         },
         {
            cartesian_derivatives.d1_gamma13,
            cartesian_derivatives.d1_gamma23,
            cartesian_derivatives.d1_gamma33,
         },
      },
      {
         {
            cartesian_derivatives.d2_gamma11,
            cartesian_derivatives.d2_gamma12,
            cartesian_derivatives.d2_gamma13,
         },
         {
            cartesian_derivatives.d2_gamma12,
            cartesian_derivatives.d2_gamma22,
            cartesian_derivatives.d2_gamma23,
         },
         {
            cartesian_derivatives.d2_gamma13,
            cartesian_derivatives.d2_gamma23,
            cartesian_derivatives.d2_gamma33,
         },
      },
      {
         {
            cartesian_derivatives.d3_gamma11,
            cartesian_derivatives.d3_gamma12,
            cartesian_derivatives.d3_gamma13,
         },
         {
            cartesian_derivatives.d3_gamma12,
            cartesian_derivatives.d3_gamma22,
            cartesian_derivatives.d3_gamma23,
         },
         {
            cartesian_derivatives.d3_gamma13,
            cartesian_derivatives.d3_gamma23,
            cartesian_derivatives.d3_gamma33,
         },
      },
   };
   const amrex::Real extrinsic_cartesian[3][3]{
      {
         cartesian_derivatives.extrinsic11,
         cartesian_derivatives.extrinsic12,
         cartesian_derivatives.extrinsic13,
      },
      {
         cartesian_derivatives.extrinsic12,
         cartesian_derivatives.extrinsic22,
         cartesian_derivatives.extrinsic23,
      },
      {
         cartesian_derivatives.extrinsic13,
         cartesian_derivatives.extrinsic23,
         cartesian_derivatives.extrinsic33,
      },
   };

   amrex::Real shift[3]{};
   amrex::Real gamma[3][3]{};
   amrex::Real inverse_gamma[3][3]{};
   amrex::Real lapse_gradient[3]{};
   amrex::Real shift_derivative[3][3]{};
   amrex::Real gamma_derivative[3][3][3]{};
   amrex::Real extrinsic[3][3]{};
   for (int a = 0; a < 3; ++a) {
      for (int i = 0; i < 3; ++i) {
         const amrex::Real e_ai = frame.mesh_to_cartesian[a][i];
         shift[a] += e_ai * shift_cartesian[i];
         lapse_gradient[a] += e_ai * lapse_gradient_cartesian[i];
         for (int b = 0; b < 3; ++b) {
            for (int j = 0; j < 3; ++j) {
               const amrex::Real e_bj = frame.mesh_to_cartesian[b][j];
               gamma[a][b] += e_ai * e_bj * gamma_cartesian[i][j];
               inverse_gamma[a][b] +=
                  e_ai * e_bj * inverse_gamma_cartesian[i][j];
               extrinsic[a][b] +=
                  e_ai * e_bj * extrinsic_cartesian[i][j];
               shift_derivative[a][b] +=
                  e_ai * e_bj * shift_derivative_cartesian[i][j];
               for (int c = 0; c < 3; ++c) {
                  for (int k = 0; k < 3; ++k) {
                     gamma_derivative[a][b][c] +=
                        e_ai * e_bj * frame.mesh_to_cartesian[c][k]
                        * gamma_derivative_cartesian[i][j][k];
                  }
               }
            }
         }
      }
   }

   ReferenceMetricSpacetimeGeometry geometry{};
   geometry.metric = {
      cartesian_metric.lapse,
      shift[0],
      shift[1],
      shift[2],
      gamma[0][0],
      gamma[0][1],
      gamma[0][2],
      gamma[1][1],
      gamma[1][2],
      gamma[2][2],
      inverse_gamma[0][0],
      inverse_gamma[0][1],
      inverse_gamma[0][2],
      inverse_gamma[1][1],
      inverse_gamma[1][2],
      inverse_gamma[2][2],
      cartesian_metric.sqrt_det_gamma,
   };
   geometry.reference_derivatives = {
      lapse_gradient[0],
      lapse_gradient[1],
      lapse_gradient[2],
      shift_derivative[0][0],
      shift_derivative[1][0],
      shift_derivative[2][0],
      shift_derivative[0][1],
      shift_derivative[1][1],
      shift_derivative[2][1],
      shift_derivative[0][2],
      shift_derivative[1][2],
      shift_derivative[2][2],
      gamma_derivative[0][0][0],
      gamma_derivative[0][0][1],
      gamma_derivative[0][0][2],
      gamma_derivative[0][1][1],
      gamma_derivative[0][1][2],
      gamma_derivative[0][2][2],
      gamma_derivative[1][0][0],
      gamma_derivative[1][0][1],
      gamma_derivative[1][0][2],
      gamma_derivative[1][1][1],
      gamma_derivative[1][1][2],
      gamma_derivative[1][2][2],
      gamma_derivative[2][0][0],
      gamma_derivative[2][0][1],
      gamma_derivative[2][0][2],
      gamma_derivative[2][1][1],
      gamma_derivative[2][1][2],
      gamma_derivative[2][2][2],
      extrinsic[0][0],
      extrinsic[0][1],
      extrinsic[0][2],
      extrinsic[1][1],
      extrinsic[1][2],
      extrinsic[2][2],
   };
   return geometry;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ReferenceMetricSpacetimeGeometry reference_metric_fixed_spacetime_geometry(
   FixedSpacetimeParameters const& parameters,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, 3> const& q,
   amrex::Real time
) noexcept
{
   const auto cartesian_position = mesh_coordinate_triplet_to_cartesian(
      coordinate, q
   );
   const FixedSpacetimeGeometry cartesian_geometry = fixed_spacetime_geometry(
      parameters,
      cartesian_position[0],
      cartesian_position[1],
      cartesian_position[2],
      time
   );
   return reference_metric_spacetime_geometry_from_cartesian(
      cartesian_geometry, coordinate, axes, q
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ReferenceMetricSpacetimeGeometry reference_metric_fixed_spacetime_geometry_at_cell(
   FixedSpacetimeParameters const& parameters,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& lower,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& spacing,
   int i,
   int j,
   int k,
   amrex::Real time
) noexcept
{
   const auto q = mesh_coordinate_triplet_at_location(
      coordinate, axes, lower, spacing, i, j, k, false, false, false
   );
   return reference_metric_fixed_spacetime_geometry(
      parameters, coordinate, axes, q, time
   );
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ReferenceMetricSpacetimeGeometry reference_metric_fixed_spacetime_geometry_at_face(
   FixedSpacetimeParameters const& parameters,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& lower,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& spacing,
   int i,
   int j,
   int k,
   amrex::Real time
) noexcept
{
   static_assert(Direction >= 0 && Direction < AMREX_SPACEDIM);
   const auto q = mesh_coordinate_triplet_at_location(
      coordinate,
      axes,
      lower,
      spacing,
      i,
      j,
      k,
      Direction == 0,
      Direction == 1,
      Direction == 2
   );
   return reference_metric_fixed_spacetime_geometry(
      parameters, coordinate, axes, q, time
   );
}

} // namespace pyfluid_amrex
