#pragma once

#include "pyfluid_amrex/reference_chart.hpp"
#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/fccz4_point.hpp"
#include "pyfluid_amrex/kernels_generated/spacetime_geometry.hpp"
#include "pyfluid_amrex/spacetime_fields.hpp"
#include "pyfluid_amrex/z4c_state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int symmetric_tensor_component(int first, int second) noexcept
{
   if (first > second) {
      const int temporary = first;
      first = second;
      second = temporary;
   }
   return first == 0 ? second : (first == 1 ? second + 2 : 5);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
EulerianMatter canonical_matter(
   EulerianMatter const& source,
   ReferenceChartPoint const& chart
) noexcept
{
   const amrex::Real momentum[3]{source.momentum1, source.momentum2, source.momentum3};
   const amrex::Real stress[6]{source.stress11, source.stress12, source.stress13,
                             source.stress22, source.stress23, source.stress33};
   amrex::Real canonical_momentum[3]{};
   amrex::Real canonical_stress[6]{};
   for (int i = 0; i < 3; ++i) {
      canonical_momentum[chart.canonical[i]] = momentum[i];
      for (int j = i; j < 3; ++j) {
         canonical_stress[symmetric_tensor_component(chart.canonical[i], chart.canonical[j])]
            = stress[symmetric_tensor_component(i, j)];
      }
   }
   return {source.rho, canonical_momentum[0], canonical_momentum[1], canonical_momentum[2],
           canonical_stress[0], canonical_stress[1], canonical_stress[2],
           canonical_stress[3], canonical_stress[4], canonical_stress[5]};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ReferenceScalarJet reference_scalar_jet(
   ReferenceChartPoint const& chart,
   ReferenceScalarJet const& input
) noexcept
{
   const auto q = chart.position;
   if (chart.coordinate == MeshSpherical) {
      return generated::spherical_reference_scalar_jet(q[0], q[1], q[2], input);
   }
   if (chart.coordinate == MeshCylindrical) {
      return generated::cylindrical_reference_scalar_jet(q[0], q[1], q[2], input);
   }
   return generated::cartesian_reference_scalar_jet(q[0], q[1], q[2], input);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ReferenceVectorJet reference_vector_jet(
   ReferenceChartPoint const& chart,
   ReferenceVectorJet const& input
) noexcept
{
   const auto q = chart.position;
   if (chart.coordinate == MeshSpherical) {
      return generated::spherical_reference_vector_jet(q[0], q[1], q[2], input);
   }
   if (chart.coordinate == MeshCylindrical) {
      return generated::cylindrical_reference_vector_jet(q[0], q[1], q[2], input);
   }
   return generated::cartesian_reference_vector_jet(q[0], q[1], q[2], input);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SymmetricTensorJet reference_tensor_jet(
   ReferenceChartPoint const& chart,
   SymmetricTensorJet const& input
) noexcept
{
   const auto q = chart.position;
   if (chart.coordinate == MeshSpherical) {
      return generated::spherical_reference_metric_jet(q[0], q[1], q[2], input);
   }
   if (chart.coordinate == MeshCylindrical) {
      return generated::cylindrical_reference_metric_jet(q[0], q[1], q[2], input);
   }
   return generated::cartesian_reference_metric_jet(q[0], q[1], q[2], input);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ReferenceScalarJet gather_reference_scalar(
   ReferenceChartPoint const& chart, amrex::Real value, amrex::Real const* first,
   amrex::Real const (*second)[3] = nullptr
) noexcept
{
   ReferenceScalarJet result{};
   result.value = value;
   for (int d = 0; d < 3; ++d) {
      result.first[chart.canonical[d]] = first[d];
      for (int e = 0; e < 3; ++e) {
         result.second[chart.canonical[d]][chart.canonical[e]] =
            second ? second[d][e] : 0.0_rt;
      }
   }
   return reference_scalar_jet(chart, result);
}

// Shared conformal storage layout: K_hat is explicitly decoded to physical K.
// The connection buffer stores Lambda on fCCZ4, with its formulation/basis
// protected by restart metadata. All tensor and derivative indices are mapped.
template <typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Fccz4Point gather_fccz4_point(
   Array const& values,
   int i, int j, int k,
   Z4cSpatialDerivatives const& derivatives,
   ReferenceChartPoint const& chart
) noexcept
{
   Fccz4Point point{};
   amrex::Real d_k[3]{};
   for (int d = 0; d < 3; ++d) {
      d_k[d] = derivatives.d_k_hat[d] + 2.0_rt * derivatives.d_theta[d];
   }
   point.chi = gather_reference_scalar(
      chart, values(i, j, k, Z4C_CHI), derivatives.d_chi, derivatives.dd_chi
   );
   point.lapse = gather_reference_scalar(
      chart, values(i, j, k, Z4C_LAPSE), derivatives.d_lapse, derivatives.dd_lapse
   );
   point.trace_k = gather_reference_scalar(
      chart, values(i, j, k, Z4C_K_HAT) + 2.0_rt * values(i, j, k, Z4C_THETA), d_k
   );
   point.theta = gather_reference_scalar(
      chart, values(i, j, k, Z4C_THETA), derivatives.d_theta
   );
   for (int a = 0; a < 3; ++a) {
      const int ca = chart.canonical[a];
      point.shift.value[ca] = values(i, j, k, z4c_shift_components()[a]);
      point.connection.value[ca] = values(i, j, k, z4c_gamma_tilde_vector_components()[a]);
      for (int d = 0; d < 3; ++d) {
         const int cd = chart.canonical[d];
         point.shift.first[cd][ca] = derivatives.d_shift[d][a];
         point.connection.first[cd][ca] = derivatives.d_gamma_tilde_vector[d][a];
         for (int e = 0; e < 3; ++e) {
            point.shift.second[cd][chart.canonical[e]][ca] = derivatives.dd_shift[d][e][a];
         }
      }
      for (int b = a; b < 3; ++b) {
         const int component = symmetric_tensor_component(a, b);
         const int canonical_component = symmetric_tensor_component(
            ca, chart.canonical[b]
         );
         point.metric.value[canonical_component] =
            values(i, j, k, z4c_gamma_tilde_components()[component]);
         point.a_tilde.value[canonical_component] =
            values(i, j, k, z4c_a_tilde_components()[component]);
         for (int d = 0; d < 3; ++d) {
            const int cd = chart.canonical[d];
            point.metric.first[cd][canonical_component] = derivatives.d_gamma_tilde[d][component];
            point.a_tilde.first[cd][canonical_component] = derivatives.d_a_tilde[d][component];
            for (int e = 0; e < 3; ++e) {
               point.metric.second[cd][chart.canonical[e]][canonical_component] =
                  derivatives.dd_gamma_tilde[d][e][component];
            }
         }
      }
   }
   point.metric = reference_tensor_jet(chart, point.metric);
   point.a_tilde = reference_tensor_jet(chart, point.a_tilde);
   point.shift = reference_vector_jet(chart, point.shift);
   point.connection = reference_vector_jet(chart, point.connection);
   return point;
}

// Convert the derivative input of the shared conformal-to-physical metric
// algebra. Its time derivative still follows ADM; only spatial derivatives
// change from coordinate partials to reference-covariant unit components.
template <typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cSpatialDerivatives reference_spacetime_derivatives(
   Array const& values, int i, int j, int k, Z4cSpatialDerivatives const& partial,
   ReferenceChartPoint const& chart
) noexcept
{
   const auto point = gather_fccz4_point(values, i, j, k, partial, chart);
   auto result = partial;
   for (int d = 0; d < 3; ++d) {
      const int cd = chart.canonical[d];
      result.d_chi[d] = point.chi.first[cd];
      result.d_lapse[d] = point.lapse.first[cd];
      for (int a = 0; a < 3; ++a) {
         result.d_shift[d][a] = point.shift.first[cd][chart.canonical[a]];
         for (int b = a; b < 3; ++b) {
            result.d_gamma_tilde[d][symmetric_tensor_component(a, b)] =
               point.metric.first[cd][symmetric_tensor_component(
                  chart.canonical[a], chart.canonical[b]
               )];
         }
      }
   }
   return result;
}

} // namespace pyfluid_amrex
