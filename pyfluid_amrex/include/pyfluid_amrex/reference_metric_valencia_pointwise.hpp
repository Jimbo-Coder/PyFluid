#pragma once

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/kernels_generated/relativistic/flux.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/reference_metric.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real reference_metric_runtime_vector_component_for_axis(
   int coordinate,
   MeshAxes const& axes,
   int target_axis,
   amrex::Real component1,
   amrex::Real component2,
   amrex::Real component3
) noexcept
{
   const amrex::Real values[3] = {component1, component2, component3};
   for (int component = 0; component < 3; ++component) {
      if (mesh_vector_axis_for_component(coordinate, axes, component) == target_axis) {
         return values[component];
      }
   }
   return 0.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDPrimitive rhd_primitive_in_canonical_reference_basis(
   RHDPrimitive const& primitive,
   int coordinate,
   MeshAxes const& axes
) noexcept
{
   const auto canonical_axes = mesh_canonical_vector_axes(coordinate);
   return {
      primitive.rho,
      primitive.eps,
      primitive.pressure,
      reference_metric_runtime_vector_component_for_axis(
         coordinate, axes, canonical_axes[0], primitive.v1, primitive.v2, primitive.v3
      ),
      reference_metric_runtime_vector_component_for_axis(
         coordinate, axes, canonical_axes[1], primitive.v1, primitive.v2, primitive.v3
      ),
      reference_metric_runtime_vector_component_for_axis(
         coordinate, axes, canonical_axes[2], primitive.v1, primitive.v2, primitive.v3
      ),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 9> rhd_reference_metric_nonpressure_mixed_flux(
   RHDPrimitive const& primitive,
   SpacetimeMetric const& metric,
   int coordinate,
   MeshAxes const& axes
) noexcept
{
   // The generated Valencia algebra owns the physics in runtime component
   // order. Reorder only its mixed momentum-flux tensor into the canonical
   // reference basis expected by the generated connection correction. This
   // keeps lower-dimensional axis subsets from silently changing tensor
   // meaning.
   const RHDConserved runtime_flux[3]{
      generated::rhd_flux<0>(primitive, metric),
      generated::rhd_flux<1>(primitive, metric),
      generated::rhd_flux<2>(primitive, metric),
   };
   amrex::GpuArray<amrex::Real, 9> flux_component_normal{};
   for (int runtime_component = 0; runtime_component < 3; ++runtime_component) {
      const int component_axis = mesh_vector_axis_for_component(
         coordinate, axes, runtime_component
      );
      const int canonical_component = mesh_canonical_axis_index(
         coordinate, component_axis
      );
      for (int runtime_normal = 0; runtime_normal < 3; ++runtime_normal) {
         const int normal_axis = mesh_vector_axis_for_component(
            coordinate, axes, runtime_normal
         );
         const int canonical_normal = mesh_canonical_axis_index(
            coordinate, normal_axis
         );
         const RHDConserved& flux = runtime_flux[runtime_normal];
         const amrex::Real momentum = runtime_component == 0
            ? flux.mom1
            : (runtime_component == 1 ? flux.mom2 : flux.mom3);
         if (canonical_component >= 0 && canonical_normal >= 0) {
            flux_component_normal[3 * canonical_component + canonical_normal] =
               momentum;
         }
      }
   }
   const amrex::Real pressure_flux =
      metric.lapse * metric.sqrt_det_gamma * primitive.pressure;
   flux_component_normal[0] -= pressure_flux;
   flux_component_normal[4] -= pressure_flux;
   flux_component_normal[8] -= pressure_flux;
   return flux_component_normal;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 9> rhd_reference_metric_nonpressure_mixed_flux(
   RHDPrimitive const& primitive,
   int coordinate,
   MeshAxes const& axes
) noexcept
{
   return rhd_reference_metric_nonpressure_mixed_flux(
      primitive, SpacetimeMetric{}, coordinate, axes
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> rhd_reference_metric_momentum_basis_divergence(
   RHDPrimitive const& primitive,
   SpacetimeMetric const& metric,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, 3> const& q
) noexcept
{
   const auto flux_component_normal = rhd_reference_metric_nonpressure_mixed_flux(
      primitive, metric, coordinate, axes
   );
   amrex::GpuArray<amrex::Real, 3> canonical{};
   for (int component = 0; component < 3; ++component) {
      canonical[component] = reference_metric_mixed_basis_divergence_component(
         coordinate, component, q, flux_component_normal
      );
   }

   amrex::GpuArray<amrex::Real, 3> runtime{};
   for (int component = 0; component < 3; ++component) {
      const int axis = mesh_vector_axis_for_component(coordinate, axes, component);
      const int canonical_component = mesh_canonical_axis_index(coordinate, axis);
      runtime[component] = canonical_component >= 0
         ? canonical[canonical_component]
         : 0.0_rt;
   }
   return runtime;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> rhd_reference_metric_momentum_basis_divergence(
   RHDPrimitive const& primitive,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, 3> const& q
) noexcept
{
   return rhd_reference_metric_momentum_basis_divergence(
      primitive, SpacetimeMetric{}, coordinate, axes, q
   );
}

} // namespace pyfluid_amrex
