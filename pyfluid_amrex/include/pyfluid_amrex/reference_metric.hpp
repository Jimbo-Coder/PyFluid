#pragma once

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/kernels_generated/vector/reference_metric.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"

namespace pyfluid_amrex {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real reference_metric_tensor_divergence_connection_component(
   int coordinate,
   int component,
   amrex::GpuArray<amrex::Real, 3> const& q,
   amrex::GpuArray<amrex::Real, 9> const& tensor_normal_component
) noexcept
{
#define PYFLUID_REFERENCE_METRIC_TENSOR_ARGS                                      \
   q[0], q[1], q[2], tensor_normal_component[0], tensor_normal_component[1],     \
      tensor_normal_component[2], tensor_normal_component[3],                    \
      tensor_normal_component[4], tensor_normal_component[5],                    \
      tensor_normal_component[6], tensor_normal_component[7],                    \
      tensor_normal_component[8]
   if (coordinate == MeshCylindrical) {
      if (component == 0) {
         return generated::cylindrical_tensor_divergence_connection_1(
            PYFLUID_REFERENCE_METRIC_TENSOR_ARGS
         );
      }
      if (component == 1) {
         return generated::cylindrical_tensor_divergence_connection_2(
            PYFLUID_REFERENCE_METRIC_TENSOR_ARGS
         );
      }
      return generated::cylindrical_tensor_divergence_connection_3(
         PYFLUID_REFERENCE_METRIC_TENSOR_ARGS
      );
   }
   if (coordinate == MeshSpherical) {
      if (component == 0) {
         return generated::spherical_tensor_divergence_connection_1(
            PYFLUID_REFERENCE_METRIC_TENSOR_ARGS
         );
      }
      if (component == 1) {
         return generated::spherical_tensor_divergence_connection_2(
            PYFLUID_REFERENCE_METRIC_TENSOR_ARGS
         );
      }
      return generated::spherical_tensor_divergence_connection_3(
         PYFLUID_REFERENCE_METRIC_TENSOR_ARGS
      );
   }
   if (component == 0) {
      return generated::cartesian_tensor_divergence_connection_1(
         PYFLUID_REFERENCE_METRIC_TENSOR_ARGS
      );
   }
   if (component == 1) {
      return generated::cartesian_tensor_divergence_connection_2(
         PYFLUID_REFERENCE_METRIC_TENSOR_ARGS
      );
   }
   return generated::cartesian_tensor_divergence_connection_3(
      PYFLUID_REFERENCE_METRIC_TENSOR_ARGS
   );
#undef PYFLUID_REFERENCE_METRIC_TENSOR_ARGS
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real reference_metric_mixed_basis_divergence_component(
   int coordinate,
   int component,
   amrex::GpuArray<amrex::Real, 3> const& q,
   amrex::GpuArray<amrex::Real, 9> const& flux_component_normal
) noexcept
{
   // The generated leaf also serves the historical viscous convention
   // tensor[normal, component]. Valencia stores the explicit mixed ordering
   // flux[component, normal], so transpose once at this shared boundary.
   const amrex::GpuArray<amrex::Real, 9> tensor_normal_component{{
      flux_component_normal[0], flux_component_normal[3], flux_component_normal[6],
      flux_component_normal[1], flux_component_normal[4], flux_component_normal[7],
      flux_component_normal[2], flux_component_normal[5], flux_component_normal[8],
   }};
   return reference_metric_tensor_divergence_connection_component(
      coordinate, component, q, tensor_normal_component
   );
}

} // namespace pyfluid_amrex
