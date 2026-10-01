#pragma once

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/coordinate_locations.hpp"
#include "pyfluid_amrex/metric_reductions.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

// Static chart sampling and component permutation, independent of the
// evolved field layout and of any spacetime formulation.
struct ReferenceChartPoint {
   int coordinate;
   amrex::GpuArray<int, 3> canonical;
   amrex::GpuArray<amrex::Real, 3> position;
};

struct ReferenceChartGeometry {
   int coordinate;
   MeshAxes axes;
   amrex::GpuArray<int, 3> canonical;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lower;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> upper;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> spacing;
   amrex::GpuArray<int, AMREX_SPACEDIM> periodic;
   amrex::GpuArray<amrex::Real, 3> inverse_spacing;
   amrex::Dim3 domain_lower;

   ReferenceChartGeometry(
      amrex::Geometry const& geometry,
      int coordinate_kind,
      MeshAxes const& active_axes
   )
      : coordinate(coordinate_kind), axes(active_axes),
        lower(geometry.ProbLoArray()), upper(geometry.ProbHiArray()),
        spacing(geometry.CellSizeArray()), periodic(geometry.isPeriodicArray()),
        domain_lower(amrex::lbound(geometry.Domain()))
   {
      for (int d = 0; d < 3; ++d) {
         canonical[d] = mesh_canonical_axis_index(
            coordinate, mesh_vector_axis_for_component(coordinate, axes, d)
         );
         inverse_spacing[d] = d < AMREX_SPACEDIM ? 1.0_rt / spacing[d] : 1.0_rt;
      }
   }

   ReferenceChartGeometry(amrex::Geometry const& geometry, RunConfig const& config)
      : ReferenceChartGeometry(
           geometry, mesh_coordinate_code(config.mesh_coordinate), mesh_axes_layout(config)
        )
   {}

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   ReferenceChartPoint at(int i, int j, int k, bool nodal = true) const noexcept
   {
      return {coordinate, canonical, mesh_coordinate_triplet_at_location(
         coordinate, axes, lower, spacing, i - domain_lower.x, j - domain_lower.y,
         k - domain_lower.z, nodal, nodal, nodal)};
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real nodal_volume(int i, int j, int k) const noexcept
   {
      const int index[3]{i - domain_lower.x, j - domain_lower.y, k - domain_lower.z};
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> cell_lower{};
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> cell_width{};
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
         const amrex::Real center = lower[d] + index[d] * spacing[d];
         cell_lower[d] = center - 0.5_rt * spacing[d];
         amrex::Real cell_upper = center + 0.5_rt * spacing[d];
         if (!periodic[d]) {
            cell_lower[d] = amrex::max(cell_lower[d], lower[d]);
            cell_upper = amrex::min(cell_upper, upper[d]);
         }
         cell_width[d] = cell_upper - cell_lower[d];
      }
      return mesh_cell_volume_for_reduction(
         coordinate, axes, cell_lower, cell_width, 0, 0, 0
      );
   }

};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real reference_inverse_scale(
   ReferenceChartPoint const& chart,
   int runtime_direction
) noexcept
{
   const auto q = chart.position;
   const auto scale = coordinate_scale_factors(chart.coordinate, q[0], q[1], q[2]);
   return 1.0_rt / scale[chart.canonical[runtime_direction]];
}

} // namespace pyfluid_amrex
