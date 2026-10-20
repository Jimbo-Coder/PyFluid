#pragma once

#include "pyfluid_amrex/relativistic_flux_limiter.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/reference_metric_spacetime.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {

struct StoredLimiterMetricAccessor {
   amrex::Array4<amrex::Real const> spacetime;
   amrex::Array4<amrex::Real const> candidate_spacetime;
   ExcisionGridView excision{};
   RelativisticDiodeBoundary diode{};

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
         spacetime, i, j, k, excision
      );
   }
};

struct FixedLimiterMetricAccessor {
   FixedSpacetimeParameters spacetime;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lower;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> spacing;
   int coordinate;
   MeshAxes axes;
   amrex::Real time;
   ExcisionGridView excision{};
   RelativisticDiodeBoundary diode{};

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric cell(int i, int j, int k) const noexcept
   {
      return reference_metric_fixed_spacetime_geometry_at_cell(
         spacetime, coordinate, axes, lower, spacing, i, j, k, time
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
      return reference_metric_fixed_spacetime_geometry_at_face<Direction>(
         spacetime, coordinate, axes, lower, spacing, i, j, k, time
      ).metric;
   }
};

} // namespace pyfluid_amrex
