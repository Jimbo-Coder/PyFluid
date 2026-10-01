#pragma once

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace pyfluid_amrex {

using namespace amrex::literals;

// Fixed coordinate regions, independent of resolution and of the metric.
// The runtime validates parameter ordering before enabling this diagnostic.
struct SpacetimeConstraintRegion {
   amrex::GpuArray<amrex::Real, 3> center{};
   amrex::Real extent = -1.0_rt;
   amrex::Real exclusion_radius = 0.0_rt;
   amrex::Real split_radius = 0.0_rt;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   bool enabled() const noexcept { return extent > 0.0_rt; }

   // -1: outside; 0: near; 1: far. The exclusion and box boundaries are
   // included, and points exactly on the split sphere belong to far.
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   int classify(amrex::GpuArray<amrex::Real, 3> const& position) const noexcept
   {
      if (!enabled()) return -1;
      amrex::Real radius_squared = 0.0_rt;
      for (int direction = 0; direction < 3; ++direction) {
         const amrex::Real displacement = position[direction] - center[direction];
         if (displacement < -extent || displacement > extent) return -1;
         radius_squared += displacement * displacement;
      }
      if (radius_squared < exclusion_radius * exclusion_radius) return -1;
      return radius_squared < split_radius * split_radius ? 0 : 1;
   }
};

} // namespace pyfluid_amrex
