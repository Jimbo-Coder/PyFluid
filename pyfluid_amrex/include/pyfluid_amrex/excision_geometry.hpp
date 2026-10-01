#pragma once

#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace pyfluid_amrex {

using namespace amrex::literals;

// A fixed coordinate worldtube. Horizon containment and characteristic
// outflow are runtime requirements, not consequences of choosing a radius.
struct SphericalExcisionRegion {
   amrex::GpuArray<amrex::Real, 3> center{};
   amrex::Real radius = 0.0_rt;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   bool enabled() const noexcept { return radius > 0.0_rt; }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   bool contains(amrex::GpuArray<amrex::Real, 3> const& position) const noexcept
   {
      amrex::Real radius_squared = 0.0_rt;
      for (int direction = 0; direction < 3; ++direction) {
         const amrex::Real delta = position[direction] - center[direction];
         radius_squared += delta * delta;
      }
      return enabled() && radius_squared <= radius * radius;
   }

   // Does an axis-aligned stencil footprint touch the inactive region?
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   bool intersects(amrex::GpuArray<amrex::Real, 3> const& lower,
                   amrex::GpuArray<amrex::Real, 3> const& upper) const noexcept
   {
      amrex::Real distance_squared = 0.0_rt;
      for (int direction = 0; direction < 3; ++direction) {
         const amrex::Real nearest = amrex::max(lower[direction],
            amrex::min(center[direction], upper[direction]));
         const amrex::Real delta = nearest - center[direction];
         distance_squared += delta * delta;
      }
      return enabled() && distance_squared <= radius * radius;
   }
};

struct ExcisionStencilPlan {
   amrex::GpuArray<int, 3> sides{};
   // Pairs (0,1), (0,2), (1,2), two directions per pair.
   amrex::GpuArray<int, 6> mixed_sides{};
};

// Centering is geometry data, not a separate cell/node implementation.
struct ExcisionGridView {
   SphericalExcisionRegion region;
   amrex::GpuArray<amrex::Real, 3> lower{};
   amrex::GpuArray<amrex::Real, 3> spacing{};
   amrex::GpuArray<amrex::Real, 3> centering_offset{};

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::GpuArray<amrex::Real, 3> position(int i, int j, int k) const noexcept
   {
      const int index[3]{i, j, k};
      amrex::GpuArray<amrex::Real, 3> point{};
      for (int direction = 0; direction < 3; ++direction) {
         point[direction] = lower[direction]
            + (index[direction] + centering_offset[direction]) * spacing[direction];
      }
      return point;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   bool operator()(int i, int j, int k, int) const noexcept
   {
      return !region.contains(position(i, j, k));
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   ExcisionStencilPlan derivative_stencil(int i, int j, int k,
                                         int stencil_radius) const noexcept
   {
      const auto point = position(i, j, k);
      ExcisionStencilPlan plan;
      for (int direction = 0; direction < 3; ++direction) {
         auto stencil_lower = point;
         auto stencil_upper = point;
         stencil_lower[direction] -= stencil_radius * spacing[direction];
         stencil_upper[direction] += stencil_radius * spacing[direction];
         if (region.intersects(stencil_lower, stencil_upper)) {
            plan.sides[direction] = point[direction] >= region.center[direction] ? 1 : -1;
         }
      }
      int pair = 0;
      for (int first = 0; first < 3; ++first) {
         for (int second = first + 1; second < 3; ++second, ++pair) {
            auto stencil_lower = point;
            auto stencil_upper = point;
            const int directions[2]{first, second};
            for (int axis = 0; axis < 2; ++axis) {
               const int direction = directions[axis];
               const int side = plan.sides[direction];
               plan.mixed_sides[2 * pair + axis] = side;
               if (side <= 0) stencil_lower[direction] -= stencil_radius * spacing[direction];
               if (side >= 0) stencil_upper[direction] += stencil_radius * spacing[direction];
            }
            if (region.intersects(stencil_lower, stencil_upper)) {
               // A mixed stencil can cross the cut even when both axis
               // lines miss it. Use the exterior orthant for this pair only.
               for (int axis = 0; axis < 2; ++axis) {
                  const int direction = directions[axis];
                  plan.mixed_sides[2 * pair + axis] =
                     point[direction] >= region.center[direction] ? 1 : -1;
               }
            }
         }
      }
      return plan;
   }

   // Restrict a tensor-product sampling stencil to the exterior orthant.
   // Coordinates and centerings belong to the source grid; the target can
   // lie on another centering or AMR level. Coincident axes use one point.
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   void restrict_stencil(amrex::GpuArray<amrex::Real, 3> const& target,
                         amrex::GpuArray<int, 3>& first,
                         amrex::GpuArray<int, 3> const& count) const noexcept
   {
      const auto stencil_lower = position(first[0], first[1], first[2]);
      const auto stencil_upper = position(first[0] + count[0] - 1,
         first[1] + count[1] - 1, first[2] + count[2] - 1);
      if (!region.intersects(stencil_lower, stencil_upper)) return;
      for (int direction = 0; direction < 3; ++direction) {
         if (count[direction] == 1) continue;
         const amrex::Real coordinate = (target[direction] - lower[direction])
            / spacing[direction] - centering_offset[direction];
         first[direction] = target[direction] >= region.center[direction]
            ? static_cast<int>(std::ceil(coordinate))
            : static_cast<int>(std::floor(coordinate)) - count[direction] + 1;
      }
   }
};

} // namespace pyfluid_amrex
