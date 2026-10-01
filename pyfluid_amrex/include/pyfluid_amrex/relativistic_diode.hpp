#pragma once

#include <AMReX_Geometry.H>
#include <cmath>
#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void apply_relativistic_diode_velocity(
   amrex::Real& velocity,
   amrex::Real lapse,
   amrex::Real shift,
   bool low_face,
   bool high_face
) noexcept
{
   // The boundary is stationary in coordinates, not in the Eulerian frame.
   // Valencia baryon transport is D (alpha v^i - beta^i).
   const amrex::Real transport = lapse * velocity - shift;
   if ((low_face && transport > 0.0_rt) || (high_face && transport < 0.0_rt)) {
      velocity = shift / lapse;
   }
}

template <typename Primitive>
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE inline
void apply_relativistic_diode_velocity(
   Primitive& primitive,
   SpacetimeMetric const& metric,
   bool xlo, bool xhi, bool ylo, bool yhi, bool zlo, bool zhi
) noexcept
{
   if (!(xlo || xhi || ylo || yhi || zlo || zhi)) return;
   const amrex::Real original_speed_squared = spatial_vector_norm_squared(
      primitive.v1, primitive.v2, primitive.v3, metric
   );
   apply_relativistic_diode_velocity(
      primitive.v1, metric.lapse, metric.shift1, xlo, xhi
   );
   apply_relativistic_diode_velocity(
      primitive.v2, metric.lapse, metric.shift2, ylo, yhi
   );
   apply_relativistic_diode_velocity(
      primitive.v3, metric.lapse, metric.shift3, zlo, zhi
   );
   const amrex::Real clipped_speed_squared = spatial_vector_norm_squared(
      primitive.v1, primitive.v2, primitive.v3, metric
   );
   if (clipped_speed_squared >= 1.0_rt) {
      // A shifted normal clip can leave too much tangential velocity. Complete
      // this boundary state along its coordinate-transport direction toward
      // the coordinate-stationary observer. This preserves every diode sign
      // (including corners), never alters thermodynamics/B, and avoids a new
      // arbitrary Lorentz ceiling. It is ghost BC construction, not recovery.
      const amrex::Real rest1 = metric.shift1 / metric.lapse;
      const amrex::Real rest2 = metric.shift2 / metric.lapse;
      const amrex::Real rest3 = metric.shift3 / metric.lapse;
      const amrex::Real rest_squared = spatial_vector_norm_squared(
         rest1, rest2, rest3, metric
      );
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
         rest_squared < 1.0_rt && original_speed_squared < 1.0_rt,
         "relativistic diode cannot construct a causal coordinate-stationary boundary state"
      );
      const amrex::Real delta1 = primitive.v1 - rest1;
      const amrex::Real delta2 = primitive.v2 - rest2;
      const amrex::Real delta3 = primitive.v3 - rest3;
      const amrex::Real delta_squared = spatial_vector_norm_squared(
         delta1, delta2, delta3, metric
      );
      const amrex::Real cross = 0.5_rt * (
         clipped_speed_squared - rest_squared - delta_squared
      );
      const amrex::Real allowance = amrex::max(
         original_speed_squared, rest_squared
      ) - rest_squared;
      const amrex::Real root = std::sqrt(cross * cross + delta_squared * allowance);
      const amrex::Real fraction = cross >= 0.0_rt
         ? (allowance == 0.0_rt ? 0.0_rt : allowance / (root + cross))
         : (root - cross) / delta_squared;
      primitive.v1 = rest1 + fraction * delta1;
      primitive.v2 = rest2 + fraction * delta2;
      primitive.v3 = rest3 + fraction * delta3;
   }
}

struct RelativisticDiodeBoundary {
   amrex::GpuArray<int, 3> lower{{0, 0, 0}};
   amrex::GpuArray<int, 3> upper{{0, 0, 0}};
   int low_faces = 0;
   int high_faces = 0;

   template <int Direction>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   int side(int i, int j, int k) const noexcept
   {
      if (((low_faces | high_faces) & (1 << Direction)) == 0) return 0;
      const int point[3]{i, j, k};
      if ((low_faces & (1 << Direction)) && point[Direction] == lower[Direction]) return -1;
      if ((high_faces & (1 << Direction)) && point[Direction] == upper[Direction]) return 1;
      return 0;
   }
};

inline RelativisticDiodeBoundary make_relativistic_diode_boundary(
   RunConfig const& cfg, amrex::Geometry const& geometry
)
{
   RelativisticDiodeBoundary result;
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      result.lower[direction] = geometry.Domain().smallEnd(direction);
      result.upper[direction] = geometry.Domain().bigEnd(direction) + 1;
      if (!geometry.isPeriodic(direction)) {
         if (side_boundary_kind(cfg, direction, false) == BoundaryKind::Diode)
            result.low_faces |= 1 << direction;
         if (side_boundary_kind(cfg, direction, true) == BoundaryKind::Diode)
            result.high_faces |= 1 << direction;
      }
   }
   return result;
}

// At a diode face the exterior supplies no independent state. Use the
// interior cell's extrapolated state on both sides, with incoming coordinate
// transport removed. Every consistent Riemann solver then returns its physical
// flux, including pressure and magnetic stresses, without numerical mass influx.
// This is the boundary condition, not a reconstruction fallback in active cells.
template <int Direction, typename Primitive>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool apply_relativistic_diode_face(
   Primitive& left, Primitive& right, SpacetimeMetric const& metric,
   RelativisticDiodeBoundary const& boundary, int i, int j, int k
) noexcept
{
   const int side = boundary.template side<Direction>(i, j, k);
   if (side == 0) return false;
   Primitive state = side < 0 ? right : left;
   apply_relativistic_diode_velocity(
      state, metric,
      Direction == 0 && side < 0, Direction == 0 && side > 0,
      Direction == 1 && side < 0, Direction == 1 && side > 0,
      Direction == 2 && side < 0, Direction == 2 && side > 0
   );
   left = state;
   right = state;
   return true;
}

} // namespace pyfluid_amrex
