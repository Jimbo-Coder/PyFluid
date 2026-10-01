#pragma once

#include <AMReX_BC_TYPES.H>

#include "pyfluid_amrex/interpolation_weights.hpp"
#include "pyfluid_amrex/generated/spatial_order_catalog.hpp"
#include "pyfluid_amrex/z4c_runtime.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Z4cFillPatchBoundary::operator()(
   amrex::IntVect const& index,
   amrex::Array4<amrex::Real> const& values,
   int destination_component,
   int num_components,
   amrex::GeometryData const& geometry,
   amrex::Real,
   amrex::BCRec const* boundary_records,
   int,
   int
) const noexcept
{
   // AMReX's all-node GpuBndryFuncFab path initializes physical ghosts by
   // constant extrapolation and delegates reflection to this functor. The
   // cell/face paths apply BCRec themselves. Own both reflection and
   // polynomial extrapolation here. Tensor-product corner fills read only
   // physical-domain nodes (or already exchanged tangential/periodic ghosts),
   // never another physical ghost being written by this launch. These are
   // finite-difference extensions, not positive physical point values. Keep
   // raw variables here: exponentiating high-order log extrapolation into
   // wide AMR halos can overflow. Physical-point centering reconstruction
   // owns its separate positive-scalar representation and metric checks.
   const amrex::Box& cell_domain = geometry.Domain();
   const int node_lo[3]{
      cell_domain.smallEnd(0),
#if (AMREX_SPACEDIM >= 2)
      cell_domain.smallEnd(1),
#else
      0,
#endif
#if (AMREX_SPACEDIM == 3)
      cell_domain.smallEnd(2),
#else
      0,
#endif
   };
   const int node_hi[3]{
      cell_domain.bigEnd(0) + 1,
#if (AMREX_SPACEDIM >= 2)
      cell_domain.bigEnd(1) + 1,
#else
      0,
#endif
#if (AMREX_SPACEDIM == 3)
      cell_domain.bigEnd(2) + 1,
#else
      0,
#endif
   };
   const int point[3]{
      index[0],
#if (AMREX_SPACEDIM >= 2)
      index[1],
#else
      0,
#endif
#if (AMREX_SPACEDIM == 3)
      index[2],
#else
      0,
#endif
   };
   for (int component = 0; component < num_components; ++component) {
      const amrex::BCRec& boundary = boundary_records[component];
      int source[3]{point[0], point[1], point[2]};
      int count[3]{1, 1, 1};
      // A second derivative loses two powers of spacing. Use order+2
      // physical nodes for the finite-difference ghost extension.
      constexpr int maximum_points = maximum_spatial_order + 2;
      amrex::GpuArray<amrex::Real, maximum_points> weights[3]{};
      for (int direction = 0; direction < 3; ++direction) {
         weights[direction][0] = 1.0_rt;
      }
      amrex::Real parity = 1.0_rt;
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         if (geometry.isPeriodic(direction)) {
            continue;
         }
         while (source[direction] < node_lo[direction] || source[direction] > node_hi[direction]) {
            const bool low = source[direction] < node_lo[direction];
            const int boundary_kind = low ? boundary.lo(direction) : boundary.hi(direction);
            const int boundary_node = low ? node_lo[direction] : node_hi[direction];
            if (boundary_kind == amrex::BCType::reflect_even
                || boundary_kind == amrex::BCType::reflect_odd) {
               source[direction] = 2 * boundary_node - source[direction];
               if (boundary_kind == amrex::BCType::reflect_odd) parity = -parity;
               // A one-cell invariant direction can cross both faces. Fold
               // again instead of reading an unfilled opposite-side ghost.
               continue;
            }
            if (boundary_kind == amrex::BCType::hoextrap) {
               count[direction] = amrex::min(derivative_order + 2, node_hi[direction] - node_lo[direction] + 1);
               const int first = low ? node_lo[direction] : node_hi[direction] - count[direction] + 1;
               weights[direction] = uniform_lagrange_weights<maximum_points>(source[direction], first, count[direction]);
               source[direction] = first;
            } else {
               source[direction] = boundary_node;
            }
            break;
         }
      }
      const int output_component = destination_component + component;
      amrex::Real value = 0.0_rt;
      for (int c = 0; c < count[2]; ++c) {
         amrex::Real plane = 0.0_rt;
         for (int b = 0; b < count[1]; ++b) {
            amrex::Real line = 0.0_rt;
            for (int a = 0; a < count[0]; ++a) {
               line += weights[0][a]
                  * values(source[0] + a, source[1] + b, source[2] + c, output_component);
            }
            plane += weights[1][b] * line;
         }
         value += weights[2][c] * plane;
      }
      values(point[0], point[1], point[2], output_component) = parity * value;
   }
}

} // namespace pyfluid_amrex
