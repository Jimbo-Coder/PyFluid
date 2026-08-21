#pragma once

#include <AMReX_BC_TYPES.H>

#include "pyfluid_amrex/z4c_runtime.hpp"

namespace pyfluid_amrex {

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
   // cell/face paths apply BCRec reflection themselves. Mirror about the
   // boundary node (not the adjacent cell face) and compose parities at
   // corners so nodal Z4c symmetry is decomposition independent.
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
      amrex::Real parity = 1.0;
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         if (geometry.isPeriodic(direction)) {
            continue;
         }
         int boundary_kind = amrex::BCType::int_dir;
         if (point[direction] < node_lo[direction]) {
            boundary_kind = boundary.lo(direction);
            source[direction] = (
               boundary_kind == amrex::BCType::reflect_even
               || boundary_kind == amrex::BCType::reflect_odd
            )
               ? 2 * node_lo[direction] - point[direction]
               : node_lo[direction];
         } else if (point[direction] > node_hi[direction]) {
            boundary_kind = boundary.hi(direction);
            source[direction] = (
               boundary_kind == amrex::BCType::reflect_even
               || boundary_kind == amrex::BCType::reflect_odd
            )
               ? 2 * node_hi[direction] - point[direction]
               : node_hi[direction];
         }
         if (boundary_kind == amrex::BCType::reflect_odd) {
            parity = -parity;
         }
      }
      const int output_component = destination_component + component;
      values(
         point[0], point[1], point[2], output_component
      ) = parity * values(
         source[0], source[1], source[2], output_component
      );
   }
}

} // namespace pyfluid_amrex
