#include "pyfluid_amrex/amr_solver_base.hpp"
#include "diagnostics_impl.hpp"

namespace pyfluid_amrex {

bool AmrSolverBase::symmetry_axis_active(int dir) const
{
   if (dir == 0) {
      return side_boundary_kind(cfg_, 0, false) == BoundaryKind::Symmetry ||
             side_boundary_kind(cfg_, 0, true) == BoundaryKind::Symmetry;
   }
#if (AMREX_SPACEDIM >= 2)
   if (dir == 1) {
      return side_boundary_kind(cfg_, 1, false) == BoundaryKind::Symmetry ||
             side_boundary_kind(cfg_, 1, true) == BoundaryKind::Symmetry;
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (dir == 2) {
      return side_boundary_kind(cfg_, 2, false) == BoundaryKind::Symmetry ||
             side_boundary_kind(cfg_, 2, true) == BoundaryKind::Symmetry;
   }
#endif
   return false;
}

amrex::Real AmrSolverBase::symmetry_volume_factor() const
{
   amrex::Real factor = 1.0;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (symmetry_axis_active(dir)) {
         factor *= 2.0;
      }
   }
   return factor;
}

amrex::Real AmrSolverBase::symmetry_momentum_factor(int dir, amrex::Real volume_factor) const
{
   return symmetry_axis_active(dir) ? 0.0 : volume_factor;
}

amrex::Real AmrSolverBase::symmetry_angular_momentum_factor(int dir, amrex::Real volume_factor) const
{
   for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
      if (symmetry_axis_active(axis) && axis != dir) {
         return 0.0;
      }
   }
   return volume_factor;
}

#include "amr_solver_diagnostics_impl.hpp"

} // namespace pyfluid_amrex
