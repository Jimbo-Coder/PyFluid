#pragma once

#include <AMReX_Geometry.H>

#include "pyfluid_amrex/excision_geometry.hpp"

namespace pyfluid_amrex {

struct RunConfig;

SphericalExcisionRegion spacetime_excision_region(RunConfig const& config);

inline ExcisionGridView excision_grid_view(SphericalExcisionRegion const& region,
   amrex::Geometry const& geometry, amrex::IndexType const& centering)
{
   ExcisionGridView view;
   view.region = region;
   view.spacing = {{1.0, 1.0, 1.0}};
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      view.lower[direction] = geometry.ProbLo(direction);
      view.spacing[direction] = geometry.CellSize(direction);
      view.centering_offset[direction] = centering.nodeCentered(direction) ? 0.0 : 0.5;
   }
   return view;
}

} // namespace pyfluid_amrex
