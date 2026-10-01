#include "pyfluid_amrex/spacetime_coupling_storage.hpp"

#include <AMReX.H>

#include "pyfluid_amrex/relativistic_state.hpp"
#include "pyfluid_amrex/spacetime_fields.hpp"

namespace pyfluid_amrex {

void SpacetimeCouplingStorage::configure(int num_levels, int cell_ghost_width, int matter_ghost_width)
{
   if (num_levels <= 0 || cell_ghost_width < 2 || matter_ghost_width < 2) {
      amrex::Abort("invalid spacetime-coupling storage configuration");
   }
   active_ = true;
   cell_ghost_width_ = cell_ghost_width;
   matter_ghost_width_ = matter_ghost_width;
   cell_spacetime_.resize(num_levels);
   cell_matter_.resize(num_levels);
   vertex_matter_.resize(num_levels);
   cell_spacetime_valid_.resize(num_levels, 0);
}

void SpacetimeCouplingStorage::allocate_level(
   int level,
   amrex::BoxArray const& cell_boxes,
   amrex::DistributionMapping const& distribution
)
{
   if (!active_ || level < 0 || level >= static_cast<int>(cell_spacetime_.size())) {
      amrex::Abort("invalid spacetime-coupling level allocation");
   }
   cell_spacetime_[level] = std::make_unique<amrex::MultiFab>(
      cell_boxes, distribution, Z4C_NUM_COMPONENTS, cell_ghost_width_
   );
   cell_matter_[level] = std::make_unique<amrex::MultiFab>(
      cell_boxes, distribution, EULERIAN_MATTER_NUM_COMPONENTS, matter_ghost_width_
   );
   vertex_matter_[level] = std::make_unique<amrex::MultiFab>(
      amrex::convert(cell_boxes, amrex::IntVect::TheNodeVector()),
      distribution,
      EULERIAN_MATTER_NUM_COMPONENTS,
      0
   );
   cell_spacetime_valid_[level] = 0;
}

void SpacetimeCouplingStorage::clear_level(int level)
{
   if (!active_ || level < 0 || level >= static_cast<int>(cell_spacetime_.size())) {
      return;
   }
   cell_spacetime_[level].reset();
   cell_matter_[level].reset();
   vertex_matter_[level].reset();
   cell_spacetime_valid_[level] = 0;
}

bool SpacetimeCouplingStorage::cell_spacetime_valid(int level) const noexcept
{
   return active_ && level >= 0
      && level < static_cast<int>(cell_spacetime_valid_.size())
      && cell_spacetime_valid_[level] != 0;
}

void SpacetimeCouplingStorage::mark_cell_spacetime_valid(int level) noexcept
{
   if (active_ && level >= 0
       && level < static_cast<int>(cell_spacetime_valid_.size())) {
      cell_spacetime_valid_[level] = 1;
   }
}

void SpacetimeCouplingStorage::invalidate_cell_spacetime(int level) noexcept
{
   if (active_ && level >= 0
       && level < static_cast<int>(cell_spacetime_valid_.size())) {
      cell_spacetime_valid_[level] = 0;
   }
}

} // namespace pyfluid_amrex
