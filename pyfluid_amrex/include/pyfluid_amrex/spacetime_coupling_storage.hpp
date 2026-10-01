#pragma once

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MultiFab.H>

#include <memory>
#include <vector>

namespace pyfluid_amrex {

class SpacetimeCouplingStorage {
public:
   using LevelData = std::vector<std::unique_ptr<amrex::MultiFab>>;

   void configure(int num_levels, int cell_ghost_width, int matter_ghost_width = 2);
   void allocate_level(
      int level,
      amrex::BoxArray const& cell_boxes,
      amrex::DistributionMapping const& distribution
   );
   void clear_level(int level);
   bool cell_spacetime_valid(int level) const noexcept;
   void mark_cell_spacetime_valid(int level) noexcept;
   void invalidate_cell_spacetime(int level) noexcept;

   bool active() const noexcept { return active_; }
   int cell_ghost_width() const noexcept { return cell_ghost_width_; }
   LevelData& cell_spacetime() noexcept { return cell_spacetime_; }
   LevelData const& cell_spacetime() const noexcept { return cell_spacetime_; }
   LevelData& cell_matter() noexcept { return cell_matter_; }
   LevelData const& cell_matter() const noexcept { return cell_matter_; }
   LevelData& vertex_matter() noexcept { return vertex_matter_; }
   LevelData const& vertex_matter() const noexcept { return vertex_matter_; }

private:
   bool active_ = false;
   int cell_ghost_width_ = 0;
   int matter_ghost_width_ = 2;
   LevelData cell_spacetime_;
   LevelData cell_matter_;
   LevelData vertex_matter_;
   std::vector<unsigned char> cell_spacetime_valid_;
};

} // namespace pyfluid_amrex
