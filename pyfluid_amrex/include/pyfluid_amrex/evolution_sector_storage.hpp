#pragma once

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "pyfluid_amrex/fields.hpp"

namespace pyfluid_amrex {

enum class EvolutionStageSlot : std::uint8_t {
   State,
   Next,
   Scratch,
};

class EvolutionSectorStorage {
public:
   using LevelData = std::vector<std::unique_ptr<amrex::MultiFab>>;

   void configure(
      GridFieldSetInfo const& field_set,
      int num_levels,
      int state_ghost_width = -1,
      bool evolves = true
   );

   bool active() const noexcept;
   bool evolves() const noexcept;
   EvolutionSector sector() const noexcept;
   GridFieldCentering centering() const noexcept;
   int num_components() const noexcept;
   int ghost_width() const noexcept;
   GridFieldSetInfo const& field_set() const noexcept;

   void allocate_level(
      int level,
      amrex::BoxArray const& cell_boxes,
      amrex::DistributionMapping const& distribution,
      bool allocate_source_state,
      bool allocate_rk4_subcycling_state
   );
   void clear_level(int level);
   void copy_state_to_old(int level);
   void copy_next_to_state(int level);
   void average_down_level(int fine_level, amrex::IntVect const& refinement_ratio);
   void copy(
      LevelData& destination,
      LevelData const& source,
      int level_begin,
      int level_end
   ) const;
   void linear_combination(
      LevelData& destination,
      amrex::Real weight_a,
      LevelData const& state_a,
      amrex::Real weight_b,
      LevelData const& state_b,
      int level_begin,
      int level_end
   ) const;
   void add_scaled(
      LevelData& destination,
      amrex::Real scale,
      LevelData const& increment,
      int level_begin,
      int level_end
   ) const;
   bool rk4_stage_data_ready(int level, amrex::Real coarse_dt) const;
   void store_rk4_rhs_stage(int stage, int level_begin, int level_end);
   void fill_rk4_dense_state(
      int coarse_level,
      amrex::Real fine_step_start,
      int rk_stage,
      amrex::Real coarse_old_time,
      amrex::Real coarse_time,
      amrex::Real coarse_dt,
      amrex::Real fine_dt
   );

   void invalidate_rk4_stage_cache(int level);
   bool rk4_stage_cache_matches(
      int level,
      int stage,
      amrex::Real fine_start,
      amrex::Real coarse_time,
      amrex::Real coarse_dt
   ) const;
   void set_rk4_stage_cache(
      int level,
      int stage,
      amrex::Real fine_start,
      amrex::Real coarse_time,
      amrex::Real coarse_dt
   );

   LevelData& state() noexcept { return state_; }
   LevelData const& state() const noexcept { return state_; }
   LevelData& old_state() noexcept { return old_state_; }
   LevelData const& old_state() const noexcept { return old_state_; }
   LevelData& source_state() noexcept { return source_state_; }
   LevelData const& source_state() const noexcept { return source_state_; }
   LevelData& rk4_coarse_stage_state() noexcept { return rk4_coarse_stage_state_; }
   LevelData const& rk4_coarse_stage_state() const noexcept
   {
      return rk4_coarse_stage_state_;
   }
   LevelData& next() noexcept { return next_; }
   LevelData const& next() const noexcept { return next_; }
   LevelData& scratch() noexcept { return scratch_; }
   LevelData const& scratch() const noexcept { return scratch_; }
   LevelData& rhs() noexcept { return rhs_; }
   LevelData const& rhs() const noexcept { return rhs_; }
   LevelData& stage(EvolutionStageSlot slot) noexcept;
   LevelData const& stage(EvolutionStageSlot slot) const noexcept;
   std::array<LevelData, 4>& rk4_rhs_stages() noexcept { return rk4_rhs_stages_; }
   std::array<LevelData, 4> const& rk4_rhs_stages() const noexcept
   {
      return rk4_rhs_stages_;
   }

private:
   GridFieldSetInfo field_set_{};
   GridFieldCentering centering_ = GridFieldCentering::Cell;
   int num_components_ = 0;
   int ghost_width_ = 0;
   bool active_ = false;
   bool evolves_ = false;

   LevelData state_;
   LevelData old_state_;
   LevelData source_state_;
   LevelData rk4_coarse_stage_state_;
   LevelData next_;
   LevelData scratch_;
   LevelData rhs_;
   std::array<LevelData, 4> rk4_rhs_stages_;
   amrex::Vector<int> rk4_cache_stage_;
   amrex::Vector<amrex::Real> rk4_cache_fine_start_;
   amrex::Vector<amrex::Real> rk4_cache_time_;
   amrex::Vector<amrex::Real> rk4_cache_dt_;
};

} // namespace pyfluid_amrex
