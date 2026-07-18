#include "pyfluid_amrex/evolution_sector_storage.hpp"

#include <AMReX.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFabUtil.H>

#include <algorithm>

namespace pyfluid_amrex {

void EvolutionSectorStorage::configure(
   GridFieldSetInfo const& field_set,
   int num_levels,
   int state_ghost_width,
   bool evolves
)
{
   if (field_set.sector == EvolutionSector::None || field_set.fields.empty()) {
      amrex::Abort("an evolved sector requires a nonempty first-class field set");
   }
   if (num_levels <= 0) {
      amrex::Abort("an evolved sector requires at least one AMR level slot");
   }

   bool found_component = false;
   GridFieldCentering common_centering = GridFieldCentering::Cell;
   int descriptor_ghost_width = 0;
   int component_count = 0;
   for (GridFieldInfo const& field : field_set.fields) {
      if (field.role != FieldRole::EvolvedState) {
         amrex::Abort("an evolved sector field set may contain only evolved state fields");
      }
      descriptor_ghost_width = std::max(descriptor_ghost_width, field.ghost_width);
      for (GridFieldComponentInfo const& component : field.components) {
         if (!found_component) {
            common_centering = component.centering;
            found_component = true;
         } else if (component.centering != common_centering) {
            amrex::Abort(
               "one evolved sector state vector must have one centering; split distinct topologies into distinct field sets"
            );
         }
         ++component_count;
      }
   }
   if (!found_component || component_count == 0) {
      amrex::Abort("an evolved sector field set has no components");
   }

   field_set_ = field_set;
   centering_ = common_centering;
   num_components_ = component_count;
   ghost_width_ = state_ghost_width >= 0 ? state_ghost_width : descriptor_ghost_width;
   active_ = true;
   evolves_ = evolves;

   state_.resize(num_levels);
   old_state_.resize(num_levels);
   source_state_.resize(num_levels);
   rk4_coarse_stage_state_.resize(num_levels);
   next_.resize(num_levels);
   scratch_.resize(num_levels);
   rhs_.resize(num_levels);
   for (LevelData& stage : rk4_rhs_stages_) {
      stage.resize(num_levels);
   }
   rk4_cache_stage_.resize(num_levels, -1);
   rk4_cache_fine_start_.resize(num_levels, 0.0);
   rk4_cache_time_.resize(num_levels, 0.0);
   rk4_cache_dt_.resize(num_levels, 0.0);
}

bool EvolutionSectorStorage::active() const noexcept
{
   return active_;
}

bool EvolutionSectorStorage::evolves() const noexcept
{
   return evolves_;
}

EvolutionSector EvolutionSectorStorage::sector() const noexcept
{
   return field_set_.sector;
}

GridFieldCentering EvolutionSectorStorage::centering() const noexcept
{
   return centering_;
}

int EvolutionSectorStorage::num_components() const noexcept
{
   return num_components_;
}

int EvolutionSectorStorage::ghost_width() const noexcept
{
   return ghost_width_;
}

GridFieldSetInfo const& EvolutionSectorStorage::field_set() const noexcept
{
   return field_set_;
}

EvolutionSectorStorage::LevelData& EvolutionSectorStorage::stage(
   EvolutionStageSlot slot
) noexcept
{
   if (!evolves_) {
      return state_;
   }
   switch (slot) {
   case EvolutionStageSlot::State:
      return state_;
   case EvolutionStageSlot::Next:
      return next_;
   case EvolutionStageSlot::Scratch:
      return scratch_;
   }
   amrex::Abort("invalid evolution stage slot");
   return state_;
}

EvolutionSectorStorage::LevelData const& EvolutionSectorStorage::stage(
   EvolutionStageSlot slot
) const noexcept
{
   if (!evolves_) {
      return state_;
   }
   switch (slot) {
   case EvolutionStageSlot::State:
      return state_;
   case EvolutionStageSlot::Next:
      return next_;
   case EvolutionStageSlot::Scratch:
      return scratch_;
   }
   amrex::Abort("invalid evolution stage slot");
   return state_;
}

void EvolutionSectorStorage::allocate_level(
   int level,
   amrex::BoxArray const& cell_boxes,
   amrex::DistributionMapping const& distribution,
   bool allocate_source_state,
   bool allocate_rk4_subcycling_state
)
{
   if (!active_ || level < 0 || level >= static_cast<int>(state_.size())) {
      amrex::Abort("invalid evolved-sector level allocation");
   }

   amrex::BoxArray boxes = amrex::convert(cell_boxes, grid_field_index_type(centering_));
   state_[level] = std::make_unique<amrex::MultiFab>(boxes, distribution, num_components_, ghost_width_);
   if (!evolves_) {
      old_state_[level].reset();
      source_state_[level].reset();
      rk4_coarse_stage_state_[level].reset();
      next_[level].reset();
      scratch_[level].reset();
      rhs_[level].reset();
      for (LevelData& stage : rk4_rhs_stages_) {
         stage[level].reset();
      }
      return;
   }
   old_state_[level] = std::make_unique<amrex::MultiFab>(
      boxes, distribution, num_components_, ghost_width_
   );
   next_[level] = std::make_unique<amrex::MultiFab>(boxes, distribution, num_components_, ghost_width_);
   scratch_[level] = std::make_unique<amrex::MultiFab>(boxes, distribution, num_components_, ghost_width_);
   rhs_[level] = std::make_unique<amrex::MultiFab>(boxes, distribution, num_components_, 0);

   source_state_[level].reset();
   if (allocate_source_state) {
      source_state_[level] = std::make_unique<amrex::MultiFab>(
         boxes, distribution, num_components_, ghost_width_
      );
   }

   rk4_coarse_stage_state_[level].reset();
   for (LevelData& stage : rk4_rhs_stages_) {
      stage[level].reset();
   }
   if (allocate_rk4_subcycling_state) {
      rk4_coarse_stage_state_[level] = std::make_unique<amrex::MultiFab>(
         boxes, distribution, num_components_, ghost_width_
      );
      for (LevelData& stage : rk4_rhs_stages_) {
         stage[level] = std::make_unique<amrex::MultiFab>(
            boxes, distribution, num_components_, 0
         );
      }
   }
}

void EvolutionSectorStorage::clear_level(int level)
{
   if (!active_ || level < 0 || level >= static_cast<int>(state_.size())) {
      return;
   }
   state_[level].reset();
   old_state_[level].reset();
   source_state_[level].reset();
   rk4_coarse_stage_state_[level].reset();
   next_[level].reset();
   scratch_[level].reset();
   rhs_[level].reset();
   for (LevelData& stage : rk4_rhs_stages_) {
      stage[level].reset();
   }
   invalidate_rk4_stage_cache(level);
}

void EvolutionSectorStorage::copy_state_to_old(int level)
{
   if (!active_ || !state_.at(level) || !old_state_.at(level)) {
      return;
   }
   amrex::MultiFab::Copy(
      *old_state_[level],
      *state_[level],
      0,
      0,
      num_components_,
      state_[level]->nGrowVect()
   );
}

void EvolutionSectorStorage::copy_next_to_state(int level)
{
   if (!active_ || !state_.at(level) || !next_.at(level)) {
      return;
   }
   amrex::MultiFab::Copy(*state_[level], *next_[level], 0, 0, num_components_, 0);
}

void EvolutionSectorStorage::copy(
   LevelData& destination,
   LevelData const& source,
   int level_begin,
   int level_end
) const
{
   for (int level = level_begin; level < level_end; ++level) {
      amrex::MultiFab::Copy(
         *destination.at(level), *source.at(level), 0, 0, num_components_, 0
      );
   }
}

void EvolutionSectorStorage::linear_combination(
   LevelData& destination,
   amrex::Real weight_a,
   LevelData const& state_a,
   amrex::Real weight_b,
   LevelData const& state_b,
   int level_begin,
   int level_end
) const
{
   for (int level = level_begin; level < level_end; ++level) {
      amrex::MultiFab::LinComb(
         *destination.at(level),
         weight_a,
         *state_a.at(level),
         0,
         weight_b,
         *state_b.at(level),
         0,
         0,
         num_components_,
         0
      );
   }
}

void EvolutionSectorStorage::add_scaled(
   LevelData& destination,
   amrex::Real scale,
   LevelData const& increment,
   int level_begin,
   int level_end
) const
{
   for (int level = level_begin; level < level_end; ++level) {
      amrex::MultiFab::Saxpy(
         *destination.at(level),
         scale,
         *increment.at(level),
         0,
         0,
         num_components_,
         0
      );
   }
}

bool EvolutionSectorStorage::rk4_stage_data_ready(
   int level,
   amrex::Real coarse_dt
) const
{
   if (coarse_dt <= 0.0 || !rk4_coarse_stage_state_.at(level)
       || !old_state_.at(level)) {
      return false;
   }
   for (LevelData const& stage : rk4_rhs_stages_) {
      if (!stage.at(level)) {
         return false;
      }
   }
   return true;
}

void EvolutionSectorStorage::store_rk4_rhs_stage(
   int stage,
   int level_begin,
   int level_end
)
{
   if (stage < 0 || stage >= static_cast<int>(rk4_rhs_stages_.size())) {
      amrex::Abort("invalid RK4 stage index");
   }
   copy(rk4_rhs_stages_[stage], rhs_, level_begin, level_end);
}

void EvolutionSectorStorage::fill_rk4_dense_state(
   int coarse_level,
   amrex::Real fine_step_start,
   int rk_stage,
   amrex::Real coarse_old_time,
   amrex::Real coarse_time,
   amrex::Real coarse_dt,
   amrex::Real fine_dt
)
{
   if (!rk4_stage_data_ready(coarse_level, coarse_dt)) {
      amrex::Abort("RK4 subcycling requires stored coarse stage data");
   }
   if (rk4_stage_cache_matches(
          coarse_level,
          rk_stage,
          fine_step_start,
          coarse_time,
          coarse_dt
       )) {
      return;
   }

   amrex::Real xsi = (fine_step_start - coarse_old_time) / coarse_dt;
   xsi = amrex::min(amrex::max(xsi, 0.0), 1.0);
   const amrex::Real r = fine_dt / coarse_dt;
   const amrex::Real xsi2 = xsi * xsi;
   const amrex::Real xsi3 = xsi2 * xsi;
   const amrex::GpuArray<amrex::Real, 4> b{{
      xsi - 1.5 * xsi2 + (2.0 / 3.0) * xsi3,
      xsi2 - (2.0 / 3.0) * xsi3,
      xsi2 - (2.0 / 3.0) * xsi3,
      -0.5 * xsi2 + (2.0 / 3.0) * xsi3,
   }};
   const amrex::GpuArray<amrex::Real, 4> c{{
      1.0 - 3.0 * xsi + 2.0 * xsi2,
      2.0 * xsi - 2.0 * xsi2,
      2.0 * xsi - 2.0 * xsi2,
      -xsi + 2.0 * xsi2,
   }};
   const amrex::GpuArray<amrex::Real, 4> d{{
      -3.0 + 4.0 * xsi,
      2.0 - 4.0 * xsi,
      2.0 - 4.0 * xsi,
      -1.0 + 4.0 * xsi,
   }};
   constexpr amrex::GpuArray<amrex::Real, 4> e{{4.0, -4.0, -4.0, 4.0}};
   amrex::MultiFab& dense_state = *rk4_coarse_stage_state_[coarse_level];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(dense_state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto output = dense_state.array(mfi);
      const auto initial = old_state_[coarse_level]->const_array(mfi);
      const auto k1 = rk4_rhs_stages_[0][coarse_level]->const_array(mfi);
      const auto k2 = rk4_rhs_stages_[1][coarse_level]->const_array(mfi);
      const auto k3 = rk4_rhs_stages_[2][coarse_level]->const_array(mfi);
      const auto k4 = rk4_rhs_stages_[3][coarse_level]->const_array(mfi);
      amrex::ParallelFor(
         box,
         num_components_,
         [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
            const amrex::Real stages[4] = {
               k1(i, j, k, component),
               k2(i, j, k, component),
               k3(i, j, k, component),
               k4(i, j, k, component),
            };
            amrex::Real state_sum = 0.0;
            amrex::Real first_time_derivative = 0.0;
            amrex::Real second_time_derivative = 0.0;
            amrex::Real third_time_derivative = 0.0;
            for (int stage = 0; stage < 4; ++stage) {
               state_sum += b[stage] * stages[stage];
               first_time_derivative += c[stage] * stages[stage];
               second_time_derivative += d[stage] * stages[stage];
               third_time_derivative += e[stage] * stages[stage];
            }
            amrex::Real value = initial(i, j, k, component) + coarse_dt * state_sum;
            if (rk_stage == 2) {
               value += coarse_dt * 0.5 * r * first_time_derivative;
            } else if (rk_stage == 3 || rk_stage == 4) {
               const amrex::Real r2 = r * r;
               const amrex::Real r3 = r2 * r;
               const amrex::Real at = rk_stage == 3 ? 0.5 * r : r;
               const amrex::Real att = rk_stage == 3 ? 0.25 * r2 : 0.5 * r2;
               const amrex::Real attt = rk_stage == 3 ? 0.0625 * r3 : 0.125 * r3;
               const amrex::Real stage_correction =
                  (rk_stage == 3 ? -4.0 : 4.0) * (stages[2] - stages[1]);
               value += coarse_dt * (
                  at * first_time_derivative
                  + att * second_time_derivative
                  + attt * (third_time_derivative + stage_correction)
               );
            }
            output(i, j, k, component) = value;
         }
      );
   }
   set_rk4_stage_cache(
      coarse_level, rk_stage, fine_step_start, coarse_time, coarse_dt
   );
}

void EvolutionSectorStorage::average_down_level(
   int fine_level,
   amrex::IntVect const& refinement_ratio
)
{
   if (!active_ || fine_level <= 0 || !state_.at(fine_level)
       || !state_.at(fine_level - 1)) {
      return;
   }
   if (centering_ == GridFieldCentering::Vertex) {
      amrex::average_down_nodal(
         *state_[fine_level], *state_[fine_level - 1], refinement_ratio
      );
      return;
   }
   amrex::average_down(
      *state_[fine_level],
      *state_[fine_level - 1],
      0,
      num_components_,
      refinement_ratio
   );
}

void EvolutionSectorStorage::invalidate_rk4_stage_cache(int level)
{
   if (level < 0 || level >= static_cast<int>(rk4_cache_stage_.size())) {
      return;
   }
   rk4_cache_stage_[level] = -1;
   rk4_cache_fine_start_[level] = 0.0;
   rk4_cache_time_[level] = 0.0;
   rk4_cache_dt_[level] = 0.0;
}

bool EvolutionSectorStorage::rk4_stage_cache_matches(
   int level,
   int stage,
   amrex::Real fine_start,
   amrex::Real coarse_time,
   amrex::Real coarse_dt
) const
{
   if (level < 0 || level >= static_cast<int>(rk4_cache_stage_.size())) {
      return false;
   }
   const auto close = [](amrex::Real left, amrex::Real right) {
      const amrex::Real scale = amrex::max(
         1.0, amrex::max(amrex::Math::abs(left), amrex::Math::abs(right))
      );
      return amrex::Math::abs(left - right)
         <= 64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
   };
   return rk4_cache_stage_[level] == stage
      && close(rk4_cache_fine_start_[level], fine_start)
      && close(rk4_cache_time_[level], coarse_time)
      && close(rk4_cache_dt_[level], coarse_dt);
}

void EvolutionSectorStorage::set_rk4_stage_cache(
   int level,
   int stage,
   amrex::Real fine_start,
   amrex::Real coarse_time,
   amrex::Real coarse_dt
)
{
   rk4_cache_stage_.at(level) = stage;
   rk4_cache_fine_start_.at(level) = fine_start;
   rk4_cache_time_.at(level) = coarse_time;
   rk4_cache_dt_.at(level) = coarse_dt;
}

} // namespace pyfluid_amrex
