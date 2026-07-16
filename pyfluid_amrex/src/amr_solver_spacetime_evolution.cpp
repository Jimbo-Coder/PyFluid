#include "pyfluid_amrex/amr_solver_base.hpp"

#include "pyfluid_amrex/relativistic_boundary.hpp"
#include "pyfluid_amrex/spacetime_coupling.hpp"
#include "pyfluid_amrex/z4c_pointwise.hpp"

namespace pyfluid_amrex {

// Stage orchestration only. Pointwise metric, matter, and Z4c algebra is
// generated and shared with fixed-spacetime relativistic evolution.

void AmrSolverBase::project_spacetime_state(amrex::MultiFab& state)
   {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
         for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox();
            const auto state_array = state.array(mfi);
            amrex::ParallelFor(
               box,
               [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  project_z4c_state_at(state_array, i, j, k);
               }
            );
         }
      }
   }

void AmrSolverBase::project_spacetime_state_range(
      LevelData& state,
      int lev_begin,
      int lev_end
   )
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         project_spacetime_state(*state[lev]);
      }
   }

void AmrSolverBase::prepare_spacetime_coupling_range(
   LevelData& state,
   int lev_begin,
   int lev_end,
   amrex::Real stage_time,
   amrex::Real fine_step_start,
   int rk_stage
)
{
   const bool synchronous_levels = lev_end - lev_begin > 1;
   for (int lev = lev_begin; lev < lev_end; ++lev) {
      project_spacetime_state(*state[lev]);
      fill_spacetime_level_boundaries(
         state,
         lev,
         stage_time,
         synchronous_levels,
         fine_step_start,
         rk_stage
      );
      transfer_z4c_vertex_to_cell<4, AMREX_SPACEDIM>(
         *spacetime_coupling_.cell_spacetime()[lev], *state[lev]
      );
   }
}

void AmrSolverBase::apply_evolved_spacetime_fluid_boundaries_range(
   LevelData& state,
   int lev_begin,
   int lev_end
)
{
   if (!spacetime_matter_is_coupled(cfg_)) {
      return;
   }
   for (int lev = lev_begin; lev < lev_end; ++lev) {
      if (physics_system_ == PhysicalSystem::RHD) {
         const int eos_method = eos_method_code(cfg_);
         if (eos_method == EOSPolytrope) {
            apply_evolved_metric_state_boundary<PhysicalSystem::RHD, EOSPolytrope>(
               *state[lev], Geom(lev), boundary_data_, eos_params(cfg_),
               *spacetime_coupling_.cell_spacetime()[lev]
            );
         } else if (eos_method == EOSHybridPolytrope) {
            apply_evolved_metric_state_boundary<PhysicalSystem::RHD, EOSHybridPolytrope>(
               *state[lev], Geom(lev), boundary_data_, eos_params(cfg_),
               *spacetime_coupling_.cell_spacetime()[lev]
            );
         } else {
            apply_evolved_metric_state_boundary<PhysicalSystem::RHD, EOSIdealGas>(
               *state[lev], Geom(lev), boundary_data_, eos_params(cfg_),
               *spacetime_coupling_.cell_spacetime()[lev]
            );
         }
      } else if (physics_system_ == PhysicalSystem::RMHD) {
         apply_evolved_metric_state_boundary<PhysicalSystem::RMHD, EOSIdealGas>(
            *state[lev],
            Geom(lev),
            boundary_data_,
            eos_params(cfg_),
            *spacetime_coupling_.cell_spacetime()[lev]
         );
      }
   }
}

void AmrSolverBase::fill_spacetime_matter_range(int lev_begin, int lev_end)
{
   for (int lev = lev_begin; lev < lev_end; ++lev) {
      amrex::MultiFab& vertex_matter = *spacetime_coupling_.vertex_matter()[lev];
      if (!spacetime_matter_is_coupled(cfg_)) {
         vertex_matter.setVal(0.0);
         continue;
      }
      amrex::MultiFab& cell_matter = *spacetime_coupling_.cell_matter()[lev];
      if (physics_system_ == PhysicalSystem::RHD) {
         const int eos_method = eos_method_code(cfg_);
         if (eos_method == EOSPolytrope) {
            fill_eulerian_matter_cells<PhysicalSystem::RHD, EOSPolytrope>(
               cell_matter, *primitive_[lev],
               *spacetime_coupling_.cell_spacetime()[lev], eos_params(cfg_)
            );
         } else if (eos_method == EOSHybridPolytrope) {
            fill_eulerian_matter_cells<PhysicalSystem::RHD, EOSHybridPolytrope>(
               cell_matter, *primitive_[lev],
               *spacetime_coupling_.cell_spacetime()[lev], eos_params(cfg_)
            );
         } else {
            fill_eulerian_matter_cells<PhysicalSystem::RHD, EOSIdealGas>(
               cell_matter, *primitive_[lev],
               *spacetime_coupling_.cell_spacetime()[lev], eos_params(cfg_)
            );
         }
      } else if (physics_system_ == PhysicalSystem::RMHD) {
         fill_eulerian_matter_cells<PhysicalSystem::RMHD, EOSIdealGas>(
            cell_matter,
            *primitive_[lev],
            *spacetime_coupling_.cell_spacetime()[lev],
            eos_params(cfg_)
         );
      } else {
         amrex::Abort("evolved spacetime matter coupling requires RHD or RMHD");
      }
      transfer_matter_cell_to_vertex<4, AMREX_SPACEDIM>(
         vertex_matter, cell_matter
      );
   }
}

void AmrSolverBase::add_spacetime_ko_dissipation(
   amrex::MultiFab const& state,
   int lev
)
   {
      const amrex::Real coefficient = cfg_.spacetime_ko_dissipation;
      if (coefficient == 0.0) {
         return;
      }
      amrex::MultiFab& rhs = *spacetime_state_.rhs()[lev];
      const auto spacing = Geom(lev).CellSizeArray();
      const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
         1.0 / spacing[0],
#if (AMREX_SPACEDIM >= 2)
         1.0 / spacing[1],
#else
         1.0,
#endif
#if (AMREX_SPACEDIM == 3)
         1.0 / spacing[2],
#else
         1.0,
#endif
      }};
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto input = state.const_array(mfi);
         const auto output = rhs.array(mfi);
         amrex::ParallelFor(
            box,
            static_cast<int>(Z4C_NUM_COMPONENTS),
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
               amrex::Real dissipation = generated::kreiss_oliger<4, 0>(
                  input, i, j, k, component, inv_spacing[0]
               );
#if (AMREX_SPACEDIM >= 2)
               dissipation += generated::kreiss_oliger<4, 1>(
                  input, i, j, k, component, inv_spacing[1]
               );
#endif
#if (AMREX_SPACEDIM == 3)
               dissipation += generated::kreiss_oliger<4, 2>(
                  input, i, j, k, component, inv_spacing[2]
               );
#endif
               output(i, j, k, component) += coefficient * dissipation;
            }
         );
      }
   }

template <Z4cLapseGauge LapseGauge, Z4cShiftGauge ShiftGauge>
void AmrSolverBase::compute_spacetime_rhs_level(
      LevelData& input,
      int lev,
      amrex::Real stage_time
   )
   {
      amrex::MultiFab const& state = *input[lev];
      amrex::MultiFab const& matter = *spacetime_coupling_.vertex_matter()[lev];
      amrex::MultiFab& rhs = *spacetime_state_.rhs()[lev];
      const auto spacing = Geom(lev).CellSizeArray();
      const amrex::GpuArray<amrex::Real, 3> inv_spacing{
         1.0 / spacing[0],
#if (AMREX_SPACEDIM >= 2)
         1.0 / spacing[1],
#else
         1.0,
#endif
#if (AMREX_SPACEDIM >= 3)
         1.0 / spacing[2]
#else
         1.0
#endif
      };
      const Z4cEvolutionParameters parameters = z4c_evolution_parameters(cfg_);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
         for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox();
            const auto state_array = state.const_array(mfi);
            const auto matter_array = matter.const_array(mfi);
            const auto rhs_array = rhs.array(mfi);
            amrex::ParallelFor(
               box,
               [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  store_z4c_rhs(
                     rhs_array,
                     i,
                     j,
                     k,
                     evaluate_z4c_rhs_at<
                        4,
                        AMREX_SPACEDIM,
                        LapseGauge,
                        ShiftGauge
                     >(
                        state_array,
                        i,
                        j,
                        k,
                        inv_spacing,
                        load_eulerian_matter(matter_array, i, j, k),
                        stage_time,
                        parameters
                     )
                  );
               }
            );
         }
      }
      add_spacetime_ko_dissipation(state, lev);
   }

template <Z4cLapseGauge LapseGauge, Z4cShiftGauge ShiftGauge>
void AmrSolverBase::compute_spacetime_rhs_gauge_range(
      LevelData& input,
      int lev_begin,
      int lev_end,
      amrex::Real stage_time
   )
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         compute_spacetime_rhs_level<LapseGauge, ShiftGauge>(
            input, lev, stage_time
         );
      }
   }

template <Z4cLapseGauge LapseGauge>
void AmrSolverBase::dispatch_spacetime_rhs_shift_gauge(
   LevelData& input,
   int lev_begin,
   int lev_end,
   amrex::Real stage_time
)
{
   if (z4c_shift_gauge(cfg_) == Z4cShiftGauge::Frozen) {
      compute_spacetime_rhs_gauge_range<
         LapseGauge,
         Z4cShiftGauge::Frozen
      >(
         input,
         lev_begin,
         lev_end,
         stage_time
      );
   } else {
      compute_spacetime_rhs_gauge_range<
         LapseGauge,
         Z4cShiftGauge::GammaDriver
      >(
         input,
         lev_begin,
         lev_end,
         stage_time
      );
   }
}

void AmrSolverBase::compute_spacetime_rhs_range(
   LevelData& input,
   int lev_begin,
   int lev_end,
   amrex::Real stage_time
)
{
   switch (z4c_lapse_gauge(cfg_)) {
   case Z4cLapseGauge::Frozen:
      dispatch_spacetime_rhs_shift_gauge<Z4cLapseGauge::Frozen>(
         input, lev_begin, lev_end, stage_time
      );
      break;
   case Z4cLapseGauge::Harmonic:
      dispatch_spacetime_rhs_shift_gauge<Z4cLapseGauge::Harmonic>(
         input, lev_begin, lev_end, stage_time
      );
      break;
   case Z4cLapseGauge::OnePlusLog:
      dispatch_spacetime_rhs_shift_gauge<Z4cLapseGauge::OnePlusLog>(
         input, lev_begin, lev_end, stage_time
      );
      break;
   case Z4cLapseGauge::SlowStart:
      dispatch_spacetime_rhs_shift_gauge<Z4cLapseGauge::SlowStart>(
         input, lev_begin, lev_end, stage_time
      );
      break;
   }
}

} // namespace pyfluid_amrex
