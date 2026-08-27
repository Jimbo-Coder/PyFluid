#include "amr/amr_solver_base.hpp"

#include <AMReX_BC_TYPES.H>
#include <AMReX_PhysBCFunct.H>

#include "pyfluid_amrex/centering_transfer.hpp"
#include "pyfluid_amrex/initial_data/provider.hpp"
#include "pyfluid_amrex/relativistic_boundary.hpp"
#include "pyfluid_amrex/spacetime_coupling.hpp"
#include "pyfluid_amrex/z4c_evolution_pointwise.hpp"

namespace pyfluid_amrex {

// Stage orchestration only. Pointwise metric, matter, and Z4c algebra is
// generated and shared with fixed-spacetime relativistic evolution.

bool AmrSolverBase::uses_analytic_fixed_spacetime_kernels() const noexcept
{
   return analytic_fixed_spacetime_kernels_;
}

void AmrSolverBase::project_spacetime_state(amrex::MultiFab& state)
   {
      const amrex::Real chi_floor = cfg_.spacetime_chi_floor;
      const amrex::Real lapse_floor = cfg_.spacetime_lapse_floor;
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
                  project_z4c_state_at(
                     state_array, i, j, k, chi_floor, lapse_floor
                  );
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
   const bool fixed_spacetime_from_stored_state =
      !spacetime_state_.evolves()
      && !uses_analytic_fixed_spacetime_kernels()
      && cfg_.spacetime_fixed_metric == "minkowski"
      && (!cfg_.initial_data_input_file.empty()
          || cfg_.spacetime_initial_data == "ctt"
          || cfg_.spacetime_initial_data == "cttk"
          || cfg_.spacetime_initial_data == "xcts");
   for (int lev = lev_begin; lev < lev_end; ++lev) {
      if (!spacetime_state_.evolves()
          && spacetime_coupling_.cell_spacetime_valid(lev)) {
         continue;
      }
      if (!spacetime_state_.evolves()
          && !fixed_spacetime_from_stored_state) {
         initialize_z4c_cell_state(
            *spacetime_coupling_.cell_spacetime()[lev],
            Geom(lev),
            cfg_,
            stage_time,
            initial_data_provider_->view()
         );
         spacetime_coupling_.mark_cell_spacetime_valid(lev);
         continue;
      }
      if (spacetime_state_.evolves()) {
         project_spacetime_state(*state[lev]);
      }
      fill_spacetime_level_boundaries(
         state,
         lev,
         stage_time,
         synchronous_levels,
         fine_step_start,
         rk_stage
      );
      transfer_grid_field_centering<4, AMREX_SPACEDIM>(
         *spacetime_coupling_.cell_spacetime()[lev],
         *state[lev],
         Geom(lev),
         true
      );
      spacetime_coupling_.mark_cell_spacetime_valid(lev);
   }
}

void AmrSolverBase::apply_stored_spacetime_fluid_boundaries_range(
   LevelData& state,
   int lev_begin,
   int lev_end
)
{
   if (!spacetime_state_.active()) {
      return;
   }
   for (int lev = lev_begin; lev < lev_end; ++lev) {
      if (physics_system_ == PhysicalSystem::RHD) {
         apply_stored_metric_state_boundary<PhysicalSystem::RHD, EOSRuntime>(
            *state[lev], Geom(lev), boundary_data_, eos_params(cfg_),
            *spacetime_coupling_.cell_spacetime()[lev]
         );
      } else if (physics_system_ == PhysicalSystem::RMHD) {
         apply_stored_metric_state_boundary<PhysicalSystem::RMHD, EOSRuntime>(
            *state[lev],
            Geom(lev),
            boundary_data_,
            eos_params(cfg_),
            *spacetime_coupling_.cell_spacetime()[lev]
         );
      }
   }
}

void AmrSolverBase::refresh_stored_spacetime_fluid_boundary(
   int lev,
   amrex::Real time
)
{
   if (!spacetime_state_.active()) {
      return;
   }
   prepare_spacetime_coupling_range(
      spacetime_state_.state(), lev, lev + 1, time, -1.0, 0
   );
   apply_stored_spacetime_fluid_boundaries_range(
      fluid_state_.state(), lev, lev + 1
   );
}

void AmrSolverBase::fill_spacetime_matter_range(int lev_begin, int lev_end)
{
   for (int lev = lev_begin; lev < lev_end; ++lev) {
      if (!spacetime_includes_matter(cfg_)) {
         spacetime_coupling_.vertex_matter()[lev]->setVal(0.0);
         continue;
      }
      amrex::MultiFab& cell_matter = *spacetime_coupling_.cell_matter()[lev];
      if (physics_system_ == PhysicalSystem::RHD) {
         fill_eulerian_matter_cells<PhysicalSystem::RHD, EOSRuntime>(
            cell_matter, *primitive_[lev],
            *spacetime_coupling_.cell_spacetime()[lev], eos_params(cfg_)
         );
      } else if (physics_system_ == PhysicalSystem::RMHD) {
         fill_eulerian_matter_cells<PhysicalSystem::RMHD, EOSRuntime>(
            cell_matter,
            *primitive_[lev],
            *spacetime_coupling_.cell_spacetime()[lev],
            eos_params(cfg_)
         );
      } else {
         amrex::Abort("evolved spacetime matter coupling requires RHD or RMHD");
      }
   }

   amrex::Vector<amrex::BCRec> boundary_conditions(
      EULERIAN_MATTER_NUM_COMPONENTS
   );
   for (amrex::BCRec& boundary : boundary_conditions) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         const int boundary_type = Geom(0).isPeriodic(direction)
            ? amrex::BCType::int_dir
            : amrex::BCType::foextrap;
         boundary.setLo(direction, boundary_type);
         boundary.setHi(direction, boundary_type);
      }
   }
   amrex::GpuBndryFuncFab<amrex::FabFillNoOp> boundary_function;
   using MatterPhysicalBoundary = amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<amrex::FabFillNoOp>
   >;
   for (int lev = lev_begin; lev < lev_end; ++lev) {
      if (!spacetime_includes_matter(cfg_)) {
         continue;
      }
      amrex::MultiFab& cell_matter = *spacetime_coupling_.cell_matter()[lev];
      cell_matter.OverrideSync(Geom(lev).periodicity());
      MatterPhysicalBoundary fine_boundary(
         Geom(lev), boundary_conditions, boundary_function
      );
      amrex::Vector<amrex::MultiFab*> fine_data{&cell_matter};
      amrex::Vector<amrex::Real> fine_time{0.0};
      if (lev == 0) {
         amrex::FillPatchSingleLevel(
            cell_matter,
            cell_matter.nGrowVect(),
            0.0,
            fine_data,
            fine_time,
            0,
            0,
            EULERIAN_MATTER_NUM_COMPONENTS,
            Geom(lev),
            fine_boundary,
            0
         );
      } else {
         MatterPhysicalBoundary coarse_boundary(
            Geom(lev - 1), boundary_conditions, boundary_function
         );
         amrex::Vector<amrex::MultiFab*> coarse_data{
            spacetime_coupling_.cell_matter()[lev - 1].get()
         };
         amrex::Vector<amrex::Real> coarse_time{0.0};
         amrex::FillPatchTwoLevels(
            cell_matter,
            cell_matter.nGrowVect(),
            0.0,
            coarse_data,
            coarse_time,
            fine_data,
            fine_time,
            0,
            0,
            EULERIAN_MATTER_NUM_COMPONENTS,
            Geom(lev - 1),
            Geom(lev),
            coarse_boundary,
            0,
            fine_boundary,
            0,
            refRatio(lev - 1),
            selected_state_interpolater(),
            boundary_conditions,
            0
         );
      }
      amrex::MultiFab& vertex_matter = *spacetime_coupling_.vertex_matter()[lev];
      transfer_grid_field_centering<4, AMREX_SPACEDIM>(
         vertex_matter, cell_matter, Geom(lev)
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
      const Z4cLapseGauge lapse_gauge = z4c_lapse_gauge(cfg_);
      const Z4cShiftGauge shift_gauge = z4c_shift_gauge(cfg_);

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
                     evaluate_z4c_rhs_at<4, AMREX_SPACEDIM>(
                        state_array,
                        i,
                        j,
                        k,
                        inv_spacing,
                        load_eulerian_matter(matter_array, i, j, k),
                        stage_time,
                        parameters,
                        lapse_gauge,
                        shift_gauge
                     )
                  );
               }
            );
         }
      }
      add_spacetime_ko_dissipation(state, lev);
   }

void AmrSolverBase::compute_spacetime_rhs_gauge_range(
      LevelData& input,
      int lev_begin,
      int lev_end,
      amrex::Real stage_time
   )
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         compute_spacetime_rhs_level(input, lev, stage_time);
      }
   }

void AmrSolverBase::compute_spacetime_rhs_range(
   LevelData& input,
   int lev_begin,
   int lev_end,
   amrex::Real stage_time
)
{
   compute_spacetime_rhs_gauge_range(input, lev_begin, lev_end, stage_time);
}

} // namespace pyfluid_amrex
