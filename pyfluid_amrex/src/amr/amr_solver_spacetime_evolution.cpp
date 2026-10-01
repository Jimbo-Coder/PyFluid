#include "pyfluid_amrex/spacetime_derivatives.hpp"
#include "amr/amr_solver_base.hpp"

#include <AMReX_BC_TYPES.H>
#include <AMReX_PhysBCFunct.H>

#include "pyfluid_amrex/centering_transfer.hpp"
#include "pyfluid_amrex/fccz4_runtime.hpp"
#include "pyfluid_amrex/reference_chart.hpp"
#include "pyfluid_amrex/initial_data/provider.hpp"
#include "pyfluid_amrex/relativistic_boundary.hpp"
#include "pyfluid_amrex/spacetime_coupling.hpp"
#include "pyfluid_amrex/spacetime_excision.hpp"
#include "pyfluid_amrex/z4c_excision_derivatives.hpp"
#include "pyfluid_amrex/z4c_evolution_pointwise.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

// Stage orchestration only. Pointwise metric, matter, and Z4c algebra is
// generated and shared with fixed-spacetime relativistic evolution.

bool AmrSolverBase::uses_analytic_fixed_spacetime_kernels() const noexcept
{
   return analytic_fixed_spacetime_kernels_;
}

void AmrSolverBase::select_analytic_fixed_spacetime_kernels(
   std::string const& source_ic,
   std::string const& source_spacetime
) noexcept
{
   analytic_fixed_spacetime_kernels_ =
      analytic_fixed_spacetime_kernels_selected(
         cfg_, source_ic, source_spacetime
      );
}

void AmrSolverBase::project_spacetime_state(amrex::MultiFab& state)
   {
      debug_check_mfab_finite(state, Z4C_NUM_COMPONENTS, amrex::IntVect(0),
         "Z4c before algebraic projection");
      debug_check_spacetime_metric(state, amrex::IntVect(0),
         cfg_.debug_check_finite || cfg_.debug_check_physical, true);
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
      debug_check_mfab_finite(state, Z4C_NUM_COMPONENTS, amrex::IntVect(0),
         "Z4c after algebraic projection");
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
      debug_check_mfab_finite(*state[lev], Z4C_NUM_COMPONENTS,
         state[lev]->nGrowVect(), "Z4c stage ghost fill");
      transfer_grid_field_centering<
         4, AMREX_SPACEDIM, Z4cPositiveScalarCenteringRepresentation
      >(
         *spacetime_coupling_.cell_spacetime()[lev],
         *state[lev],
         Geom(lev),
         true,
         spacetime_excision_region(cfg_)
      );
      debug_check_mfab_finite(*spacetime_coupling_.cell_spacetime()[lev],
         Z4C_NUM_COMPONENTS, spacetime_coupling_.cell_spacetime()[lev]->nGrowVect(),
         "Z4c vertex-to-cell transfer");
      debug_check_spacetime_metric(*spacetime_coupling_.cell_spacetime()[lev],
         // Primitive recovery evaluates metrics in the fluid ghost halo.
         // Additional excision storage supplies linear derivative samples,
         // not pointwise metrics; remote extrapolated corners need only be
         // finite and are covered by the preceding check.
         amrex::IntVect(cfg_.ng),
         cfg_.debug_check_finite || cfg_.debug_check_physical);
      spacetime_coupling_.mark_cell_spacetime_valid(lev);
   }
}

void AmrSolverBase::apply_stored_spacetime_fluid_boundaries_range(
   LevelData& state,
   int lev_begin,
   int lev_end,
   amrex::Real time
)
{
   if (!spacetime_state_.active()) {
      return;
   }
   const bool analytic_fixed_spacetime =
      uses_analytic_fixed_spacetime_kernels();
   for (int lev = lev_begin; lev < lev_end; ++lev) {
      if (physics_system_ == PhysicalSystem::RHD) {
         if (analytic_fixed_spacetime) {
            apply_fixed_metric_state_boundary<PhysicalSystem::RHD, EOSRuntime>(
               *state[lev],
               Geom(lev),
               boundary_data_,
               fixed_spacetime_parameters(cfg_),
               eos_params(cfg_),
               time,
               mesh_coordinate_code(cfg_.mesh_coordinate),
               mesh_axes_layout(cfg_)
            );
         } else {
            apply_stored_metric_state_boundary<PhysicalSystem::RHD, EOSRuntime>(
               *state[lev], Geom(lev), boundary_data_, eos_params(cfg_),
               *spacetime_coupling_.cell_spacetime()[lev]
            );
         }
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
      spacetime_state_.state(), lev, lev + 1, time, -1.0_rt, 0
   );
   apply_stored_spacetime_fluid_boundaries_range(
      fluid_state_.state(), lev, lev + 1, time
   );
}

void AmrSolverBase::fill_spacetime_matter_range(int lev_begin, int lev_end)
{
   for (int lev = lev_begin; lev < lev_end; ++lev) {
      if (!spacetime_includes_matter(cfg_)) {
         spacetime_coupling_.vertex_matter()[lev]->setVal(0.0_rt);
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
      amrex::Vector<amrex::Real> fine_time{0.0_rt};
      if (lev == 0) {
         amrex::FillPatchSingleLevel(
            cell_matter,
            cell_matter.nGrowVect(),
            0.0_rt,
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
         amrex::Vector<amrex::Real> coarse_time{0.0_rt};
         amrex::FillPatchTwoLevels(
            cell_matter,
            cell_matter.nGrowVect(),
            0.0_rt,
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
         vertex_matter, cell_matter, Geom(lev), false,
         spacetime_excision_region(cfg_)
      );
   }
}

void AmrSolverBase::add_spacetime_ko_dissipation(
   amrex::MultiFab const& state,
   int lev
)
   {
      const int derivative_order = cfg_.spacetime_derivative_order;
      const amrex::Real coefficient = cfg_.spacetime_ko_dissipation;
      if (coefficient == 0.0_rt) {
         return;
      }
      amrex::MultiFab& rhs = *spacetime_state_.rhs()[lev];
      const auto excision = excision_grid_view(spacetime_excision_region(cfg_), Geom(lev), state.ixType());
      const auto spacing = Geom(lev).CellSizeArray();
      const ReferenceChartGeometry reference_chart(Geom(lev), cfg_);
      const bool use_reference_metric = cfg_.spacetime_formulation == "fccz4";
      const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
         1.0_rt / spacing[0],
#if (AMREX_SPACEDIM >= 2)
         1.0_rt / spacing[1],
#else
         1.0_rt,
#endif
#if (AMREX_SPACEDIM == 3)
         1.0_rt / spacing[2],
#else
         1.0_rt,
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
               auto physical_inverse_spacing = inv_spacing;
               if (use_reference_metric) {
                  const auto chart = reference_chart.at(i, j, k);
                  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                     physical_inverse_spacing[d] *= reference_inverse_scale(chart, d);
                  }
               }
               amrex::Real dissipation = spacetime_dissipation<0>(
                  derivative_order, input, excision, i, j, k, component, physical_inverse_spacing[0]
               );
#if (AMREX_SPACEDIM >= 2)
               dissipation += spacetime_dissipation<1>(
                  derivative_order, input, excision, i, j, k, component, physical_inverse_spacing[1]
               );
#endif
#if (AMREX_SPACEDIM == 3)
               dissipation += spacetime_dissipation<2>(
                  derivative_order, input, excision, i, j, k, component, physical_inverse_spacing[2]
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
      if (cfg_.spacetime_formulation == "fccz4") {
         compute_fccz4_rhs(state, matter, rhs, Geom(lev), cfg_, stage_time);
      } else {
      const auto spacing = Geom(lev).CellSizeArray();
      const amrex::GpuArray<amrex::Real, 3> inv_spacing{
         1.0_rt / spacing[0],
#if (AMREX_SPACEDIM >= 2)
         1.0_rt / spacing[1],
#else
         1.0_rt,
#endif
#if (AMREX_SPACEDIM >= 3)
         1.0_rt / spacing[2]
#else
         1.0_rt
#endif
      };
      const int derivative_order = cfg_.spacetime_derivative_order;
      const Z4cEvolutionParameters parameters = z4c_evolution_parameters(cfg_);
      const Z4cLapseGauge lapse_gauge = z4c_lapse_gauge(cfg_);
      const Z4cShiftGauge shift_gauge = z4c_shift_gauge(cfg_);
      const auto excision = excision_grid_view(spacetime_excision_region(cfg_), Geom(lev), state.ixType());

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
                  if (excision.region.enabled() && !excision(i, j, k, 0)) {
                     for (int component = 0; component < Z4C_NUM_COMPONENTS; ++component) {
                        rhs_array(i, j, k, component) = 0.0_rt;
                     }
                     return;
                  }
                  const Z4cState local_state = load_z4c_state(state_array, i, j, k);
                  if (excision.region.enabled()) {
                     const bool inner_boundary = !excision(i - 1, j, k, 0) || !excision(i + 1, j, k, 0)
                        || !excision(i, j - 1, k, 0) || !excision(i, j + 1, k, 0)
                        || !excision(i, j, k - 1, 0) || !excision(i, j, k + 1, 0);
                     if (inner_boundary) {
                        auto normal = excision.position(i, j, k);
                        amrex::Real radius_squared = 0.0_rt;
                        for (int direction = 0; direction < 3; ++direction) {
                           normal[direction] -= excision.region.center[direction];
                           radius_squared += normal[direction] * normal[direction];
                        }
                        const amrex::Real radius = std::sqrt(radius_squared);
                        for (int direction = 0; direction < 3; ++direction) normal[direction] /= radius;
                        if (!(z4c_excision_outflow_margin(local_state, normal, parameters,
                              lapse_gauge, shift_gauge) > 0.0_rt)) {
                           amrex::Abort("spacetime excision has an incoming physical or gauge mode; "
                              "resolve a smaller inner boundary or use an outflow-compatible gauge");
                        }
                     }
                  }
                  const auto derivatives = gather_spacetime_derivatives(
                     derivative_order, state_array, i, j, k,
                     inv_spacing, local_state, excision);
                  store_z4c_rhs(
                     rhs_array,
                     i,
                     j,
                     k,
                     generated::evaluate_z4c_rhs(
                        local_state,
                        derivatives,
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
      }
      add_spacetime_ko_dissipation(state, lev);
      apply_z4c_outer_boundary_rhs(state, rhs, Geom(lev), cfg_);
      debug_check_mfab_finite(rhs, Z4C_NUM_COMPONENTS, amrex::IntVect(0),
         "Z4c RHS assembly");
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
