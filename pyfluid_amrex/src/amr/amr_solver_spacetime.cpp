#include "amr/amr_solver_base.hpp"
#include "pyfluid_amrex/initial_data.hpp"

#include <AMReX_Math.H>
#include <AMReX_MultiFabUtil.H>

#include <cmath>

#include "pyfluid_amrex/spacetime_coupling.hpp"
#include "pyfluid_amrex/initial_data/materialization.hpp"
#include "pyfluid_amrex/nodal_fill.hpp"
#include "pyfluid_amrex/nodal_interpolater.hpp"
#include "pyfluid_amrex/z4c_constraint_pointwise.hpp"

namespace pyfluid_amrex {

amrex::MFInterpolater* spacetime_interpolater()
{
   return fourth_order_nodal_interpolater();
}

void AmrSolverBase::fill_spacetime_level_boundaries(
   LevelData& data,
   int lev,
   amrex::Real time,
   bool synchronous_levels,
   amrex::Real fine_step_start,
   int rk_stage
)
{
   if (!spacetime_state_.active() || !data.at(lev)) {
      return;
   }
   SpacetimeFillPatchPhysBC fine_bc(
      Geom(lev), spacetime_fillpatch_bcs_, spacetime_fillpatch_func_
   );
   if (lev == 0) {
      amrex::Vector<amrex::MultiFab*> source{data[0].get()};
      amrex::Vector<amrex::Real> source_time{time};
      amrex::FillPatchSingleLevel(
         *data[0],
         data[0]->nGrowVect(),
         time,
         source,
         source_time,
         0,
         0,
         spacetime_state_.num_components(),
         Geom(0),
         fine_bc,
         0
      );
      return;
   }

   amrex::Vector<amrex::MultiFab*> coarse_data;
   amrex::Vector<amrex::Real> coarse_time;
   if (!spacetime_state_.evolves()) {
      coarse_data.push_back(data[lev - 1].get());
      coarse_time.push_back(time);
   } else if (synchronous_levels) {
      coarse_data.push_back(data[lev - 1].get());
      coarse_time.push_back(time);
   } else if (
      rk_stage > 0
      && spacetime_state_.rk4_stage_data_ready(lev - 1, level_dts_[lev - 1])
   ) {
      spacetime_state_.fill_rk4_dense_state(
         lev - 1,
         fine_step_start,
         rk_stage,
         level_old_times_[lev - 1],
         level_times_[lev - 1],
         level_dts_[lev - 1],
         level_dts_[lev]
      );
      coarse_data.push_back(
         spacetime_state_.rk4_coarse_stage_state()[lev - 1].get()
      );
      coarse_time.push_back(time);
   } else if (coarse_time_interval_contains(lev - 1, time)) {
      coarse_data.push_back(spacetime_state_.old_state()[lev - 1].get());
      coarse_data.push_back(spacetime_state_.state()[lev - 1].get());
      coarse_time.push_back(level_old_times_[lev - 1]);
      coarse_time.push_back(level_times_[lev - 1]);
      time = clamp_to_coarse_time_interval(lev - 1, time);
   } else {
      coarse_data.push_back(data[lev - 1].get());
      coarse_time.push_back(time);
   }
   amrex::Vector<amrex::MultiFab*> fine_data{data[lev].get()};
   amrex::Vector<amrex::Real> fine_time{time};
   SpacetimeFillPatchPhysBC coarse_bc(
      Geom(lev - 1), spacetime_fillpatch_bcs_, spacetime_fillpatch_func_
   );
   fill_nodal_two_levels(
      *data[lev],
      data[lev]->nGrowVect(),
      time,
      coarse_data,
      coarse_time,
      fine_data,
      fine_time,
      Geom(lev - 1),
      Geom(lev),
      coarse_bc,
      fine_bc,
      refRatio(lev - 1),
      spacetime_fillpatch_bcs_,
      spacetime_state_.num_components()
   );
}

void AmrSolverBase::fill_all_spacetime_boundaries(LevelData& data, amrex::Real time)
{
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      fill_spacetime_level_boundaries(data, lev, time);
   }
}

void AmrSolverBase::initialize_spacetime_level(int lev, amrex::Real time)
{
   if (!spacetime_state_.active()) {
      return;
   }
   initialize_z4c_state(
      *spacetime_state_.state()[lev], Geom(lev), cfg_, time,
      initial_data_provider_.view()
   );
   spacetime_coupling_.invalidate_cell_spacetime(lev);
   fill_spacetime_level_boundaries(spacetime_state_.state(), lev, time);
   complete_z4c_contracted_connections(*spacetime_state_.state()[lev], Geom(lev));
   fill_spacetime_level_boundaries(spacetime_state_.state(), lev, time);
}

void AmrSolverBase::materialize_constraint_solved_initial_data()
{
   const bool uses_constraint_solve = cfg_.spacetime_initial_data == "ctt"
      || cfg_.spacetime_initial_data == "cttk"
      || cfg_.spacetime_initial_data == "xcts";
   if (!uses_constraint_solve
       || !initial_data_provider_.view().fields.spacetime_from_id) {
      return;
   }
   if (!spacetime_state_.active()) {
      amrex::Abort(
         "constraint-solved initial data requires stored fixed or evolved spacetime"
      );
   }

   InitialDataEllipticHierarchy hierarchy;
   amrex::Vector<amrex::MultiFab*> fluid;
   amrex::Vector<amrex::MultiFab const*> primitive;
   amrex::Vector<amrex::MultiFab*> spacetime;
   amrex::Vector<amrex::MultiFab*> cell_spacetime;
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      hierarchy.geometry.push_back(Geom(lev));
      hierarchy.grids.push_back(boxArray(lev));
      hierarchy.distribution.push_back(DistributionMap(lev));
      if (initial_data_provider_.view().has_fluid()) {
         initialize_relativistic_primitive_cache_from_provider(
            *primitive_[lev], Geom(lev), cfg_, initial_data_provider_.view()
         );
      } else {
         fill_level_primitive_cache(lev, 0.0);
      }
      fluid.push_back(fluid_state_.state()[lev].get());
      primitive.push_back(primitive_[lev].get());
      spacetime.push_back(spacetime_state_.state()[lev].get());
      cell_spacetime.push_back(spacetime_coupling_.cell_spacetime()[lev].get());
      if (lev < finestLevel()) {
         hierarchy.refinement_ratio.push_back(refRatio(lev));
      }
   }
   hierarchy.cell_interpolater = state_interpolater();

   ConformalConstraintFormulation formulation =
      ConformalConstraintFormulation::ctt;
   if (cfg_.spacetime_initial_data == "cttk") {
      formulation = ConformalConstraintFormulation::cttk;
   } else if (cfg_.spacetime_initial_data == "xcts") {
      formulation = ConformalConstraintFormulation::xcts;
   }
   const ConformalMaterializationResult result = materialize_conformal_initial_data(
      hierarchy,
      fluid,
      primitive,
      spacetime,
      cell_spacetime,
      physics_system_,
      eos_method_,
      eos_params(cfg_),
      initial_data_provider_.view().fields,
      formulation
   );
   if (amrex::ParallelDescriptor::IOProcessor()) {
      amrex::Print()
         << cfg_.spacetime_initial_data
         << " initial data: matter_iterations=" << result.matter_iterations
         << " relative_matter_update=" << result.relative_matter_update
         << " coupled_iterations=" << result.constraints.coupled_iterations
         << " hamiltonian_residual=" << result.constraints.hamiltonian_residual
         << " lapse_residual=" << result.constraints.lapse_residual
         << " momentum_residual=" << result.constraints.momentum_residual << '\n';
   }

   for (int lev = 0; lev <= finestLevel(); ++lev) {
      spacetime_coupling_.invalidate_cell_spacetime(lev);
   }
   fill_all_spacetime_boundaries(spacetime_state_.state(), 0.0);
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      complete_z4c_contracted_connections(
         *spacetime_state_.state()[lev], Geom(lev)
      );
   }
   fill_all_spacetime_boundaries(spacetime_state_.state(), 0.0);
   prepare_spacetime_coupling_range(
      spacetime_state_.state(), 0, finestLevel() + 1, 0.0, -1.0, 0
   );
   apply_stored_spacetime_fluid_boundaries_range(
      fluid_state_.state(), 0, finestLevel() + 1
   );
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      copy_level_state_to_old(lev);
   }
}

void AmrSolverBase::fill_new_spacetime_level_from_coarse(int lev, amrex::Real time)
{
   if (!spacetime_state_.active()) {
      return;
   }
   if (!spacetime_state_.evolves() || lev == 0 || time == 0.0) {
      initialize_spacetime_level(lev, time);
      return;
   }
   fill_spacetime_level_boundaries(spacetime_state_.state(), lev - 1, time);
   SpacetimeFillPatchPhysBC coarse_bc(
      Geom(lev - 1), spacetime_fillpatch_bcs_, spacetime_fillpatch_func_
   );
   SpacetimeFillPatchPhysBC fine_bc(
      Geom(lev), spacetime_fillpatch_bcs_, spacetime_fillpatch_func_
   );
   amrex::InterpFromCoarseLevel(
      *spacetime_state_.state()[lev],
      spacetime_state_.state()[lev]->nGrowVect(),
      time,
      *spacetime_state_.state()[lev - 1],
      0,
      0,
      spacetime_state_.num_components(),
      Geom(lev - 1),
      Geom(lev),
      coarse_bc,
      0,
      fine_bc,
      0,
      refRatio(lev - 1),
      spacetime_interpolater(),
      spacetime_fillpatch_bcs_,
      0
   );
}

void AmrSolverBase::fill_remade_spacetime_level(
   int lev,
   amrex::Real time,
   amrex::MultiFab& old_state
)
{
   if (!spacetime_state_.active()) {
      return;
   }
   if (!spacetime_state_.evolves()) {
      initialize_spacetime_level(lev, time);
      return;
   }
   if (lev == 0) {
      spacetime_state_.state()[0]->ParallelCopy(
         old_state,
         0,
         0,
         spacetime_state_.num_components(),
         0,
         0,
         Geom(0).periodicity()
      );
      fill_spacetime_level_boundaries(spacetime_state_.state(), 0, time);
      return;
   }
   fill_spacetime_level_boundaries(spacetime_state_.state(), lev - 1, time);
   amrex::Vector<amrex::MultiFab*> coarse_data{
      spacetime_state_.state()[lev - 1].get()
   };
   amrex::Vector<amrex::Real> coarse_time{time};
   amrex::Vector<amrex::MultiFab*> fine_data{&old_state};
   amrex::Vector<amrex::Real> fine_time{time};
   SpacetimeFillPatchPhysBC coarse_bc(
      Geom(lev - 1), spacetime_fillpatch_bcs_, spacetime_fillpatch_func_
   );
   SpacetimeFillPatchPhysBC fine_bc(
      Geom(lev), spacetime_fillpatch_bcs_, spacetime_fillpatch_func_
   );
   fill_nodal_two_levels(
      *spacetime_state_.state()[lev],
      spacetime_state_.state()[lev]->nGrowVect(),
      time,
      coarse_data,
      coarse_time,
      fine_data,
      fine_time,
      Geom(lev - 1),
      Geom(lev),
      coarse_bc,
      fine_bc,
      refRatio(lev - 1),
      spacetime_fillpatch_bcs_,
      spacetime_state_.num_components()
   );
}

void AmrSolverBase::average_down_spacetime_level(int fine_lev)
{
   if (spacetime_state_.evolves()) {
      spacetime_state_.average_down_level(fine_lev, refRatio(fine_lev - 1));
      spacetime_coupling_.invalidate_cell_spacetime(fine_lev - 1);
   }
}

void AmrSolverBase::average_down_spacetime()
{
   for (int fine_lev = finestLevel(); fine_lev > 0; --fine_lev) {
      average_down_spacetime_level(fine_lev);
   }
}

SpacetimeConstraintDiagnostics
AmrSolverBase::compute_spacetime_constraint_diagnostics(amrex::Real time)
{
   SpacetimeConstraintDiagnostics diagnostics;
   if (!spacetime_state_.active()) {
      return diagnostics;
   }

   fill_all_spacetime_boundaries(spacetime_state_.state(), time);
   amrex::Real hamiltonian_squared_integral = 0.0;
   amrex::Real composite_volume = 0.0;
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      amrex::MultiFab const& state = *spacetime_state_.state()[lev];
      amrex::iMultiFab valid(state.boxArray(), state.DistributionMap(), 1, 0);
      if (lev < finestLevel()) {
         const amrex::BoxArray fine_nodes = amrex::convert(
            boxArray(lev + 1), amrex::IntVect::TheNodeVector()
         );
         valid = amrex::makeFineMask(
            state,
            fine_nodes,
            refRatio(lev),
            Geom(lev).periodicity(),
            1,
            0
         );
      } else {
         valid.setVal(1);
      }
      const std::unique_ptr<amrex::iMultiFab> owner = state.OwnerMask(
         Geom(lev).periodicity()
      );
      amrex::MultiFab norms(
         state.boxArray(), state.DistributionMap(), 7, 0
      );
      norms.setVal(0.0);
      const amrex::Box nodal_domain = amrex::convert(
         Geom(lev).Domain(), amrex::IntVect::TheNodeVector()
      );
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
         amrex::Box box = mfi.tilebox();
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            if (!Geom(lev).isPeriodic(direction)
                && box.smallEnd(direction) == nodal_domain.smallEnd(direction)) {
               box.setSmall(direction, box.smallEnd(direction) + 2);
            }
            if (!Geom(lev).isPeriodic(direction)
                && box.bigEnd(direction) == nodal_domain.bigEnd(direction)) {
               box.setBig(direction, box.bigEnd(direction) - 2);
            }
         }
         const auto input = state.const_array(mfi);
         const auto output = norms.array(mfi);
         const auto valid_points = valid.const_array(mfi);
         const auto owned_points = owner->const_array(mfi);
         const bool matter_coupled = spacetime_includes_matter(cfg_);
         const auto matter = spacetime_coupling_.vertex_matter()[lev]->const_array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (valid_points(i, j, k, 0) == 0
                || owned_points(i, j, k, 0) == 0) {
               return;
            }
            const Z4cMatter point_matter = matter_coupled
               ? load_eulerian_matter(matter, i, j, k)
               : vacuum_z4c_matter();
            const Z4cConstraints c = evaluate_z4c_constraints_at<4, AMREX_SPACEDIM>(
               input, i, j, k, inv_spacing, point_matter
            );
            output(i, j, k, 0) = amrex::Math::abs(c.hamiltonian);
            output(i, j, k, 1) = c.momentum1 * c.momentum1
               + c.momentum2 * c.momentum2 + c.momentum3 * c.momentum3;
            output(i, j, k, 2) = c.z_tilde1 * c.z_tilde1
               + c.z_tilde2 * c.z_tilde2 + c.z_tilde3 * c.z_tilde3;
            output(i, j, k, 3) = amrex::Math::abs(c.theta);
            output(i, j, k, 4) = amrex::Math::abs(
               c.log_conformal_determinant
            );
            output(i, j, k, 5) = amrex::Math::abs(c.trace_a_tilde);
            output(i, j, k, 6) = 1.0;
         });
      }
      diagnostics.hamiltonian_linf = amrex::max(
         diagnostics.hamiltonian_linf, norms.norm0(0, 0, false)
      );
      const amrex::Real level_volume = cartesian_cell_volume(Geom(lev));
      const amrex::Real level_hamiltonian_norm = norms.norm2(0);
      hamiltonian_squared_integral += level_volume
         * level_hamiltonian_norm * level_hamiltonian_norm;
      composite_volume += level_volume * norms.sum(6);
      diagnostics.momentum_linf = amrex::max(
         diagnostics.momentum_linf, std::sqrt(norms.norm0(1, 0, false))
      );
      diagnostics.z_linf = amrex::max(
         diagnostics.z_linf, std::sqrt(norms.norm0(2, 0, false))
      );
      diagnostics.theta_linf = amrex::max(
         diagnostics.theta_linf, norms.norm0(3, 0, false)
      );
      diagnostics.log_conformal_determinant_linf = amrex::max(
         diagnostics.log_conformal_determinant_linf,
         norms.norm0(4, 0, false)
      );
      diagnostics.trace_a_tilde_linf = amrex::max(
         diagnostics.trace_a_tilde_linf, norms.norm0(5, 0, false)
      );
   }
   if (composite_volume > 0.0) {
      diagnostics.hamiltonian_l2 = std::sqrt(
         hamiltonian_squared_integral / composite_volume
      );
   }
   return diagnostics;
}

} // namespace pyfluid_amrex
