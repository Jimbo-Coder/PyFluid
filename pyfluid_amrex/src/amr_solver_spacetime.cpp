#include "pyfluid_amrex/amr_solver_base.hpp"

#include <AMReX_Math.H>

#include <cmath>

#include "pyfluid_amrex/spacetime_coupling.hpp"
#include "pyfluid_amrex/nodal_interpolater.hpp"
#include "pyfluid_amrex/z4c_pointwise.hpp"

namespace pyfluid_amrex {

namespace {

amrex::MFInterpolater* spacetime_interpolater()
{
   return fourth_order_nodal_interpolater();
}

template <typename PhysicalBoundary>
void fill_nodal_two_levels(
   amrex::MultiFab& destination,
   amrex::IntVect const& num_ghost,
   amrex::Real time,
   amrex::Vector<amrex::MultiFab*> const& coarse_data,
   amrex::Vector<amrex::Real> const& coarse_times,
   amrex::Vector<amrex::MultiFab*> const& fine_data,
   amrex::Vector<amrex::Real> const& fine_times,
   amrex::Geometry const& coarse_geometry,
   amrex::Geometry const& fine_geometry,
   PhysicalBoundary& coarse_bc,
   PhysicalBoundary& fine_bc,
   amrex::IntVect const& refinement_ratio,
   amrex::Vector<amrex::BCRec> const& boundary_conditions,
   int num_components
)
{
   amrex::MultiFab coarse_at_time(
      coarse_data.front()->boxArray(),
      coarse_data.front()->DistributionMap(),
      num_components,
      0
   );
   amrex::FillPatchSingleLevel(
      coarse_at_time,
      amrex::IntVect(0),
      time,
      coarse_data,
      coarse_times,
      0,
      0,
      num_components,
      coarse_geometry,
      coarse_bc,
      0
   );
   amrex::InterpFromCoarseLevel(
      destination,
      num_ghost,
      time,
      coarse_at_time,
      0,
      0,
      num_components,
      coarse_geometry,
      fine_geometry,
      coarse_bc,
      0,
      fine_bc,
      0,
      refinement_ratio,
      spacetime_interpolater(),
      boundary_conditions,
      0
   );
   amrex::FillPatchSingleLevel(
      destination,
      num_ghost,
      time,
      fine_data,
      fine_times,
      0,
      0,
      num_components,
      fine_geometry,
      fine_bc,
      0
   );
}

} // namespace

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
   if (synchronous_levels) {
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
      tov_initial_data_.view()
   );
   fill_spacetime_level_boundaries(spacetime_state_.state(), lev, time);
   complete_z4c_contracted_connections(*spacetime_state_.state()[lev], Geom(lev));
   fill_spacetime_level_boundaries(spacetime_state_.state(), lev, time);
}

void AmrSolverBase::fill_new_spacetime_level_from_coarse(int lev, amrex::Real time)
{
   if (!spacetime_state_.active()) {
      return;
   }
   if (lev == 0 || time == 0.0) {
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
   if (spacetime_state_.active()) {
      spacetime_state_.average_down_level(fine_lev, refRatio(fine_lev - 1));
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
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      amrex::MultiFab const& state = *spacetime_state_.state()[lev];
      amrex::MultiFab norms(
         state.boxArray(), state.DistributionMap(), 6, 0
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
         const bool matter_coupled = spacetime_matter_is_coupled(cfg_);
         const auto matter = spacetime_coupling_.vertex_matter()[lev]->const_array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
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
         });
      }
      diagnostics.hamiltonian_linf = amrex::max(
         diagnostics.hamiltonian_linf, norms.norm0(0, 0, false)
      );
      const amrex::Long point_count = state.boxArray().numPts();
      if (point_count > 0) {
         diagnostics.hamiltonian_l2 = amrex::max(
            diagnostics.hamiltonian_l2,
            norms.norm2(0) / std::sqrt(static_cast<amrex::Real>(point_count))
         );
      }
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
   return diagnostics;
}

} // namespace pyfluid_amrex
