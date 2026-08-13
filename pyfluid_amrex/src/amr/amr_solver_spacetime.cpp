#include "amr/amr_solver_base.hpp"
#include "pyfluid_amrex/initial_data.hpp"

#include <AMReX_Math.H>
#include <AMReX_MultiFabUtil.H>

#include <cmath>

#include "pyfluid_amrex/spacetime_coupling.hpp"
#include "pyfluid_amrex/hierarchy_interpolation.hpp"
#include "pyfluid_amrex/initial_data/materialization.hpp"
#include "pyfluid_amrex/initial_data/physical_id.hpp"
#include "pyfluid_amrex/kernels_generated/diagnostics.hpp"
#include "pyfluid_amrex/nodal_fill.hpp"
#include "pyfluid_amrex/nodal_interpolater.hpp"
#include "pyfluid_amrex/surface_geometry.hpp"
#include "pyfluid_amrex/z4c_constraint_pointwise.hpp"

namespace pyfluid_amrex {

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

void AmrSolverBase::materialize_constraint_solved_initial_data(
   bool seed_from_loaded_id
)
{
   initial_data_rotating_star_summary_.reset();
   const bool uses_constraint_solve = cfg_.spacetime_initial_data == "ctt"
      || cfg_.spacetime_initial_data == "cttk"
      || cfg_.spacetime_initial_data == "xcts";
#if (AMREX_SPACEDIM == 1)
   if (cfg_.initial_data_radial_grid == "adaptive" && cfg_.ic == "tov"
       && !seed_from_loaded_id) {
      const RelativisticInitialDataView provider = initial_data_provider_.view();
      if (finestLevel() != 0 || !provider.tov.active()) {
         amrex::Abort(
            "adaptive TOV ID requires one level and a native TOV provider"
         );
      }
      InitialDataEllipticHierarchy mapped_hierarchy;
      mapped_hierarchy.geometry.push_back(Geom(0));
      mapped_hierarchy.grids.push_back(boxArray(0));
      mapped_hierarchy.distribution.push_back(DistributionMap(0));
      mapped_hierarchy.physical_coordinate_faces.resize(1);
      mapped_hierarchy.physical_coordinate_faces[0][0] =
         stellar_adaptive_radial_faces(
            provider.tov.surface_radius,
            Geom(0).ProbHi(0),
            Geom(0).Domain().length(0)
         );
      initial_data_coordinate_faces_ =
         mapped_hierarchy.physical_coordinate_faces;
      initialize_relativistic_primitive_cache_from_provider(
         *primitive_[0], mapped_hierarchy, 0, cfg_, provider
      );
      initialize_mapped_z4c_state_from_provider(
         *spacetime_state_.state()[0],
         mapped_hierarchy,
         0,
         cfg_,
         0.0,
         provider
      );
      const InitialDataCoordinateLine radial = mapped_hierarchy.coordinate_line(
         0, 0
      );
      amrex::Print()
         << "adaptive TOV ID radial_cells=" << radial.centers.size()
         << " surface=" << provider.tov.surface_radius
         << " inner_dr=" << radial.widths.front()
         << " outer_dr=" << radial.widths.back() << '\n';
      finalize_constraint_solved_initial_data_state();
      return;
   }
#endif
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
   amrex::Vector<amrex::MultiFab*> primitive;
   amrex::Vector<amrex::MultiFab*> spacetime;
   amrex::Vector<amrex::MultiFab*> cell_spacetime;
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      hierarchy.geometry.push_back(Geom(lev));
      hierarchy.grids.push_back(boxArray(lev));
      hierarchy.distribution.push_back(DistributionMap(lev));
      if (seed_from_loaded_id) {
         fill_level_primitive_cache(lev, 0.0);
      } else if (initial_data_provider_.view().has_fluid()) {
         initialize_relativistic_primitive_cache_from_provider(
            *primitive_[lev], Geom(lev), cfg_, initial_data_provider_.view()
         );
      } else {
         fill_level_primitive_cache(lev, 0.0);
      }
      primitive.push_back(primitive_[lev].get());
      spacetime.push_back(spacetime_state_.state()[lev].get());
      cell_spacetime.push_back(spacetime_coupling_.cell_spacetime()[lev].get());
      if (lev < finestLevel()) {
         hierarchy.refinement_ratio.push_back(refRatio(lev));
      }
   }
   hierarchy.cell_interpolater = selected_state_interpolater();
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      const BoundaryKind lower = side_boundary_kind(cfg_, direction, false);
      const BoundaryKind upper = side_boundary_kind(cfg_, direction, true);
      hierarchy.reflection_lo[direction] =
         lower == BoundaryKind::Symmetry || lower == BoundaryKind::Reflecting
         || lower == BoundaryKind::Origin;
      hierarchy.reflection_hi[direction] =
         upper == BoundaryKind::Symmetry || upper == BoundaryKind::Reflecting
         || upper == BoundaryKind::Origin;
   }
#if (AMREX_SPACEDIM == 2)
   if (cfg_.initial_data_radial_grid == "adaptive"
       || !initial_data_coordinate_faces_.empty()) {
      const RelativisticInitialDataView provider = initial_data_provider_.view();
      if (finestLevel() != 0 || !provider.tov.active()) {
         amrex::Abort(
            "adaptive stellar ID requires one level and a native TOV seed"
         );
      }
      if (seed_from_loaded_id) {
         if (initial_data_coordinate_faces_.empty()) {
            amrex::Abort(
               "adaptive stellar ID continuation requires mapped coordinate "
               "metadata in the saved artifact"
            );
         }
         hierarchy.physical_coordinate_faces = initial_data_coordinate_faces_;
      } else {
         hierarchy.physical_coordinate_faces.resize(1);
         hierarchy.physical_coordinate_faces[0][0] =
            stellar_adaptive_radial_faces(
               provider.tov.surface_radius,
               hierarchy.geometry[0].ProbHi(0),
               hierarchy.geometry[0].Domain().length(0)
            );
         initial_data_coordinate_faces_ = hierarchy.physical_coordinate_faces;
         initialize_relativistic_primitive_cache_from_provider(
            *primitive_[0], hierarchy, 0, cfg_, provider
         );
      }
      if (amrex::ParallelDescriptor::IOProcessor()) {
         const InitialDataCoordinateLine radial = hierarchy.coordinate_line(
            0, 0
         );
         amrex::Print()
            << "adaptive stellar ID radial_cells=" << radial.centers.size()
            << " seed_surface=" << provider.tov.surface_radius
            << " inner_dr=" << radial.widths.front()
            << " outer_dr=" << radial.widths.back() << '\n';
      }
   }
#endif
   bool rotating_xy_reflection =
      hierarchy.reflection_lo[0] != 0 || hierarchy.reflection_hi[0] != 0;
#if (AMREX_SPACEDIM >= 2)
   rotating_xy_reflection = rotating_xy_reflection
      || hierarchy.reflection_lo[1] != 0
      || hierarchy.reflection_hi[1] != 0;
#endif
   if (cfg_.ic == "rotating_star"
       && cfg_.initial_data_angular_velocity != 0.0
       && cfg_.mesh_coordinate == "cartesian"
       && rotating_xy_reflection) {
      amrex::Abort(
         "a nonzero star rotating about z is not invariant under independent "
         "x/y reflection boundaries; use full x/y and optional z symmetry"
      );
   }

   ConformalConstraintFormulation formulation =
      ConformalConstraintFormulation::ctt;
   if (cfg_.spacetime_initial_data == "cttk") {
      formulation = ConformalConstraintFormulation::cttk;
   } else if (cfg_.spacetime_initial_data == "xcts") {
      formulation = ConformalConstraintFormulation::xcts;
   }
   ConformalContinuationObserver continuation_observer;
   if (physical_initial_data_sequence_output_enabled()) {
      initialize_physical_initial_data_sequence_output();
      continuation_observer = [this](
         int continuation_step,
         int continuation_steps,
         amrex::Real angular_velocity,
         RotatingStarUpdateSummary const& rotating_star
      ) {
         finalize_constraint_solved_initial_data_state();
         run_evolution_preflight(0.0);
         write_physical_initial_data_sequence_model(
            continuation_step,
            continuation_steps,
            angular_velocity,
            rotating_star
         );
      };
   }
   const ConformalMaterializationResult result = materialize_conformal_initial_data(
      hierarchy,
      primitive,
      spacetime,
      cell_spacetime,
      physics_system_,
      eos_method_,
      eos_params(cfg_),
      initial_data_provider_.view().fields,
      formulation,
      rotating_star_parameters(cfg_, initial_data_provider_.view()),
      seed_from_loaded_id,
      continuation_observer
   );
   if (cfg_.ic == "rotating_star") {
      initial_data_rotating_star_summary_ =
         std::make_unique<RotatingStarUpdateSummary>(result.rotating_star);
   }
   if (amrex::ParallelDescriptor::IOProcessor()) {
      amrex::Print()
         << cfg_.spacetime_initial_data
         << " initial data: matter_iterations=" << result.matter_iterations
         << " relative_matter_update=" << result.relative_matter_update
         << " coupled_iterations=" << result.constraints.coupled_iterations
         << " hamiltonian_residual=" << result.constraints.hamiltonian_residual
         << " lapse_residual=" << result.constraints.lapse_residual
         << " momentum_residual=" << result.constraints.momentum_residual << '\n';
      if (cfg_.ic == "rotating_star"
          && cfg_.initial_data_angular_velocity != 0.0) {
         if (!(result.rotating_star.equatorial_radius > 0.0)
             || !(result.rotating_star.polar_radius > 0.0)) {
            amrex::Abort("rotating-star matter support has no resolved surface extent");
         }
         amrex::Print()
            << "rotating star: omega=" << cfg_.initial_data_angular_velocity
            << " maximum_density=" << result.rotating_star.maximum_density
            << " maximum_velocity=" << result.rotating_star.maximum_velocity
            << " injection_energy=" << result.rotating_star.injection_energy
            << " boundary_density_excess="
            << result.rotating_star.boundary_density_excess
            << " equatorial_radius=" << result.rotating_star.equatorial_radius
            << " polar_radius=" << result.rotating_star.polar_radius
            << " coordinate_axis_ratio="
            << result.rotating_star.equatorial_radius
               / result.rotating_star.polar_radius
            << " mass_shedding_indicator="
            << result.rotating_star.mass_shedding_indicator
            << " adm_mass=" << result.rotating_star.adm_mass
            << " baryonic_mass=" << result.rotating_star.baryonic_mass
            << " angular_momentum=" << result.rotating_star.angular_momentum
            << " dimensionless_spin="
            << result.rotating_star.dimensionless_spin
            << " equatorial_log_enthalpy_gradient="
            << result.rotating_star.equatorial_log_enthalpy_gradient
            << " polar_log_enthalpy_gradient="
            << result.rotating_star.polar_log_enthalpy_gradient
            << '\n';
      }
   }

   finalize_constraint_solved_initial_data_state();
}

void AmrSolverBase::finalize_constraint_solved_initial_data_state()
{
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      spacetime_coupling_.invalidate_cell_spacetime(lev);
   }
   fill_all_spacetime_boundaries(spacetime_state_.state(), 0.0);
   if (initial_data_coordinate_faces_.empty()) {
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         complete_z4c_contracted_connections(
            *spacetime_state_.state()[lev], Geom(lev)
         );
      }
   }
   fill_all_spacetime_boundaries(spacetime_state_.state(), 0.0);
   prepare_spacetime_coupling_range(
      spacetime_state_.state(), 0, finestLevel() + 1, 0.0, -1.0, 0
   );
   amrex::Vector<amrex::Geometry> geometry;
   ConstPhysicalIdField primitive;
   ConstPhysicalIdField cell_spacetime;
   MutablePhysicalIdField fluid;
   geometry.reserve(finestLevel() + 1);
   primitive.reserve(finestLevel() + 1);
   cell_spacetime.reserve(finestLevel() + 1);
   fluid.reserve(finestLevel() + 1);
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      geometry.push_back(Geom(lev));
      primitive.push_back(primitive_[lev].get());
      cell_spacetime.push_back(
         spacetime_coupling_.cell_spacetime()[lev].get()
      );
      fluid.push_back(fluid_state_.state()[lev].get());
   }
   materialize_physical_initial_data_fluid(
      cfg_, physics_system_, geometry, primitive, cell_spacetime, fluid
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
      nodal_interpolater(),
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
   amrex::Real interior_hamiltonian_squared_integral = 0.0;
   amrex::Real interior_volume = 0.0;
   amrex::Real cf_hamiltonian_squared_integral = 0.0;
   amrex::Real cf_volume = 0.0;
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
      amrex::iMultiFab stencil_valid(
         state.boxArray(), state.DistributionMap(), 1, 2
      );
      stencil_valid.setVal(0);
      stencil_valid.ParallelCopy(valid, 0, 0, 1, 0, 0);
      stencil_valid.FillBoundary(Geom(lev).periodicity());
      amrex::MultiFab norms(
         state.boxArray(), state.DistributionMap(), 11, 0
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
         const auto same_level_points = stencil_valid.const_array(mfi);
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
            bool same_level_stencil = true;
            for (int offset = 1; offset <= 2; ++offset) {
               same_level_stencil = same_level_stencil
                  && same_level_points(i - offset, j, k, 0) != 0
                  && same_level_points(i + offset, j, k, 0) != 0;
#if (AMREX_SPACEDIM >= 2)
               same_level_stencil = same_level_stencil
                  && same_level_points(i, j - offset, k, 0) != 0
                  && same_level_points(i, j + offset, k, 0) != 0;
#endif
#if (AMREX_SPACEDIM == 3)
               same_level_stencil = same_level_stencil
                  && same_level_points(i, j, k - offset, 0) != 0
                  && same_level_points(i, j, k + offset, 0) != 0;
#endif
            }
            const Z4cConstraints c = evaluate_z4c_constraints_at<
               4, AMREX_SPACEDIM
            >(input, i, j, k, inv_spacing, point_matter);
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
            const int value_component = same_level_stencil ? 7 : 9;
            const int weight_component = same_level_stencil ? 8 : 10;
            output(i, j, k, value_component) = amrex::Math::abs(c.hamiltonian);
            output(i, j, k, weight_component) = 1.0;
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
      diagnostics.hamiltonian_interior_linf = amrex::max(
         diagnostics.hamiltonian_interior_linf, norms.norm0(7, 0, false)
      );
      const amrex::Real level_interior_norm = norms.norm2(7);
      interior_hamiltonian_squared_integral += level_volume
         * level_interior_norm * level_interior_norm;
      interior_volume += level_volume * norms.sum(8);
      diagnostics.hamiltonian_cf_linf = amrex::max(
         diagnostics.hamiltonian_cf_linf, norms.norm0(9, 0, false)
      );
      const amrex::Real level_cf_norm = norms.norm2(9);
      cf_hamiltonian_squared_integral += level_volume
         * level_cf_norm * level_cf_norm;
      cf_volume += level_volume * norms.sum(10);
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
   if (interior_volume > 0.0) {
      diagnostics.hamiltonian_interior_l2 = std::sqrt(
         interior_hamiltonian_squared_integral / interior_volume
      );
   }
   if (cf_volume > 0.0) {
      diagnostics.hamiltonian_cf_l2 = std::sqrt(
         cf_hamiltonian_squared_integral / cf_volume
      );
   }
   return diagnostics;
}

SpacetimeGlobalDiagnostics
AmrSolverBase::compute_spacetime_global_diagnostics(
   amrex::Real time,
   bool surface_requested,
   bool volume_requested
)
{
   SpacetimeGlobalDiagnostics diagnostics;
   if (!spacetime_state_.active()) {
      return diagnostics;
   }
#if (AMREX_SPACEDIM != 3)
   amrex::Abort("global_gr diagnostics currently require a three-dimensional spacetime hierarchy");
#else
   if (mesh_coordinate_kind() != MeshCartesian || mixed_patch_system_enabled()) {
      amrex::Abort("global_gr diagnostics currently require a single Cartesian patch");
   }
   for (int direction = 0; direction < 3; ++direction) {
      if (Geom(0).isPeriodic(direction)) {
         amrex::Abort("global_gr diagnostics require nonperiodic asymptotic boundaries");
      }
   }

   fill_all_spacetime_boundaries(spacetime_state_.state(), time);
   enum IntegralComponent : int {
      AdmMassVolume = 0,
      AdmMomentum1Volume,
      AdmMomentum2Volume,
      AdmMomentum3Volume,
      AdmAngularMomentum1Volume,
      AdmAngularMomentum2Volume,
      AdmAngularMomentum3Volume,
      ProperMassVolume,
      KomarMassVolume,
      Count,
   };
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      amrex::MultiFab const& state = *spacetime_state_.state()[lev];
      if (!volume_requested) {
         continue;
      }
      amrex::iMultiFab valid(state.boxArray(), state.DistributionMap(), 1, 0);
      if (lev < finestLevel()) {
         const amrex::BoxArray fine_nodes = amrex::convert(
            boxArray(lev + 1), amrex::IntVect::TheNodeVector()
         );
         valid = amrex::makeFineMask(
            state, fine_nodes, refRatio(lev), Geom(lev).periodicity(), 1, 0
         );
      } else {
         valid.setVal(1);
      }
      const std::unique_ptr<amrex::iMultiFab> owner = state.OwnerMask(
         Geom(lev).periodicity()
      );
      amrex::MultiFab integrals(
         state.boxArray(), state.DistributionMap(), IntegralComponent::Count, 0
      );
      integrals.setVal(0.0);
      const amrex::Box nodal_domain = amrex::convert(
         Geom(lev).Domain(), amrex::IntVect::TheNodeVector()
      );
      const auto spacing = Geom(lev).CellSizeArray();
      const auto lower = Geom(lev).ProbLoArray();
      const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
         1.0 / spacing[0], 1.0 / spacing[1], 1.0 / spacing[2]
      }};
      const amrex::GpuArray<int, 3> domain_lower{{
         nodal_domain.smallEnd(0),
         nodal_domain.smallEnd(1),
         nodal_domain.smallEnd(2),
      }};
      const amrex::GpuArray<int, 3> domain_upper{{
         nodal_domain.bigEnd(0),
         nodal_domain.bigEnd(1),
         nodal_domain.bigEnd(2),
      }};
      const amrex::Real cell_volume = spacing[0] * spacing[1] * spacing[2];
      const bool matter_coupled = spacetime_includes_matter(cfg_);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto input = state.const_array(mfi);
         const auto output = integrals.array(mfi);
         const auto valid_points = valid.const_array(mfi);
         const auto owned_points = owner->const_array(mfi);
         const auto matter = spacetime_coupling_.vertex_matter()[lev]->const_array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (valid_points(i, j, k, 0) == 0 || owned_points(i, j, k, 0) == 0) {
               return;
            }

            const int indices[3] = {i, j, k};
            const Z4cSpatialDerivatives derivatives =
               gather_z4c_geometry_derivatives<4, 3>(
                  input,
                  i,
                  j,
                  k,
                  inv_spacing,
                  domain_lower,
                  domain_upper
               );
            const Z4cState point_state = load_z4c_state(input, i, j, k);
            const Vector3 position{
               lower[0] + static_cast<amrex::Real>(i - nodal_domain.smallEnd(0)) * spacing[0],
               lower[1] + static_cast<amrex::Real>(j - nodal_domain.smallEnd(1)) * spacing[1],
               lower[2] + static_cast<amrex::Real>(k - nodal_domain.smallEnd(2)) * spacing[2],
            };
            const Z4cMatter point_matter = matter_coupled
               ? load_eulerian_matter(matter, i, j, k)
               : vacuum_z4c_matter();
            const AdmVolumeIntegrands point_volume =
               generated::adm_constraint_volume_integrands(
                  point_state, derivatives, point_matter, position
               );
            amrex::Real volume_weight = cell_volume;
            for (int direction = 0; direction < 3; ++direction) {
               if (indices[direction] == nodal_domain.smallEnd(direction)
                   || indices[direction] == nodal_domain.bigEnd(direction)) {
                  volume_weight *= 0.5;
               }
            }
            output(i, j, k, IntegralComponent::AdmMassVolume) =
               volume_weight * point_volume.mass;
            output(i, j, k, IntegralComponent::AdmMomentum1Volume) =
               volume_weight * point_volume.linear_momentum1;
            output(i, j, k, IntegralComponent::AdmMomentum2Volume) =
               volume_weight * point_volume.linear_momentum2;
            output(i, j, k, IntegralComponent::AdmMomentum3Volume) =
               volume_weight * point_volume.linear_momentum3;
            output(i, j, k, IntegralComponent::AdmAngularMomentum1Volume) =
               volume_weight * point_volume.angular_momentum1;
            output(i, j, k, IntegralComponent::AdmAngularMomentum2Volume) =
               volume_weight * point_volume.angular_momentum2;
            output(i, j, k, IntegralComponent::AdmAngularMomentum3Volume) =
               volume_weight * point_volume.angular_momentum3;
            output(i, j, k, IntegralComponent::ProperMassVolume) =
               volume_weight * point_volume.proper_mass;
            output(i, j, k, IntegralComponent::KomarMassVolume) =
               volume_weight * point_volume.komar_mass;
         });
      }
      diagnostics.adm_mass_volume += integrals.sum(IntegralComponent::AdmMassVolume);
      diagnostics.adm_momentum_volume[0] += integrals.sum(IntegralComponent::AdmMomentum1Volume);
      diagnostics.adm_momentum_volume[1] += integrals.sum(IntegralComponent::AdmMomentum2Volume);
      diagnostics.adm_momentum_volume[2] += integrals.sum(IntegralComponent::AdmMomentum3Volume);
      diagnostics.adm_angular_momentum_volume[0] +=
         integrals.sum(IntegralComponent::AdmAngularMomentum1Volume);
      diagnostics.adm_angular_momentum_volume[1] +=
         integrals.sum(IntegralComponent::AdmAngularMomentum2Volume);
      diagnostics.adm_angular_momentum_volume[2] +=
         integrals.sum(IntegralComponent::AdmAngularMomentum3Volume);
      diagnostics.proper_mass_volume += integrals.sum(IntegralComponent::ProperMassVolume);
      diagnostics.komar_mass_volume += integrals.sum(IntegralComponent::KomarMassVolume);
   }
   if (volume_requested) {
      const amrex::Real volume_factor = symmetry_volume_factor();
      diagnostics.adm_mass_volume *= volume_factor;
      diagnostics.proper_mass_volume *= volume_factor;
      diagnostics.komar_mass_volume *= volume_factor;
      for (int axis = 0; axis < 3; ++axis) {
         diagnostics.adm_momentum_volume[axis] *= symmetry_momentum_factor(
            axis, volume_factor
         );
         diagnostics.adm_angular_momentum_volume[axis] *=
            symmetry_angular_momentum_factor(axis, volume_factor);
      }
   }
   if (surface_requested) {
      if (symmetry_volume_factor() != 1.0) {
         amrex::Abort(
            "spherical GR surface diagnostics do not yet reconstruct omitted symmetry sectors"
         );
      }
      enum SurfaceFluxComponent : int {
         AdmMassFlux1 = 0,
         AdmMassFlux2,
         AdmMassFlux3,
         KomarMassFlux1,
         KomarMassFlux2,
         KomarMassFlux3,
         AdmMomentumFlux11,
         AdmMomentumFlux12,
         AdmMomentumFlux13,
         AdmMomentumFlux21,
         AdmMomentumFlux22,
         AdmMomentumFlux23,
         AdmMomentumFlux31,
         AdmMomentumFlux32,
         AdmMomentumFlux33,
         AdmAngularMomentumFlux11,
         AdmAngularMomentumFlux12,
         AdmAngularMomentumFlux13,
         AdmAngularMomentumFlux21,
         AdmAngularMomentumFlux22,
         AdmAngularMomentumFlux23,
         AdmAngularMomentumFlux31,
         AdmAngularMomentumFlux32,
         AdmAngularMomentumFlux33,
         SurfaceFluxCount,
      };
      LevelData surface_flux_hierarchy(finestLevel() + 1);
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         amrex::MultiFab const& state = *spacetime_state_.state()[lev];
         auto flux = std::make_unique<amrex::MultiFab>(
            state.boxArray(), state.DistributionMap(), SurfaceFluxCount, 1
         );
         flux->setVal(0.0);
         const amrex::Box nodal_domain = amrex::convert(
            Geom(lev).Domain(), amrex::IntVect::TheNodeVector()
         );
         const auto spacing = Geom(lev).CellSizeArray();
         const auto lower = Geom(lev).ProbLoArray();
         const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
            1.0 / spacing[0], 1.0 / spacing[1], 1.0 / spacing[2]
         }};
         const amrex::GpuArray<int, 3> domain_lower{{
            nodal_domain.smallEnd(0),
            nodal_domain.smallEnd(1),
            nodal_domain.smallEnd(2),
         }};
         const amrex::GpuArray<int, 3> domain_upper{{
            nodal_domain.bigEnd(0),
            nodal_domain.bigEnd(1),
            nodal_domain.bigEnd(2),
         }};
         for (amrex::MFIter mfi(state, false); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto input = state.const_array(mfi);
            const auto output = flux->array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const Z4cSpatialDerivatives derivatives =
                  gather_z4c_geometry_derivatives<4, 3>(
                     input, i, j, k, inv_spacing, domain_lower, domain_upper
                  );
               const Z4cState point_state = load_z4c_state(input, i, j, k);
               const Vector3 position{
                  lower[0] + static_cast<amrex::Real>(i - nodal_domain.smallEnd(0)) * spacing[0],
                  lower[1] + static_cast<amrex::Real>(j - nodal_domain.smallEnd(1)) * spacing[1],
                  lower[2] + static_cast<amrex::Real>(k - nodal_domain.smallEnd(2)) * spacing[2],
               };
               const AdmFluxIntegrands point_flux = generated::adm_flux_integrands(
                  point_state, derivatives, position
               );
               output(i, j, k, AdmMassFlux1) = point_flux.mass1;
               output(i, j, k, AdmMassFlux2) = point_flux.mass2;
               output(i, j, k, AdmMassFlux3) = point_flux.mass3;
               output(i, j, k, KomarMassFlux1) = point_flux.komar_mass1;
               output(i, j, k, KomarMassFlux2) = point_flux.komar_mass2;
               output(i, j, k, KomarMassFlux3) = point_flux.komar_mass3;
               output(i, j, k, AdmMomentumFlux11) = point_flux.linear_momentum11;
               output(i, j, k, AdmMomentumFlux12) = point_flux.linear_momentum12;
               output(i, j, k, AdmMomentumFlux13) = point_flux.linear_momentum13;
               output(i, j, k, AdmMomentumFlux21) = point_flux.linear_momentum21;
               output(i, j, k, AdmMomentumFlux22) = point_flux.linear_momentum22;
               output(i, j, k, AdmMomentumFlux23) = point_flux.linear_momentum23;
               output(i, j, k, AdmMomentumFlux31) = point_flux.linear_momentum31;
               output(i, j, k, AdmMomentumFlux32) = point_flux.linear_momentum32;
               output(i, j, k, AdmMomentumFlux33) = point_flux.linear_momentum33;
               output(i, j, k, AdmAngularMomentumFlux11) = point_flux.angular_momentum11;
               output(i, j, k, AdmAngularMomentumFlux12) = point_flux.angular_momentum12;
               output(i, j, k, AdmAngularMomentumFlux13) = point_flux.angular_momentum13;
               output(i, j, k, AdmAngularMomentumFlux21) = point_flux.angular_momentum21;
               output(i, j, k, AdmAngularMomentumFlux22) = point_flux.angular_momentum22;
               output(i, j, k, AdmAngularMomentumFlux23) = point_flux.angular_momentum23;
               output(i, j, k, AdmAngularMomentumFlux31) = point_flux.angular_momentum31;
               output(i, j, k, AdmAngularMomentumFlux32) = point_flux.angular_momentum32;
               output(i, j, k, AdmAngularMomentumFlux33) = point_flux.angular_momentum33;
            });
         }
         flux->FillBoundary(Geom(lev).periodicity());
         surface_flux_hierarchy[lev] = std::move(flux);
      }

      const ParametricSurface surface = diagnostic_extraction_surface();
      const HierarchyInterpolationPlan plan =
         diagnostic_surface_interpolation_plan(surface);
      const std::vector<amrex::Real> flux = plan.interpolate(
         surface_flux_hierarchy, 0, SurfaceFluxCount
      );
      const std::vector<amrex::Real> integrals = surface.integrate_normal_fluxes(
         flux, SurfaceFluxCount / 3
      );
      diagnostics.adm_mass_surface = integrals[0];
      diagnostics.komar_mass_surface = integrals[1];
      for (int axis = 0; axis < 3; ++axis) {
         diagnostics.adm_momentum_surface[axis] = integrals[2 + axis];
         diagnostics.adm_angular_momentum_surface[axis] = integrals[5 + axis];
      }
   }
#endif
   return diagnostics;
}

} // namespace pyfluid_amrex
