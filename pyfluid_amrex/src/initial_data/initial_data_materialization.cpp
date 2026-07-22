#include "pyfluid_amrex/initial_data/materialization.hpp"

#include <AMReX.H>
#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include "pyfluid_amrex/centering_transfer.hpp"
#include "pyfluid_amrex/initial_data/centering.hpp"
#include "pyfluid_amrex/kernels_generated/initial_data.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/sources.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "pyfluid_amrex/spacetime_coupling.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {

namespace {

char const* conformal_constraint_formulation_name(
   ConformalConstraintFormulation formulation
) noexcept
{
   switch (formulation) {
   case ConformalConstraintFormulation::ctt:
      return "ctt";
   case ConformalConstraintFormulation::cttk:
      return "cttk";
   case ConformalConstraintFormulation::xcts:
      return "xcts";
   }
   return "unknown";
}

amrex::Vector<amrex::BCRec> extrapolating_boundary_records(
   InitialDataEllipticHierarchy const& hierarchy,
   int num_components
)
{
   amrex::Vector<amrex::BCRec> records(num_components);
   for (amrex::BCRec& record : records) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         const int type = hierarchy.geometry[0].isPeriodic(direction)
            ? amrex::BCType::int_dir
            : amrex::BCType::foextrap;
         record.setLo(direction, type);
         record.setHi(direction, type);
      }
   }
   return records;
}

void materialize_conformal_geometry(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab const*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab const*> const& shift,
   amrex::Vector<amrex::MultiFab const*> const& mean_curvature,
   amrex::Vector<amrex::MultiFab const*> const& conformal_extrinsic,
   amrex::Vector<amrex::MultiFab*> const& spacetime_state,
   amrex::Vector<amrex::MultiFab*> const& cell_spacetime
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   for (int level = 0; level < num_levels; ++level) {
      spacetime_state[level]->setVal(0.0);
      for (amrex::MFIter mfi(
              *spacetime_state[level], amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto psi = conformal_factor[level]->const_array(mfi);
         const auto lapse_psi =
            lapse_times_conformal_factor[level]->const_array(mfi);
         const auto shift_vector = shift[level]->const_array(mfi);
         const auto curvature = mean_curvature[level]->const_array(mfi);
         const auto extrinsic = conformal_extrinsic[level]->const_array(mfi);
         const auto geometry = spacetime_state[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            store_z4c_state(
               geometry,
               i,
               j,
               k,
               generated::ctt_canonical_z4c_state(
                  psi(i, j, k, 0),
                  extrinsic(i, j, k, CONFORMAL_A11),
                  extrinsic(i, j, k, CONFORMAL_A12),
                  extrinsic(i, j, k, CONFORMAL_A13),
                  extrinsic(i, j, k, CONFORMAL_A22),
                  extrinsic(i, j, k, CONFORMAL_A23),
                  extrinsic(i, j, k, CONFORMAL_A33),
                  curvature(i, j, k, 0),
                  generated::xcts_lapse_from_conformal_unknowns(
                     psi(i, j, k, 0), lapse_psi(i, j, k, 0)
                  ),
                  shift_vector(i, j, k, 0),
                  shift_vector(i, j, k, 1),
                  shift_vector(i, j, k, 2)
               )
            );
         });
      }
   }
   const auto boundary = extrapolating_boundary_records(
      hierarchy, Z4C_NUM_COMPONENTS
   );
   fill_initial_data_ghosts(hierarchy, spacetime_state, boundary);
   for (int level = 0; level < num_levels; ++level) {
      transfer_grid_field_centering(
         *cell_spacetime[level],
         *spacetime_state[level],
         hierarchy.geometry[level],
         2
      );
   }
   fill_initial_data_ghosts(hierarchy, cell_spacetime, boundary);
}

template <PhysicalSystem System, int EOSMethod>
ConformalMaterializationResult materialize_conformal_initial_data_method(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& fluid_state,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab*> const& spacetime_state,
   amrex::Vector<amrex::MultiFab*> const& cell_spacetime,
   EOSParams const& eos,
   InitialDataFieldSelection const& fields,
   ConformalConstraintFormulation formulation,
   RotatingStarParameters const& rotating_star
)
{
   static_assert(System == PhysicalSystem::RHD || System == PhysicalSystem::RMHD);
   amrex::ignore_unused(fields);
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   if (num_levels == 0
       || static_cast<int>(fluid_state.size()) != num_levels
       || static_cast<int>(primitive_state.size()) != num_levels
       || static_cast<int>(spacetime_state.size()) != num_levels
       || static_cast<int>(cell_spacetime.size()) != num_levels) {
      amrex::Abort("invalid conformal initial-data hierarchy materialization");
   }

   amrex::Vector<std::unique_ptr<amrex::MultiFab>> conformal_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> vector_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> lapse_conformal_storage(
      num_levels
   );
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> shift_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> curvature_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> extrinsic_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> energy_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> momentum_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> stress_trace_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> vertex_energy_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> vertex_momentum_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> vertex_stress_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> previous_conformal(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> previous_vector(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> previous_lapse(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> previous_curvature(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> previous_primitive(num_levels);
   amrex::Vector<amrex::MultiFab*> conformal_factor(num_levels);
   amrex::Vector<amrex::MultiFab*> vector_potential(num_levels);
   amrex::Vector<amrex::MultiFab*> xcts_shift(num_levels);
   amrex::Vector<amrex::MultiFab*> lapse_times_conformal_factor(num_levels);
   amrex::Vector<amrex::MultiFab const*> lapse_conformal_view(num_levels);
   amrex::Vector<amrex::MultiFab const*> shift_view(num_levels);
   amrex::Vector<amrex::MultiFab*> mean_curvature_mutable(num_levels);
   amrex::Vector<amrex::MultiFab const*> mean_curvature(num_levels);
   amrex::Vector<amrex::MultiFab*> conformal_extrinsic(num_levels);
   amrex::Vector<amrex::MultiFab const*> conformal_factor_const(num_levels);
   amrex::Vector<amrex::MultiFab const*> conformal_extrinsic_const(num_levels);
   amrex::Vector<amrex::MultiFab*> cell_energy(num_levels);
   amrex::Vector<amrex::MultiFab*> cell_momentum(num_levels);
   amrex::Vector<amrex::MultiFab*> cell_stress(num_levels);
   amrex::Vector<amrex::MultiFab const*> cell_energy_density(num_levels);
   amrex::Vector<amrex::MultiFab const*> cell_momentum_density(num_levels);
   amrex::Vector<amrex::MultiFab const*> cell_stress_trace(num_levels);
   amrex::Vector<amrex::MultiFab*> vertex_energy(num_levels);
   amrex::Vector<amrex::MultiFab*> vertex_momentum(num_levels);
   amrex::Vector<amrex::MultiFab*> vertex_stress(num_levels);
   amrex::Vector<amrex::MultiFab const*> energy_density(num_levels);
   amrex::Vector<amrex::MultiFab const*> momentum_density(num_levels);
   amrex::Vector<amrex::MultiFab const*> stress_trace(num_levels);

   for (int level = 0; level < num_levels; ++level) {
      if (fluid_state[level] == nullptr || primitive_state[level] == nullptr
          || spacetime_state[level] == nullptr || cell_spacetime[level] == nullptr) {
         amrex::Abort(
            "conformal initial data received an unallocated hierarchy level"
         );
      }
      const amrex::BoxArray vertex_grids = amrex::convert(
         hierarchy.grids[level], amrex::IntVect::TheNodeVector()
      );
      if (fluid_state[level]->boxArray() != hierarchy.grids[level]
          || primitive_state[level]->boxArray() != hierarchy.grids[level]
          || cell_spacetime[level]->boxArray() != hierarchy.grids[level]
          || spacetime_state[level]->boxArray() != vertex_grids) {
         amrex::Abort(
            "conformal initial-data fields do not match the staggered hierarchy"
         );
      }
      conformal_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 1
      );
      vector_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 3, 1
      );
      lapse_conformal_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 1
      );
      shift_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 3, 1
      );
      curvature_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 1
      );
      extrinsic_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level],
         CONFORMAL_A_NUM_COMPONENTS, 1
      );
      energy_storage[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 1, 1
      );
      momentum_storage[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 3, 1
      );
      stress_trace_storage[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 1, 1
      );
      vertex_energy_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 0
      );
      vertex_momentum_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 3, 0
      );
      vertex_stress_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 0
      );
      previous_conformal[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 0
      );
      previous_vector[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 3, 0
      );
      previous_lapse[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 0
      );
      previous_curvature[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 0
      );
      previous_primitive[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level],
         hierarchy.distribution[level],
         primitive_state[level]->nComp(),
         0
      );
      conformal_storage[level]->setVal(1.0);
      vector_storage[level]->setVal(0.0);
      lapse_conformal_storage[level]->setVal(1.0);
      shift_storage[level]->setVal(0.0);
      curvature_storage[level]->setVal(0.0);
      extrinsic_storage[level]->setVal(0.0);
      conformal_factor[level] = conformal_storage[level].get();
      vector_potential[level] = vector_storage[level].get();
      xcts_shift[level] = vector_storage[level].get();
      lapse_times_conformal_factor[level] = lapse_conformal_storage[level].get();
      lapse_conformal_view[level] = formulation != ConformalConstraintFormulation::xcts
         ? conformal_storage[level].get()
         : lapse_conformal_storage[level].get();
      shift_view[level] = formulation != ConformalConstraintFormulation::xcts
         ? shift_storage[level].get()
         : vector_storage[level].get();
      mean_curvature_mutable[level] = curvature_storage[level].get();
      mean_curvature[level] = curvature_storage[level].get();
      conformal_extrinsic[level] = extrinsic_storage[level].get();
      conformal_factor_const[level] = conformal_storage[level].get();
      conformal_extrinsic_const[level] = extrinsic_storage[level].get();
      cell_energy[level] = energy_storage[level].get();
      cell_momentum[level] = momentum_storage[level].get();
      cell_stress[level] = stress_trace_storage[level].get();
      cell_energy_density[level] = energy_storage[level].get();
      cell_momentum_density[level] = momentum_storage[level].get();
      cell_stress_trace[level] = stress_trace_storage[level].get();
      vertex_energy[level] = vertex_energy_storage[level].get();
      vertex_momentum[level] = vertex_momentum_storage[level].get();
      vertex_stress[level] = vertex_stress_storage[level].get();
      energy_density[level] = vertex_energy_storage[level].get();
      momentum_density[level] = vertex_momentum_storage[level].get();
      stress_trace[level] = vertex_stress_storage[level].get();
   }

   auto conformal_boundary = InitialDataEllipticBoundaryConditions::constant(
      hierarchy,
      1,
      1.0,
      0.0,
      0.0,
      InitialDataEllipticCentering::vertex
   );
   set_centered_power_law_robin_boundary(
      hierarchy, conformal_boundary, amrex::Vector<amrex::Real>{0.0}, 1.0
   );
   auto lapse_boundary = InitialDataEllipticBoundaryConditions::constant(
      hierarchy,
      1,
      1.0,
      0.0,
      0.0,
      InitialDataEllipticCentering::vertex
   );
   set_centered_power_law_robin_boundary(
      hierarchy, lapse_boundary, amrex::Vector<amrex::Real>{0.0}, 1.0
   );
   auto vector_boundary = InitialDataEllipticBoundaryConditions::constant(
      hierarchy,
      3,
      1.0,
      0.0,
      0.0,
      InitialDataEllipticCentering::vertex
   );

   constexpr int max_constraint_iterations = 100;
   CttOptions options;
   options.max_coupled_iterations = max_constraint_iterations;
   options.relative_update_tolerance = 1.0e-10;
   options.relaxation = 0.5;
   options.linear.boundary_interpolation_order = 4;
   options.linear.relative_tolerance = 1.0e-12;
   XctsOptions xcts_options;
   xcts_options.max_coupled_iterations = options.max_coupled_iterations;
   xcts_options.relative_update_tolerance = options.relative_update_tolerance;
   xcts_options.relaxation = options.relaxation;
   xcts_options.linear = options.linear;
   CttkOptions cttk_options;
   cttk_options.max_coupled_iterations = options.max_coupled_iterations;
   cttk_options.relative_update_tolerance = options.relative_update_tolerance;
   cttk_options.relaxation = options.relaxation;
   cttk_options.linear = options.linear;

   ConformalMaterializationResult result;
   const bool update_rotating_matter = rotating_star.active
      && rotating_star.angular_velocity != 0.0;
   const int max_matter_iterations = update_rotating_matter
      ? rotating_star.max_iterations
      : 12;
   const amrex::Real matter_tolerance = update_rotating_matter
      ? rotating_star.relative_tolerance
      : 1.0e-9;
   const int continuation_steps = update_rotating_matter
      ? rotating_star.continuation_steps
      : 1;
   for (int continuation_step = 1;
        continuation_step <= continuation_steps;
        ++continuation_step) {
      RotatingStarParameters continued_rotating_star = rotating_star;
      continued_rotating_star.angular_velocity =
         rotating_star.angular_velocity
         * static_cast<amrex::Real>(continuation_step)
         / static_cast<amrex::Real>(continuation_steps);
      result.relative_matter_update = std::numeric_limits<amrex::Real>::max();
      int step_matter_iterations = 0;
      for (int matter_iteration = 1;
           matter_iteration <= max_matter_iterations;
           ++matter_iteration) {
         materialize_conformal_geometry(
            hierarchy,
            conformal_factor_const,
            lapse_conformal_view,
            shift_view,
            mean_curvature,
            conformal_extrinsic_const,
            spacetime_state,
            cell_spacetime
         );
         for (int level = 0; level < num_levels; ++level) {
            amrex::MultiFab::Copy(
               *previous_primitive[level],
               *primitive_state[level],
               0,
               0,
               primitive_state[level]->nComp(),
               0
            );
         }
         // The conformal unknowns start from flat space.  Preserve the spherical
         // seed through the first constraint solve so the Euler update sees the
         // star's lapse and shift rather than erasing it in the flat seed metric.
         const bool preserve_spherical_seed =
            continuation_step == 1 && matter_iteration == 1;
         if (update_rotating_matter && !preserve_spherical_seed) {
            amrex::Vector<amrex::MultiFab const*> cell_spacetime_const(num_levels);
            for (int level = 0; level < num_levels; ++level) {
               cell_spacetime_const[level] = cell_spacetime[level];
            }
            result.rotating_star = update_rotating_star_primitives(
               hierarchy,
               primitive_state,
               cell_spacetime_const,
               EOSMethod,
               eos,
               continued_rotating_star
            );
            if (amrex::ParallelDescriptor::IOProcessor()) {
               amrex::Print()
                  << "rotating continuation step=" << continuation_step
                  << "/" << continuation_steps
                  << " matter_iteration=" << matter_iteration
                  << " maximum_density=" << result.rotating_star.maximum_density
                  << " maximum_velocity=" << result.rotating_star.maximum_velocity
                  << " injection_energy=" << result.rotating_star.injection_energy
                  << " boundary_density_excess="
                  << result.rotating_star.boundary_density_excess << '\n';
            }
         }
         for (int level = 0; level < num_levels; ++level) {
            if (primitive_state[level]->contains_nan(
                   0, primitive_state[level]->nComp(), 0
                )) {
               amrex::Abort("conformal initial-data primitive state is nonfinite");
            }
            if (cell_spacetime[level]->contains_nan(
                   0, cell_spacetime[level]->nComp(), 0
                )) {
               amrex::Abort("conformal initial-data cell geometry is nonfinite");
            }
            amrex::MultiFab::Copy(
               *previous_conformal[level], *conformal_storage[level], 0, 0, 1, 0
            );
            amrex::MultiFab::Copy(
               *previous_vector[level], *vector_storage[level], 0, 0, 3, 0
            );
            amrex::MultiFab::Copy(
               *previous_lapse[level],
               *lapse_conformal_storage[level],
               0,
               0,
               1,
               0
            );
            if (formulation == ConformalConstraintFormulation::cttk) {
               amrex::MultiFab::Copy(
                  *previous_curvature[level], *curvature_storage[level], 0, 0, 1, 0
               );
            }
            for (amrex::MFIter mfi(
                    *energy_storage[level], amrex::TilingIfNotGPU()
                 ); mfi.isValid(); ++mfi) {
               const amrex::Box box = mfi.tilebox();
               const auto primitive = primitive_state[level]->const_array(mfi);
               const auto cell_geometry = cell_spacetime[level]->const_array(mfi);
               const auto energy = energy_storage[level]->array(mfi);
               const auto momentum = momentum_storage[level]->array(mfi);
               const auto stress_trace_field = stress_trace_storage[level]->array(mfi);
               amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  const Z4cState geometry = load_z4c_state(
                     cell_geometry, i, j, k
                  );
                  const SpacetimeMetric metric = generated::z4c_spacetime_metric(geometry);
                  EulerianMatter matter;
                  if constexpr (System == PhysicalSystem::RHD) {
                     matter = generated::rhd_eulerian_matter(
                        rhd_primitive_at<EOSMethod>(primitive, i, j, k, eos), metric
                     );
                  } else {
                     matter = generated::rmhd_eulerian_matter(
                        rmhd_primitive_at(primitive, i, j, k, eos), metric
                     );
                  }
                  energy(i, j, k, 0) = matter.rho;
                  momentum(i, j, k, 0) = matter.momentum1;
                  momentum(i, j, k, 1) = matter.momentum2;
                  momentum(i, j, k, 2) = matter.momentum3;
                  stress_trace_field(i, j, k, 0) =
                     matter.stress11 * metric.inv_gamma11
                     + 2.0 * matter.stress12 * metric.inv_gamma12
                     + 2.0 * matter.stress13 * metric.inv_gamma13
                     + matter.stress22 * metric.inv_gamma22
                     + 2.0 * matter.stress23 * metric.inv_gamma23
                     + matter.stress33 * metric.inv_gamma33;
               });
            }
            if (energy_storage[level]->contains_nan(0, 1, 0)) {
               amrex::Abort("conformal initial-data energy source is nonfinite");
            }
            if (momentum_storage[level]->contains_nan(0, 3, 0)) {
               amrex::Abort("conformal initial-data momentum source is nonfinite");
            }
            if (stress_trace_storage[level]->contains_nan(0, 1, 0)) {
               amrex::Abort("conformal initial-data stress trace is nonfinite");
            }
         }
         fill_initial_data_ghosts(
            hierarchy,
            cell_energy,
            extrapolating_boundary_records(hierarchy, 1)
         );
         fill_initial_data_ghosts(
            hierarchy,
            cell_momentum,
            extrapolating_boundary_records(hierarchy, 3)
         );
         fill_initial_data_ghosts(
            hierarchy,
            cell_stress,
            extrapolating_boundary_records(hierarchy, 1)
         );
         materialize_initial_data_vertices(
            hierarchy,
            vertex_energy,
            cell_energy_density,
            extrapolating_boundary_records(hierarchy, 1),
            2
         );
         materialize_initial_data_vertices(
            hierarchy,
            vertex_momentum,
            cell_momentum_density,
            extrapolating_boundary_records(hierarchy, 3),
            2
         );
         materialize_initial_data_vertices(
            hierarchy,
            vertex_stress,
            cell_stress_trace,
            extrapolating_boundary_records(hierarchy, 1),
            2
         );
         for (int level = 0; level < num_levels; ++level) {
            if (vertex_energy_storage[level]->contains_nan(0, 1, 0)) {
               amrex::Abort("vertex initial-data energy source is nonfinite");
            }
            if (vertex_momentum_storage[level]->contains_nan(0, 3, 0)) {
               amrex::Abort("vertex initial-data momentum source is nonfinite");
            }
            if (vertex_stress_storage[level]->contains_nan(0, 1, 0)) {
               amrex::Abort("vertex initial-data stress trace is nonfinite");
            }
         }

         if (formulation == ConformalConstraintFormulation::ctt) {
            const CttResult constraints = solve_ctt_constraints(
               hierarchy,
               conformal_factor,
               vector_potential,
               conformal_extrinsic,
               energy_density,
               momentum_density,
               conformal_boundary,
               vector_boundary,
               options
            );
            result.constraints = {
               constraints.coupled_iterations,
               constraints.relative_update,
               constraints.hamiltonian_residual,
               0.0,
               constraints.momentum_residual,
               constraints.converged,
            };
         } else if (formulation == ConformalConstraintFormulation::cttk) {
            const CttkResult constraints = solve_cttk_hybrid_constraints(
               hierarchy,
               conformal_factor,
               mean_curvature_mutable,
               vector_potential,
               conformal_extrinsic,
               energy_density,
               momentum_density,
               conformal_boundary,
               vector_boundary,
               cttk_options
            );
            result.constraints = {
               constraints.coupled_iterations,
               constraints.relative_update,
               constraints.hamiltonian_residual,
               0.0,
               constraints.momentum_residual,
               constraints.converged,
            };
         } else {
            const XctsResult constraints = solve_xcts_constraints(
               hierarchy,
               conformal_factor,
               lapse_times_conformal_factor,
               xcts_shift,
               conformal_extrinsic,
               energy_density,
               momentum_density,
               stress_trace,
               conformal_boundary,
               lapse_boundary,
               vector_boundary,
               xcts_options
            );
            result.constraints = {
               constraints.coupled_iterations,
               constraints.relative_update,
               constraints.hamiltonian_residual,
               constraints.lapse_residual,
               constraints.momentum_residual,
               constraints.converged,
            };
         }
         if (!result.constraints.converged) {
            amrex::Print()
               << conformal_constraint_formulation_name(formulation)
               << " initial-data solve failed: iterations="
               << result.constraints.coupled_iterations
               << " relative_update=" << result.constraints.relative_update
               << " hamiltonian_residual="
               << result.constraints.hamiltonian_residual
               << " lapse_residual=" << result.constraints.lapse_residual
               << " momentum_residual=" << result.constraints.momentum_residual
               << '\n';
            amrex::Abort("conformal constraint initial-data solve did not converge");
         }

         amrex::Real update_norm = 0.0;
         amrex::Real solution_norm = 0.0;
         for (int level = 0; level < num_levels; ++level) {
            amrex::MultiFab::Subtract(
               *previous_primitive[level],
               *primitive_state[level],
               0,
               0,
               primitive_state[level]->nComp(),
               0
            );
            for (int component = 0;
                 component < primitive_state[level]->nComp();
                 ++component) {
               update_norm = std::max(
                  update_norm,
                  previous_primitive[level]->norm0(component, 0, false)
               );
               solution_norm = std::max(
                  solution_norm,
                  primitive_state[level]->norm0(component, 0, false)
               );
            }
            amrex::MultiFab::Subtract(
               *previous_conformal[level], *conformal_storage[level], 0, 0, 1, 0
            );
            update_norm = std::max(
               update_norm, previous_conformal[level]->norm0(0, 0, false)
            );
            solution_norm = std::max(
               solution_norm, conformal_storage[level]->norm0(0, 0, false)
            );
            amrex::MultiFab::Subtract(
               *previous_vector[level], *vector_storage[level], 0, 0, 3, 0
            );
            for (int component = 0; component < 3; ++component) {
               update_norm = std::max(
                  update_norm, previous_vector[level]->norm0(component, 0, false)
               );
               solution_norm = std::max(
                  solution_norm, vector_storage[level]->norm0(component, 0, false)
               );
            }
            if (formulation == ConformalConstraintFormulation::xcts) {
               amrex::MultiFab::Subtract(
                  *previous_lapse[level],
                  *lapse_conformal_storage[level],
                  0,
                  0,
                  1,
                  0
               );
               update_norm = std::max(
                  update_norm, previous_lapse[level]->norm0(0, 0, false)
               );
               solution_norm = std::max(
                  solution_norm, lapse_conformal_storage[level]->norm0(0, 0, false)
               );
            } else if (formulation == ConformalConstraintFormulation::cttk) {
               amrex::MultiFab::Subtract(
                  *previous_curvature[level], *curvature_storage[level], 0, 0, 1, 0
               );
               update_norm = std::max(
                  update_norm, previous_curvature[level]->norm0(0, 0, false)
               );
               solution_norm = std::max(
                  solution_norm, curvature_storage[level]->norm0(0, 0, false)
               );
            }
         }
         step_matter_iterations = matter_iteration;
         result.relative_matter_update = update_norm
            / std::max(amrex::Real(1.0), solution_norm);
         if (update_rotating_matter && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print()
               << "rotating continuation step=" << continuation_step
               << "/" << continuation_steps
               << " coupled_update_iteration=" << matter_iteration
               << " relative_update=" << result.relative_matter_update << '\n';
         }
         if (result.relative_matter_update <= matter_tolerance) {
            break;
         }
      }
      result.matter_iterations += step_matter_iterations;
      if (result.relative_matter_update > matter_tolerance) {
         amrex::Abort(
            "matter-coupled conformal fixed-point iteration did not converge "
            "at rotation continuation step "
            + std::to_string(continuation_step)
         );
      }
   }

   materialize_conformal_geometry(
      hierarchy,
      conformal_factor_const,
      lapse_conformal_view,
      shift_view,
      mean_curvature,
      conformal_extrinsic_const,
      spacetime_state,
      cell_spacetime
   );
   if (update_rotating_matter) {
      RotatingStarParameters diagnostic_parameters = rotating_star;
      diagnostic_parameters.injection_energy = result.rotating_star.injection_energy;
      amrex::Vector<amrex::MultiFab const*> final_cell_spacetime(num_levels);
      for (int level = 0; level < num_levels; ++level) {
         final_cell_spacetime[level] = cell_spacetime[level];
      }
      result.rotating_star.mass_shedding_indicator =
         rotating_star_mass_shedding_indicator(
            hierarchy, final_cell_spacetime, diagnostic_parameters
         );
   }
   for (int level = 0; level < num_levels; ++level) {
      for (amrex::MFIter mfi(*fluid_state[level], amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto cell_geometry = cell_spacetime[level]->const_array(mfi);
         const auto primitive = primitive_state[level]->const_array(mfi);
         const auto fluid = fluid_state[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const Z4cState geometry = load_z4c_state(
               cell_geometry, i, j, k
            );
            const SpacetimeMetric metric = generated::z4c_spacetime_metric(geometry);
            if constexpr (System == PhysicalSystem::RHD) {
               store_rhd_conserved(
                  fluid,
                  i,
                  j,
                  k,
                  generated::rhd_primitive_to_conserved(
                     rhd_primitive_at<EOSMethod>(primitive, i, j, k, eos), metric
                  )
               );
            } else {
               store_rmhd_conserved(
                  fluid,
                  i,
                  j,
                  k,
                  generated::rmhd_primitive_to_conserved(
                     rmhd_primitive_at(primitive, i, j, k, eos), metric
                  )
               );
            }
         });
      }
   }
   return result;
}

} // namespace

ConformalMaterializationResult materialize_conformal_initial_data(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& fluid_state,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab*> const& spacetime_state,
   amrex::Vector<amrex::MultiFab*> const& cell_spacetime,
   PhysicalSystem system,
   int eos_method,
   EOSParams const& eos,
   InitialDataFieldSelection const& fields,
   ConformalConstraintFormulation formulation,
   RotatingStarParameters const& rotating_star
)
{
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(
      hierarchy, fluid_state, primitive_state, spacetime_state,
      cell_spacetime, system, eos_method, eos, fields, formulation,
      rotating_star
   );
   amrex::Abort("conformal constraint initial data requires the 3D executable");
   return {};
#else
   if (system == PhysicalSystem::RMHD) {
      return materialize_conformal_initial_data_method<
         PhysicalSystem::RMHD,
         EOSIdealGas
      >(
         hierarchy, fluid_state, primitive_state, spacetime_state,
         cell_spacetime, eos, fields, formulation, rotating_star
      );
   }
   if (system != PhysicalSystem::RHD) {
      amrex::Abort("conformal constraint initial data requires RHD or RMHD");
   }
   if (eos_method == EOSPolytrope) {
      return materialize_conformal_initial_data_method<
         PhysicalSystem::RHD,
         EOSPolytrope
      >(
         hierarchy, fluid_state, primitive_state, spacetime_state,
         cell_spacetime, eos, fields, formulation, rotating_star
      );
   }
   if (eos_method == EOSHybridPolytrope) {
      return materialize_conformal_initial_data_method<
         PhysicalSystem::RHD,
         EOSHybridPolytrope
      >(
         hierarchy, fluid_state, primitive_state, spacetime_state,
         cell_spacetime, eos, fields, formulation, rotating_star
      );
   }
   return materialize_conformal_initial_data_method<
      PhysicalSystem::RHD,
      EOSIdealGas
   >(
      hierarchy, fluid_state, primitive_state, spacetime_state,
      cell_spacetime, eos, fields, formulation, rotating_star
   );
#endif
}

} // namespace pyfluid_amrex
