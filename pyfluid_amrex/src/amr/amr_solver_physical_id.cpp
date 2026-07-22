#include "amr/amr_solver_base.hpp"

#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/initial_data/physical_id.hpp"
#include "pyfluid_amrex/z4c_runtime.hpp"

#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace pyfluid_amrex {

void AmrSolverBase::materialize_configured_initial_data()
{
   if (cfg_.initial_data_input_file.empty()) {
      materialize_constraint_solved_initial_data();
      return;
   }
   if (load_physical_initial_data_artifact()) {
      materialize_constraint_solved_initial_data(true);
   }
}

bool AmrSolverBase::load_physical_initial_data_artifact()
{
   amrex::Vector<amrex::Geometry> geometry;
   amrex::Vector<amrex::IntVect> refinement_ratios;
   MutablePhysicalIdField primitive;
   MutablePhysicalIdField fluid;
   MutablePhysicalIdField spacetime;
   MutablePhysicalIdFaceFields face_magnetic;
   for (int level = 0; level <= finestLevel(); ++level) {
      geometry.push_back(Geom(level));
      primitive.push_back(primitive_[level].get());
      fluid.push_back(fluid_state_.state()[level].get());
      if (spacetime_state_.active()) {
         spacetime.push_back(spacetime_state_.state()[level].get());
      }
      if (ct_enabled(cfg_)) {
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            face_magnetic[direction].push_back(face_b_[level].b[direction].get());
         }
      }
      if (level < finestLevel()) {
         refinement_ratios.push_back(refRatio(level));
      }
   }
   const PhysicalInitialDataMetadata metadata = read_physical_initial_data(
      cfg_.initial_data_input_file,
      cfg_,
      physics_system_,
      geometry,
      refinement_ratios,
      primitive,
      fluid,
      spacetime,
      face_magnetic
   );
   if (!metadata.used_exact_cache) {
      ConstPhysicalIdField cell_spacetime;
      if (spacetime_state_.active()) {
         for (int level = 0; level <= finestLevel(); ++level) {
            spacetime_coupling_.invalidate_cell_spacetime(level);
         }
         fill_all_spacetime_boundaries(spacetime_state_.state(), 0.0);
         for (int level = 0; level <= finestLevel(); ++level) {
            complete_z4c_contracted_connections(
               *spacetime_state_.state()[level], Geom(level)
            );
         }
         fill_all_spacetime_boundaries(spacetime_state_.state(), 0.0);
         prepare_spacetime_coupling_range(
            spacetime_state_.state(),
            0,
            finestLevel() + 1,
            0.0,
            -1.0,
            0
         );
         for (int level = 0; level <= finestLevel(); ++level) {
            cell_spacetime.push_back(
               spacetime_coupling_.cell_spacetime()[level].get()
            );
         }
      }
      materialize_physical_initial_data_fluid(
         cfg_,
         physics_system_,
         geometry,
         ConstPhysicalIdField(primitive.begin(), primitive.end()),
         cell_spacetime,
         fluid
      );
      if (ct_enabled(cfg_)) {
         project_materialized_initial_face_b();
      }
      amrex::Print()
         << "materialized physical initial data onto target hierarchy\n";
   }
   if (ct_enabled(cfg_)) {
      sync_all_cell_b_from_faces(fluid_state_.state(), false);
   }
   for (int level = 0; level <= finestLevel(); ++level) {
      if (spacetime_state_.active()) {
         spacetime_coupling_.invalidate_cell_spacetime(level);
      }
      copy_level_state_to_old(level);
   }
   amrex::Print()
      << "loaded physical initial data: " << cfg_.initial_data_input_file << '\n';
   const bool configured_rotation = metadata.source_ic == "rotating_star"
      && cfg_.ic == "rotating_star";
   if (configured_rotation
       && metadata.rotation_law != cfg_.initial_data_rotation_law) {
      amrex::Abort(
         "rotating-star physical ID continuation cannot change rotation law"
      );
   }
   if (configured_rotation
       && (metadata.rotation_center[0] != cfg_.initial_data_rotation_center1
           || metadata.rotation_center[1] != cfg_.initial_data_rotation_center2
           || metadata.rotation_center[2] != cfg_.initial_data_rotation_center3)) {
      amrex::Abort(
         "rotating-star physical ID continuation cannot change the rotation center"
      );
   }
   const bool continue_rotation = configured_rotation
      && metadata.angular_velocity != cfg_.initial_data_angular_velocity;
   if (continue_rotation) {
      if (cfg_.spacetime_initial_data != "xcts") {
         amrex::Abort(
            "saved rotating-star continuation currently requires XCTS"
         );
      }
      if (metadata.angular_velocity * cfg_.initial_data_angular_velocity < 0.0
          || std::abs(metadata.angular_velocity)
             >= std::abs(cfg_.initial_data_angular_velocity)) {
         amrex::Abort(
            "rotating-star physical ID continuation requires a larger target "
            "angular velocity with the saved rotation sign"
         );
      }
      amrex::Real previous_magnitude = std::abs(metadata.angular_velocity);
      for (amrex::Real target :
           cfg_.initial_data_rotation_continuation_targets) {
         if (!(std::abs(target) > previous_magnitude)) {
            amrex::Abort(
               "saved rotating-star continuation targets must all exceed the "
               "saved angular-velocity magnitude"
            );
         }
         previous_magnitude = std::abs(target);
      }
      amrex::Print()
         << "continuing rotating physical initial data: source_omega="
         << metadata.angular_velocity
         << " target_omega=" << cfg_.initial_data_angular_velocity << '\n';
   }
   return continue_rotation;
}

void AmrSolverBase::project_materialized_initial_face_b()
{
#if (AMREX_SPACEDIM > 1)
   const int num_levels = finestLevel() + 1;
   amrex::Vector<amrex::Geometry> geometry(num_levels);
   amrex::Vector<amrex::BoxArray> grids(num_levels);
   amrex::Vector<amrex::DistributionMapping> distribution(num_levels);
   amrex::Vector<FaceMagneticField> logical_face_b(num_levels);
   amrex::Vector<FaceMagneticField> gradient(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> potential_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> rhs_storage(num_levels);
   amrex::Vector<amrex::MultiFab*> potential(num_levels);
   amrex::Vector<amrex::MultiFab const*> rhs(num_levels);
   amrex::Real maximum_divergence = 0.0;
   for (int level = 0; level < num_levels; ++level) {
      geometry[level] = Geom(level);
      grids[level] = boxArray(level);
      distribution[level] = DistributionMap(level);
      allocate_face_magnetic_field_like(logical_face_b[level], face_b_[level]);
      allocate_face_magnetic_field_like(gradient[level], face_b_[level]);
      fill_face_fillpatch_from_face_b(
         logical_face_b[level], face_b_[level], Geom(level), cfg_
      );
      rhs_storage[level] = std::make_unique<amrex::MultiFab>(
         grids[level], distribution[level], 1, 0
      );
      fill_face_divergence(
         *rhs_storage[level], logical_face_b[level], Geom(level), false
      );
      maximum_divergence = std::max(
         maximum_divergence, rhs_storage[level]->norm0(0, 0, true)
      );
      potential_storage[level] = std::make_unique<amrex::MultiFab>(
         grids[level], distribution[level], 1, 1
      );
      potential_storage[level]->setVal(0.0);
      potential[level] = potential_storage[level].get();
      rhs[level] = rhs_storage[level].get();
   }
   amrex::ParallelDescriptor::ReduceRealMax(maximum_divergence);
   if (maximum_divergence <= 64.0 * std::numeric_limits<amrex::Real>::epsilon()) {
      return;
   }
   amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lower_boundary{};
   amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> upper_boundary{};
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      const amrex::LinOpBCType type = Geom(0).isPeriodic(direction)
         ? amrex::LinOpBCType::Periodic
         : amrex::LinOpBCType::Dirichlet;
      lower_boundary[direction] = type;
      upper_boundary[direction] = type;
   }
   amrex::LPInfo info;
   amrex::MLPoisson poisson(geometry, grids, distribution, info);
   poisson.setMaxOrder(2);
   poisson.setDomainBC(lower_boundary, upper_boundary);
   for (int level = 0; level < num_levels; ++level) {
      poisson.setLevelBC(level, nullptr);
   }
   amrex::MLMG mlmg(poisson);
   mlmg.setMaxIter(200);
   mlmg.setMaxFmgIter(1);
   mlmg.setVerbose(0);
   const amrex::Real residual = mlmg.solve(potential, rhs, 1.0e-12, 0.0);
   if (!std::isfinite(residual)) {
      amrex::Abort("physical-ID magnetic projection produced a nonfinite residual");
   }
   amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> gradients(
      num_levels
   );
   for (int level = 0; level < num_levels; ++level) {
      gradients[level] = face_b_array(gradient[level]);
   }
   mlmg.getGradSolution(gradients);
   for (int level = 0; level < num_levels; ++level) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         amrex::MultiFab::Add(
            *logical_face_b[level].b[direction],
            *gradient[level].b[direction],
            0,
            0,
            1,
            0
         );
      }
      fill_face_b_from_face_fillpatch(
         face_b_[level], logical_face_b[level], Geom(level), cfg_
      );
   }
   amrex::Print()
      << "projected materialized physical-ID magnetic field: initial_divb_linf="
      << maximum_divergence << " residual=" << residual << '\n';
#endif
}

void AmrSolverBase::write_physical_initial_data_artifact()
{
   amrex::Vector<amrex::Geometry> geometry;
   amrex::Vector<amrex::IntVect> refinement_ratios;
   ConstPhysicalIdField primitive;
   ConstPhysicalIdField fluid;
   ConstPhysicalIdField spacetime;
   ConstPhysicalIdFaceFields face_magnetic;
   for (int level = 0; level <= finestLevel(); ++level) {
      fill_level_primitive_cache(level, 0.0);
      geometry.push_back(Geom(level));
      primitive.push_back(primitive_[level].get());
      fluid.push_back(fluid_state_.state()[level].get());
      if (spacetime_state_.active()) {
         spacetime.push_back(spacetime_state_.state()[level].get());
      }
      if (ct_enabled(cfg_)) {
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            face_magnetic[direction].push_back(
               face_b_[level].b[direction].get()
            );
         }
      }
      if (level < finestLevel()) {
         refinement_ratios.push_back(refRatio(level));
      }
   }
   write_physical_initial_data(
      cfg_.initial_data_output_file,
      cfg_,
      physics_system_,
      geometry,
      refinement_ratios,
      primitive,
      fluid,
      spacetime,
      face_magnetic
   );
   amrex::Print()
      << "wrote physical initial data: " << cfg_.initial_data_output_file << '\n';
}

} // namespace pyfluid_amrex
