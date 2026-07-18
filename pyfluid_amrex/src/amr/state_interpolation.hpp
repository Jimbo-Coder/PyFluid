#pragma once

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Interpolater.H>

#include "pyfluid_amrex/coordinate_locations.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

class MeshBasisStateInterpolater final : public amrex::Interpolater {
public:
   void configure(
      amrex::Interpolater* base,
      int coordinate,
      MeshAxes const& axes,
      bool transform_state_vectors
   )
   {
      base_ = base;
      coordinate_ = coordinate;
      axes_ = axes;
      transform_state_vectors_ = transform_state_vectors;
   }

   amrex::Box CoarseBox(amrex::Box const& fine, int ratio) override
   {
      return base().CoarseBox(fine, ratio);
   }

   amrex::Box CoarseBox(amrex::Box const& fine, amrex::IntVect const& ratio) override
   {
      return base().CoarseBox(fine, ratio);
   }

   void interp(
      amrex::FArrayBox const& coarse,
      int coarse_component,
      amrex::FArrayBox& fine,
      int fine_component,
      int num_components,
      amrex::Box const& fine_region,
      amrex::IntVect const& ratio,
      amrex::Geometry const& coarse_geometry,
      amrex::Geometry const& fine_geometry,
      amrex::Vector<amrex::BCRec> const& boundary_conditions,
      int actual_component,
      int actual_state,
      amrex::RunOn run_on
   ) override
   {
      if (!transform_state_vectors_ || coordinate_ == MeshCartesian) {
         base().interp(
            coarse,
            coarse_component,
            fine,
            fine_component,
            num_components,
            fine_region,
            ratio,
            coarse_geometry,
            fine_geometry,
            boundary_conditions,
            actual_component,
            actual_state,
            run_on
         );
         return;
      }

      amrex::FArrayBox cartesian_coarse(coarse.box(), num_components);
      transform_state_vectors(
         coarse,
         coarse_component,
         cartesian_coarse,
         0,
         num_components,
         coarse.box(),
         coarse_geometry,
         true
      );
      amrex::FArrayBox cartesian_fine(fine.box(), num_components);
      base().interp(
         cartesian_coarse,
         0,
         cartesian_fine,
         0,
         num_components,
         fine_region,
         ratio,
         coarse_geometry,
         fine_geometry,
         boundary_conditions,
         actual_component,
         actual_state,
         run_on
      );
      transform_state_vectors(
         cartesian_fine,
         0,
         fine,
         fine_component,
         num_components,
         fine_region,
         fine_geometry,
         false
      );
   }

private:
   amrex::Interpolater& base() const
   {
      AMREX_ALWAYS_ASSERT(base_ != nullptr);
      return *base_;
   }

   void transform_state_vectors(
      amrex::FArrayBox const& source,
      int source_component,
      amrex::FArrayBox& destination,
      int destination_component,
      int num_components,
      amrex::Box const& box,
      amrex::Geometry const& geometry,
      bool to_cartesian
   ) const
   {
      const auto source_array = source.const_array();
      const auto destination_array = destination.array();
      const auto problo = geometry.ProbLoArray();
      const auto dx = geometry.CellSizeArray();
      const int coordinate = coordinate_;
      const MeshAxes axes = axes_;
      amrex::ParallelFor(box, num_components, [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
         const int state_component = component;
         const int vector_begin =
            state_component >= MOM1 && state_component <= MOM3
               ? MOM1
               : (state_component >= B1 && state_component <= B3 ? B1 : -1);
         if (vector_begin < 0 || vector_begin + 2 >= num_components) {
            destination_array(i, j, k, destination_component + component) =
               source_array(i, j, k, source_component + component);
            return;
         }

         const StateVector3 source_vector{
            source_array(i, j, k, source_component + vector_begin),
            source_array(i, j, k, source_component + vector_begin + 1),
            source_array(i, j, k, source_component + vector_begin + 2),
         };
         const auto q = mesh_coordinate_triplet_at_location(
            coordinate, axes, problo, dx, i, j, k, false, false, false
         );
         const StateVector3 transformed = to_cartesian
            ? mesh_basis_vector_to_cartesian(coordinate, axes, q, source_vector)
            : cartesian_vector_to_mesh_basis(coordinate, axes, q, source_vector);
         const int vector_component = state_component - vector_begin;
         destination_array(i, j, k, destination_component + component) = vector_component == 0
            ? transformed.x
            : (vector_component == 1 ? transformed.y : transformed.z);
      });
   }

   amrex::Interpolater* base_ = nullptr;
   int coordinate_ = MeshCartesian;
   MeshAxes axes_{};
   bool transform_state_vectors_ = false;
};

} // namespace pyfluid_amrex
