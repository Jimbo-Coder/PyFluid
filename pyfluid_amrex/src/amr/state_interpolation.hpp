#pragma once

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_Interpolater.H>

#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/fixed_spacetime_types.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

class MeshBasisStateInterpolater final : public amrex::Interpolater {
public:
   void configure(
      amrex::Interpolater* base,
      int coordinate,
      MeshAxes const& axes,
      bool transform_state_vectors,
      PhysicalSystem physical_system,
      int eos_method,
      EOSParams const& eos,
      FixedSpacetimeParameters const& fixed_spacetime,
      bool preserve_relativistic_admissibility
   );

   amrex::Box CoarseBox(amrex::Box const& fine, int ratio) override;

   amrex::Box CoarseBox(
      amrex::Box const& fine,
      amrex::IntVect const& ratio
   ) override;

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
   ) override;

private:
   void limit_relativistic_child_deviations(
      amrex::FArrayBox const& coarse,
      int coarse_component,
      amrex::FArrayBox& fine,
      int fine_component,
      int num_components,
      amrex::Box const& fine_region,
      amrex::IntVect const& ratio,
      amrex::Geometry const& fine_geometry,
      int actual_component,
      amrex::RunOn run_on
   ) const;

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
   ) const;

   amrex::Interpolater* base_ = nullptr;
   int coordinate_ = MeshCartesian;
   MeshAxes axes_{};
   bool transform_state_vectors_ = false;
   PhysicalSystem physical_system_ = PhysicalSystem::HD;
   int eos_method_ = 0;
   EOSParams eos_{};
   FixedSpacetimeParameters fixed_spacetime_{};
   bool preserve_relativistic_admissibility_ = false;
};

} // namespace pyfluid_amrex
