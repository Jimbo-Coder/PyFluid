#pragma once

#include <AMReX_FillPatchUtil.H>
#include <AMReX_MultiFab.H>

#include <memory>

#include "pyfluid_amrex/nodal_interpolater.hpp"

namespace pyfluid_amrex {

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
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> aliased_fine_storage(
      fine_data.size()
   );
   amrex::Vector<amrex::MultiFab*> fine_sources = fine_data;
   for (int source_index = 0;
        source_index < static_cast<int>(fine_data.size());
        ++source_index) {
      if (fine_data[source_index] == &destination) {
         aliased_fine_storage[source_index] = std::make_unique<amrex::MultiFab>(
            fine_data[source_index]->boxArray(),
            fine_data[source_index]->DistributionMap(),
            num_components,
            0
         );
         amrex::MultiFab::Copy(
            *aliased_fine_storage[source_index],
            *fine_data[source_index],
            0,
            0,
            num_components,
            0
         );
         fine_sources[source_index] = aliased_fine_storage[source_index].get();
      }
   }
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
      nodal_interpolater(),
      boundary_conditions,
      0
   );
   amrex::FillPatchSingleLevel(
      destination,
      num_ghost,
      time,
      fine_sources,
      fine_times,
      0,
      0,
      num_components,
      fine_geometry,
      fine_bc,
      0
   );
}

} // namespace pyfluid_amrex
