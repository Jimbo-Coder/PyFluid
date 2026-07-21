#include "pyfluid_amrex/initial_data/centering.hpp"

#include <AMReX.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PhysBCFunct.H>

#include "pyfluid_amrex/centering_transfer.hpp"
#include "pyfluid_amrex/nodal_fill.hpp"
#include "pyfluid_amrex/nodal_interpolater.hpp"

namespace pyfluid_amrex {

void detail::fill_initial_data_vertex_ghosts(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& fields,
   amrex::Vector<amrex::BCRec> const& boundary_conditions
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   if (num_levels == 0
       || static_cast<int>(hierarchy.grids.size()) != num_levels
       || static_cast<int>(hierarchy.distribution.size()) != num_levels
       || static_cast<int>(hierarchy.refinement_ratio.size()) != num_levels - 1
       || static_cast<int>(fields.size()) != num_levels
       || fields.front() == nullptr) {
      amrex::Abort("invalid initial-data vertex hierarchy ghost fill");
   }
   const int num_components = fields.front()->nComp();
   if (num_components <= 0
       || static_cast<int>(boundary_conditions.size()) != num_components) {
      amrex::Abort("invalid initial-data vertex boundary component layout");
   }

   amrex::GpuBndryFuncFab<amrex::FabFillNoOp> boundary_function;
   for (int level = 0; level < num_levels; ++level) {
      if (fields[level] == nullptr
          || fields[level]->nComp() != num_components
          || fields[level]->nGrow() < 1
          || centering_mask(fields[level]->ixType())
             != (1 << AMREX_SPACEDIM) - 1) {
         amrex::Abort("incompatible initial-data vertex field hierarchy");
      }
      fields[level]->OverrideSync(hierarchy.geometry[level].periodicity());
      amrex::PhysBCFunct<amrex::GpuBndryFuncFab<amrex::FabFillNoOp>> fine_bc(
         hierarchy.geometry[level], boundary_conditions, boundary_function
      );
      amrex::Vector<amrex::MultiFab*> fine_data{fields[level]};
      amrex::Vector<amrex::Real> fine_time{0.0};
      if (level == 0) {
         amrex::FillPatchSingleLevel(
            *fields[level],
            fields[level]->nGrowVect(),
            0.0,
            fine_data,
            fine_time,
            0,
            0,
            num_components,
            hierarchy.geometry[level],
            fine_bc,
            0
         );
         continue;
      }
      amrex::PhysBCFunct<amrex::GpuBndryFuncFab<amrex::FabFillNoOp>> coarse_bc(
         hierarchy.geometry[level - 1], boundary_conditions, boundary_function
      );
      amrex::Vector<amrex::MultiFab*> coarse_data{fields[level - 1]};
      amrex::Vector<amrex::Real> coarse_time{0.0};
      fill_nodal_two_levels(
         *fields[level],
         fields[level]->nGrowVect(),
         0.0,
         coarse_data,
         coarse_time,
         fine_data,
         fine_time,
         hierarchy.geometry[level - 1],
         hierarchy.geometry[level],
         coarse_bc,
         fine_bc,
         hierarchy.refinement_ratio[level - 1],
         boundary_conditions,
         num_components
      );
   }
}

void materialize_initial_data_vertices(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& vertex_fields,
   amrex::Vector<amrex::MultiFab const*> const& source_fields,
   amrex::Vector<amrex::BCRec> const& boundary_conditions,
   int transfer_order
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   if (num_levels == 0
       || static_cast<int>(hierarchy.grids.size()) != num_levels
       || static_cast<int>(hierarchy.distribution.size()) != num_levels
       || static_cast<int>(hierarchy.refinement_ratio.size()) != num_levels - 1
       || static_cast<int>(vertex_fields.size()) != num_levels
       || static_cast<int>(source_fields.size()) != num_levels) {
      amrex::Abort("invalid initial-data vertex hierarchy materialization");
   }
   if (boundary_conditions.empty()) {
      amrex::Abort("initial-data vertex hierarchy requires boundary conditions");
   }
   for (int level = 0; level < num_levels; ++level) {
      if (vertex_fields[level] == nullptr || source_fields[level] == nullptr) {
         amrex::Abort("initial-data vertex hierarchy contains an unallocated level");
      }
      if (centering_mask(vertex_fields[level]->ixType())
          != (1 << AMREX_SPACEDIM) - 1) {
         amrex::Abort("initial-data vertex hierarchy requires nodal destinations");
      }
      if (centering_mask(source_fields[level]->ixType()) != 0) {
         amrex::Abort("initial-data vertex hierarchy requires cell-centered sources");
      }
      if (static_cast<int>(boundary_conditions.size())
          != vertex_fields[level]->nComp()) {
         amrex::Abort(
            "initial-data vertex hierarchy boundary/component count mismatch"
         );
      }
      transfer_grid_field_centering(
         *vertex_fields[level],
         *source_fields[level],
         hierarchy.geometry[level],
         transfer_order
      );
   }
   for (int level = num_levels - 1; level > 0; --level) {
      amrex::average_down_nodal(
         *vertex_fields[level],
         *vertex_fields[level - 1],
         hierarchy.refinement_ratio[level - 1]
      );
   }
}

} // namespace pyfluid_amrex
