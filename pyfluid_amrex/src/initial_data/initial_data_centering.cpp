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

namespace {

amrex::Vector<amrex::BCRec> reflection_only_records(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::BCRec> records
)
{
   for (amrex::BCRec& record : records) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         if (hierarchy.reflection_lo[direction] == 0) {
            record.setLo(direction, amrex::BCType::int_dir);
         }
         if (hierarchy.reflection_hi[direction] == 0) {
            record.setHi(direction, amrex::BCType::int_dir);
         }
      }
   }
   return records;
}

void fill_nodal_physical_boundaries(
   amrex::MultiFab& field,
   amrex::Geometry const& geometry,
   amrex::Vector<amrex::BCRec> const& boundary_conditions
)
{
   bool has_reflection = false;
   for (amrex::BCRec const& record : boundary_conditions) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         has_reflection = has_reflection
            || record.lo(direction) == amrex::BCType::reflect_even
            || record.lo(direction) == amrex::BCType::reflect_odd
            || record.hi(direction) == amrex::BCType::reflect_even
            || record.hi(direction) == amrex::BCType::reflect_odd;
      }
   }
   if (!has_reflection) {
      return;
   }
   const amrex::Box domain = amrex::surroundingNodes(geometry.Domain());
   const amrex::Dim3 domain_lo = amrex::lbound(domain);
   const amrex::Dim3 domain_hi = amrex::ubound(domain);
   for (int component = 0; component < field.nComp(); ++component) {
      amrex::GpuArray<int, AMREX_SPACEDIM> lower_type{};
      amrex::GpuArray<int, AMREX_SPACEDIM> upper_type{};
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         lower_type[direction] = boundary_conditions[component].lo(direction);
         upper_type[direction] = boundary_conditions[component].hi(direction);
      }
      for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.fabbox();
         const auto values = field.array(mfi);
         amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               int source_i = i;
               int source_j = j;
               int source_k = k;
               int parity = 1;
               bool physical = false;
               bool odd_plane = false;
               auto reflect = [&] AMREX_GPU_DEVICE(
                  int index,
                  int lo,
                  int hi,
                  int lo_type,
                  int hi_type,
                  int& source
               ) noexcept {
                  if (lo_type == amrex::BCType::int_dir
                      && hi_type == amrex::BCType::int_dir) {
                     return;
                  }
                  if (index < lo) {
                     if (lo_type == amrex::BCType::reflect_even
                         || lo_type == amrex::BCType::reflect_odd) {
                        physical = true;
                        source = 2 * lo - index;
                        if (lo_type == amrex::BCType::reflect_odd) {
                           parity *= -1;
                        }
                     }
                  } else if (index > hi) {
                     if (hi_type == amrex::BCType::reflect_even
                         || hi_type == amrex::BCType::reflect_odd) {
                        physical = true;
                        source = 2 * hi - index;
                        if (hi_type == amrex::BCType::reflect_odd) {
                           parity *= -1;
                        }
                     }
                  } else if (
                     (index == lo && lo_type == amrex::BCType::reflect_odd)
                     || (index == hi && hi_type == amrex::BCType::reflect_odd)
                  ) {
                     odd_plane = true;
                  }
               };
               reflect(
                  i, domain_lo.x, domain_hi.x,
                  lower_type[0], upper_type[0], source_i
               );
#if (AMREX_SPACEDIM >= 2)
               reflect(
                  j, domain_lo.y, domain_hi.y,
                  lower_type[1], upper_type[1], source_j
               );
#endif
#if (AMREX_SPACEDIM == 3)
               reflect(
                  k, domain_lo.z, domain_hi.z,
                  lower_type[2], upper_type[2], source_k
               );
#endif
               if (odd_plane) {
                  values(i, j, k, component) = 0.0;
               } else if (physical) {
                  values(i, j, k, component) =
                     parity * values(source_i, source_j, source_k, component);
               }
            }
         );
      }
   }
}

} // namespace

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
   const amrex::Vector<amrex::BCRec> reflection_boundaries =
      reflection_only_records(hierarchy, boundary_conditions);

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
         fill_nodal_physical_boundaries(
            *fields[level], hierarchy.geometry[level], reflection_boundaries
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
      fill_nodal_physical_boundaries(
         *fields[level], hierarchy.geometry[level], reflection_boundaries
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
