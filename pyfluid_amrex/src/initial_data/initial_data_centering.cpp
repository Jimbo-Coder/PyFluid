#include "pyfluid_amrex/initial_data/centering.hpp"

#include <AMReX.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_GpuAsyncArray.H>
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

template <int Order>
void transfer_initial_data_cell_to_vertices_with_reflection(
   amrex::MultiFab& destination,
   amrex::MultiFab const& source,
   amrex::Geometry const& geometry,
   amrex::GpuArray<int, AMREX_SPACEDIM> const& reflection_lo,
   amrex::GpuArray<int, AMREX_SPACEDIM> const& reflection_hi
)
{
   static_assert(Order == 2 || Order == 4 || Order == 6 || Order == 8);
   constexpr int destination_mask = (1 << AMREX_SPACEDIM) - 1;
   if (source.nComp() != destination.nComp()
       || centering_mask(source.ixType()) != 0
       || centering_mask(destination.ixType()) != destination_mask) {
      amrex::Abort("invalid reflected cell-to-vertex transfer layout");
   }
   amrex::BoxArray expected_boxes = source.boxArray();
   expected_boxes.convert(destination.ixType());
   if (destination.boxArray() != expected_boxes) {
      amrex::Abort("reflected centering transfer requires one cell layout");
   }
   const amrex::Box cell_domain = geometry.Domain();
   amrex::GpuArray<int, 3> source_lo{{0, 0, 0}};
   amrex::GpuArray<int, 3> source_hi{{0, 0, 0}};
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      const int ghost_cells = source.nGrowVect()[direction];
      if (ghost_cells < Order / 2) {
         amrex::Abort(
            "reflected centering transfer source has insufficient ghosts"
         );
      }
      source_lo[direction] = cell_domain.smallEnd(direction);
      source_hi[direction] = cell_domain.bigEnd(direction);
      if (geometry.isPeriodic(direction)
          || reflection_lo[direction] != 0) {
         source_lo[direction] -= ghost_cells;
      }
      if (geometry.isPeriodic(direction)
          || reflection_hi[direction] != 0) {
         source_hi[direction] += ghost_cells;
      }
   }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(destination, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto input = source.const_array(mfi);
      const auto output = destination.array(mfi);
      amrex::ParallelFor(
         box,
         destination.nComp(),
         [=] AMREX_GPU_DEVICE(
            int i, int j, int k, int component
         ) noexcept {
            output(i, j, k, component) =
               generated::bounded_topology_centering_transfer<
                  Order, AMREX_SPACEDIM
               >(
                  input,
                  i,
                  j,
                  k,
                  component,
                  0,
                  destination_mask,
                  source_lo,
                  source_hi
               );
         }
      );
   }
}

void transfer_initial_data_cell_to_vertices_with_reflection(
   amrex::MultiFab& destination,
   amrex::MultiFab const& source,
   amrex::Geometry const& geometry,
   int order,
   amrex::GpuArray<int, AMREX_SPACEDIM> const& reflection_lo,
   amrex::GpuArray<int, AMREX_SPACEDIM> const& reflection_hi
)
{
   switch (order) {
   case 2:
      transfer_initial_data_cell_to_vertices_with_reflection<2>(
         destination, source, geometry, reflection_lo, reflection_hi
      );
      return;
   case 4:
      transfer_initial_data_cell_to_vertices_with_reflection<4>(
         destination, source, geometry, reflection_lo, reflection_hi
      );
      return;
   case 6:
      transfer_initial_data_cell_to_vertices_with_reflection<6>(
         destination, source, geometry, reflection_lo, reflection_hi
      );
      return;
   case 8:
      transfer_initial_data_cell_to_vertices_with_reflection<8>(
         destination, source, geometry, reflection_lo, reflection_hi
      );
      return;
   default:
      amrex::Abort("initial-data centering transfer order is unsupported");
   }
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
      bool mapped = false;
      std::array<InitialDataCoordinateLine, AMREX_SPACEDIM> coordinates;
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         coordinates[direction] = hierarchy.coordinate_line(level, direction);
         mapped = mapped || !coordinates[direction].uniform;
      }
      if (!mapped) {
         bool has_reflection = false;
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            has_reflection = has_reflection
               || hierarchy.reflection_lo[direction] != 0
               || hierarchy.reflection_hi[direction] != 0;
         }
         if (has_reflection) {
            transfer_initial_data_cell_to_vertices_with_reflection(
               *vertex_fields[level],
               *source_fields[level],
               hierarchy.geometry[level],
               transfer_order,
               hierarchy.reflection_lo,
               hierarchy.reflection_hi
            );
         } else {
            transfer_grid_field_centering(
               *vertex_fields[level],
               *source_fields[level],
               hierarchy.geometry[level],
               transfer_order
            );
         }
         continue;
      }
      if (num_levels != 1 || transfer_order != 2
          || source_fields[level]->nGrow() < 1) {
         amrex::Abort(
            "mapped initial-data vertex materialization currently requires "
            "one level, second order, and one source ghost cell"
         );
      }
      const amrex::Dim3 cell_lo = amrex::lbound(
         hierarchy.geometry[level].Domain()
      );
      const amrex::Dim3 cell_hi = amrex::ubound(
         hierarchy.geometry[level].Domain()
      );
      amrex::Gpu::AsyncArray<amrex::Real> faces1_storage(
         coordinates[0].faces.data(), coordinates[0].faces.size()
      );
      amrex::Gpu::AsyncArray<amrex::Real> centers1_storage(
         coordinates[0].centers.data(), coordinates[0].centers.size()
      );
      const amrex::Real* faces1 = faces1_storage.data();
      const amrex::Real* centers1 = centers1_storage.data();
#if (AMREX_SPACEDIM >= 2)
      amrex::Gpu::AsyncArray<amrex::Real> faces2_storage(
         coordinates[1].faces.data(), coordinates[1].faces.size()
      );
      amrex::Gpu::AsyncArray<amrex::Real> centers2_storage(
         coordinates[1].centers.data(), coordinates[1].centers.size()
      );
      const amrex::Real* faces2 = faces2_storage.data();
      const amrex::Real* centers2 = centers2_storage.data();
#endif
#if (AMREX_SPACEDIM == 3)
      amrex::Gpu::AsyncArray<amrex::Real> faces3_storage(
         coordinates[2].faces.data(), coordinates[2].faces.size()
      );
      amrex::Gpu::AsyncArray<amrex::Real> centers3_storage(
         coordinates[2].centers.data(), coordinates[2].centers.size()
      );
      const amrex::Real* faces3 = faces3_storage.data();
      const amrex::Real* centers3 = centers3_storage.data();
#endif
      for (amrex::MFIter mfi(*vertex_fields[level], amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto source = source_fields[level]->const_array(mfi);
         const auto vertex = vertex_fields[level]->array(mfi);
         const int num_components = vertex_fields[level]->nComp();
         amrex::ParallelFor(box, num_components, [=] AMREX_GPU_DEVICE(
            int i, int j, int k, int component
         ) noexcept {
            const int face1_index = i - cell_lo.x;
            const amrex::Real face1 = faces1[face1_index];
            const amrex::Real lower1 = i - 1 < cell_lo.x
               ? 2.0 * faces1[0] - centers1[0]
               : centers1[i - 1 - cell_lo.x];
            const amrex::Real upper1 = i > cell_hi.x
               ? 2.0 * faces1[cell_hi.x - cell_lo.x + 1]
                    - centers1[cell_hi.x - cell_lo.x]
               : centers1[i - cell_lo.x];
            const amrex::Real weight1 = (face1 - lower1) / (upper1 - lower1);
#if (AMREX_SPACEDIM >= 2)
            const int face2_index = j - cell_lo.y;
            const amrex::Real face2 = faces2[face2_index];
            const amrex::Real lower2 = j - 1 < cell_lo.y
               ? 2.0 * faces2[0] - centers2[0]
               : centers2[j - 1 - cell_lo.y];
            const amrex::Real upper2 = j > cell_hi.y
               ? 2.0 * faces2[cell_hi.y - cell_lo.y + 1]
                    - centers2[cell_hi.y - cell_lo.y]
               : centers2[j - cell_lo.y];
            const amrex::Real weight2 = (face2 - lower2) / (upper2 - lower2);
#endif
#if (AMREX_SPACEDIM == 3)
            const int face3_index = k - cell_lo.z;
            const amrex::Real face3 = faces3[face3_index];
            const amrex::Real lower3 = k - 1 < cell_lo.z
               ? 2.0 * faces3[0] - centers3[0]
               : centers3[k - 1 - cell_lo.z];
            const amrex::Real upper3 = k > cell_hi.z
               ? 2.0 * faces3[cell_hi.z - cell_lo.z + 1]
                    - centers3[cell_hi.z - cell_lo.z]
               : centers3[k - cell_lo.z];
            const amrex::Real weight3 = (face3 - lower3) / (upper3 - lower3);
#endif
            amrex::Real value = 0.0;
#if (AMREX_SPACEDIM == 1)
            value = (1.0 - weight1) * source(i - 1, j, k, component)
               + weight1 * source(i, j, k, component);
#elif (AMREX_SPACEDIM == 2)
            for (int offset2 = 0; offset2 <= 1; ++offset2) {
               const amrex::Real angular_weight = offset2 == 0
                  ? 1.0 - weight2 : weight2;
               value += angular_weight * (
                  (1.0 - weight1)
                     * source(i - 1, j - 1 + offset2, k, component)
                  + weight1
                     * source(i, j - 1 + offset2, k, component)
               );
            }
#else
            for (int offset3 = 0; offset3 <= 1; ++offset3) {
               const amrex::Real third_weight = offset3 == 0
                  ? 1.0 - weight3 : weight3;
               for (int offset2 = 0; offset2 <= 1; ++offset2) {
                  const amrex::Real second_weight = offset2 == 0
                     ? 1.0 - weight2 : weight2;
                  value += third_weight * second_weight * (
                     (1.0 - weight1) * source(
                        i - 1, j - 1 + offset2, k - 1 + offset3, component
                     )
                     + weight1 * source(
                        i, j - 1 + offset2, k - 1 + offset3, component
                     )
                  );
               }
            }
#endif
            vertex(i, j, k, component) = value;
         });
      }
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
