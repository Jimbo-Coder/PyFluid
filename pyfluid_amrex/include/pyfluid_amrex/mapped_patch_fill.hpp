#pragma once

#include <AMReX_Box.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <limits>
#include <vector>

#include "pyfluid_amrex/patch_grid.hpp"
#include "pyfluid_amrex/patch_layout_builder.hpp"
#include "pyfluid_amrex/patch_maps.hpp"

namespace pyfluid_amrex {

struct MappedPatchSourceTable {
   amrex::Vector<amrex::Long> offsets;
   amrex::Vector<amrex::Long> strides;
   amrex::Vector<amrex::Real> values;
   amrex::Vector<int> valid;
   int ncomp = 0;
};

inline amrex::Long patch_grid_flat_index(PatchGridDescriptor const& patch, amrex::IntVect const& local_cell)
{
   amrex::Long index = local_cell[0];
#if (AMREX_SPACEDIM >= 2)
   index += static_cast<amrex::Long>(patch.cells[0]) * local_cell[1];
#endif
#if (AMREX_SPACEDIM == 3)
   index += static_cast<amrex::Long>(patch.cells[0])
      * static_cast<amrex::Long>(patch.cells[1]) * local_cell[2];
#endif
   return index;
}

inline amrex::IntVect patch_grid_nearest_cell(PatchGridDescriptor const& patch, PatchCoordinate const& local)
{
   amrex::IntVect cell(AMREX_D_DECL(0, 0, 0));
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const int component = patch_local_component_for_dir(patch.local_axes, dir);
      const amrex::Real q = patch_coordinate_component(local, component);
      const amrex::Real scaled = (q - patch_coordinate_component(patch.local_lo, component))
         / patch_grid_cell_size(patch, dir);
      int index = static_cast<int>(std::floor(scaled));
      index = amrex::max(0, amrex::min(index, patch.cells[dir] - 1));
      cell[dir] = index;
   }
   return cell;
}

inline bool patch_grid_contains_local_cell(PatchGridDescriptor const& patch, amrex::IntVect const& local_cell)
{
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (local_cell[dir] < 0 || local_cell[dir] >= patch.cells[dir]) {
         return false;
      }
   }
   return true;
}

inline amrex::IntVect patch_grid_clamped_local_cell(
   PatchGridDescriptor const& patch,
   amrex::IntVect local_cell
)
{
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      local_cell[dir] = amrex::max(0, amrex::min(local_cell[dir], patch.cells[dir] - 1));
   }
   return local_cell;
}

inline MappedPatchSourceTable make_mapped_patch_source_table(
   amrex::MultiFab const& state,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout
)
{
   validate_patch_grid_system(grid_system);
   validate_patch_layout(layout);
   if (grid_system.patches.size() != layout.patches.size()) {
      amrex::Abort("mapped patch source table requires one layout patch per grid patch");
   }

   MappedPatchSourceTable table;
   table.ncomp = state.nComp();
   table.offsets.resize(grid_system.patches.size());
   table.strides.resize(grid_system.patches.size());
   amrex::Long total_cells = 0;
   for (int patch_id = 0; patch_id < static_cast<int>(grid_system.patches.size()); ++patch_id) {
      table.offsets[patch_id] = total_cells;
      table.strides[patch_id] = patch_grid_cell_count(grid_system.patches[patch_id]);
      total_cells += table.strides[patch_id];
   }
   const amrex::Long total_values = total_cells * static_cast<amrex::Long>(table.ncomp);
   if (total_values > static_cast<amrex::Long>(std::numeric_limits<int>::max())) {
      amrex::Abort("mapped patch source table is too large for the current all-reduce prototype");
   }
   table.values.assign(static_cast<std::size_t>(total_values), 0.0);
   table.valid.assign(static_cast<std::size_t>(total_cells), 0);

   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      PatchDescriptor const& layout_patch = patch_layout_descriptor_containing_box(layout, bx);
      PatchGridDescriptor const& grid_patch = patch_grid_descriptor_for_layout_patch(grid_system, layout_patch);
      const auto src = state.const_array(mfi);
      const int patch_id = layout_patch.nominal_patch_id;
      for (amrex::BoxIterator bit(bx); bit.ok(); ++bit) {
         const amrex::IntVect cell = bit();
         const int i = cell[0];
         int j = 0;
         int k = 0;
#if (AMREX_SPACEDIM >= 2)
         j = cell[1];
#endif
#if (AMREX_SPACEDIM == 3)
         k = cell[2];
#endif
         const amrex::IntVect local_cell = patch_layout_cell_to_patch_local_cell(layout_patch, cell);
         const PatchCoordinate local = patch_grid_cell_center(grid_patch, local_cell);
         const PatchCoordinate global = patch_local_to_global(grid_patch.map, local);
         const int owner_patch_id = patch_grid_system_owner_patch_id(grid_system, global);
         if (owner_patch_id != patch_id) {
            continue;
         }
         const amrex::Long cell_index = table.offsets[patch_id] + patch_grid_flat_index(grid_patch, local_cell);
         table.valid[static_cast<std::size_t>(cell_index)] = 1;
         for (int n = 0; n < table.ncomp; ++n) {
            table.values[static_cast<std::size_t>(cell_index * table.ncomp + n)] =
               src(i, j, k, n);
         }
      }
   }

   amrex::ParallelDescriptor::ReduceRealSum(table.values.data(), static_cast<int>(table.values.size()));
   amrex::ParallelDescriptor::ReduceIntSum(table.valid.data(), static_cast<int>(table.valid.size()));
   return table;
}

inline bool mapped_patch_source_valid(
   MappedPatchSourceTable const& table,
   PatchGridSystem const& grid_system,
   int patch_id,
   amrex::IntVect const& local_cell
)
{
   const amrex::Long cell_index = table.offsets[patch_id]
      + patch_grid_flat_index(grid_system.patches[patch_id], local_cell);
   return table.valid[static_cast<std::size_t>(cell_index)] > 0;
}

inline amrex::Real mapped_patch_source_value(
   MappedPatchSourceTable const& table,
   PatchGridSystem const& grid_system,
   int patch_id,
   amrex::IntVect const& local_cell,
   int component
)
{
   const amrex::Long cell_index = table.offsets[patch_id]
      + patch_grid_flat_index(grid_system.patches[patch_id], local_cell);
   return table.values[static_cast<std::size_t>(cell_index * table.ncomp + component)];
}

inline bool mapped_patch_cell_is_finite(
   amrex::Array4<amrex::Real const> const& values,
   int i,
   int j,
   int k,
   int ncomp
)
{
   for (int n = 0; n < ncomp; ++n) {
      if (!std::isfinite(values(i, j, k, n))) {
         return false;
      }
   }
   return true;
}

inline void copy_mapped_patch_source_cell(
   amrex::Array4<amrex::Real> const& dst,
   MappedPatchSourceTable const& table,
   PatchGridSystem const& grid_system,
   int source_patch_id,
   amrex::IntVect const& source_cell,
   int i,
   int j,
   int k,
   int ncomp
)
{
   for (int n = 0; n < ncomp; ++n) {
      dst(i, j, k, n) =
         mapped_patch_source_value(table, grid_system, source_patch_id, source_cell, n);
   }
}

inline void fill_mapped_patch_state_ghosts(
   amrex::MultiFab& state,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout
)
{
   MappedPatchSourceTable const table = make_mapped_patch_source_table(state, grid_system, layout);
   const int ncomp = state.nComp();
   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const amrex::Box& valid = mfi.validbox();
      PatchDescriptor const& layout_patch = patch_layout_descriptor_containing_box(layout, valid);
      PatchGridDescriptor const& grid_patch = patch_grid_descriptor_for_layout_patch(grid_system, layout_patch);
      const amrex::Box grown = amrex::grow(valid, state.nGrowVect());
      const auto dst = state.array(mfi);
      for (amrex::BoxIterator bit(grown); bit.ok(); ++bit) {
         const amrex::IntVect cell = bit();
         const int i = cell[0];
         int j = 0;
         int k = 0;
#if (AMREX_SPACEDIM >= 2)
         j = cell[1];
#endif
#if (AMREX_SPACEDIM == 3)
         k = cell[2];
#endif
         const PatchCoordinate local = patch_layout_local_coordinate(
            grid_patch,
            layout_patch,
            cell,
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>{AMREX_D_DECL(0.5, 0.5, 0.5)}
         );
         const PatchCoordinate global = patch_local_to_global(grid_patch.map, local);
         const bool is_owned_valid_cell =
            valid.contains(cell)
            && patch_grid_system_owner_patch_id(grid_system, global) == layout_patch.nominal_patch_id;
         if (is_owned_valid_cell) {
            continue;
         }
         int source_patch_id = patch_grid_system_owner_patch_id(grid_system, global);
         PatchCoordinate source_local = local;
         if (source_patch_id < 0 || source_patch_id >= static_cast<int>(grid_system.patches.size())) {
            source_patch_id = layout_patch.nominal_patch_id;
         } else {
            source_local = patch_global_to_local(grid_system.patches[source_patch_id].map, global);
         }
         PatchGridDescriptor const& source_patch = grid_system.patches[source_patch_id];
         const amrex::IntVect source_cell = patch_grid_nearest_cell(source_patch, source_local);
         if (mapped_patch_source_valid(table, grid_system, source_patch_id, source_cell)) {
            copy_mapped_patch_source_cell(
               dst,
               table,
               grid_system,
               source_patch_id,
               source_cell,
               i,
               j,
               k,
               ncomp
            );
            continue;
         }

         if (mapped_patch_cell_is_finite(dst, i, j, k, ncomp)) {
            continue;
         }

         amrex::IntVect local_cell = cell;
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            local_cell[dir] -= layout_patch.owned_box.smallEnd(dir);
         }
         const amrex::IntVect fallback_cell =
            patch_grid_clamped_local_cell(grid_patch, local_cell);
         if (mapped_patch_source_valid(table, grid_system, layout_patch.nominal_patch_id, fallback_cell)) {
            copy_mapped_patch_source_cell(
               dst,
               table,
               grid_system,
               layout_patch.nominal_patch_id,
               fallback_cell,
               i,
               j,
               k,
               ncomp
            );
            continue;
         }

         amrex::IntVect nearest_valid = cell;
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            nearest_valid[dir] = amrex::max(
               valid.smallEnd(dir),
               amrex::min(nearest_valid[dir], valid.bigEnd(dir))
            );
         }
         const int vi = nearest_valid[0];
#if (AMREX_SPACEDIM >= 2)
         const int vj = nearest_valid[1];
#else
         constexpr int vj = 0;
#endif
#if (AMREX_SPACEDIM == 3)
         const int vk = nearest_valid[2];
#else
         constexpr int vk = 0;
#endif
         for (int n = 0; n < ncomp; ++n) {
            dst(i, j, k, n) = dst(vi, vj, vk, n);
         }
      }
   }
}

} // namespace pyfluid_amrex
