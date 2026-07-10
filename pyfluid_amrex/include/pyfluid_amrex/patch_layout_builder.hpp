#pragma once

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <sstream>

#include "pyfluid_amrex/multipatch.hpp"
#include "pyfluid_amrex/patch_grid.hpp"

namespace pyfluid_amrex {

struct PatchLogicalLayoutOptions {
   int pack_direction = 0;
   int gap_cells = 0;
};

inline amrex::Box patch_grid_index_box(PatchGridDescriptor const& patch)
{
   amrex::IntVect small(AMREX_D_DECL(0, 0, 0));
   amrex::IntVect big = patch.cells;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      big[dir] -= 1;
   }
   return amrex::Box(small, big);
}

inline amrex::Long patch_grid_cell_count(PatchGridDescriptor const& patch)
{
   return patch_grid_index_box(patch).numPts();
}

inline amrex::Box shifted_patch_grid_index_box(
   PatchGridDescriptor const& patch,
   amrex::IntVect const& offset
)
{
   amrex::IntVect small(AMREX_D_DECL(0, 0, 0));
   amrex::IntVect big = patch.cells;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      small[dir] += offset[dir];
      big[dir] += offset[dir] - 1;
   }
   return amrex::Box(small, big);
}

inline amrex::Box patch_layout_bounding_box(amrex::Vector<PatchDescriptor> const& patches)
{
   if (patches.empty()) {
      amrex::Abort("cannot compute a patch-layout bounding box with no patches");
   }
   amrex::Box box = patches.front().owned_box;
   for (int patch_id = 1; patch_id < static_cast<int>(patches.size()); ++patch_id) {
      PatchDescriptor const& patch = patches[patch_id];
      if (!patch.owned_box.ok()) {
         amrex::Abort("cannot bound an invalid patch owned box");
      }
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         box.setSmall(dir, std::min(box.smallEnd(dir), patch.owned_box.smallEnd(dir)));
         box.setBig(dir, std::max(box.bigEnd(dir), patch.owned_box.bigEnd(dir)));
      }
   }
   return box;
}

inline void validate_patch_grid_system(PatchGridSystem const& grid_system)
{
   if (grid_system.patches.empty()) {
      amrex::Abort("patch-grid system must contain at least one patch");
   }
   if (grid_system.patches.size() != grid_system.map_system.maps.size()) {
      amrex::Abort("patch-grid system patch count must match map count");
   }
   for (int patch_id = 0; patch_id < static_cast<int>(grid_system.patches.size()); ++patch_id) {
      PatchGridDescriptor const& patch = grid_system.patches[patch_id];
      if (patch.patch_id != patch_id) {
         amrex::Abort("patch-grid ids must be dense and ordered");
      }
      if (patch.map.coordinate_map != grid_system.map_system.maps[patch_id].coordinate_map) {
         amrex::Abort("patch-grid descriptor map must match its map-system entry");
      }
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         if (patch.cells[dir] <= 0) {
            amrex::Abort("patch-grid cells must be positive in every active direction");
         }
         if (patch_grid_cell_size(patch, dir) <= 0.0) {
            amrex::Abort("patch-grid local spacing must be positive in every active direction");
         }
      }
   }
}

inline int patch_grid_system_owner_patch_id(
   PatchGridSystem const& grid_system,
   PatchCoordinate const& global
)
{
   validate_patch_grid_system(grid_system);
   PatchGridDescriptor const& central_patch = grid_system.patches.front();
   if (
      patch_coordinate_map_is_cartesian(central_patch.map.coordinate_map)
      && patch_grid_contains_global_point(central_patch, global, 1.0e-12)
   ) {
      return central_patch.patch_id;
   }
   return patch_system_owner_patch_id(grid_system.map_system, global);
}

inline PatchLayout make_patch_layout_from_grid_system(
   PatchGridSystem const& grid_system,
   PatchLogicalLayoutOptions options = {}
)
{
   validate_patch_grid_system(grid_system);
   if (options.pack_direction < 0 || options.pack_direction >= AMREX_SPACEDIM) {
      amrex::Abort("patch-grid logical-layout pack_direction is outside the executable dimension");
   }
   if (options.gap_cells < 0) {
      amrex::Abort("patch-grid logical-layout gap_cells must be non-negative");
   }

   PatchLayout layout;
   layout.patches.reserve(grid_system.patches.size());
   int next_small = 0;
   for (PatchGridDescriptor const& grid_patch : grid_system.patches) {
      amrex::IntVect offset(AMREX_D_DECL(0, 0, 0));
      offset[options.pack_direction] = next_small;
      const amrex::Box owned_box = shifted_patch_grid_index_box(grid_patch, offset);
      layout.patches.push_back(
         make_patch_descriptor(
            grid_patch.patch_id,
            owned_box,
            owned_box,
            grid_patch.patch_id,
            grid_patch.map.coordinate_map
         )
      );
      next_small += grid_patch.cells[options.pack_direction] + options.gap_cells;
   }
   layout.global_box = patch_layout_bounding_box(layout.patches);
   validate_patch_layout(layout);
   return layout;
}

inline PatchGridDescriptor const& patch_grid_descriptor_for_layout_patch(
   PatchGridSystem const& grid_system,
   PatchDescriptor const& patch
)
{
   validate_patch_grid_system(grid_system);
   if (patch.nominal_patch_id < 0 || patch.nominal_patch_id >= static_cast<int>(grid_system.patches.size())) {
      amrex::Abort("patch-layout nominal_patch_id is outside the patch-grid system");
   }
   PatchGridDescriptor const& grid_patch = grid_system.patches[patch.nominal_patch_id];
   if (grid_patch.map.coordinate_map != patch.coordinate_map) {
      amrex::Abort("patch-layout coordinate map does not match the patch-grid descriptor");
   }
   return grid_patch;
}

inline amrex::IntVect patch_layout_cell_to_patch_local_cell(
   PatchDescriptor const& patch,
   amrex::IntVect const& cell
)
{
   if (!patch.owned_box.contains(cell)) {
      amrex::Abort("patch-layout cell is outside its owned patch");
   }
   amrex::IntVect local = cell;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      local[dir] -= patch.owned_box.smallEnd(dir);
   }
   return local;
}

inline PatchDescriptor const& patch_layout_descriptor_containing_box(
   PatchLayout const& layout,
   amrex::Box const& box
)
{
   validate_patch_layout(layout);
   for (PatchDescriptor const& patch : layout.patches) {
      if (patch.owned_box.contains(box)) {
         return patch;
      }
   }
   std::ostringstream message;
   message << "AMReX box " << box << " is not contained in any multipatch owned box:";
   for (PatchDescriptor const& patch : layout.patches) {
      message << " patch" << patch.patch_id << "=" << patch.owned_box;
   }
   const std::string text = message.str();
   amrex::Abort(text.c_str());
   return layout.patches.front();
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_grid_local_coordinate(
   PatchGridDescriptor const& patch,
   amrex::IntVect const& cell,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& offsets
) noexcept
{
   PatchCoordinate local = make_patch_coordinate(
      0.5 * (patch.local_lo.q1 + patch.local_hi.q1),
      0.5 * (patch.local_lo.q2 + patch.local_hi.q2),
      0.5 * (patch.local_lo.q3 + patch.local_hi.q3)
   );
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const int component = patch_local_component_for_dir(patch.local_axes, dir);
      const amrex::Real coordinate = patch_coordinate_component(patch.local_lo, component)
         + (static_cast<amrex::Real>(cell[dir]) + offsets[dir]) * patch_grid_cell_size(patch, dir);
      set_patch_coordinate_component(local, component, coordinate);
   }
   return local;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_layout_local_coordinate(
   PatchGridDescriptor const& grid_patch,
   PatchDescriptor const& layout_patch,
   amrex::IntVect const& layout_cell,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& offsets
) noexcept
{
   amrex::IntVect local_cell = layout_cell;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      local_cell[dir] -= layout_patch.owned_box.smallEnd(dir);
   }
   return patch_grid_local_coordinate(grid_patch, local_cell, offsets);
}

inline PatchCoordinate patch_layout_cell_center(
   PatchGridDescriptor const& grid_patch,
   PatchDescriptor const& layout_patch,
   amrex::IntVect const& layout_cell
)
{
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> offsets{};
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      offsets[dir] = 0.5;
   }
   return patch_layout_local_coordinate(grid_patch, layout_patch, layout_cell, offsets);
}

} // namespace pyfluid_amrex
