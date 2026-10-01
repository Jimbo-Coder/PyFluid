#pragma once

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <string>
#include <utility>
#include <vector>

namespace pyfluid_amrex {

enum PatchCoordinateMapKind {
   PatchMapCartesianIdentity = 0,
   PatchMapThornburgCentralCartesian = 1,
   PatchMapThornburgShellPlusX = 2,
   PatchMapThornburgShellMinusX = 3,
   PatchMapThornburgShellPlusY = 4,
   PatchMapThornburgShellMinusY = 5,
   PatchMapThornburgShellPlusZ = 6,
   PatchMapThornburgShellMinusZ = 7,
   PatchMapCylindricalShellPlusX = 8,
   PatchMapCylindricalShellMinusX = 9,
   PatchMapCylindricalShellPlusY = 10,
   PatchMapCylindricalShellMinusY = 11,
};

inline char const* patch_coordinate_map_name(int coordinate_map)
{
   switch (coordinate_map) {
   case PatchMapCartesianIdentity:
      return "cartesian_identity";
   case PatchMapThornburgCentralCartesian:
      return "thornburg_central_cartesian";
   case PatchMapThornburgShellPlusX:
      return "thornburg_shell_plus_x";
   case PatchMapThornburgShellMinusX:
      return "thornburg_shell_minus_x";
   case PatchMapThornburgShellPlusY:
      return "thornburg_shell_plus_y";
   case PatchMapThornburgShellMinusY:
      return "thornburg_shell_minus_y";
   case PatchMapThornburgShellPlusZ:
      return "thornburg_shell_plus_z";
   case PatchMapThornburgShellMinusZ:
      return "thornburg_shell_minus_z";
   case PatchMapCylindricalShellPlusX:
      return "cylindrical_shell_plus_x";
   case PatchMapCylindricalShellMinusX:
      return "cylindrical_shell_minus_x";
   case PatchMapCylindricalShellPlusY:
      return "cylindrical_shell_plus_y";
   case PatchMapCylindricalShellMinusY:
      return "cylindrical_shell_minus_y";
   default:
      return "unknown";
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool patch_coordinate_map_supported(int coordinate_map)
{
   return coordinate_map >= PatchMapCartesianIdentity
      && coordinate_map <= PatchMapCylindricalShellMinusY;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool patch_coordinate_map_is_cartesian(int coordinate_map)
{
   return coordinate_map == PatchMapCartesianIdentity
      || coordinate_map == PatchMapThornburgCentralCartesian;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool patch_coordinate_map_is_thornburg(int coordinate_map)
{
   return coordinate_map >= PatchMapThornburgCentralCartesian
      && coordinate_map <= PatchMapThornburgShellMinusZ;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool patch_coordinate_map_is_thornburg_shell(int coordinate_map)
{
   return coordinate_map >= PatchMapThornburgShellPlusX
      && coordinate_map <= PatchMapThornburgShellMinusZ;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool patch_coordinate_map_is_cylindrical_shell(int coordinate_map)
{
   return coordinate_map >= PatchMapCylindricalShellPlusX
      && coordinate_map <= PatchMapCylindricalShellMinusY;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool patch_coordinate_map_is_shell(int coordinate_map)
{
   return patch_coordinate_map_is_thornburg_shell(coordinate_map)
      || patch_coordinate_map_is_cylindrical_shell(coordinate_map);
}

struct PatchDescriptor {
   // Dense evolved-fragment id. This indexes AMReX owned boxes.
   int patch_id = -1;
   // Unique finite-volume ownership. Conserved totals, refluxing, and updates
   // operate on this box exactly once.
   amrex::Box owned_box;
   // Full patch extent used as interpatch source/storage context. This can
   // include overlap cells for CapyrX-style ghost interpolation.
   amrex::Box patch_box;
   // Logical patch that supplied patch_box. Overlap ownership can split one
   // nominal patch into multiple evolved fragments.
   int nominal_patch_id = -1;
   // Coordinate map for this logical patch. Production Cartesian multipatch
   // uses identity maps; Thornburg-style mixed patches will add curved maps
   // without changing ownership semantics.
   int coordinate_map = PatchMapCartesianIdentity;
};

struct PatchLayout {
   amrex::Box global_box;
   amrex::Vector<PatchDescriptor> patches;
};

using CartesianPatchDescriptor = PatchDescriptor;
using CartesianMultipatchLayout = PatchLayout;

inline PatchDescriptor make_patch_descriptor(
   int patch_id,
   amrex::Box const& owned_box,
   amrex::Box const& patch_box,
   int nominal_patch_id = -1,
   int coordinate_map = PatchMapCartesianIdentity
)
{
   if (nominal_patch_id < 0) {
      nominal_patch_id = patch_id;
   }
   return PatchDescriptor{patch_id, owned_box, patch_box, nominal_patch_id, coordinate_map};
}

inline PatchDescriptor make_cartesian_patch_descriptor(
   int patch_id,
   amrex::Box const& owned_box,
   amrex::Box const& patch_box,
   int nominal_patch_id = -1
)
{
   return make_patch_descriptor(
      patch_id,
      owned_box,
      patch_box,
      nominal_patch_id,
      PatchMapCartesianIdentity
   );
}

inline void validate_patch_layout(PatchLayout const& layout)
{
   if (!layout.global_box.ok()) {
      amrex::Abort("multipatch layout has an invalid global box");
   }
   if (layout.patches.empty()) {
      amrex::Abort("multipatch layout must contain at least one patch");
   }
   for (int patch_index = 0; patch_index < static_cast<int>(layout.patches.size()); ++patch_index) {
      PatchDescriptor const& patch = layout.patches[patch_index];
      if (patch.patch_id != patch_index) {
         amrex::Abort("multipatch patch_id values must be dense and ordered");
      }
      if (patch.nominal_patch_id < 0) {
         amrex::Abort("multipatch nominal_patch_id values must be non-negative");
      }
      if (!patch_coordinate_map_supported(patch.coordinate_map)) {
         amrex::Abort(
            std::string("unsupported multipatch coordinate map: ")
            + patch_coordinate_map_name(patch.coordinate_map)
         );
      }
      if (!patch.owned_box.ok()) {
         amrex::Abort("multipatch patch has an invalid owned box");
      }
      if (!patch.patch_box.ok()) {
         amrex::Abort("multipatch patch has an invalid patch box");
      }
      if (!patch.patch_box.contains(patch.owned_box)) {
         amrex::Abort("multipatch patch box must contain its owned box");
      }
      if (!layout.global_box.contains(patch.owned_box)) {
         amrex::Abort("multipatch patch owned box extends outside the global box");
      }
      for (int other_index = patch_index + 1; other_index < static_cast<int>(layout.patches.size()); ++other_index) {
         amrex::Box overlap = patch.owned_box;
         overlap &= layout.patches[other_index].owned_box;
         if (overlap.ok()) {
            amrex::Abort("multipatch owned boxes must not overlap");
         }
      }
   }
}

inline void validate_cartesian_patch_layout(CartesianMultipatchLayout const& layout)
{
   validate_patch_layout(layout);
   for (PatchDescriptor const& patch : layout.patches) {
      if (patch.coordinate_map != PatchMapCartesianIdentity) {
         amrex::Abort("Cartesian multipatch layout requires identity Cartesian patch maps");
      }
   }
}

inline amrex::BoxArray patch_owned_box_array(PatchLayout const& layout)
{
   validate_patch_layout(layout);
   amrex::BoxList boxes(layout.global_box.ixType());
   for (PatchDescriptor const& patch : layout.patches) {
      boxes.push_back(patch.owned_box);
   }
   return amrex::BoxArray(boxes);
}

inline CartesianMultipatchLayout make_aligned_cartesian_split_layout(
   amrex::Box const& global_box,
   int split_direction,
   int high_patch_small_index
)
{
   if (!global_box.ok()) {
      amrex::Abort("cannot split an invalid Cartesian multipatch global box");
   }
   if (split_direction < 0 || split_direction >= AMREX_SPACEDIM) {
      amrex::Abort("Cartesian multipatch split_direction is outside the executable dimension");
   }
   const int low_edge = global_box.smallEnd(split_direction);
   const int high_edge = global_box.bigEnd(split_direction);
   if (high_patch_small_index <= low_edge || high_patch_small_index > high_edge) {
      amrex::Abort("Cartesian multipatch split index must leave non-empty low and high patches");
   }

   amrex::Box low_patch = global_box;
   amrex::Box high_patch = global_box;
   low_patch.setBig(split_direction, high_patch_small_index - 1);
   high_patch.setSmall(split_direction, high_patch_small_index);

   CartesianMultipatchLayout layout;
   layout.global_box = global_box;
   layout.patches.push_back(make_cartesian_patch_descriptor(0, low_patch, low_patch));
   layout.patches.push_back(make_cartesian_patch_descriptor(1, high_patch, high_patch));
   validate_cartesian_patch_layout(layout);
   return layout;
}

inline CartesianMultipatchLayout make_overlapping_cartesian_split_layout(
   amrex::Box const& global_box,
   int split_direction,
   int high_patch_small_index,
   int overlap_cells
)
{
   if (overlap_cells < 0) {
      amrex::Abort("Cartesian multipatch overlap_cells must be non-negative");
   }
   CartesianMultipatchLayout layout = make_aligned_cartesian_split_layout(
      global_box,
      split_direction,
      high_patch_small_index
   );
   for (CartesianPatchDescriptor& patch : layout.patches) {
      patch.patch_box = patch.owned_box;
      patch.patch_box.grow(split_direction, overlap_cells);
      patch.patch_box &= global_box;
   }
   validate_cartesian_patch_layout(layout);
   return layout;
}

inline CartesianMultipatchLayout make_centered_cartesian_split_layout(
   amrex::Box const& global_box,
   int split_direction
)
{
   const int low_edge = global_box.smallEnd(split_direction);
   const int cell_count = global_box.length(split_direction);
   return make_aligned_cartesian_split_layout(global_box, split_direction, low_edge + cell_count / 2);
}

inline CartesianMultipatchLayout make_aligned_cartesian_tiled_layout(
   amrex::Box const& global_box,
   amrex::IntVect const& patch_counts,
   int overlap_cells = 0
)
{
   if (!global_box.ok()) {
      amrex::Abort("cannot tile an invalid Cartesian multipatch global box");
   }
   if (overlap_cells < 0) {
      amrex::Abort("Cartesian multipatch overlap_cells must be non-negative");
   }
   int total_patches = 1;
   amrex::IntVect patch_size(AMREX_D_DECL(1, 1, 1));
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (patch_counts[dir] <= 0) {
         amrex::Abort("Cartesian multipatch patch_counts must be positive in every active direction");
      }
      const int cells = global_box.length(dir);
      if (cells % patch_counts[dir] != 0) {
         amrex::Abort("Cartesian multipatch tile counts must evenly divide the global box");
      }
      patch_size[dir] = cells / patch_counts[dir];
      total_patches *= patch_counts[dir];
   }

   CartesianMultipatchLayout layout;
   layout.global_box = global_box;
   for (int patch_id = 0; patch_id < total_patches; ++patch_id) {
      int remainder = patch_id;
      amrex::IntVect tile_index(AMREX_D_DECL(0, 0, 0));
      amrex::Box patch_box = global_box;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         tile_index[dir] = remainder % patch_counts[dir];
         remainder /= patch_counts[dir];
         const int patch_small = global_box.smallEnd(dir) + tile_index[dir] * patch_size[dir];
         patch_box.setSmall(dir, patch_small);
         patch_box.setBig(dir, patch_small + patch_size[dir] - 1);
      }
      amrex::Box owned_box = patch_box;
      if (overlap_cells > 0) {
         patch_box.grow(overlap_cells);
         patch_box &= global_box;
      }
      layout.patches.push_back(make_cartesian_patch_descriptor(patch_id, owned_box, patch_box));
   }
   validate_cartesian_patch_layout(layout);
   return layout;
}

inline CartesianMultipatchLayout make_cartesian_layout_from_boxes(
   amrex::Box const& global_box,
   amrex::Vector<amrex::Box> const& patch_boxes
)
{
   if (!global_box.ok()) {
      amrex::Abort("cannot build a Cartesian multipatch layout with an invalid global box");
   }
   CartesianMultipatchLayout layout;
   layout.global_box = global_box;
   layout.patches.reserve(patch_boxes.size());
   for (int patch_id = 0; patch_id < static_cast<int>(patch_boxes.size()); ++patch_id) {
      layout.patches.push_back(make_cartesian_patch_descriptor(patch_id, patch_boxes[patch_id], patch_boxes[patch_id]));
   }
   validate_cartesian_patch_layout(layout);
   return layout;
}

inline amrex::Long box_list_num_pts(amrex::BoxList const& boxes)
{
   amrex::Long points = 0;
   for (amrex::Box const& box : boxes) {
      points += box.numPts();
   }
   return points;
}

inline amrex::BoxList box_list_difference(
   amrex::BoxList boxes,
   amrex::BoxList const& subtractors
)
{
   for (amrex::Box const& subtractor : subtractors) {
      amrex::BoxList remaining(boxes.ixType());
      for (amrex::Box const& box : boxes) {
         amrex::BoxList diff = amrex::boxDiff(box, subtractor);
         remaining.join(diff);
      }
      boxes = std::move(remaining);
      boxes.removeEmpty();
      if (boxes.isEmpty()) {
         break;
      }
   }
   return boxes;
}

inline CartesianMultipatchLayout make_cartesian_layout_from_overlapping_boxes(
   amrex::Box const& global_box,
   amrex::Vector<amrex::Box> const& nominal_boxes,
   bool later_boxes_own_overlap
)
{
   if (!global_box.ok()) {
      amrex::Abort("cannot build a Cartesian overlap layout with an invalid global box");
   }
   if (nominal_boxes.size() < 2) {
      amrex::Abort("Cartesian overlap layout requires at least two nominal patch boxes");
   }
   for (amrex::Box const& box : nominal_boxes) {
      if (!box.ok() || !global_box.contains(box)) {
         amrex::Abort("Cartesian overlap nominal patch boxes must be valid and inside the global box");
      }
   }

   amrex::BoxList claimed(global_box.ixType());
   amrex::Vector<amrex::BoxList> owned_by_nominal;
   owned_by_nominal.reserve(nominal_boxes.size());
   for (int patch = 0; patch < static_cast<int>(nominal_boxes.size()); ++patch) {
      owned_by_nominal.emplace_back(global_box.ixType());
   }

   const auto assign_patch = [&](int patch) {
      amrex::BoxList nominal(nominal_boxes[patch]);
      owned_by_nominal[patch] = box_list_difference(nominal, claimed);
      if (owned_by_nominal[patch].isEmpty()) {
         amrex::Abort("Cartesian overlap nominal patch owns no cells after overlap ownership is applied");
      }
      claimed.push_back(nominal_boxes[patch]);
      claimed = amrex::removeOverlap(claimed);
   };

   if (later_boxes_own_overlap) {
      for (int patch = static_cast<int>(nominal_boxes.size()) - 1; patch >= 0; --patch) {
         assign_patch(patch);
      }
   } else {
      for (int patch = 0; patch < static_cast<int>(nominal_boxes.size()); ++patch) {
         assign_patch(patch);
      }
   }

   amrex::BoxList uncovered = amrex::complementIn(global_box, claimed);
   uncovered.removeEmpty();
   if (uncovered.isNotEmpty()) {
      amrex::Abort("Cartesian overlap nominal patch boxes must cover the level-0 domain without holes");
   }

   CartesianMultipatchLayout layout;
   layout.global_box = global_box;
   for (int patch = 0; patch < static_cast<int>(nominal_boxes.size()); ++patch) {
      for (amrex::Box const& owned : owned_by_nominal[patch]) {
         const int patch_id = static_cast<int>(layout.patches.size());
         layout.patches.push_back(make_cartesian_patch_descriptor(patch_id, owned, nominal_boxes[patch], patch));
      }
   }
   validate_cartesian_patch_layout(layout);
   amrex::Long owned_points = 0;
   for (CartesianPatchDescriptor const& patch : layout.patches) {
      owned_points += patch.owned_box.numPts();
   }
   if (owned_points != global_box.numPts()) {
      amrex::Abort("Cartesian overlap owned boxes must exactly cover the level-0 domain");
   }
   return layout;
}

inline CartesianMultipatchLayout make_cartesian_layout_from_owned_and_patch_boxes(
   amrex::Box const& global_box,
   amrex::Vector<amrex::Box> const& owned_boxes,
   amrex::Vector<amrex::Box> const& patch_boxes
)
{
   if (owned_boxes.size() != patch_boxes.size()) {
      amrex::Abort("Cartesian multipatch owned and patch box lists must have the same size");
   }
   CartesianMultipatchLayout layout;
   layout.global_box = global_box;
   layout.patches.reserve(owned_boxes.size());
   for (int patch_id = 0; patch_id < static_cast<int>(owned_boxes.size()); ++patch_id) {
      layout.patches.push_back(make_cartesian_patch_descriptor(patch_id, owned_boxes[patch_id], patch_boxes[patch_id]));
   }
   validate_cartesian_patch_layout(layout);
   return layout;
}

inline amrex::BoxArray cartesian_patch_box_array(CartesianMultipatchLayout const& layout)
{
   validate_cartesian_patch_layout(layout);
   return patch_owned_box_array(layout);
}

inline amrex::BoxArray cartesian_selected_patch_box_array(
   CartesianMultipatchLayout const& layout,
   amrex::Vector<int> const& patch_ids
)
{
   validate_cartesian_patch_layout(layout);
   amrex::BoxList boxes;
   boxes.reserve(patch_ids.size());
   for (int patch_id : patch_ids) {
      if (patch_id < 0 || patch_id >= static_cast<int>(layout.patches.size())) {
         amrex::Abort("selected Cartesian multipatch patch id is outside the layout");
      }
      boxes.push_back(layout.patches[patch_id].owned_box);
   }
   return amrex::BoxArray(boxes);
}

inline amrex::Box cartesian_neighbor_overlap_box(
   CartesianPatchDescriptor const& destination_patch,
   CartesianPatchDescriptor const& source_patch,
   amrex::IntVect const& grow
)
{
   amrex::Box destination_ghost_region = destination_patch.patch_box;
   destination_ghost_region.grow(grow);
   amrex::Box overlap = destination_ghost_region;
   overlap &= source_patch.owned_box;
   return overlap;
}

inline void fill_aligned_cartesian_neighbor_ghosts(
   amrex::Vector<amrex::MultiFab*> const& patch_data,
   CartesianMultipatchLayout const& layout,
   int source_component,
   int destination_component,
   int num_components,
   amrex::IntVect const& ghost_width
)
{
   validate_cartesian_patch_layout(layout);
   if (patch_data.size() != layout.patches.size()) {
      amrex::Abort("Cartesian multipatch patch_data size does not match the layout");
   }
   for (amrex::MultiFab* data : patch_data) {
      if (data == nullptr) {
         amrex::Abort("Cartesian multipatch patch_data contains a null MultiFab");
      }
      if (data->nComp() < source_component + num_components || data->nComp() < destination_component + num_components) {
         amrex::Abort("Cartesian multipatch component range exceeds a patch MultiFab component count");
      }
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         if (data->nGrowVect()[dir] < ghost_width[dir]) {
            amrex::Abort("Cartesian multipatch patch MultiFab does not have the requested ghost width");
         }
      }
   }

   amrex::Vector<amrex::MultiFab> owned_source_data;
   owned_source_data.reserve(layout.patches.size());
   for (int source_index = 0; source_index < static_cast<int>(layout.patches.size()); ++source_index) {
      amrex::BoxArray owned_box_array(layout.patches[source_index].owned_box);
      amrex::DistributionMapping distribution(owned_box_array);
      owned_source_data.emplace_back(owned_box_array, distribution, patch_data[source_index]->nComp(), 0);
      owned_source_data.back().ParallelCopy(
         *patch_data[source_index],
         source_component,
         source_component,
         num_components,
         amrex::IntVect::TheZeroVector(),
         amrex::IntVect::TheZeroVector()
      );
   }

   for (int destination_index = 0; destination_index < static_cast<int>(layout.patches.size()); ++destination_index) {
      CartesianPatchDescriptor const& destination_patch = layout.patches[destination_index];
      for (int source_index = 0; source_index < static_cast<int>(layout.patches.size()); ++source_index) {
         if (source_index == destination_index) {
            continue;
         }
         CartesianPatchDescriptor const& source_patch = layout.patches[source_index];
         amrex::Box overlap = cartesian_neighbor_overlap_box(destination_patch, source_patch, ghost_width);
         if (!overlap.ok()) {
            continue;
         }
         patch_data[destination_index]->ParallelCopy(
            owned_source_data[source_index],
            source_component,
            destination_component,
            num_components,
            amrex::IntVect::TheZeroVector(),
            ghost_width
         );
      }
   }
}

} // namespace pyfluid_amrex
