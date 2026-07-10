#include "pyfluid_amrex/amr_solver.hpp"
#include "pyfluid_amrex/multipatch.hpp"
#include "pyfluid_amrex/patch_domain.hpp"
#include "pyfluid_amrex/patch_grid.hpp"
#include "pyfluid_amrex/patch_grid_config.hpp"
#include "pyfluid_amrex/patch_layout_builder.hpp"

namespace pyfluid_amrex {

namespace {

amrex::IntVect configured_patch_counts(RunConfig const& cfg, amrex::Box const& domain)
{
   amrex::IntVect counts(AMREX_D_DECL(1, 1, 1));
   bool configured = false;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (cfg.mesh_patch_counts[dir] != 0) {
         configured = true;
      }
   }
   if (configured) {
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         counts[dir] = cfg.mesh_patch_counts[dir];
      }
      return counts;
   }

   int split_dir = 0;
   for (int dir = 1; dir < AMREX_SPACEDIM; ++dir) {
      if (domain.length(dir) > domain.length(split_dir)) {
         split_dir = dir;
      }
   }
   counts[split_dir] = 2;
   return counts;
}

PatchDomainBoxes make_patch_domain_boxes(RunConfig const& cfg, amrex::IntVect const& refinement)
{
   PatchDomainBoxes boxes;
   boxes.count = cfg.mesh_patch_box_count;
   for (int patch = 0; patch < cfg.mesh_patch_box_count; ++patch) {
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         const int idx = patch * AMREX_SPACEDIM + dir;
         boxes.small[idx] = cfg.mesh_patch_box_small[idx] * refinement[dir];
         boxes.big[idx] = (cfg.mesh_patch_box_big[idx] + 1) * refinement[dir] - 1;
      }
   }
   return boxes;
}

amrex::Vector<amrex::Box> configured_patch_boxes(RunConfig const& cfg)
{
   return patch_domain_box_vector(make_patch_domain_boxes(cfg, amrex::IntVect(AMREX_D_DECL(1, 1, 1))));
}

amrex::Long validate_patch_boxes_disjoint_inside(amrex::Box const& domain, amrex::Vector<amrex::Box> const& boxes)
{
   amrex::BoxList claimed(domain.ixType());
   for (amrex::Box const& box : boxes) {
      if (!domain.contains(box)) {
         amrex::Abort("Cartesian multipatch patch boxes must be inside the level-0 domain");
      }
      claimed.push_back(box);
   }
   const amrex::Long raw_points = box_list_num_pts(claimed);
   claimed = amrex::removeOverlap(claimed);
   const amrex::Long unique_points = box_list_num_pts(claimed);
   if (raw_points != unique_points) {
      amrex::Abort("Cartesian multipatch exact patch boxes must not overlap; use mesh.patch_box_ownership for overlap");
   }
   return unique_points;
}

void validate_patch_box_rectangular_coverage(amrex::Box const& domain, amrex::Vector<amrex::Box> const& boxes)
{
   amrex::BoxList claimed(domain.ixType());
   for (amrex::Box const& box : boxes) {
      claimed.push_back(box);
   }
   const amrex::Long unique_points = validate_patch_boxes_disjoint_inside(domain, boxes);
   amrex::BoxList uncovered = amrex::complementIn(domain, claimed);
   uncovered.removeEmpty();
   if (uncovered.isNotEmpty() || unique_points != domain.numPts()) {
      amrex::Abort("Cartesian multipatch patch boxes must exactly cover the level-0 domain without holes");
   }
}

void validate_patch_counts(amrex::Box const& domain, amrex::IntVect const& counts)
{
   int total_patches = 1;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (counts[dir] <= 0) {
         amrex::Abort("Cartesian multipatch patch counts must be positive");
      }
      if (domain.length(dir) % counts[dir] != 0) {
         amrex::Abort("Cartesian multipatch patch counts must evenly divide the level-0 domain");
      }
      total_patches *= counts[dir];
   }
   if (total_patches <= 1) {
      amrex::Abort("Cartesian multipatch requires at least two patches");
   }
}

void print_box_array(char const* label, int lev, amrex::BoxArray const& box_array)
{
   if (!amrex::ParallelDescriptor::IOProcessor()) {
      return;
   }
   amrex::Print() << label << " level " << lev << " boxes=" << box_array.size() << "\n";
   for (int index = 0; index < static_cast<int>(box_array.size()); ++index) {
      amrex::Print() << "  [" << index << "] " << box_array[index] << "\n";
   }
}

} // namespace

amrex::IntVect AmrSolver::patch_union_refinement_to_level(int lev) const
{
   amrex::IntVect refinement(AMREX_D_DECL(1, 1, 1));
   for (int coarse_lev = 0; coarse_lev < lev; ++coarse_lev) {
      refinement *= refRatio(coarse_lev);
   }
   return refinement;
}

PatchDomainBoxes AmrSolver::patch_union_domain_boxes(int lev) const
{
   return make_patch_domain_boxes(cfg_, patch_union_refinement_to_level(lev));
}

PatchGridSystem AmrSolver::production_patch_grid_system() const
{
   return production_patch_grid_system_for_level(0);
}

PatchGridSystem AmrSolver::production_patch_grid_system_for_level(int lev) const
{
   if (!mixed_patch_system_enabled()) {
      amrex::Abort("production_patch_grid_system is only defined for mixed patch systems");
   }
   return configured_mixed_patch_grid_system(cfg_, patch_union_refinement_to_level(lev));
}

PatchLayout AmrSolver::production_patch_layout(amrex::Box const& domain) const
{
   if (mesh_patch_system_kind() == MeshPatchHomogeneousCartesian) {
      CartesianMultipatchLayout layout;
      if (cfg_.mesh_patch_box_count > 0) {
         const amrex::Vector<amrex::Box> boxes = configured_patch_boxes(cfg_);
         if (cfg_.mesh_patch_box_ownership == "exact") {
            if (cfg_.mesh_patch_box_domain == "union") {
               validate_patch_boxes_disjoint_inside(domain, boxes);
            } else {
               validate_patch_box_rectangular_coverage(domain, boxes);
            }
            layout = make_cartesian_layout_from_boxes(domain, boxes);
         } else {
            layout = make_cartesian_layout_from_overlapping_boxes(
               domain,
               boxes,
               cfg_.mesh_patch_box_ownership == "owner_last"
            );
         }
      } else {
         const amrex::IntVect counts = configured_patch_counts(cfg_, domain);
         validate_patch_counts(domain, counts);
         layout = make_aligned_cartesian_tiled_layout(domain, counts, cfg_.mesh_patch_overlap_cells);
      }
      return layout;
   }

   if (!mixed_patch_system_enabled()) {
      amrex::Abort("production_patch_layout requires a multipatch system");
   }
   PatchLayout layout = configured_mixed_patch_layout(cfg_);
   if (layout.global_box != domain) {
      amrex::Abort(
         "mixed patch logical domain does not match amr.n_cell; "
         "set amr.n_cell to the packed patch-layout domain size"
      );
   }
   return layout;
}

PatchLayout AmrSolver::production_patch_layout_for_level(int lev) const
{
   if (mixed_patch_system_enabled()) {
      const amrex::IntVect refinement = patch_union_refinement_to_level(lev);
      PatchLayout layout = make_patch_layout_from_grid_system(
         production_patch_grid_system_for_level(lev),
         configured_patch_logical_layout_options(cfg_, refinement)
      );
      if (layout.global_box != Geom(lev).Domain()) {
         amrex::Abort(
            "mixed patch logical domain does not match the AMR level domain; "
            "set amr.n_cell to the level-0 packed patch-layout domain size"
         );
      }
      return layout;
   }
   return production_patch_layout(Geom(lev).Domain());
}

void AmrSolver::PostProcessBaseGrids(amrex::BoxArray& box_array) const
{
   if (!production_multipatch_enabled()) {
      return;
   }
   if (mesh_coordinate_kind() != MeshCartesian) {
      amrex::Abort("multipatch base-grid construction requires mesh.coordinate = cartesian");
   }
#if (AMREX_SPACEDIM == 1)
   amrex::ignore_unused(box_array);
   amrex::Abort("multipatch production runs are supported for 2D and 3D executables");
#else
   const amrex::Box domain = Geom(0).Domain();
   PatchLayout const layout = production_patch_layout(domain);
   box_array = patch_owned_box_array(layout);
   box_array.maxSize(maxGridSize(0));
#endif
}

amrex::BoxArray AmrSolver::clip_to_patch_union_domain(int lev, amrex::BoxArray const& box_array) const
{
   if (!patch_union_domain_enabled() || box_array.empty()) {
      return box_array;
   }

   amrex::Vector<amrex::Box> domain_boxes = patch_domain_box_vector(patch_union_domain_boxes(lev));
   for (amrex::Box& box : domain_boxes) {
      box &= Geom(lev).Domain();
   }

   amrex::BoxList clipped(box_array.ixType());
   for (int index = 0; index < static_cast<int>(box_array.size()); ++index) {
      for (amrex::Box const& domain_box : domain_boxes) {
         amrex::Box intersection = box_array[index];
         intersection &= domain_box;
         if (intersection.ok()) {
            clipped.push_back(intersection);
         }
      }
   }
   clipped = amrex::removeOverlap(clipped);
   clipped.removeEmpty();
   if (clipped.isEmpty()) {
      return amrex::BoxArray();
   }
   amrex::BoxArray result(clipped);
   result.maxSize(maxGridSize(lev));
   return result;
}

amrex::BoxArray AmrSolver::clip_to_mixed_patch_domain(int lev, amrex::BoxArray const& box_array) const
{
   if (!mixed_patch_system_enabled() || box_array.empty()) {
      return box_array;
   }

   PatchLayout const layout = production_patch_layout_for_level(lev);
   amrex::Vector<amrex::Box> owned_boxes;
   owned_boxes.reserve(layout.patches.size());
   for (PatchDescriptor const& patch : layout.patches) {
      amrex::Box owned = patch.owned_box;
      owned &= Geom(lev).Domain();
      if (owned.ok()) {
         owned_boxes.push_back(owned);
      }
   }

   amrex::Vector<amrex::Box> clipped_boxes;
   for (amrex::Box const& owned : owned_boxes) {
      amrex::BoxList patch_clipped(box_array.ixType());
      for (int index = 0; index < static_cast<int>(box_array.size()); ++index) {
         amrex::Box intersection = box_array[index];
         intersection &= owned;
         if (intersection.ok()) {
            patch_clipped.push_back(intersection);
         }
      }
      if (patch_clipped.isNotEmpty()) {
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            int strip_cells = std::max(1, cfg_.ng);
            if (lev > 0) {
               const int ratio = std::max(1, refRatio(lev - 1)[dir]);
               strip_cells = ((strip_cells + ratio - 1) / ratio) * ratio;
            }
            const int width = std::min(strip_cells, owned.length(dir));
            amrex::Box lo_strip = owned;
            lo_strip.setBig(dir, owned.smallEnd(dir) + width - 1);
            patch_clipped.push_back(lo_strip);
            amrex::Box hi_strip = owned;
            hi_strip.setSmall(dir, owned.bigEnd(dir) - width + 1);
            patch_clipped.push_back(hi_strip);
         }
      }
      patch_clipped = amrex::removeOverlap(patch_clipped);
      patch_clipped.removeEmpty();
      if (patch_clipped.isEmpty()) {
         continue;
      }
      patch_clipped.simplify();
      amrex::BoxArray patch_boxes(patch_clipped);
      patch_boxes.maxSize(maxGridSize(lev));
      for (int index = 0; index < static_cast<int>(patch_boxes.size()); ++index) {
         clipped_boxes.push_back(patch_boxes[index]);
      }
   }
   if (clipped_boxes.empty()) {
      return amrex::BoxArray();
   }
   amrex::BoxList result_list(box_array.ixType());
   for (amrex::Box const& box : clipped_boxes) {
      result_list.push_back(box);
   }
   return amrex::BoxArray(result_list);
}

amrex::BoxArray AmrSolver::clip_to_active_multipatch_domain(
   int lev,
   amrex::BoxArray const& box_array
) const
{
   amrex::BoxArray clipped = clip_to_patch_union_domain(lev, box_array);
   clipped = clip_to_mixed_patch_domain(lev, clipped);
   return clipped;
}

void AmrSolver::clip_multipatch_initial_hierarchy(amrex::Real time)
{
   if (!patch_union_domain_enabled() && !mixed_patch_system_enabled()) {
      return;
   }
   const int old_finest = finestLevel();
   int new_finest = old_finest;
   for (int lev = 1; lev <= old_finest; ++lev) {
      amrex::BoxArray clipped = clip_to_active_multipatch_domain(lev, grids[lev]);
      if (clipped.empty()) {
         new_finest = lev - 1;
         break;
      }
      if (clipped != grids[lev]) {
         if (Verbose() > 0) {
            print_box_array("multipatch clipped initial AMR", lev, clipped);
         }
         const amrex::DistributionMapping dm = MakeDistributionMap(lev, clipped);
         MakeNewLevelFromScratch(lev, time, clipped, dm);
         SetBoxArray(lev, clipped);
         SetDistributionMap(lev, dm);
      }
   }
   for (int lev = new_finest + 1; lev <= old_finest; ++lev) {
      ClearLevel(lev);
      ClearBoxArray(lev);
      ClearDistributionMap(lev);
   }
   finest_level = new_finest;
}

void AmrSolver::regrid(int lbase, amrex::Real time, bool initial)
{
   if (!patch_union_domain_enabled() && !mixed_patch_system_enabled()) {
      amrex::AmrCore::regrid(lbase, time, initial);
      return;
   }
   amrex::ignore_unused(initial);
   if (lbase >= max_level) {
      return;
   }

   int new_finest;
   amrex::Vector<amrex::BoxArray> new_grids(finest_level + 2);
   MakeNewGrids(lbase, time, new_finest, new_grids);
   BL_ASSERT(new_finest <= finest_level + 1);

   for (int lev = lbase + 1; lev <= new_finest; ++lev) {
      new_grids[lev] = clip_to_active_multipatch_domain(lev, new_grids[lev]);
      if (Verbose() > 0) {
         print_box_array("multipatch clipped regrid AMR", lev, new_grids[lev]);
      }
      if (new_grids[lev].empty()) {
         new_finest = lev - 1;
         break;
      }
   }

   bool coarse_ba_changed = false;
   for (int lev = lbase + 1; lev <= new_finest; ++lev) {
      if (lev <= finest_level) {
         bool ba_changed = (new_grids[lev] != grids[lev]);
         if (ba_changed || coarse_ba_changed) {
            amrex::BoxArray level_grids = grids[lev];
            amrex::DistributionMapping level_dmap = dmap[lev];
            if (ba_changed) {
               level_grids = new_grids[lev];
               level_dmap = MakeDistributionMap(lev, level_grids);
            }
            const auto old_num_setdm = num_setdm;
            RemakeLevel(lev, time, level_grids, level_dmap);
            SetBoxArray(lev, level_grids);
            if (old_num_setdm == num_setdm) {
               SetDistributionMap(lev, level_dmap);
            }
         }
         coarse_ba_changed = ba_changed;
      } else {
         amrex::DistributionMapping new_dmap = MakeDistributionMap(lev, new_grids[lev]);
         const auto old_num_setdm = num_setdm;
         MakeNewLevelFromCoarse(lev, time, new_grids[lev], new_dmap);
         SetBoxArray(lev, new_grids[lev]);
         if (old_num_setdm == num_setdm) {
            SetDistributionMap(lev, new_dmap);
         }
      }
   }

   for (int lev = new_finest + 1; lev <= finest_level; ++lev) {
      ClearLevel(lev);
      ClearBoxArray(lev);
      ClearDistributionMap(lev);
   }

   finest_level = new_finest;
}

} // namespace pyfluid_amrex
