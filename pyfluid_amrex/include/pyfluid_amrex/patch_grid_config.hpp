#pragma once

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/patch_grid.hpp"
#include "pyfluid_amrex/patch_layout_builder.hpp"

namespace pyfluid_amrex {

inline PatchCoordinate configured_patch_origin(RunConfig const& cfg)
{
   return make_patch_coordinate(cfg.mesh_patch_origin1, cfg.mesh_patch_origin2, cfg.mesh_patch_origin3);
}

inline int cartesian_component_for_mesh_axis(int axis)
{
   if (axis == MeshAxisX) {
      return 0;
   }
   if (axis == MeshAxisY) {
      return 1;
   }
   if (axis == MeshAxisZ) {
      return 2;
   }
   amrex::Abort("mixed patch mesh.axes must use Cartesian axes x/y/z");
   return 0;
}

inline PatchCartesianAxes configured_patch_cartesian_axes(RunConfig const& cfg)
{
   const MeshAxes mesh_axes = mesh_axes_layout(MeshCartesian, cfg.mesh_axes);
   PatchCartesianAxes axes{};
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      axes.component[dir] = cartesian_component_for_mesh_axis(mesh_axis(mesh_axes, dir));
   }
   return axes;
}

inline PatchGridResolution configured_patch_grid_resolution(RunConfig const& cfg)
{
   PatchGridResolution resolution;
   resolution.central_cells = amrex::IntVect(
      AMREX_D_DECL(
         cfg.mesh_patch_central_cells[0],
         cfg.mesh_patch_central_cells[1],
         cfg.mesh_patch_central_cells[2]
      )
   );
   resolution.angular_cells = cfg.mesh_patch_angular_cells;
   resolution.radial_cells = cfg.mesh_patch_radial_cells;
   resolution.outer_radial_cells = cfg.mesh_patch_outer_radial_cells;
   resolution.axial_cells = cfg.mesh_patch_axial_cells;
   resolution.axial_lo = cfg.mesh_patch_axial_lo;
   resolution.axial_hi = cfg.mesh_patch_axial_hi;
   return resolution;
}

inline bool config_uses_mixed_patch_system(RunConfig const& cfg)
{
   const int patch_system = mesh_patch_system_code(cfg.mesh_patch_system);
   return patch_system == MeshPatchMixedThornburg04
      || patch_system == MeshPatchMixedThornburg13
      || patch_system == MeshPatchMixedCylindrical5
      || patch_system == MeshPatchMixedCylindrical9;
}

inline PatchGridSystem configured_mixed_patch_grid_system(RunConfig const& cfg)
{
   const PatchGridResolution resolution = configured_patch_grid_resolution(cfg);
   const PatchCoordinate origin = configured_patch_origin(cfg);
   const PatchCartesianAxes active_axes = configured_patch_cartesian_axes(cfg);
   switch (mesh_patch_system_code(cfg.mesh_patch_system)) {
   case MeshPatchMixedThornburg04:
      return make_thornburg04_patch_grid_system(
         cfg.mesh_patch_inner_radius,
         cfg.mesh_patch_outer_radius,
         resolution,
         origin,
         active_axes
      );
   case MeshPatchMixedThornburg13:
      return make_thornburg13_patch_grid_system(
         cfg.mesh_patch_inner_radius,
         cfg.mesh_patch_medium_radius,
         cfg.mesh_patch_outer_radius,
         resolution,
         origin,
         active_axes
      );
   case MeshPatchMixedCylindrical5:
      return make_cylindrical5_patch_grid_system(
         cfg.mesh_patch_inner_radius,
         cfg.mesh_patch_outer_radius,
         resolution,
         origin,
         active_axes
      );
   case MeshPatchMixedCylindrical9:
      return make_cylindrical9_patch_grid_system(
         cfg.mesh_patch_inner_radius,
         cfg.mesh_patch_medium_radius,
         cfg.mesh_patch_outer_radius,
         resolution,
         origin,
         active_axes
      );
   default:
      amrex::Abort("configured_mixed_patch_grid_system requires a mixed patch system");
      return PatchGridSystem{};
   }
}

inline void refine_patch_grid_system(PatchGridSystem& grid_system, amrex::IntVect const& refinement)
{
   for (PatchGridDescriptor& patch : grid_system.patches) {
      patch.cells *= refinement;
   }
}

inline PatchGridSystem configured_mixed_patch_grid_system(
   RunConfig const& cfg,
   amrex::IntVect const& refinement
)
{
   PatchGridSystem grid_system = configured_mixed_patch_grid_system(cfg);
   refine_patch_grid_system(grid_system, refinement);
   return grid_system;
}

inline PatchLogicalLayoutOptions configured_patch_logical_layout_options(
   RunConfig const& cfg,
   amrex::IntVect const& refinement = amrex::IntVect(AMREX_D_DECL(1, 1, 1))
)
{
   PatchLogicalLayoutOptions options;
   options.pack_direction = cfg.mesh_patch_pack_direction;
   options.gap_cells = cfg.mesh_patch_gap_cells * refinement[options.pack_direction];
   return options;
}

inline PatchLayout configured_mixed_patch_layout(
   RunConfig const& cfg,
   amrex::IntVect const& refinement = amrex::IntVect(AMREX_D_DECL(1, 1, 1))
)
{
   return make_patch_layout_from_grid_system(
      configured_mixed_patch_grid_system(cfg, refinement),
      configured_patch_logical_layout_options(cfg, refinement)
   );
}

inline amrex::IntVect configured_mixed_patch_refinement_for_domain(
   RunConfig const& cfg,
   amrex::Box const& domain
)
{
   PatchLayout const base_layout = configured_mixed_patch_layout(cfg);
   amrex::IntVect refinement(AMREX_D_DECL(1, 1, 1));
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const int base_length = base_layout.global_box.length(dir);
      const int domain_length = domain.length(dir);
      if (base_length <= 0 || domain_length % base_length != 0) {
         amrex::Abort("mixed patch domain does not refine the level-0 packed patch layout by an integer ratio");
      }
      refinement[dir] = domain_length / base_length;
   }
   return refinement;
}

inline PatchLayout configured_mixed_patch_layout_for_domain(
   RunConfig const& cfg,
   amrex::Box const& domain
)
{
   const amrex::IntVect refinement = configured_mixed_patch_refinement_for_domain(cfg, domain);
   PatchLayout layout = configured_mixed_patch_layout(cfg, refinement);
   if (layout.global_box != domain) {
      amrex::Abort("mixed patch domain does not match the refined packed patch layout");
   }
   return layout;
}

} // namespace pyfluid_amrex
