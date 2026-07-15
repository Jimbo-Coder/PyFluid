// Included inside pyfluid_amrex::AmrSolver private section.
// Adaptive timestep and subcycle-count selection helpers.

   template <int NComp, int EOSMethod, int DiffusionMethod>
   amrex::Real adaptive_dt_all_levels()
   {
      amrex::Real dt = std::numeric_limits<amrex::Real>::max();
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         if (solid_active(lev) && !has_mhd_components<NComp>()) {
            const auto solid_apertures = solid_aperture_ptrs(lev);
            dt = amrex::min(
               dt,
               adaptive_dt_cutcell_masked<NComp, EOSMethod, DiffusionMethod>(
                  *frequency_[lev],
                  *state_[lev],
                  *solid_mask_[lev],
                  *solid_volume_fraction_[lev],
                  solid_apertures,
                  Geom(lev),
                  cfg_,
                  &solid_valid_kind_[lev]
               )
            );
         } else {
            if constexpr (!has_mhd_components<NComp>()) {
               if (mixed_patch_system_enabled()) {
                  const PatchGridSystem patch_grid_system = production_patch_grid_system_for_level(lev);
                  const PatchLayout patch_layout = production_patch_layout_for_level(lev);
                  dt = amrex::min(
                     dt,
                     adaptive_dt_mapped_hydro<NComp, EOSMethod, DiffusionMethod>(
                        *frequency_[lev],
                        *state_[lev],
                        cfg_,
                        patch_grid_system,
                        patch_layout
                     )
                  );
               } else {
                  dt = amrex::min(
                     dt,
                     adaptive_dt<NComp, EOSMethod, DiffusionMethod>(
                        *frequency_[lev], *state_[lev], Geom(lev), cfg_, level_times_[lev]
                     )
                  );
               }
            } else {
               dt = amrex::min(
                  dt,
                  adaptive_dt<NComp, EOSMethod, DiffusionMethod>(
                     *frequency_[lev], *state_[lev], Geom(lev), cfg_, level_times_[lev]
                  )
               );
            }
         }
         dt = amrex::min(dt, solid_geometry_motion_dt(lev));
      }
      return dt;
   }

   template <int NComp, int EOSMethod, int DiffusionMethod>
   amrex::Real adaptive_dt_for_level(int lev)
   {
      if (solid_active(lev) && !has_mhd_components<NComp>()) {
         const auto solid_apertures = solid_aperture_ptrs(lev);
         return amrex::min(
            adaptive_dt_cutcell_masked<NComp, EOSMethod, DiffusionMethod>(
               *frequency_[lev],
               *state_[lev],
               *solid_mask_[lev],
               *solid_volume_fraction_[lev],
               solid_apertures,
               Geom(lev),
               cfg_,
               &solid_valid_kind_[lev]
            ),
            solid_geometry_motion_dt(lev)
         );
      }
      if constexpr (!has_mhd_components<NComp>()) {
         if (mixed_patch_system_enabled()) {
            const PatchGridSystem patch_grid_system = production_patch_grid_system_for_level(lev);
            const PatchLayout patch_layout = production_patch_layout_for_level(lev);
            return amrex::min(
               adaptive_dt_mapped_hydro<NComp, EOSMethod, DiffusionMethod>(
                  *frequency_[lev],
                  *state_[lev],
                  cfg_,
                  patch_grid_system,
                  patch_layout
               ),
               solid_geometry_motion_dt(lev)
            );
         }
      }
      return amrex::min(
         adaptive_dt<NComp, EOSMethod, DiffusionMethod>(
            *frequency_[lev], *state_[lev], Geom(lev), cfg_, level_times_[lev]
         ),
         solid_geometry_motion_dt(lev)
      );
   }

   template <int NComp, int EOSMethod, int DiffusionMethod>
   amrex::Real adaptive_base_dt_all_levels()
   {
      amrex::Real base_dt = std::numeric_limits<amrex::Real>::max();
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         base_dt = amrex::min(base_dt, adaptive_dt_for_level<NComp, EOSMethod, DiffusionMethod>(lev) * level_time_factor(lev));
      }
      return base_dt;
   }

   template <int NComp, int EOSMethod, int DiffusionMethod>
   amrex::Real adaptive_base_dt_for_subcycling()
   {
      if (local_cfl_subcycling()) {
         return adaptive_dt_for_level<NComp, EOSMethod, DiffusionMethod>(0);
      }
      return adaptive_base_dt_all_levels<NComp, EOSMethod, DiffusionMethod>();
   }

   template <int NComp, int EOSMethod, int DiffusionMethod>
   int subcycle_step_count(int coarse_lev, amrex::Real coarse_dt)
   {
      if (!local_cfl_subcycling()) {
         return temporal_ref_ratio(coarse_lev);
      }
      const amrex::Real fine_dt_allowed =
         adaptive_dt_for_level<NComp, EOSMethod, DiffusionMethod>(coarse_lev + 1);
      if (!std::isfinite(fine_dt_allowed) || fine_dt_allowed <= 0.0) {
         amrex::Abort(
            "local-CFL subcycling found an invalid timestep for fine level "
            + std::to_string(coarse_lev + 1)
            + ": allowed_dt=" + std::to_string(fine_dt_allowed)
            + " coarse_dt=" + std::to_string(coarse_dt)
         );
      }
      const amrex::Real raw_count = coarse_dt / fine_dt_allowed;
      const amrex::Real eps =
         64.0 * std::numeric_limits<amrex::Real>::epsilon() * amrex::max(1.0, amrex::Math::abs(raw_count));
      int count = std::max(1, static_cast<int>(std::ceil(raw_count - eps)));
      amrex::Real fine_dt = coarse_dt / static_cast<amrex::Real>(count);
      const amrex::Real safety =
         64.0 * std::numeric_limits<amrex::Real>::epsilon() * amrex::max(1.0, amrex::Math::abs(fine_dt_allowed));
      if (fine_dt > fine_dt_allowed + safety) {
         ++count;
      }
      return count;
   }
