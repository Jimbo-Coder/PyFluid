// Included inside pyfluid_amrex::AmrSolver private section.
// Adaptive timestep and subcycle-count selection helpers.

   amrex::Real spacetime_dt_for_level(int lev)
   {
      if (!spacetime_state_.evolves()) {
         return std::numeric_limits<amrex::Real>::max();
      }
      amrex::MultiFab const& cell_spacetime =
         *spacetime_coupling_.cell_spacetime()[lev];
      amrex::MultiFab& frequency = *frequency_[lev];
      return compute_spacetime_dt(
         frequency, cell_spacetime, Geom(lev), cfg_
      );
   }

   amrex::Real adaptive_dt_all_levels()
   {
      amrex::Real dt = std::numeric_limits<amrex::Real>::max();
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         dt = amrex::min(
            dt,
            adaptive_dt_for_level(lev)
         );
      }
      return dt;
   }

   amrex::Real adaptive_dt_for_level(int lev)
   {
      amrex::iMultiFab const* inactive_mask = nullptr;
      if (cfg_.spacetime_black_hole_matter == "excision") {
         inactive_mask = apparent_horizon_tracker_->excision_mask(
            lev, *fluid_state_.state()[lev], Geom(lev)
         );
         AMREX_ALWAYS_ASSERT(inactive_mask != nullptr);
      }
      const SignalSpeedLaunch signal_launch{
         *frequency_[lev],
         *fluid_state_.state()[lev],
         cfg_,
         num_state_components(),
         diffusion_method_,
         inactive_mask,
      };
      amrex::MultiFab const* stored_spacetime = nullptr;
      if (spacetime_state_.active()) {
         prepare_spacetime_coupling_range(
            spacetime_state_.state(),
            lev,
            lev + 1,
            level_times_[lev],
            -1.0_rt,
            0
         );
         stored_spacetime = spacetime_coupling_.cell_spacetime()[lev].get();
      }
      const amrex::Real spacetime_dt = spacetime_dt_for_level(lev);
      if (solid_active(lev) && num_state_components() < NVAR_MHD) {
         const auto solid_apertures = solid_aperture_ptrs(lev);
         return amrex::min(
            spacetime_dt,
            amrex::min(
               compute_cutcell_adaptive_dt(
                  signal_launch,
                  *solid_mask_[lev],
                  *solid_volume_fraction_[lev],
                  solid_apertures,
                  Geom(lev),
                  &solid_valid_kind_[lev]
               ),
               solid_geometry_motion_dt(lev)
            )
         );
      }
      if (num_state_components() < NVAR_MHD) {
         if (mixed_patch_system_enabled()) {
            const PatchGridSystem patch_grid_system = production_patch_grid_system_for_level(lev);
            const PatchLayout patch_layout = production_patch_layout_for_level(lev);
            return amrex::min(
               spacetime_dt,
               amrex::min(
                  compute_mapped_hydro_adaptive_dt(
                     signal_launch,
                     patch_grid_system,
                     patch_layout
                  ),
                  solid_geometry_motion_dt(lev)
               )
            );
         }
      }
      return amrex::min(
         spacetime_dt,
         amrex::min(
            compute_adaptive_dt(
               signal_launch, Geom(lev), level_times_[lev], stored_spacetime
            ),
            solid_geometry_motion_dt(lev)
         )
      );
   }

   amrex::Real adaptive_base_dt_all_levels()
   {
      amrex::Real base_dt = std::numeric_limits<amrex::Real>::max();
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         base_dt = amrex::min(base_dt, adaptive_dt_for_level(lev) * level_time_factor(lev));
      }
      return base_dt;
   }

   amrex::Real adaptive_base_dt_for_subcycling()
   {
      if (local_cfl_subcycling()) {
         return adaptive_dt_for_level(0);
      }
      return adaptive_base_dt_all_levels();
   }

   int subcycle_step_count(int coarse_lev, amrex::Real coarse_dt)
   {
      if (!local_cfl_subcycling()) {
         return temporal_ref_ratio(coarse_lev);
      }
      const amrex::Real fine_dt_allowed =
         adaptive_dt_for_level(coarse_lev + 1);
      if (!std::isfinite(fine_dt_allowed) || fine_dt_allowed <= 0.0_rt) {
         amrex::Abort(
            "local-CFL subcycling found an invalid timestep for fine level "
            + std::to_string(coarse_lev + 1)
            + ": allowed_dt=" + std::to_string(fine_dt_allowed)
            + " coarse_dt=" + std::to_string(coarse_dt)
         );
      }
      const amrex::Real raw_count = coarse_dt / fine_dt_allowed;
      const amrex::Real eps =
         64.0_rt * std::numeric_limits<amrex::Real>::epsilon() * amrex::max(1.0_rt, amrex::Math::abs(raw_count));
      int count = std::max(1, static_cast<int>(std::ceil(raw_count - eps)));
      amrex::Real fine_dt = coarse_dt / static_cast<amrex::Real>(count);
      const amrex::Real safety =
         64.0_rt * std::numeric_limits<amrex::Real>::epsilon() * amrex::max(1.0_rt, amrex::Math::abs(fine_dt_allowed));
      if (fine_dt > fine_dt_allowed + safety) {
         ++count;
      }
      return count;
   }
