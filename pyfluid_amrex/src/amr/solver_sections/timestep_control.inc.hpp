// Included inside pyfluid_amrex::AmrSolver private section.
// Adaptive timestep and subcycle-count selection helpers.

   amrex::Real spacetime_dt_for_level(int lev)
   {
      if (!spacetime_state_.evolves()) {
         return std::numeric_limits<amrex::Real>::max();
      }
      amrex::Real gauge_speed = 1.0;
      const Z4cLapseGauge lapse_gauge = z4c_lapse_gauge(cfg_);
      if (lapse_gauge == Z4cLapseGauge::OnePlusLog
          || lapse_gauge == Z4cLapseGauge::Trumpet
          || lapse_gauge == Z4cLapseGauge::SlowStart) {
         gauge_speed = std::sqrt(2.0);
      }
      if (z4c_shift_gauge(cfg_) == Z4cShiftGauge::GammaDriver) {
         gauge_speed = amrex::max(gauge_speed, 2.0);
      }
      amrex::MultiFab const& cell_spacetime =
         *spacetime_coupling_.cell_spacetime()[lev];
      amrex::MultiFab& frequency = *frequency_[lev];
      return compute_spacetime_dt(
         frequency, cell_spacetime, Geom(lev), cfg_.cfl, gauge_speed
      );
   }

   template <int NComp, int EOSMethod, int DiffusionMethod>
   amrex::Real adaptive_dt_all_levels()
   {
      amrex::Real dt = std::numeric_limits<amrex::Real>::max();
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         dt = amrex::min(
            dt,
            adaptive_dt_for_level<NComp, EOSMethod, DiffusionMethod>(lev)
         );
      }
      return dt;
   }

   template <int NComp, int EOSMethod, int DiffusionMethod>
   amrex::Real adaptive_dt_for_level(int lev)
   {
      const SignalSpeedLaunch signal_launch{
         *frequency_[lev],
         *fluid_state_.state()[lev],
         cfg_,
         NComp,
         DiffusionMethod,
      };
      amrex::MultiFab const* stored_spacetime = nullptr;
      if (spacetime_state_.active()) {
         prepare_spacetime_coupling_range(
            spacetime_state_.state(),
            lev,
            lev + 1,
            level_times_[lev],
            -1.0,
            0
         );
         stored_spacetime = spacetime_coupling_.cell_spacetime()[lev].get();
      }
      const amrex::Real spacetime_dt = spacetime_dt_for_level(lev);
      if (solid_active(lev) && !has_mhd_components<NComp>()) {
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
      if constexpr (!has_mhd_components<NComp>()) {
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
