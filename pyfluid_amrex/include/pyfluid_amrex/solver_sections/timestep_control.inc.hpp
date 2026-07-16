// Included inside pyfluid_amrex::AmrSolver private section.
// Adaptive timestep and subcycle-count selection helpers.

   amrex::Real spacetime_dt_for_level(int lev)
   {
      if (!spacetime_state_.active()) {
         return std::numeric_limits<amrex::Real>::max();
      }
      amrex::Real gauge_speed = 1.0;
      const Z4cLapseGauge lapse_gauge = z4c_lapse_gauge(cfg_);
      if (lapse_gauge == Z4cLapseGauge::OnePlusLog
          || lapse_gauge == Z4cLapseGauge::SlowStart) {
         gauge_speed = std::sqrt(2.0);
      }
      if (z4c_shift_gauge(cfg_) == Z4cShiftGauge::GammaDriver) {
         gauge_speed = amrex::max(gauge_speed, 2.0);
      }
      const auto spacing = Geom(lev).CellSizeArray();
      amrex::MultiFab const& cell_spacetime =
         *spacetime_coupling_.cell_spacetime()[lev];
      amrex::MultiFab& frequency = *frequency_[lev];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(frequency, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto spacetime = cell_spacetime.const_array(mfi);
         const auto output = frequency.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const SpacetimeMetric metric = generated::z4c_spacetime_metric(
               load_z4c_state(spacetime, i, j, k)
            );
            amrex::Real value = (
               amrex::Math::abs(metric.shift1)
               + gauge_speed * metric.lapse * std::sqrt(metric.inv_gamma11)
            ) / spacing[0];
#if (AMREX_SPACEDIM >= 2)
            value += (
               amrex::Math::abs(metric.shift2)
               + gauge_speed * metric.lapse * std::sqrt(metric.inv_gamma22)
            ) / spacing[1];
#endif
#if (AMREX_SPACEDIM == 3)
            value += (
               amrex::Math::abs(metric.shift3)
               + gauge_speed * metric.lapse * std::sqrt(metric.inv_gamma33)
            ) / spacing[2];
#endif
            output(i, j, k, 0) = value;
         });
      }
      const amrex::Real max_frequency = frequency.max(0);
      return cfg_.cfl / max_frequency;
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
      amrex::MultiFab const* evolved_spacetime = nullptr;
      if (spacetime_state_.active()) {
         prepare_spacetime_coupling_range(
            spacetime_state_.state(),
            lev,
            lev + 1,
            level_times_[lev],
            -1.0,
            0
         );
         if (spacetime_matter_is_coupled(cfg_)) {
            evolved_spacetime = spacetime_coupling_.cell_spacetime()[lev].get();
         }
      }
      const amrex::Real spacetime_dt = spacetime_dt_for_level(lev);
      if (solid_active(lev) && !has_mhd_components<NComp>()) {
         const auto solid_apertures = solid_aperture_ptrs(lev);
         return amrex::min(
            spacetime_dt,
            amrex::min(
               adaptive_dt_cutcell_masked<NComp, EOSMethod, DiffusionMethod>(
                  *frequency_[lev],
                  *fluid_state_.state()[lev],
                  *solid_mask_[lev],
                  *solid_volume_fraction_[lev],
                  solid_apertures,
                  Geom(lev),
                  cfg_,
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
                  adaptive_dt_mapped_hydro<NComp, EOSMethod, DiffusionMethod>(
                     *frequency_[lev],
                     *fluid_state_.state()[lev],
                     cfg_,
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
            adaptive_dt<NComp, EOSMethod, DiffusionMethod>(
               *frequency_[lev], *fluid_state_.state()[lev], Geom(lev), cfg_,
               level_times_[lev], evolved_spacetime
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
