// Included inside pyfluid_amrex::AmrSolver private section.
// Recursive AMR time-subcycling orchestration.

   template <int NComp, bool CTEnabled, int TimeMethod, int DiffusionMethod, int EOSMethod>
   DiffusionStats advance_one_level_subcycle_step(int lev, amrex::Real dt)
   {
      DiffusionStats diffusion;
      const amrex::Real time = level_times_[lev];
      level_old_times_[lev] = time;
      level_dts_[lev] = dt;
      invalidate_rk4_stage_cache(lev);
      refresh_solid_geometry_level(lev, time, true);

      if constexpr (CTEnabled) {
         const amrex::Real stage_start = profile_begin();
         apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
         sync_cell_b_from_faces(*fluid_state_.state()[lev], face_b_[lev], Geom(lev), false);
         profile_end(ProfileStage::CT, stage_start);
      }

      amrex::Real stage_start = profile_begin();
      if constexpr (TimeMethod == TimeRK4) {
         fill_level_boundaries<true>(fluid_state_.state(), lev, time, time, 1);
      } else {
         fill_level_boundaries<true>(fluid_state_.state(), lev, time);
      }
      profile_end(ProfileStage::BoundaryFill, stage_start);
      copy_level_state_to_old(lev);

      const bool next_state_needs_copy =
         advance_level_range<NComp, TimeMethod, EOSMethod, true, true, !CTEnabled>(time, dt, lev, lev + 1);

      if constexpr (CTEnabled) {
         stage_start = profile_begin();
         diffusion = combine_diffusion_stats(diffusion, advance_ct_level<DiffusionMethod>(lev, dt));
         profile_end(ProfileStage::CT, stage_start);
      }

      if (next_state_needs_copy) {
         stage_start = profile_begin();
         copy_next_to_state_level<NComp>(lev);
         reset_barotropic_energy_level<NComp, EOSMethod>(lev);
         profile_end(ProfileStage::Update, stage_start);
      } else {
         reset_barotropic_energy_level<NComp, EOSMethod>(lev);
      }

      level_times_[lev] = time + dt;
      level_steps_[lev] += 1;
      if (cfg_.regrid_mode == "aligned" && cfg_.regrid_int > 0 && lev == finestLevel()) {
         ++aligned_regrid_ticks_;
         aligned_regrid_due_ = aligned_regrid_ticks_ % cfg_.regrid_int == 0;
      }
      refresh_solid_geometry_level(lev, level_times_[lev], true);

      if constexpr (DiffusionMethod != DiffusionNone) {
         RunConfig viscosity_cfg = viscosity_diffusion_config();
         if (viscosity_enabled(viscosity_cfg)) {
            stage_start = profile_begin();
            diffusion = combine_diffusion_stats(
               diffusion,
               apply_diffusion_level<DiffusionMethod, NComp>(lev, level_times_[lev], dt, viscosity_cfg)
            );
            reset_barotropic_energy_level<NComp, EOSMethod>(lev);
            profile_end(ProfileStage::Diffusion, stage_start);
         }
         if constexpr (CTEnabled) {
            stage_start = profile_begin();
            diffusion = combine_diffusion_stats(
               diffusion,
               advance_ct_inactive_magnetic_resistivity_level<DiffusionMethod>(lev, dt)
            );
            reset_barotropic_energy_level<NComp, EOSMethod>(lev);
            profile_end(ProfileStage::Diffusion, stage_start);
         }
      }

      if (cfg_.debug_check_finite || accepted_physical_check_due(level_steps_[lev], false)) {
         stage_start = profile_begin();
         debug_check_state_level<NComp, EOSMethod>(lev, "post-subcycle accepted state", level_steps_[lev], level_times_[lev]);
         profile_end(ProfileStage::Diagnostics, stage_start);
      }

      return diffusion;
   }

   template <int NComp, bool CTEnabled, int TimeMethod, int DiffusionMethod, int EOSMethod>
   DiffusionStats advance_subcycled_level(
      int lev,
      amrex::Real dt,
      amrex::Real& active_cell_updates,
      amrex::Real& layout_cell_updates
   )
   {
      if (lev < finestLevel()) {
         reset_flux_register(lev + 1);
      }

      active_cell_updates += static_cast<amrex::Real>(active_cell_count_level(lev));
      layout_cell_updates += static_cast<amrex::Real>(boxArray(lev).numPts());
      DiffusionStats diffusion =
         advance_one_level_subcycle_step<NComp, CTEnabled, TimeMethod, DiffusionMethod, EOSMethod>(lev, dt);

      if (lev < finestLevel()) {
         const amrex::Real boundary_start = profile_begin();
         fill_level_boundaries<true>(fluid_state_.state(), lev, level_times_[lev]);
         profile_end(ProfileStage::BoundaryFill, boundary_start);

         const int substeps = subcycle_step_count<NComp, EOSMethod, DiffusionMethod>(lev, dt);
         const amrex::Real fine_dt = dt / static_cast<amrex::Real>(substeps);
         for (int substep = 0; substep < substeps; ++substep) {
            diffusion = combine_diffusion_stats(
               diffusion,
               advance_subcycled_level<NComp, CTEnabled, TimeMethod, DiffusionMethod, EOSMethod>(
                  lev + 1,
                  fine_dt,
                  active_cell_updates,
                  layout_cell_updates
               )
            );
         }

         const amrex::Real stage_start = profile_begin();
         reflux_level_to_state(lev + 1);
         average_down_state_level(lev + 1);
         average_down_spacetime_level(lev + 1);
         reset_barotropic_energy_level<NComp, EOSMethod>(lev);
         if constexpr (CTEnabled) {
            reflux_edge_flux_register_to_face_b(lev + 1);
            average_down_face_b_level(lev + 1);
            sync_cell_b_from_faces(*fluid_state_.state()[lev], face_b_[lev], Geom(lev), false);
            reset_barotropic_energy_level<NComp, EOSMethod>(lev);
         }
         profile_end(ProfileStage::AverageDown, stage_start);
      }

      if (should_regrid_aligned_descendants(lev)) {
         const amrex::Real stage_start = profile_begin();
         regrid_hierarchy<true>(level_times_[lev], level_steps_[lev], lev);
         aligned_regrid_due_ = false;
         profile_end(ProfileStage::Regrid, stage_start);
      }

      return diffusion;
   }

   template <int NComp, bool CTEnabled, int TimeMethod, int DiffusionMethod, int EOSMethod>
   DiffusionStats advance_subcycled_hierarchy(
      amrex::Real dt,
      amrex::Real& active_cell_updates,
      amrex::Real& layout_cell_updates
   )
   {
      DiffusionStats diffusion = advance_subcycled_level<NComp, CTEnabled, TimeMethod, DiffusionMethod, EOSMethod>(
         0,
         dt,
         active_cell_updates,
         layout_cell_updates
      );
      return diffusion;
   }
