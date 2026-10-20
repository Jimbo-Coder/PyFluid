// Included inside pyfluid_amrex::AmrSolver private section.
// Recursive AMR time-subcycling orchestration.

   template <int NComp, bool CTEnabled, int EOSMethod>
   DiffusionStats advance_one_level_subcycle_step(
      SubcycleOperations const& operations, int lev, amrex::Real dt
   )
   {
      DiffusionStats diffusion;
      const amrex::Real time = level_times_[lev];
      level_old_times_[lev] = time;
      level_dts_[lev] = dt;
      invalidate_rk4_stage_cache(lev);
      refresh_solid_geometry_level(lev, time, true);

      if constexpr (CTEnabled) {
         const WallTime stage_start = profile_.begin();
         apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
         sync_cell_b_from_faces(*fluid_state_.state()[lev], face_b_[lev], Geom(lev), false);
         profile_.end(ProfileStage::CT, stage_start);
      }

      WallTime stage_start = profile_.begin();
      if (timestep_method_ == TimeRK4) {
         fill_level_boundaries<true>(fluid_state_.state(), lev, time, time, 1);
      } else {
         fill_level_boundaries<true>(fluid_state_.state(), lev, time);
      }
      profile_.end(ProfileStage::BoundaryFill, stage_start);
      copy_level_state_to_old(lev);

      if constexpr (CTEnabled) {
         begin_ct_step_level(lev);
      }
      const bool next_state_needs_copy =
         (this->*operations.advance_fluid)(time, dt, lev, lev + 1);

      if constexpr (CTEnabled) {
         stage_start = profile_.begin();
         diffusion = combine_diffusion_stats(diffusion, (this->*operations.advance_ct)(lev, dt));
         profile_.end(ProfileStage::CT, stage_start);
      }

      if (next_state_needs_copy) {
         stage_start = profile_.begin();
         copy_next_to_state_level<NComp>(lev);
         reset_barotropic_energy_level<NComp, EOSMethod>(lev);
         profile_.end(ProfileStage::Update, stage_start);
      } else {
         reset_barotropic_energy_level<NComp, EOSMethod>(lev);
      }

      level_times_[lev] = time + dt;
      level_steps_[lev] += 1;
      if (cfg_.regrid_mode == "aligned" && cfg_.regrid_int > 0 && lev == finestLevel()) {
         regrid_lifecycle_.advance_finest_step(cfg_.regrid_int);
      }
      refresh_solid_geometry_level(lev, level_times_[lev], true);

      if (operations.advance_diffusion != nullptr) {
         RunConfig viscosity_cfg = viscosity_diffusion_config();
         if (viscosity_enabled(viscosity_cfg)) {
            stage_start = profile_.begin();
            diffusion = combine_diffusion_stats(
               diffusion,
               (this->*operations.advance_diffusion)(lev, level_times_[lev], dt, viscosity_cfg)
            );
            reset_barotropic_energy_level<NComp, EOSMethod>(lev);
            profile_.end(ProfileStage::Diffusion, stage_start);
         }
         if constexpr (CTEnabled) {
            stage_start = profile_.begin();
            diffusion = combine_diffusion_stats(
               diffusion,
               (this->*operations.advance_inactive_ct_resistivity)(lev, dt)
            );
            reset_barotropic_energy_level<NComp, EOSMethod>(lev);
            profile_.end(ProfileStage::Diffusion, stage_start);
         }
      }

      if constexpr (CTEnabled) {
         // The CT face field is canonical. It is installed after the fluid
         // integrator's accepted-state recovery and may also be changed by
         // resistivity, so validate the state that can seed fine-level time
         // interpolation. The RMHD atmosphere backstop retains B exactly.
         project_relativistic_exterior_range<EOSMethod>(
            fluid_state_.state(), lev, lev + 1
         );
      }

      if (cfg_.debug_check_finite || accepted_physical_check_due(level_steps_[lev], false)) {
         stage_start = profile_.begin();
         debug_check_state_level(lev, "post-subcycle accepted state", level_steps_[lev], level_times_[lev]);
         profile_.end(ProfileStage::Diagnostics, stage_start);
      }

      return diffusion;
   }

   template <int NComp, bool CTEnabled, int EOSMethod>
   DiffusionStats advance_subcycled_level(
      SubcycleOperations const& operations,
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
         advance_one_level_subcycle_step<NComp, CTEnabled, EOSMethod>(operations, lev, dt);

      if (lev < finestLevel()) {
         const WallTime boundary_start = profile_.begin();
         fill_level_boundaries<true>(fluid_state_.state(), lev, level_times_[lev]);
         profile_.end(ProfileStage::BoundaryFill, boundary_start);

         if (local_cfl_subcycling()) {
            // Re-evaluate the fine-level CFL limit after every accepted
            // substep. A rapidly accelerating wave can invalidate a count
            // chosen from the beginning of the coarse interval even though
            // every individual CFL estimate was correct when computed.
            const amrex::Real target_time = level_times_[lev];
            const amrex::Real time_scale = amrex::max(
               1.0_rt, amrex::Math::abs(target_time)
            );
            const amrex::Real time_tolerance =
               64.0_rt * std::numeric_limits<amrex::Real>::epsilon()
               * time_scale;
            int substep = 0;
            while (target_time - level_times_[lev + 1] > time_tolerance) {
               const amrex::Real remaining =
                  target_time - level_times_[lev + 1];
               const amrex::Real fine_dt_allowed = adaptive_dt_for_level(lev + 1);
               if (!std::isfinite(fine_dt_allowed)
                   || fine_dt_allowed <= 0.0_rt) {
                  amrex::Abort(
                     "local-CFL subcycling found an invalid timestep for "
                     "fine level " + std::to_string(lev + 1)
                     + ": allowed_dt=" + std::to_string(fine_dt_allowed)
                     + " remaining_dt=" + std::to_string(remaining)
                  );
               }
               const amrex::Real fine_dt = amrex::min(
                  remaining, fine_dt_allowed
               );
               diffusion = combine_diffusion_stats(
                  diffusion,
                  advance_subcycled_level<
                     NComp, CTEnabled, EOSMethod
                  >(
                     operations,
                     lev + 1,
                     fine_dt,
                     active_cell_updates,
                     layout_cell_updates
                  )
               );
               ++substep;
               if (substep > 1000000) {
                  amrex::Abort(
                     "local-CFL subcycling exceeded one million fine steps "
                     "inside one coarse interval"
                  );
               }
            }
            level_times_[lev + 1] = target_time;
         } else {
            const int substeps = subcycle_step_count(lev, dt);
            const amrex::Real fine_dt =
               dt / static_cast<amrex::Real>(substeps);
            for (int substep = 0; substep < substeps; ++substep) {
               diffusion = combine_diffusion_stats(
                  diffusion,
                  advance_subcycled_level<
                     NComp, CTEnabled, EOSMethod
                  >(
                     operations,
                     lev + 1,
                     fine_dt,
                     active_cell_updates,
                     layout_cell_updates
                  )
               );
            }
         }

         const WallTime stage_start = profile_.begin();
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
         if (spacetime_state_.evolves()) {
            prepare_spacetime_coupling_range(
               spacetime_state_.state(), lev, lev + 1,
               level_times_[lev], -1.0_rt, 0
            );
         }
         // Reflux, average-down, and optional face-B synchronization are the
         // final coarse-state owners for this parent interval. Reflux can
         // place an uncovered coarse atmosphere cell just below its physical
         // floor. The ensuing conservative-state projection is an external
         // source, so include its valid-cell change in the same ledger used
         // for explicit source terms.
         const MassEnergy projection_before =
            local_state_mass_energy_integral(
               lev, *fluid_state_.state()[lev]
            );
         debug_check_relativistic_stage_range(
            fluid_state_.state(), lev, lev + 1,
            "relativistic pre-projection refluxed parent",
            level_times_[lev]
         );
         project_relativistic_exterior_range<EOSMethod>(
            fluid_state_.state(), lev, lev + 1
         );
         accumulate_source_change(
            projection_before,
            local_state_mass_energy_integral(
               lev, *fluid_state_.state()[lev]
            ),
            1.0_rt
         );
         profile_.end(ProfileStage::AverageDown, stage_start);
      }

      if (should_regrid_aligned_descendants(lev)) {
         const WallTime stage_start = profile_.begin();
         const bool hierarchy_changed = regrid_hierarchy_and_project<
            true, EOSMethod
         >(
            level_times_[lev], level_steps_[lev], lev
         );
         static_cast<void>(hierarchy_changed);
         regrid_lifecycle_.clear_descendant_request();
         profile_.end(ProfileStage::Regrid, stage_start);
      }

      return diffusion;
   }

   template <int NComp, bool CTEnabled, int EOSMethod>
   DiffusionStats advance_subcycled_hierarchy(
      SubcycleOperations const& operations,
      amrex::Real dt,
      amrex::Real& active_cell_updates,
      amrex::Real& layout_cell_updates
   )
   {
      DiffusionStats diffusion = advance_subcycled_level<NComp, CTEnabled, EOSMethod>(
         operations,
         0,
         dt,
         active_cell_updates,
         layout_cell_updates
      );
      return diffusion;
   }
