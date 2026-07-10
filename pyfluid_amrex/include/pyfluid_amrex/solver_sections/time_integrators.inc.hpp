// Included inside pyfluid_amrex::AmrSolver private section.
// Single-rate time-integrator stage orchestration.

   template <
      int NComp,
      int TimeMethod,
      int EOSMethod,
      bool SubcyclingEnabled = false,
      bool InitialBoundariesFilled = false,
      bool FinalInPlace = false
   >
   bool advance_level_range(amrex::Real time, amrex::Real dt, int lev_begin, int lev_end)
   {
      bool use_direct_flux_stages = can_apply_direct_flux_stages<NComp>(lev_begin, lev_end);
      if constexpr (TimeMethod == TimeRK4 && SubcyclingEnabled) {
         use_direct_flux_stages = false;
      }
      if (use_direct_flux_stages) {
         LevelData& final_state = FinalInPlace ? state_ : next_;
         if constexpr (TimeMethod == TimeEuler) {
            amrex::Real stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_begin();
               fill_level_boundaries_range<SubcyclingEnabled>(state_, lev_begin, lev_end, time);
               profile_end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(state_, dt, lev_begin, lev_end, time);
            stage_start = profile_begin();
            copy_plus_flux_divergence_range<NComp>(final_state, state_, dt, lev_begin, lev_end);
            profile_end(ProfileStage::Divergence, stage_start);
         } else if constexpr (TimeMethod == TimeCorrector) {
            amrex::Real stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_begin();
               fill_level_boundaries_range<SubcyclingEnabled>(state_, lev_begin, lev_end, time);
               profile_end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(state_, 0.5 * dt, lev_begin, lev_end, time);
            stage_start = profile_begin();
            copy_plus_flux_divergence_range<NComp>(scratch_, state_, dt, lev_begin, lev_end);
            profile_end(ProfileStage::Divergence, stage_start);

            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(scratch_, lev_begin, lev_end, time + dt);
            profile_end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(scratch_, 0.5 * dt, lev_begin, lev_end, time + dt);
            stage_start = profile_begin();
            linear_combination_plus_flux_divergence_range<NComp>(
               final_state, state_, scratch_, 0.5, 0.5, 0.5 * dt, lev_begin, lev_end
            );
            profile_end(ProfileStage::Divergence, stage_start);
         } else if constexpr (TimeMethod == TimeMidpoint) {
            amrex::Real stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_begin();
               fill_level_boundaries_range<SubcyclingEnabled>(state_, lev_begin, lev_end, time);
               profile_end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(state_, 0.0, lev_begin, lev_end, time);
            stage_start = profile_begin();
            copy_plus_flux_divergence_range<NComp>(scratch_, state_, 0.5 * dt, lev_begin, lev_end);
            profile_end(ProfileStage::Divergence, stage_start);

            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(scratch_, lev_begin, lev_end, time + 0.5 * dt);
            profile_end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(scratch_, dt, lev_begin, lev_end, time + 0.5 * dt);
            stage_start = profile_begin();
            copy_plus_flux_divergence_range<NComp>(final_state, state_, dt, lev_begin, lev_end);
            profile_end(ProfileStage::Divergence, stage_start);
         } else if constexpr (TimeMethod == TimeSSPRK3) {
            amrex::Real stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_begin();
               fill_level_boundaries_range<SubcyclingEnabled>(state_, lev_begin, lev_end, time);
               profile_end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(state_, dt / 6.0, lev_begin, lev_end, time);
            stage_start = profile_begin();
            copy_plus_flux_divergence_range<NComp>(scratch_, state_, dt, lev_begin, lev_end);
            profile_end(ProfileStage::Divergence, stage_start);

            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(scratch_, lev_begin, lev_end, time + dt);
            profile_end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(scratch_, dt / 6.0, lev_begin, lev_end, time + dt);
            stage_start = profile_begin();
            linear_combination_plus_flux_divergence_range<NComp>(
               next_, state_, scratch_, 0.75, 0.25, 0.25 * dt, lev_begin, lev_end
            );
            profile_end(ProfileStage::Divergence, stage_start);

            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(next_, lev_begin, lev_end, time + 0.5 * dt);
            profile_end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(next_, 2.0 * dt / 3.0, lev_begin, lev_end, time + 0.5 * dt);
            stage_start = profile_begin();
            linear_combination_plus_flux_divergence_range<NComp>(
               final_state, state_, next_, 1.0 / 3.0, 2.0 / 3.0, 2.0 * dt / 3.0, lev_begin, lev_end
            );
            profile_end(ProfileStage::Divergence, stage_start);
         } else if constexpr (TimeMethod == TimeRK4) {
            amrex::Real stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_begin();
               fill_level_boundaries_range<SubcyclingEnabled>(state_, lev_begin, lev_end, time, time, 1);
               profile_end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(state_, dt / 6.0, lev_begin, lev_end, time);
            stage_start = profile_begin();
            copy_two_plus_flux_divergence_range<NComp>(next_, scratch_, state_, dt / 6.0, 0.5 * dt, lev_begin, lev_end);
            profile_end(ProfileStage::Divergence, stage_start);

            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(scratch_, lev_begin, lev_end, time + 0.5 * dt, time, 2);
            profile_end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(scratch_, dt / 3.0, lev_begin, lev_end, time + 0.5 * dt);
            stage_start = profile_begin();
            add_flux_divergence_and_copy_plus_flux_divergence_range<NComp>(
               next_, scratch_, state_, dt / 3.0, 0.5 * dt, lev_begin, lev_end
            );
            profile_end(ProfileStage::Divergence, stage_start);

            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(scratch_, lev_begin, lev_end, time + 0.5 * dt, time, 3);
            profile_end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(scratch_, dt / 3.0, lev_begin, lev_end, time + 0.5 * dt);
            stage_start = profile_begin();
            add_flux_divergence_and_copy_plus_flux_divergence_range<NComp>(
               next_, scratch_, state_, dt / 3.0, dt, lev_begin, lev_end
            );
            profile_end(ProfileStage::Divergence, stage_start);

            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(scratch_, lev_begin, lev_end, time + dt, time, 4);
            profile_end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(scratch_, dt / 6.0, lev_begin, lev_end, time + dt);
            stage_start = profile_begin();
            if constexpr (FinalInPlace) {
               copy_plus_flux_divergence_range<NComp>(state_, next_, dt / 6.0, lev_begin, lev_end);
            } else {
               for (int lev = lev_begin; lev < lev_end; ++lev) {
                  if (solid_active(lev) && !has_mhd_components<NComp>() && cfg_.solid_motion == "static") {
                     add_flux_divergence_to_state_masked<NComp>(
                        *next_[lev],
                        *solid_mask_[lev],
                        flux_ptrs(lev),
                        Geom(lev),
                        dt / 6.0,
                        &solid_valid_kind_[lev]
                     );
                  } else {
                     add_flux_divergence_to_state<NComp>(*next_[lev], flux_ptrs(lev), Geom(lev), dt / 6.0);
                  }
               }
            }
            profile_end(ProfileStage::Divergence, stage_start);
         }
         return !FinalInPlace;
      }

      if constexpr (TimeMethod == TimeEuler) {
         amrex::Real stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(state_, lev_begin, lev_end, time);
            profile_end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(state_, dt, lev_begin, lev_end, time);
         stage_start = profile_begin();
         copy_plus_scaled_range<NComp>(next_, state_, rhs_, dt, lev_begin, lev_end);
         profile_end(ProfileStage::Update, stage_start);
      } else if constexpr (TimeMethod == TimeCorrector) {
         amrex::Real stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(state_, lev_begin, lev_end, time);
            profile_end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(state_, 0.5 * dt, lev_begin, lev_end, time);
         stage_start = profile_begin();
         copy_plus_scaled_range<NComp>(scratch_, state_, rhs_, dt, lev_begin, lev_end);
         profile_end(ProfileStage::Update, stage_start);

         stage_start = profile_begin();
         fill_level_boundaries_range<SubcyclingEnabled>(scratch_, lev_begin, lev_end, time + dt);
         profile_end(ProfileStage::BoundaryFill, stage_start);
         compute_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(scratch_, 0.5 * dt, lev_begin, lev_end, time + dt);
         stage_start = profile_begin();
         linear_combination_plus_scaled_range<NComp>(next_, state_, scratch_, rhs_, 0.5, 0.5, 0.5 * dt, lev_begin, lev_end);
         profile_end(ProfileStage::Update, stage_start);
      } else if constexpr (TimeMethod == TimeMidpoint) {
         amrex::Real stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(state_, lev_begin, lev_end, time);
            profile_end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(state_, 0.0, lev_begin, lev_end, time);
         stage_start = profile_begin();
         copy_plus_scaled_range<NComp>(scratch_, state_, rhs_, 0.5 * dt, lev_begin, lev_end);
         profile_end(ProfileStage::Update, stage_start);

         stage_start = profile_begin();
         fill_level_boundaries_range<SubcyclingEnabled>(scratch_, lev_begin, lev_end, time + 0.5 * dt);
         profile_end(ProfileStage::BoundaryFill, stage_start);
         compute_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(scratch_, dt, lev_begin, lev_end, time + 0.5 * dt);
         stage_start = profile_begin();
         copy_plus_scaled_range<NComp>(next_, state_, rhs_, dt, lev_begin, lev_end);
         profile_end(ProfileStage::Update, stage_start);
      } else if constexpr (TimeMethod == TimeSSPRK3) {
         const bool athena_ssprk3_gravity = uses_athena_ssprk3_gravity_energy_correction();
         const bool previous_suppression = suppress_athena_gravity_energy_source_;
         suppress_athena_gravity_energy_source_ = previous_suppression || athena_ssprk3_gravity;
         amrex::Real stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(state_, lev_begin, lev_end, time);
            profile_end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(state_, dt / 6.0, lev_begin, lev_end, time);
         if (athena_ssprk3_gravity) {
            store_athena_ssprk3_start_gravity_range(lev_begin, lev_end);
            store_athena_ssprk3_mass_flux_range(0, lev_begin, lev_end);
         }
         stage_start = profile_begin();
         copy_plus_scaled_range<NComp>(scratch_, state_, rhs_, dt, lev_begin, lev_end);
         profile_end(ProfileStage::Update, stage_start);
         if (athena_ssprk3_gravity) {
            prepare_stage_sources<SubcyclingEnabled>(scratch_, lev_begin, lev_end, time + dt);
            apply_athena_ssprk3_energy_release_range(scratch_, 0, dt, lev_begin, lev_end);
         }

         stage_start = profile_begin();
         fill_level_boundaries_range<SubcyclingEnabled>(scratch_, lev_begin, lev_end, time + dt);
         profile_end(ProfileStage::BoundaryFill, stage_start);
         compute_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(scratch_, dt / 6.0, lev_begin, lev_end, time + dt);
         if (athena_ssprk3_gravity) {
            store_athena_ssprk3_mass_flux_range(1, lev_begin, lev_end);
         }
         stage_start = profile_begin();
         linear_combination_plus_scaled_range<NComp>(next_, state_, scratch_, rhs_, 0.75, 0.25, 0.25 * dt, lev_begin, lev_end);
         profile_end(ProfileStage::Update, stage_start);
         if (athena_ssprk3_gravity) {
            prepare_stage_sources<SubcyclingEnabled>(next_, lev_begin, lev_end, time + 0.5 * dt);
            apply_athena_ssprk3_energy_release_range(next_, 1, dt, lev_begin, lev_end);
         }

         stage_start = profile_begin();
         fill_level_boundaries_range<SubcyclingEnabled>(next_, lev_begin, lev_end, time + 0.5 * dt);
         profile_end(ProfileStage::BoundaryFill, stage_start);
         compute_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(next_, 2.0 * dt / 3.0, lev_begin, lev_end, time + 0.5 * dt);
         if (athena_ssprk3_gravity) {
            store_athena_ssprk3_mass_flux_range(2, lev_begin, lev_end);
         }
         stage_start = profile_begin();
         linear_combination_plus_scaled_range<NComp>(scratch_, state_, next_, rhs_, 1.0 / 3.0, 2.0 / 3.0, 2.0 * dt / 3.0, lev_begin, lev_end);
         if (athena_ssprk3_gravity) {
            prepare_stage_sources<SubcyclingEnabled>(scratch_, lev_begin, lev_end, time + dt);
            apply_athena_ssprk3_energy_release_range(scratch_, 2, dt, lev_begin, lev_end);
            add_athena_ssprk3_gravity_energy_source_register(dt, lev_begin, lev_end);
         }
         for (int lev = lev_begin; lev < lev_end; ++lev) {
            amrex::MultiFab::Copy(*next_[lev], *scratch_[lev], 0, 0, NComp, 0);
         }
         profile_end(ProfileStage::Update, stage_start);
         suppress_athena_gravity_energy_source_ = previous_suppression;
      } else if constexpr (TimeMethod == TimeRK4) {
         amrex::Real stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(state_, lev_begin, lev_end, time, time, 1);
            profile_end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(state_, dt / 6.0, lev_begin, lev_end, time);
         if constexpr (SubcyclingEnabled) {
            store_rk4_rhs_stage<NComp>(0, lev_begin, lev_end);
         }
         stage_start = profile_begin();
         copy_two_plus_scaled_range<NComp>(next_, scratch_, state_, rhs_, dt / 6.0, 0.5 * dt, lev_begin, lev_end);
         profile_end(ProfileStage::Update, stage_start);

         stage_start = profile_begin();
         fill_level_boundaries_range<SubcyclingEnabled>(scratch_, lev_begin, lev_end, time + 0.5 * dt, time, 2);
         profile_end(ProfileStage::BoundaryFill, stage_start);
         compute_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(scratch_, dt / 3.0, lev_begin, lev_end, time + 0.5 * dt);
         if constexpr (SubcyclingEnabled) {
            store_rk4_rhs_stage<NComp>(1, lev_begin, lev_end);
         }
         stage_start = profile_begin();
         add_scaled_and_copy_plus_scaled_range<NComp>(next_, scratch_, state_, rhs_, dt / 3.0, 0.5 * dt, lev_begin, lev_end);
         profile_end(ProfileStage::Update, stage_start);

         stage_start = profile_begin();
         fill_level_boundaries_range<SubcyclingEnabled>(scratch_, lev_begin, lev_end, time + 0.5 * dt, time, 3);
         profile_end(ProfileStage::BoundaryFill, stage_start);
         compute_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(scratch_, dt / 3.0, lev_begin, lev_end, time + 0.5 * dt);
         if constexpr (SubcyclingEnabled) {
            store_rk4_rhs_stage<NComp>(2, lev_begin, lev_end);
         }
         stage_start = profile_begin();
         add_scaled_and_copy_plus_scaled_range<NComp>(next_, scratch_, state_, rhs_, dt / 3.0, dt, lev_begin, lev_end);
         profile_end(ProfileStage::Update, stage_start);

         stage_start = profile_begin();
         fill_level_boundaries_range<SubcyclingEnabled>(scratch_, lev_begin, lev_end, time + dt, time, 4);
         profile_end(ProfileStage::BoundaryFill, stage_start);
         compute_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(scratch_, dt / 6.0, lev_begin, lev_end, time + dt);
         if constexpr (SubcyclingEnabled) {
            store_rk4_rhs_stage<NComp>(3, lev_begin, lev_end);
         }
         stage_start = profile_begin();
         add_scaled_range<NComp>(next_, rhs_, dt / 6.0, lev_begin, lev_end);
         profile_end(ProfileStage::Update, stage_start);
      }
      return true;
   }

   template <int NComp, int TimeMethod, int EOSMethod, bool FinalInPlace>
   bool advance_all_levels(amrex::Real time, amrex::Real dt)
   {
      if (finestLevel() > 0) {
         reset_flux_registers();
      }
      const bool next_state_needs_copy =
         advance_level_range<NComp, TimeMethod, EOSMethod, false, false, FinalInPlace>(time, dt, 0, finestLevel() + 1);
      if (finestLevel() > 0) {
         const amrex::Real stage_start = profile_begin();
         if (next_state_needs_copy) {
            reflux_next_levels();
         } else {
            for (int lev = finestLevel(); lev > 0; --lev) {
               reflux_level_to_state(lev);
            }
         }
         profile_end(ProfileStage::FluxRegister, stage_start);
      }
      return next_state_needs_copy;
   }

   template <int NComp>
   void copy_next_to_state_level(int lev)
   {
      if (solid_active(lev) && !has_mhd_components<NComp>()) {
         copy_components_masked<NComp>(*state_[lev], *next_[lev], *solid_mask_[lev], &solid_valid_kind_[lev]);
      } else {
         amrex::MultiFab::Copy(*state_[lev], *next_[lev], 0, 0, NComp, 0);
      }
   }
