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
         LevelData& final_state = FinalInPlace ? fluid_state_.state() : fluid_state_.next();
         if constexpr (TimeMethod == TimeEuler) {
            amrex::Real stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_begin();
               fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.state(), lev_begin, lev_end, time);
               profile_end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(fluid_state_.state(), dt, lev_begin, lev_end, time);
            stage_start = profile_begin();
            copy_plus_flux_divergence_range<NComp>(final_state, fluid_state_.state(), dt, lev_begin, lev_end);
            profile_end(ProfileStage::Divergence, stage_start);
         } else if constexpr (TimeMethod == TimeCorrector) {
            amrex::Real stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_begin();
               fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.state(), lev_begin, lev_end, time);
               profile_end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(fluid_state_.state(), 0.5 * dt, lev_begin, lev_end, time);
            stage_start = profile_begin();
            copy_plus_flux_divergence_range<NComp>(fluid_state_.scratch(), fluid_state_.state(), dt, lev_begin, lev_end);
            profile_end(ProfileStage::Divergence, stage_start);

            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + dt);
            profile_end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(fluid_state_.scratch(), 0.5 * dt, lev_begin, lev_end, time + dt);
            stage_start = profile_begin();
            linear_combination_plus_flux_divergence_range<NComp>(
               final_state, fluid_state_.state(), fluid_state_.scratch(), 0.5, 0.5, 0.5 * dt, lev_begin, lev_end
            );
            profile_end(ProfileStage::Divergence, stage_start);
         } else if constexpr (TimeMethod == TimeMidpoint) {
            amrex::Real stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_begin();
               fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.state(), lev_begin, lev_end, time);
               profile_end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(fluid_state_.state(), 0.0, lev_begin, lev_end, time);
            stage_start = profile_begin();
            copy_plus_flux_divergence_range<NComp>(fluid_state_.scratch(), fluid_state_.state(), 0.5 * dt, lev_begin, lev_end);
            profile_end(ProfileStage::Divergence, stage_start);

            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + 0.5 * dt);
            profile_end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(fluid_state_.scratch(), dt, lev_begin, lev_end, time + 0.5 * dt);
            stage_start = profile_begin();
            copy_plus_flux_divergence_range<NComp>(final_state, fluid_state_.state(), dt, lev_begin, lev_end);
            profile_end(ProfileStage::Divergence, stage_start);
         } else if constexpr (TimeMethod == TimeSSPRK3) {
            amrex::Real stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_begin();
               fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.state(), lev_begin, lev_end, time);
               profile_end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(fluid_state_.state(), dt / 6.0, lev_begin, lev_end, time);
            stage_start = profile_begin();
            copy_plus_flux_divergence_range<NComp>(fluid_state_.scratch(), fluid_state_.state(), dt, lev_begin, lev_end);
            profile_end(ProfileStage::Divergence, stage_start);

            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + dt);
            profile_end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(fluid_state_.scratch(), dt / 6.0, lev_begin, lev_end, time + dt);
            stage_start = profile_begin();
            linear_combination_plus_flux_divergence_range<NComp>(
               fluid_state_.next(), fluid_state_.state(), fluid_state_.scratch(), 0.75, 0.25, 0.25 * dt, lev_begin, lev_end
            );
            profile_end(ProfileStage::Divergence, stage_start);

            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.next(), lev_begin, lev_end, time + 0.5 * dt);
            profile_end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(fluid_state_.next(), 2.0 * dt / 3.0, lev_begin, lev_end, time + 0.5 * dt);
            stage_start = profile_begin();
            linear_combination_plus_flux_divergence_range<NComp>(
               final_state, fluid_state_.state(), fluid_state_.next(), 1.0 / 3.0, 2.0 / 3.0, 2.0 * dt / 3.0, lev_begin, lev_end
            );
            profile_end(ProfileStage::Divergence, stage_start);
         } else if constexpr (TimeMethod == TimeRK4) {
            amrex::Real stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_begin();
               fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.state(), lev_begin, lev_end, time, time, 1);
               profile_end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(fluid_state_.state(), dt / 6.0, lev_begin, lev_end, time);
            stage_start = profile_begin();
            copy_two_plus_flux_divergence_range<NComp>(fluid_state_.next(), fluid_state_.scratch(), fluid_state_.state(), dt / 6.0, 0.5 * dt, lev_begin, lev_end);
            profile_end(ProfileStage::Divergence, stage_start);

            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + 0.5 * dt, time, 2);
            profile_end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(fluid_state_.scratch(), dt / 3.0, lev_begin, lev_end, time + 0.5 * dt);
            stage_start = profile_begin();
            add_flux_divergence_and_copy_plus_flux_divergence_range<NComp>(
               fluid_state_.next(), fluid_state_.scratch(), fluid_state_.state(), dt / 3.0, 0.5 * dt, lev_begin, lev_end
            );
            profile_end(ProfileStage::Divergence, stage_start);

            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + 0.5 * dt, time, 3);
            profile_end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(fluid_state_.scratch(), dt / 3.0, lev_begin, lev_end, time + 0.5 * dt);
            stage_start = profile_begin();
            add_flux_divergence_and_copy_plus_flux_divergence_range<NComp>(
               fluid_state_.next(), fluid_state_.scratch(), fluid_state_.state(), dt / 3.0, dt, lev_begin, lev_end
            );
            profile_end(ProfileStage::Divergence, stage_start);

            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + dt, time, 4);
            profile_end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(fluid_state_.scratch(), dt / 6.0, lev_begin, lev_end, time + dt);
            stage_start = profile_begin();
            if constexpr (FinalInPlace) {
               copy_plus_flux_divergence_range<NComp>(fluid_state_.state(), fluid_state_.next(), dt / 6.0, lev_begin, lev_end);
            } else {
               for (int lev = lev_begin; lev < lev_end; ++lev) {
                  if (solid_active(lev) && !has_mhd_components<NComp>() && cfg_.solid_motion == "static") {
                     add_flux_divergence_to_state_masked<NComp>(
                        *fluid_state_.next()[lev],
                        *solid_mask_[lev],
                        flux_ptrs(lev),
                        Geom(lev),
                        dt / 6.0,
                        &solid_valid_kind_[lev]
                     );
                  } else {
                     add_flux_divergence_to_state<NComp>(*fluid_state_.next()[lev], flux_ptrs(lev), Geom(lev), dt / 6.0);
                  }
               }
            }
            profile_end(ProfileStage::Divergence, stage_start);
         }
         project_relativistic_atmosphere_range<EOSMethod>(
            final_state, lev_begin, lev_end, time + dt
         );
         return !FinalInPlace;
      }

      if constexpr (TimeMethod == TimeEuler) {
         amrex::Real stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.state(), lev_begin, lev_end, time);
            profile_end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::State, dt, lev_begin, lev_end, time
         );
         stage_start = profile_begin();
         evolution_copy_plus_scaled_range<NComp>(
            EvolutionStageSlot::Next, EvolutionStageSlot::State, dt, lev_begin, lev_end
         );
         profile_end(ProfileStage::Update, stage_start);
      } else if constexpr (TimeMethod == TimeCorrector) {
         amrex::Real stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.state(), lev_begin, lev_end, time);
            profile_end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::State, 0.5 * dt, lev_begin, lev_end, time
         );
         stage_start = profile_begin();
         evolution_copy_plus_scaled_range<NComp>(
            EvolutionStageSlot::Scratch, EvolutionStageSlot::State, dt, lev_begin, lev_end
         );
         profile_end(ProfileStage::Update, stage_start);

         stage_start = profile_begin();
         fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + dt);
         profile_end(ProfileStage::BoundaryFill, stage_start);
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::Scratch, 0.5 * dt, lev_begin, lev_end, time + dt
         );
         stage_start = profile_begin();
         evolution_linear_combination_plus_scaled_range<NComp>(
            EvolutionStageSlot::Next,
            EvolutionStageSlot::State,
            EvolutionStageSlot::Scratch,
            0.5,
            0.5,
            0.5 * dt,
            lev_begin,
            lev_end
         );
         profile_end(ProfileStage::Update, stage_start);
      } else if constexpr (TimeMethod == TimeMidpoint) {
         amrex::Real stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.state(), lev_begin, lev_end, time);
            profile_end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::State, 0.0, lev_begin, lev_end, time
         );
         stage_start = profile_begin();
         evolution_copy_plus_scaled_range<NComp>(
            EvolutionStageSlot::Scratch, EvolutionStageSlot::State, 0.5 * dt, lev_begin, lev_end
         );
         profile_end(ProfileStage::Update, stage_start);

         stage_start = profile_begin();
         fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + 0.5 * dt);
         profile_end(ProfileStage::BoundaryFill, stage_start);
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::Scratch, dt, lev_begin, lev_end, time + 0.5 * dt
         );
         stage_start = profile_begin();
         evolution_copy_plus_scaled_range<NComp>(
            EvolutionStageSlot::Next, EvolutionStageSlot::State, dt, lev_begin, lev_end
         );
         profile_end(ProfileStage::Update, stage_start);
      } else if constexpr (TimeMethod == TimeSSPRK3) {
         const bool athena_ssprk3_gravity = uses_athena_ssprk3_gravity_energy_correction();
         const bool previous_suppression = suppress_athena_gravity_energy_source_;
         suppress_athena_gravity_energy_source_ = previous_suppression || athena_ssprk3_gravity;
         amrex::Real stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.state(), lev_begin, lev_end, time);
            profile_end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::State, dt / 6.0, lev_begin, lev_end, time
         );
         if (athena_ssprk3_gravity) {
            store_athena_ssprk3_start_gravity_range(lev_begin, lev_end);
            store_athena_ssprk3_mass_flux_range(0, lev_begin, lev_end);
         }
         stage_start = profile_begin();
         evolution_copy_plus_scaled_range<NComp>(
            EvolutionStageSlot::Scratch, EvolutionStageSlot::State, dt, lev_begin, lev_end
         );
         profile_end(ProfileStage::Update, stage_start);
         if (athena_ssprk3_gravity) {
            prepare_stage_sources<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + dt);
            apply_athena_ssprk3_energy_release_range(fluid_state_.scratch(), 0, dt, lev_begin, lev_end);
         }

         stage_start = profile_begin();
         fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + dt);
         profile_end(ProfileStage::BoundaryFill, stage_start);
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::Scratch, dt / 6.0, lev_begin, lev_end, time + dt
         );
         if (athena_ssprk3_gravity) {
            store_athena_ssprk3_mass_flux_range(1, lev_begin, lev_end);
         }
         stage_start = profile_begin();
         evolution_linear_combination_plus_scaled_range<NComp>(
            EvolutionStageSlot::Next,
            EvolutionStageSlot::State,
            EvolutionStageSlot::Scratch,
            0.75,
            0.25,
            0.25 * dt,
            lev_begin,
            lev_end
         );
         profile_end(ProfileStage::Update, stage_start);
         if (athena_ssprk3_gravity) {
            prepare_stage_sources<SubcyclingEnabled>(fluid_state_.next(), lev_begin, lev_end, time + 0.5 * dt);
            apply_athena_ssprk3_energy_release_range(fluid_state_.next(), 1, dt, lev_begin, lev_end);
         }

         stage_start = profile_begin();
         fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.next(), lev_begin, lev_end, time + 0.5 * dt);
         profile_end(ProfileStage::BoundaryFill, stage_start);
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::Next, 2.0 * dt / 3.0, lev_begin, lev_end, time + 0.5 * dt
         );
         if (athena_ssprk3_gravity) {
            store_athena_ssprk3_mass_flux_range(2, lev_begin, lev_end);
         }
         stage_start = profile_begin();
         evolution_linear_combination_plus_scaled_range<NComp>(
            EvolutionStageSlot::Scratch,
            EvolutionStageSlot::State,
            EvolutionStageSlot::Next,
            1.0 / 3.0,
            2.0 / 3.0,
            2.0 * dt / 3.0,
            lev_begin,
            lev_end
         );
         if (athena_ssprk3_gravity) {
            prepare_stage_sources<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + dt);
            apply_athena_ssprk3_energy_release_range(fluid_state_.scratch(), 2, dt, lev_begin, lev_end);
            add_athena_ssprk3_gravity_energy_source_register(dt, lev_begin, lev_end);
         }
         evolution_copy_range(
            EvolutionStageSlot::Next, EvolutionStageSlot::Scratch, lev_begin, lev_end
         );
         profile_end(ProfileStage::Update, stage_start);
         suppress_athena_gravity_energy_source_ = previous_suppression;
      } else if constexpr (TimeMethod == TimeRK4) {
         amrex::Real stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_begin();
            fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.state(), lev_begin, lev_end, time, time, 1);
            profile_end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::State, dt / 6.0, lev_begin, lev_end, time, time, 1
         );
         if constexpr (SubcyclingEnabled) {
            store_rk4_rhs_stage<NComp>(0, lev_begin, lev_end);
            if (spacetime_state_.evolves()) {
               spacetime_state_.store_rk4_rhs_stage(0, lev_begin, lev_end);
            }
         }
         stage_start = profile_begin();
         evolution_copy_two_plus_scaled_range<NComp>(
            EvolutionStageSlot::Next,
            EvolutionStageSlot::Scratch,
            EvolutionStageSlot::State,
            dt / 6.0,
            0.5 * dt,
            lev_begin,
            lev_end
         );
         profile_end(ProfileStage::Update, stage_start);

         stage_start = profile_begin();
         fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + 0.5 * dt, time, 2);
         profile_end(ProfileStage::BoundaryFill, stage_start);
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::Scratch,
            dt / 3.0,
            lev_begin,
            lev_end,
            time + 0.5 * dt,
            time,
            2
         );
         if constexpr (SubcyclingEnabled) {
            store_rk4_rhs_stage<NComp>(1, lev_begin, lev_end);
            if (spacetime_state_.evolves()) {
               spacetime_state_.store_rk4_rhs_stage(1, lev_begin, lev_end);
            }
         }
         stage_start = profile_begin();
         evolution_add_scaled_and_copy_plus_scaled_range<NComp>(
            EvolutionStageSlot::Next,
            EvolutionStageSlot::Scratch,
            EvolutionStageSlot::State,
            dt / 3.0,
            0.5 * dt,
            lev_begin,
            lev_end
         );
         profile_end(ProfileStage::Update, stage_start);

         stage_start = profile_begin();
         fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + 0.5 * dt, time, 3);
         profile_end(ProfileStage::BoundaryFill, stage_start);
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::Scratch,
            dt / 3.0,
            lev_begin,
            lev_end,
            time + 0.5 * dt,
            time,
            3
         );
         if constexpr (SubcyclingEnabled) {
            store_rk4_rhs_stage<NComp>(2, lev_begin, lev_end);
            if (spacetime_state_.evolves()) {
               spacetime_state_.store_rk4_rhs_stage(2, lev_begin, lev_end);
            }
         }
         stage_start = profile_begin();
         evolution_add_scaled_and_copy_plus_scaled_range<NComp>(
            EvolutionStageSlot::Next,
            EvolutionStageSlot::Scratch,
            EvolutionStageSlot::State,
            dt / 3.0,
            dt,
            lev_begin,
            lev_end
         );
         profile_end(ProfileStage::Update, stage_start);

         stage_start = profile_begin();
         fill_level_boundaries_range<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + dt, time, 4);
         profile_end(ProfileStage::BoundaryFill, stage_start);
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::Scratch,
            dt / 6.0,
            lev_begin,
            lev_end,
            time + dt,
            time,
            4
         );
         if constexpr (SubcyclingEnabled) {
            store_rk4_rhs_stage<NComp>(3, lev_begin, lev_end);
            if (spacetime_state_.evolves()) {
               spacetime_state_.store_rk4_rhs_stage(3, lev_begin, lev_end);
            }
         }
         stage_start = profile_begin();
         evolution_add_scaled_range<NComp>(
            EvolutionStageSlot::Next, dt / 6.0, lev_begin, lev_end
         );
         profile_end(ProfileStage::Update, stage_start);
      }
      if (spacetime_state_.evolves()) {
         project_spacetime_state_range(
            spacetime_state_.next(), lev_begin, lev_end
         );
         for (int lev = lev_begin; lev < lev_end; ++lev) {
            spacetime_state_.copy_next_to_state(lev);
         }
      }
      if (spacetime_state_.active()) {
         prepare_spacetime_coupling_range(
            spacetime_state_.state(), lev_begin, lev_end, time + dt, -1.0, 0
         );
      }
      project_relativistic_atmosphere_range<EOSMethod>(
         fluid_state_.next(), lev_begin, lev_end, time + dt
      );
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
         copy_components_masked<NComp>(*fluid_state_.state()[lev], *fluid_state_.next()[lev], *solid_mask_[lev], &solid_valid_kind_[lev]);
      } else {
         amrex::MultiFab::Copy(*fluid_state_.state()[lev], *fluid_state_.next()[lev], 0, 0, NComp, 0);
      }
   }
