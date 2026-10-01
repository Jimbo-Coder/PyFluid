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
      constexpr auto const& stages = time_integrator_stages<TimeMethod>();
      bool use_direct_flux_stages = can_apply_direct_flux_stages<NComp>(lev_begin, lev_end);
      if constexpr (TimeMethod == TimeRK4 && SubcyclingEnabled) {
         use_direct_flux_stages = false;
      }
      if (use_direct_flux_stages) {
         LevelData& final_state = FinalInPlace ? fluid_state_.state() : fluid_state_.next();
         if constexpr (TimeMethod == TimeEuler) {
            WallTime stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_.begin();
               fill_level_boundaries_range<SubcyclingEnabled>(
                  fluid_state_.state(), lev_begin, lev_end, time + stages[0].time.apply(dt)
               );
               profile_.end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(
               fluid_state_.state(), stages[0].final_rhs_weight.apply(dt), lev_begin, lev_end,
               time + stages[0].time.apply(dt)
            );
            stage_start = profile_.begin();
            copy_plus_flux_divergence_range<NComp>(
               final_state, fluid_state_.state(), stages[0].rhs_weight.apply(dt), lev_begin, lev_end
            );
            profile_.end(ProfileStage::Divergence, stage_start);
         } else if constexpr (TimeMethod == TimeCorrector) {
            WallTime stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_.begin();
               fill_level_boundaries_range<SubcyclingEnabled>(
                  fluid_state_.state(), lev_begin, lev_end, time + stages[0].time.apply(dt)
               );
               profile_.end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(
               fluid_state_.state(), stages[0].final_rhs_weight.apply(dt), lev_begin, lev_end,
               time + stages[0].time.apply(dt)
            );
            stage_start = profile_.begin();
            copy_plus_flux_divergence_range<NComp>(
               fluid_state_.scratch(), fluid_state_.state(), stages[0].rhs_weight.apply(dt), lev_begin,
               lev_end
            );
            profile_.end(ProfileStage::Divergence, stage_start);
            if constexpr (has_mhd_components<NComp>()) {
               install_ct_stage_face_field<EOSMethod>(
                  fluid_state_.scratch(), EvolutionStageSlot::Scratch, lev_begin, lev_end,
                  time + stages[0].output_time.apply(dt), stages[0].input_weight, stages[0].base_weight,
                  stages[0].rhs_weight.apply(dt)
               );
            }

            stage_start = profile_.begin();
            fill_level_boundaries_range<SubcyclingEnabled>(
               fluid_state_.scratch(), lev_begin, lev_end, time + stages[1].time.apply(dt)
            );
            profile_.end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(
               fluid_state_.scratch(), stages[1].final_rhs_weight.apply(dt), lev_begin, lev_end,
               time + stages[1].time.apply(dt)
            );
            stage_start = profile_.begin();
            affine_combination_plus_flux_divergence_range<NComp>(
               final_state, fluid_state_.state(), fluid_state_.scratch(),
               stages[1].input_weight, stages[1].rhs_weight.apply(dt), lev_begin, lev_end
            );
            profile_.end(ProfileStage::Divergence, stage_start);
         } else if constexpr (TimeMethod == TimeMidpoint) {
            WallTime stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_.begin();
               fill_level_boundaries_range<SubcyclingEnabled>(
                  fluid_state_.state(), lev_begin, lev_end, time + stages[0].time.apply(dt)
               );
               profile_.end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(
               fluid_state_.state(), stages[0].final_rhs_weight.apply(dt), lev_begin, lev_end,
               time + stages[0].time.apply(dt)
            );
            stage_start = profile_.begin();
            copy_plus_flux_divergence_range<NComp>(
               fluid_state_.scratch(), fluid_state_.state(), stages[0].rhs_weight.apply(dt), lev_begin,
               lev_end
            );
            profile_.end(ProfileStage::Divergence, stage_start);
            if constexpr (has_mhd_components<NComp>()) {
               install_ct_stage_face_field<EOSMethod>(
                  fluid_state_.scratch(), EvolutionStageSlot::Scratch, lev_begin, lev_end,
                  time + stages[0].output_time.apply(dt), stages[0].input_weight, stages[0].base_weight,
                  stages[0].rhs_weight.apply(dt)
               );
            }

            stage_start = profile_.begin();
            fill_level_boundaries_range<SubcyclingEnabled>(
               fluid_state_.scratch(), lev_begin, lev_end, time + stages[1].time.apply(dt)
            );
            profile_.end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(
               fluid_state_.scratch(), stages[1].final_rhs_weight.apply(dt), lev_begin, lev_end,
               time + stages[1].time.apply(dt)
            );
            stage_start = profile_.begin();
            copy_plus_flux_divergence_range<NComp>(
               final_state, fluid_state_.state(), stages[1].rhs_weight.apply(dt), lev_begin, lev_end
            );
            profile_.end(ProfileStage::Divergence, stage_start);
         } else if constexpr (TimeMethod == TimeSSPRK3) {
            WallTime stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_.begin();
               fill_level_boundaries_range<SubcyclingEnabled>(
                  fluid_state_.state(), lev_begin, lev_end, time + stages[0].time.apply(dt)
               );
               profile_.end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(
               fluid_state_.state(), stages[0].final_rhs_weight.apply(dt), lev_begin, lev_end,
               time + stages[0].time.apply(dt)
            );
            stage_start = profile_.begin();
            copy_plus_flux_divergence_range<NComp>(
               fluid_state_.scratch(), fluid_state_.state(), stages[0].rhs_weight.apply(dt), lev_begin,
               lev_end
            );
            profile_.end(ProfileStage::Divergence, stage_start);
            if constexpr (has_mhd_components<NComp>()) {
               install_ct_stage_face_field<EOSMethod>(
                  fluid_state_.scratch(), EvolutionStageSlot::Scratch, lev_begin, lev_end,
                  time + stages[0].output_time.apply(dt), stages[0].input_weight, stages[0].base_weight,
                  stages[0].rhs_weight.apply(dt)
               );
            }
            write_debug_stage_state(
               fluid_state_.scratch(), "ssprk3_stage1_candidate", time + stages[0].output_time.apply(dt),
               lev_begin, lev_end
            );

            stage_start = profile_.begin();
            fill_level_boundaries_range<SubcyclingEnabled>(
               fluid_state_.scratch(), lev_begin, lev_end, time + stages[1].time.apply(dt)
            );
            profile_.end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(
               fluid_state_.scratch(), stages[1].final_rhs_weight.apply(dt), lev_begin, lev_end,
               time + stages[1].time.apply(dt)
            );
            stage_start = profile_.begin();
            affine_combination_plus_flux_divergence_range<NComp>(
               fluid_state_.next(), fluid_state_.state(), fluid_state_.scratch(),
               stages[1].input_weight, stages[1].rhs_weight.apply(dt), lev_begin,
               lev_end
            );
            profile_.end(ProfileStage::Divergence, stage_start);
            if constexpr (has_mhd_components<NComp>()) {
               install_ct_stage_face_field<EOSMethod>(
                  fluid_state_.next(), EvolutionStageSlot::Next, lev_begin, lev_end,
                  time + stages[1].output_time.apply(dt), stages[1].base_weight, stages[1].input_weight,
                  stages[1].rhs_weight.apply(dt)
               );
            }
            write_debug_stage_state(
               fluid_state_.next(), "ssprk3_stage2_candidate", time + stages[1].output_time.apply(dt),
               lev_begin, lev_end
            );

            stage_start = profile_.begin();
            fill_level_boundaries_range<SubcyclingEnabled>(
               fluid_state_.next(), lev_begin, lev_end, time + stages[2].time.apply(dt)
            );
            profile_.end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(
               fluid_state_.next(), stages[2].final_rhs_weight.apply(dt), lev_begin, lev_end,
               time + stages[2].time.apply(dt)
            );
            stage_start = profile_.begin();
            affine_combination_plus_flux_divergence_range<NComp>(
               final_state, fluid_state_.state(), fluid_state_.next(),
               stages[2].input_weight, stages[2].rhs_weight.apply(dt), lev_begin, lev_end
            );
            profile_.end(ProfileStage::Divergence, stage_start);
            write_debug_stage_state(
               final_state, "ssprk3_stage3_candidate", time + stages[2].output_time.apply(dt), lev_begin,
               lev_end
            );
         } else if constexpr (TimeMethod == TimeRK4) {
            WallTime stage_start = 0.0;
            if constexpr (!InitialBoundariesFilled) {
               stage_start = profile_.begin();
               fill_level_boundaries_range<SubcyclingEnabled>(
                  fluid_state_.state(), lev_begin, lev_end, time + stages[0].time.apply(dt), time, 1
               );
               profile_.end(ProfileStage::BoundaryFill, stage_start);
            }
            compute_flux_stage_range<NComp, EOSMethod>(
               fluid_state_.state(), stages[0].final_rhs_weight.apply(dt), lev_begin, lev_end,
               time + stages[0].time.apply(dt)
            );
            stage_start = profile_.begin();
            copy_two_plus_flux_divergence_range<NComp>(
               fluid_state_.next(), fluid_state_.scratch(), fluid_state_.state(),
               stages[0].final_rhs_weight.apply(dt), stages[0].rhs_weight.apply(dt), lev_begin, lev_end
            );
            profile_.end(ProfileStage::Divergence, stage_start);
            if constexpr (has_mhd_components<NComp>()) {
               install_ct_stage_face_field<EOSMethod>(
                  fluid_state_.scratch(), EvolutionStageSlot::Scratch, lev_begin, lev_end,
                  time + stages[0].output_time.apply(dt), stages[0].input_weight, stages[0].base_weight,
                  stages[0].rhs_weight.apply(dt)
               );
            }

            stage_start = profile_.begin();
            fill_level_boundaries_range<SubcyclingEnabled>(
               fluid_state_.scratch(), lev_begin, lev_end, time + stages[1].time.apply(dt), time, 2
            );
            profile_.end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(
               fluid_state_.scratch(), stages[1].final_rhs_weight.apply(dt), lev_begin, lev_end,
               time + stages[1].time.apply(dt)
            );
            stage_start = profile_.begin();
            add_flux_divergence_and_copy_plus_flux_divergence_range<NComp>(
               fluid_state_.next(), fluid_state_.scratch(), fluid_state_.state(),
               stages[1].final_rhs_weight.apply(dt), stages[1].rhs_weight.apply(dt), lev_begin, lev_end
            );
            profile_.end(ProfileStage::Divergence, stage_start);
            if constexpr (has_mhd_components<NComp>()) {
               install_ct_stage_face_field<EOSMethod>(
                  fluid_state_.scratch(), EvolutionStageSlot::Scratch, lev_begin, lev_end,
                  time + stages[1].output_time.apply(dt), stages[1].base_weight, stages[1].input_weight,
                  stages[1].rhs_weight.apply(dt)
               );
            }

            stage_start = profile_.begin();
            fill_level_boundaries_range<SubcyclingEnabled>(
               fluid_state_.scratch(), lev_begin, lev_end, time + stages[2].time.apply(dt), time, 3
            );
            profile_.end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(
               fluid_state_.scratch(), stages[2].final_rhs_weight.apply(dt), lev_begin, lev_end,
               time + stages[2].time.apply(dt)
            );
            stage_start = profile_.begin();
            add_flux_divergence_and_copy_plus_flux_divergence_range<NComp>(
               fluid_state_.next(), fluid_state_.scratch(), fluid_state_.state(),
               stages[2].final_rhs_weight.apply(dt), stages[2].rhs_weight.apply(dt), lev_begin, lev_end
            );
            profile_.end(ProfileStage::Divergence, stage_start);
            if constexpr (has_mhd_components<NComp>()) {
               install_ct_stage_face_field<EOSMethod>(
                  fluid_state_.scratch(), EvolutionStageSlot::Scratch, lev_begin, lev_end,
                  time + stages[2].output_time.apply(dt), stages[2].base_weight, stages[2].input_weight,
                  stages[2].rhs_weight.apply(dt)
               );
            }

            stage_start = profile_.begin();
            fill_level_boundaries_range<SubcyclingEnabled>(
               fluid_state_.scratch(), lev_begin, lev_end, time + stages[3].time.apply(dt), time, 4
            );
            profile_.end(ProfileStage::BoundaryFill, stage_start);
            compute_flux_stage_range<NComp, EOSMethod>(
               fluid_state_.scratch(), stages[3].final_rhs_weight.apply(dt), lev_begin, lev_end,
               time + stages[3].time.apply(dt)
            );
            stage_start = profile_.begin();
            if constexpr (FinalInPlace) {
               copy_plus_flux_divergence_range<NComp>(
                  fluid_state_.state(), fluid_state_.next(), stages[3].rhs_weight.apply(dt), lev_begin,
                  lev_end
               );
            } else {
               for (int lev = lev_begin; lev < lev_end; ++lev) {
                  if (solid_active(lev) && !has_mhd_components<NComp>() && cfg_.solid_motion == "static") {
                     add_flux_divergence_to_state_masked<NComp>(
                        *fluid_state_.next()[lev], *solid_mask_[lev], flux_ptrs(lev), Geom(lev),
                        stages[3].final_rhs_weight.apply(dt), &solid_valid_kind_[lev]
                     );
                  } else {
                     add_flux_divergence_to_state<NComp>(
                        *fluid_state_.next()[lev], flux_ptrs(lev), Geom(lev),
                        stages[3].final_rhs_weight.apply(dt)
                     );
                  }
               }
            }
            profile_.end(ProfileStage::Divergence, stage_start);
         }
         debug_check_relativistic_stage_range(
            final_state, lev_begin, lev_end,
            "relativistic pre-projection accepted candidate", time + dt
         );
         project_relativistic_exterior_range<EOSMethod>(
            final_state, lev_begin, lev_end
         );
         if constexpr (TimeMethod == TimeSSPRK3) {
            write_debug_stage_state(final_state, "ssprk3_accepted", time + dt, lev_begin, lev_end);
         }
         return !FinalInPlace;
      }

      if constexpr (TimeMethod == TimeEuler) {
         WallTime stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_.begin();
            fill_level_boundaries_range<SubcyclingEnabled>(
               fluid_state_.state(), lev_begin, lev_end, time + stages[0].time.apply(dt)
            );
            profile_.end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::State, stages[0], time, dt, lev_begin, lev_end,
            EvolutionStageSlot::State
         );
         stage_start = profile_.begin();
         evolution_copy_plus_scaled_range<NComp>(
            EvolutionStageSlot::Next, EvolutionStageSlot::State, stages[0].rhs_weight.apply(dt),
            lev_begin, lev_end
         );
         profile_.end(ProfileStage::Update, stage_start);
      } else if constexpr (TimeMethod == TimeCorrector) {
         WallTime stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_.begin();
            fill_level_boundaries_range<SubcyclingEnabled>(
               fluid_state_.state(), lev_begin, lev_end, time + stages[0].time.apply(dt)
            );
            profile_.end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::State, stages[0], time, dt, lev_begin, lev_end,
            EvolutionStageSlot::State
         );
         stage_start = profile_.begin();
         evolution_copy_plus_scaled_range<NComp>(
            EvolutionStageSlot::Scratch, EvolutionStageSlot::State, stages[0].rhs_weight.apply(dt),
            lev_begin, lev_end
         );
         profile_.end(ProfileStage::Update, stage_start);
         if constexpr (has_mhd_components<NComp>()) {
            install_ct_stage_face_field<EOSMethod>(
               fluid_state_.scratch(), EvolutionStageSlot::Scratch, lev_begin, lev_end,
               time + stages[0].output_time.apply(dt), stages[0].input_weight, stages[0].base_weight,
               stages[0].rhs_weight.apply(dt)
            );
         }

         stage_start = profile_.begin();
         fill_level_boundaries_range<SubcyclingEnabled>(
            fluid_state_.scratch(), lev_begin, lev_end, time + stages[1].time.apply(dt)
         );
         profile_.end(ProfileStage::BoundaryFill, stage_start);
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::Scratch, stages[1], time, dt, lev_begin, lev_end,
            EvolutionStageSlot::State
         );
         stage_start = profile_.begin();
         evolution_affine_combination_plus_scaled_range<NComp>(
            EvolutionStageSlot::Next, EvolutionStageSlot::State, EvolutionStageSlot::Scratch,
            stages[1].input_weight, stages[1].rhs_weight.apply(dt), lev_begin,
            lev_end
         );
         profile_.end(ProfileStage::Update, stage_start);
      } else if constexpr (TimeMethod == TimeMidpoint) {
         WallTime stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_.begin();
            fill_level_boundaries_range<SubcyclingEnabled>(
               fluid_state_.state(), lev_begin, lev_end, time + stages[0].time.apply(dt)
            );
            profile_.end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::State, stages[0], time, dt, lev_begin, lev_end,
            EvolutionStageSlot::State
         );
         stage_start = profile_.begin();
         evolution_copy_plus_scaled_range<NComp>(
            EvolutionStageSlot::Scratch, EvolutionStageSlot::State, stages[0].rhs_weight.apply(dt),
            lev_begin, lev_end
         );
         profile_.end(ProfileStage::Update, stage_start);
         if constexpr (has_mhd_components<NComp>()) {
            install_ct_stage_face_field<EOSMethod>(
               fluid_state_.scratch(), EvolutionStageSlot::Scratch, lev_begin, lev_end,
               time + stages[0].output_time.apply(dt), stages[0].input_weight, stages[0].base_weight,
               stages[0].rhs_weight.apply(dt)
            );
         }

         stage_start = profile_.begin();
         fill_level_boundaries_range<SubcyclingEnabled>(
            fluid_state_.scratch(), lev_begin, lev_end, time + stages[1].time.apply(dt)
         );
         profile_.end(ProfileStage::BoundaryFill, stage_start);
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::Scratch, stages[1], time, dt, lev_begin, lev_end,
            EvolutionStageSlot::State
         );
         stage_start = profile_.begin();
         evolution_copy_plus_scaled_range<NComp>(
            EvolutionStageSlot::Next, EvolutionStageSlot::State, stages[1].rhs_weight.apply(dt),
            lev_begin, lev_end
         );
         profile_.end(ProfileStage::Update, stage_start);
      } else if constexpr (TimeMethod == TimeSSPRK3) {
         const bool athena_ssprk3_gravity = uses_athena_ssprk3_gravity_energy_correction();
         const bool previous_suppression = suppress_athena_gravity_energy_source_;
         suppress_athena_gravity_energy_source_ = previous_suppression || athena_ssprk3_gravity;
         WallTime stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_.begin();
            fill_level_boundaries_range<SubcyclingEnabled>(
               fluid_state_.state(), lev_begin, lev_end, time + stages[0].time.apply(dt)
            );
            profile_.end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::State, stages[0], time, dt, lev_begin, lev_end,
            EvolutionStageSlot::State
         );
         if (athena_ssprk3_gravity) {
            store_athena_ssprk3_start_gravity_range(lev_begin, lev_end);
            store_athena_ssprk3_mass_flux_range(0, lev_begin, lev_end);
         }
         stage_start = profile_.begin();
         evolution_copy_plus_scaled_range<NComp>(
            EvolutionStageSlot::Scratch, EvolutionStageSlot::State, stages[0].rhs_weight.apply(dt),
            lev_begin, lev_end
         );
         profile_.end(ProfileStage::Update, stage_start);
         if constexpr (has_mhd_components<NComp>()) {
            install_ct_stage_face_field<EOSMethod>(
               fluid_state_.scratch(), EvolutionStageSlot::Scratch, lev_begin, lev_end,
               time + stages[0].output_time.apply(dt), stages[0].input_weight, stages[0].base_weight,
               stages[0].rhs_weight.apply(dt)
            );
         }
         if (athena_ssprk3_gravity) {
            prepare_stage_sources<SubcyclingEnabled>(fluid_state_.scratch(), lev_begin, lev_end, time + dt);
            apply_athena_ssprk3_energy_release_range(fluid_state_.scratch(), 0, dt, lev_begin, lev_end);
         }
         write_debug_stage_state(
            fluid_state_.scratch(), "ssprk3_stage1_candidate", time + stages[0].output_time.apply(dt),
            lev_begin, lev_end
         );

         stage_start = profile_.begin();
         fill_level_boundaries_range<SubcyclingEnabled>(
            fluid_state_.scratch(), lev_begin, lev_end, time + stages[1].time.apply(dt)
         );
         profile_.end(ProfileStage::BoundaryFill, stage_start);
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::Scratch, stages[1], time, dt, lev_begin, lev_end,
            EvolutionStageSlot::State
         );
         if (athena_ssprk3_gravity) {
            store_athena_ssprk3_mass_flux_range(1, lev_begin, lev_end);
         }
         stage_start = profile_.begin();
         evolution_affine_combination_plus_scaled_range<NComp>(
            EvolutionStageSlot::Next, EvolutionStageSlot::State, EvolutionStageSlot::Scratch,
            stages[1].input_weight, stages[1].rhs_weight.apply(dt), lev_begin,
            lev_end
         );
         profile_.end(ProfileStage::Update, stage_start);
         if constexpr (has_mhd_components<NComp>()) {
            install_ct_stage_face_field<EOSMethod>(
               fluid_state_.next(), EvolutionStageSlot::Next, lev_begin, lev_end,
               time + stages[1].output_time.apply(dt), stages[1].base_weight, stages[1].input_weight,
               stages[1].rhs_weight.apply(dt)
            );
         }
         if (athena_ssprk3_gravity) {
            prepare_stage_sources<SubcyclingEnabled>(fluid_state_.next(), lev_begin, lev_end, time + stages[1].output_time.apply(dt));
            apply_athena_ssprk3_energy_release_range(fluid_state_.next(), 1, dt, lev_begin, lev_end);
         }
         write_debug_stage_state(
            fluid_state_.next(), "ssprk3_stage2_candidate", time + stages[1].output_time.apply(dt),
            lev_begin, lev_end
         );

         stage_start = profile_.begin();
         fill_level_boundaries_range<SubcyclingEnabled>(
            fluid_state_.next(), lev_begin, lev_end, time + stages[2].time.apply(dt)
         );
         profile_.end(ProfileStage::BoundaryFill, stage_start);
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::Next, stages[2], time, dt, lev_begin, lev_end,
            EvolutionStageSlot::State
         );
         if (athena_ssprk3_gravity) {
            store_athena_ssprk3_mass_flux_range(2, lev_begin, lev_end);
         }
         stage_start = profile_.begin();
         evolution_affine_combination_plus_scaled_range<NComp>(
            EvolutionStageSlot::Scratch, EvolutionStageSlot::State, EvolutionStageSlot::Next,
            stages[2].input_weight, stages[2].rhs_weight.apply(dt), lev_begin,
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
         profile_.end(ProfileStage::Update, stage_start);
         write_debug_stage_state(
            fluid_state_.next(), "ssprk3_stage3_candidate", time + stages[2].output_time.apply(dt),
            lev_begin, lev_end
         );
         suppress_athena_gravity_energy_source_ = previous_suppression;
      } else if constexpr (TimeMethod == TimeRK4) {
         WallTime stage_start = 0.0;
         if constexpr (!InitialBoundariesFilled) {
            stage_start = profile_.begin();
            fill_level_boundaries_range<SubcyclingEnabled>(
               fluid_state_.state(), lev_begin, lev_end, time + stages[0].time.apply(dt), time, 1
            );
            profile_.end(ProfileStage::BoundaryFill, stage_start);
         }
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::State, stages[0], time, dt, lev_begin, lev_end,
            EvolutionStageSlot::State, time, 1
         );
         if constexpr (SubcyclingEnabled) {
            store_rk4_rhs_stage<NComp>(0, lev_begin, lev_end);
            if (spacetime_state_.evolves()) {
               spacetime_state_.store_rk4_rhs_stage(0, lev_begin, lev_end);
            }
         }
         stage_start = profile_.begin();
         evolution_copy_two_plus_scaled_range<NComp>(
            EvolutionStageSlot::Next, EvolutionStageSlot::Scratch, EvolutionStageSlot::State,
            stages[0].final_rhs_weight.apply(dt), stages[0].rhs_weight.apply(dt), lev_begin, lev_end
         );
         profile_.end(ProfileStage::Update, stage_start);
         if constexpr (has_mhd_components<NComp>()) {
            install_ct_stage_face_field<EOSMethod>(
               fluid_state_.scratch(), EvolutionStageSlot::Scratch, lev_begin, lev_end,
               time + stages[0].output_time.apply(dt), stages[0].input_weight, stages[0].base_weight,
               stages[0].rhs_weight.apply(dt)
            );
         }

         stage_start = profile_.begin();
         fill_level_boundaries_range<SubcyclingEnabled>(
            fluid_state_.scratch(), lev_begin, lev_end, time + stages[1].time.apply(dt), time, 2
         );
         profile_.end(ProfileStage::BoundaryFill, stage_start);
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::Scratch, stages[1], time, dt, lev_begin, lev_end,
            EvolutionStageSlot::State, time, 2
         );
         if constexpr (SubcyclingEnabled) {
            store_rk4_rhs_stage<NComp>(1, lev_begin, lev_end);
            if (spacetime_state_.evolves()) {
               spacetime_state_.store_rk4_rhs_stage(1, lev_begin, lev_end);
            }
         }
         stage_start = profile_.begin();
         evolution_add_scaled_and_copy_plus_scaled_range<NComp>(
            EvolutionStageSlot::Next, EvolutionStageSlot::Scratch, EvolutionStageSlot::State,
            stages[1].final_rhs_weight.apply(dt), stages[1].rhs_weight.apply(dt), lev_begin, lev_end
         );
         profile_.end(ProfileStage::Update, stage_start);
         if constexpr (has_mhd_components<NComp>()) {
            install_ct_stage_face_field<EOSMethod>(
               fluid_state_.scratch(), EvolutionStageSlot::Scratch, lev_begin, lev_end,
               time + stages[1].output_time.apply(dt), stages[1].base_weight, stages[1].input_weight,
               stages[1].rhs_weight.apply(dt)
            );
         }

         stage_start = profile_.begin();
         fill_level_boundaries_range<SubcyclingEnabled>(
            fluid_state_.scratch(), lev_begin, lev_end, time + stages[2].time.apply(dt), time, 3
         );
         profile_.end(ProfileStage::BoundaryFill, stage_start);
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::Scratch, stages[2], time, dt, lev_begin, lev_end,
            EvolutionStageSlot::State, time, 3
         );
         if constexpr (SubcyclingEnabled) {
            store_rk4_rhs_stage<NComp>(2, lev_begin, lev_end);
            if (spacetime_state_.evolves()) {
               spacetime_state_.store_rk4_rhs_stage(2, lev_begin, lev_end);
            }
         }
         stage_start = profile_.begin();
         evolution_add_scaled_and_copy_plus_scaled_range<NComp>(
            EvolutionStageSlot::Next, EvolutionStageSlot::Scratch, EvolutionStageSlot::State,
            stages[2].final_rhs_weight.apply(dt), stages[2].rhs_weight.apply(dt), lev_begin, lev_end
         );
         profile_.end(ProfileStage::Update, stage_start);
         if constexpr (has_mhd_components<NComp>()) {
            install_ct_stage_face_field<EOSMethod>(
               fluid_state_.scratch(), EvolutionStageSlot::Scratch, lev_begin, lev_end,
               time + stages[2].output_time.apply(dt), stages[2].base_weight, stages[2].input_weight,
               stages[2].rhs_weight.apply(dt)
            );
         }

         stage_start = profile_.begin();
         fill_level_boundaries_range<SubcyclingEnabled>(
            fluid_state_.scratch(), lev_begin, lev_end, time + stages[3].time.apply(dt), time, 4
         );
         profile_.end(ProfileStage::BoundaryFill, stage_start);
         compute_evolution_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
            EvolutionStageSlot::Scratch, stages[3], time, dt, lev_begin, lev_end,
            EvolutionStageSlot::Next, time, 4
         );
         if constexpr (SubcyclingEnabled) {
            store_rk4_rhs_stage<NComp>(3, lev_begin, lev_end);
            if (spacetime_state_.evolves()) {
               spacetime_state_.store_rk4_rhs_stage(3, lev_begin, lev_end);
            }
         }
         stage_start = profile_.begin();
         evolution_add_scaled_range<NComp>(
            EvolutionStageSlot::Next, stages[3].final_rhs_weight.apply(dt), lev_begin, lev_end
         );
         profile_.end(ProfileStage::Update, stage_start);
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
            spacetime_state_.state(), lev_begin, lev_end, time + dt, -1.0_rt, 0
         );
      }
      // The accepted-state projection can apply density/pressure floors
      // without constituting a failed C2P reset. Those changes are real
      // external sources to the finite-volume matter budget and must be
      // recorded once, at the state that is actually accepted.
      debug_check_relativistic_stage_range(
         fluid_state_.next(), lev_begin, lev_end,
         "relativistic pre-projection accepted candidate", time + dt
      );
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         const MassEnergy projection_before =
            local_state_mass_energy_integral(
               lev, *fluid_state_.next()[lev]
            );
         project_relativistic_exterior_range<EOSMethod>(
            fluid_state_.next(), lev, lev + 1
         );
         accumulate_source_change(
            projection_before,
            local_state_mass_energy_integral(
               lev, *fluid_state_.next()[lev]
            ),
            1.0_rt
         );
      }
      if constexpr (TimeMethod == TimeSSPRK3) {
         write_debug_stage_state(
            fluid_state_.next(), "ssprk3_accepted", time + dt, lev_begin, lev_end
         );
      }
      debug_check_relativistic_stage_range(
         fluid_state_.next(), lev_begin, lev_end,
         "relativistic accepted candidate", time + dt
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
         const WallTime stage_start = profile_.begin();
         if (next_state_needs_copy) {
            reflux_next_levels();
         } else {
            for (int lev = finestLevel(); lev > 0; --lev) {
               reflux_level_to_state(lev);
            }
         }
         profile_.end(ProfileStage::FluxRegister, stage_start);
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
