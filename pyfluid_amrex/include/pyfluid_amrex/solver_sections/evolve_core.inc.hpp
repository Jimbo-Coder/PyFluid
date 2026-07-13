// Included inside pyfluid_amrex::AmrSolver private section.
// Top-level branch dispatch and the main synchronized/subcycled evolution loop.

   void dispatch_evolve()
   {
      if (is_mhd(cfg_)) {
#if PYFLUID_AMREX_ENABLE_MHD
         if (ct_enabled(cfg_)) {
#if PYFLUID_AMREX_ENABLE_CT
            if (cfg_.passive_scalars == 1) {
               dispatch_evolve_eos<NVAR_MHD + 1, true>();
            } else {
               dispatch_evolve_eos<NVAR_MHD, true>();
            }
#else
            abort_disabled_build_feature("CT", "on");
#endif
         } else {
            if (cfg_.passive_scalars == 1) {
               dispatch_evolve_eos<NVAR_MHD + 1, false>();
            } else {
               dispatch_evolve_eos<NVAR_MHD, false>();
            }
         }
#else
         abort_disabled_build_feature("physics", "mhd");
#endif
      } else {
#if PYFLUID_AMREX_ENABLE_HYDRO
         if (cfg_.passive_scalars == 1) {
            dispatch_evolve_eos<NVAR_HYDRO + 1, false>();
         } else {
            dispatch_evolve_eos<NVAR_HYDRO, false>();
         }
#else
         abort_disabled_build_feature("physics", "hydro");
#endif
      }
   }

   template <int NComp, bool CTEnabled>
   void dispatch_evolve_eos()
   {
      if (eos_method_ == EOSPolytrope) {
         dispatch_evolve_timestep<NComp, CTEnabled, EOSPolytrope>();
      } else if (eos_method_ == EOSHybridPolytrope) {
         dispatch_evolve_timestep<NComp, CTEnabled, EOSHybridPolytrope>();
      } else {
         dispatch_evolve_timestep<NComp, CTEnabled, EOSIdealGas>();
      }
   }

   template <int NComp, bool CTEnabled, int EOSMethod>
   void dispatch_evolve_timestep()
   {
      if (timestep_method_ == TimeRK4) {
#if PYFLUID_AMREX_ENABLE_TIME_RK4
         dispatch_evolve_diffusion<NComp, CTEnabled, TimeRK4, EOSMethod>();
#else
         abort_disabled_build_feature("time integrator", "rk4");
#endif
      } else if (timestep_method_ == TimeSSPRK3) {
#if PYFLUID_AMREX_ENABLE_TIME_SSPRK3
         dispatch_evolve_diffusion<NComp, CTEnabled, TimeSSPRK3, EOSMethod>();
#else
         abort_disabled_build_feature("time integrator", "ssprk3");
#endif
      } else if (timestep_method_ == TimeMidpoint) {
#if PYFLUID_AMREX_ENABLE_TIME_MIDPOINT
         dispatch_evolve_diffusion<NComp, CTEnabled, TimeMidpoint, EOSMethod>();
#else
         abort_disabled_build_feature("time integrator", "midpoint");
#endif
      } else if (timestep_method_ == TimeCorrector) {
#if PYFLUID_AMREX_ENABLE_TIME_CORRECTOR
         dispatch_evolve_diffusion<NComp, CTEnabled, TimeCorrector, EOSMethod>();
#else
         abort_disabled_build_feature("time integrator", "corrector");
#endif
      } else {
#if PYFLUID_AMREX_ENABLE_TIME_EULER
         dispatch_evolve_diffusion<NComp, CTEnabled, TimeEuler, EOSMethod>();
#else
         abort_disabled_build_feature("time integrator", "euler");
#endif
      }
   }

   template <int NComp, bool CTEnabled, int TimeMethod, int EOSMethod>
   void dispatch_evolve_diffusion()
   {
      if (diffusion_method_ == DiffusionImplicit) {
#if PYFLUID_AMREX_ENABLE_DIFFUSION_IMPLICIT
         dispatch_evolve_subcycling<NComp, CTEnabled, TimeMethod, DiffusionImplicit, EOSMethod>();
#else
         abort_disabled_build_feature("diffusion method", "implicit");
#endif
      } else if (diffusion_method_ == DiffusionExplicit) {
#if PYFLUID_AMREX_ENABLE_DIFFUSION_EXPLICIT
         dispatch_evolve_subcycling<NComp, CTEnabled, TimeMethod, DiffusionExplicit, EOSMethod>();
#else
         abort_disabled_build_feature("diffusion method", "explicit");
#endif
      } else {
#if PYFLUID_AMREX_ENABLE_DIFFUSION_NONE
         dispatch_evolve_subcycling<NComp, CTEnabled, TimeMethod, DiffusionNone, EOSMethod>();
#else
         abort_disabled_build_feature("diffusion method", "none");
#endif
      }
   }

   template <int NComp, bool CTEnabled, int TimeMethod, int DiffusionMethod, int EOSMethod>
   void dispatch_evolve_subcycling()
   {
      if (cfg_.subcycling == "on" && maxLevel() > 0) {
#if PYFLUID_AMREX_ENABLE_SUBCYCLING_ON
         evolve_impl<NComp, CTEnabled, TimeMethod, DiffusionMethod, EOSMethod, true>();
#else
         abort_disabled_build_feature("subcycling", "on");
#endif
      } else {
#if PYFLUID_AMREX_ENABLE_SUBCYCLING_OFF
         evolve_impl<NComp, CTEnabled, TimeMethod, DiffusionMethod, EOSMethod, false>();
#else
         abort_disabled_build_feature("subcycling", "off");
#endif
      }
   }

   void abort_disabled_build_feature(char const* kind, char const* name) const
   {
      amrex::Abort(
         std::string("pyfluid.") + kind + "=" + name
         + " was selected, but that branch is disabled in this AMReX build"
      );
   }

   template <int NComp, bool CTEnabled, int TimeMethod, int DiffusionMethod, int EOSMethod, bool SubcyclingEnabled>
   void evolve_impl()
   {
      amrex::Real time = 0.0;
      amrex::Real dt = 0.0;
      if (cfg_.restart_file.empty()) {
         InitFromScratch(0.0);
         clip_multipatch_initial_hierarchy(0.0);
      } else {
         load_checkpoint(cfg_.restart_file, time, dt);
      }
      enforce_cell_budget("building initial AMR hierarchy");
      validate_active_hierarchy_for_evolution();
      define_flux_registers();
      enforce_solid_mask_nesting_from_fine();
      average_down_state();
         if constexpr (CTEnabled) {
            if (finestLevel() > 0) {
               average_down_face_b();
               sync_all_cell_b_from_faces(state_, false);
            }
         }
      reset_barotropic_energy_all_levels<NComp, EOSMethod>();
      fill_all_level_boundaries<SubcyclingEnabled>(state_, time);
      const int start_step = level_steps_[0];
      debug_check_state_all_levels<NComp, EOSMethod>("initial accepted state", start_step, time, true);

      std::ofstream scalar_file;
      std::ofstream amr_level_file;
      if (amrex::ParallelDescriptor::IOProcessor()) {
         std::filesystem::create_directories(cfg_.output_dir);
         write_run_parameter_file((std::filesystem::path(cfg_.output_dir) / "run.in").string());
         scalar_file.open(std::filesystem::path(cfg_.output_dir) / "scalars.tsv", std::ios::out | std::ios::trunc);
         write_scalar_header(scalar_file, cfg_, start_step, time);
         amr_level_file.open(std::filesystem::path(cfg_.output_dir) / "amr_levels.tsv", std::ios::out | std::ios::trunc);
         write_amr_level_header(amr_level_file, maxLevel(), cfg_, start_step, time);
      }

      if (cfg_.max_step < start_step) {
         amrex::Abort("pyfluid.max_step must be at least the checkpoint step when restarting");
      }
      amrex::Real last_scalar_wall = amrex::second();
      amrex::Real total_active_cell_updates = 0.0;
      amrex::Real total_layout_cell_updates = 0.0;
      amrex::Real last_scalar_active_cell_updates = 0.0;
      amrex::Real last_scalar_layout_cell_updates = 0.0;
      amrex::Real last_scalar_sim_time = time;
      const bool stream_scalars = cfg_.scalar_output_mode == "stream";
      bool wrote_scalars = false;
      cumulative_wall_work_ = 0.0;
      latest_wall_power_ = 0.0;
      conservation_ledger_ = {};
      if (cfg_.restart_file.empty()) {
         aligned_regrid_ticks_ = 0;
         aligned_base_regrid_tick_ = 0;
      }
      aligned_regrid_due_ = false;
      const TemporalCadence scalar_cadence{TemporalCadenceUnit::Steps, cfg_.scalar_int};
      const TemporalCadence plot_cadence{TemporalCadenceUnit::Steps, cfg_.plot_int};
      const TemporalCadence aux_cadence{TemporalCadenceUnit::Steps, cfg_.aux_gf_int};
      const TemporalCadence checkpoint_cadence{TemporalCadenceUnit::Steps, cfg_.checkpoint_int};

      auto initial = compute_current_scalars<EOSMethod>(start_step, time, dt, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, {});
      validate_scalar_health(initial);
      wall_work_reference_energy_ = initial.energy;
      conservation_ledger_.reference = {initial.mass, initial.energy};
      initial.energy_minus_wall_work = 0.0;
      initial.wall_work_rel_error = 0.0;
      initial.external_mass_residual = 0.0;
      initial.external_energy_residual = 0.0;
      initial.external_mass_rel_error = 0.0;
      initial.external_energy_rel_error = 0.0;
      RunConfig checkpoint_cfg = cfg_;
      if (!cfg_.checkpoint_dir.empty()) {
         checkpoint_cfg.output_dir = cfg_.checkpoint_dir;
      }
      const bool has_checkpoint_reference =
         std::isfinite(checkpoint_reference_mass_) && std::isfinite(checkpoint_reference_energy_);
      const amrex::Real mass0 = has_checkpoint_reference ? checkpoint_reference_mass_ : initial.mass;
      const amrex::Real energy0 = has_checkpoint_reference ? checkpoint_reference_energy_ : initial.energy;
      const amrex::Real gravity_total_energy0 = initial.gravity_total_energy;
      const amrex::Real passive_mass0 = initial.passive_mass;
      initial.mass_rel_drift = amrex::Math::abs(initial.mass - mass0) / amrex::max(amrex::Math::abs(mass0), 1.0);
      initial.energy_rel_drift = amrex::Math::abs(initial.energy - energy0) / amrex::max(amrex::Math::abs(energy0), 1.0);
      initial.gravity_total_energy_rel_drift = 0.0;
      initial.passive_mass_rel_drift = 0.0;
      if (amrex::ParallelDescriptor::IOProcessor()) {
         if (stream_scalars) {
            append_scalar_row(scalar_file, initial);
            append_current_amr_level_row(amr_level_file, start_step, time);
            wrote_scalars = true;
         }
      }
      if (plot_cadence.should_record_initial()) {
         write_current_state_plotfile(start_step, time);
      }
      if (should_refresh_initial_mhd_aux_fields()) {
         refresh_all_mhd_aux_fields_for_output();
      }
      if (aux_cadence.should_record_initial()) {
         write_current_aux_gfs(start_step, time);
      }
      if (checkpoint_cadence.should_record_initial()) {
         write_checkpoint(
            state_,
            level_steps_,
            level_times_,
            level_dts_,
            Geom(),
            refRatio(),
            finestLevel(),
            checkpoint_dir(),
            start_step,
            checkpoint_cfg,
            checkpoint_face_b(),
            mass0,
            energy0,
            aligned_regrid_ticks_,
            aligned_base_regrid_tick_,
            cfg_.checkpoint_compression
         );
      }

      const amrex::Real start_wall = amrex::second();
      const amrex::Real start_sim_time = time;
      int current_step = start_step;
      const bool has_final_time = cfg_.final_time > 0.0;
      for (int step = start_step + 1; step <= cfg_.max_step; ++step) {
         if (has_final_time && time >= cfg_.final_time) {
            break;
         }
         refresh_solid_geometry_all_levels(time, true);
         amrex::Real stage_start = profile_begin();
         if constexpr (SubcyclingEnabled) {
            dt = adaptive_base_dt_for_subcycling<NComp, EOSMethod, DiffusionMethod>();
         } else {
            dt = adaptive_dt_all_levels<NComp, EOSMethod, DiffusionMethod>();
         }
         profile_end(ProfileStage::Timestep, stage_start);
         if (has_final_time) {
            dt = amrex::min(dt, cfg_.final_time - time);
         }
         validate_timestep(dt, step, time);
         DiffusionStats diffusion;
         amrex::Real active_cell_updates_this_step = 0.0;
         amrex::Real layout_cell_updates_this_step = 0.0;
         if constexpr (SubcyclingEnabled) {
            diffusion = advance_subcycled_hierarchy<NComp, CTEnabled, TimeMethod, DiffusionMethod, EOSMethod>(
               dt,
               active_cell_updates_this_step,
               layout_cell_updates_this_step
            );
            time = level_times_[0];
            current_step = level_steps_[0];
            stage_start = profile_begin();
            const bool refreshed_solid =
               refresh_solid_geometry_synced(time, true, !has_mhd_components<NComp>() && !is_mhd(cfg_));
            if (refreshed_solid) {
               reset_barotropic_energy_all_levels<NComp, EOSMethod>();
            }
            profile_end(ProfileStage::AverageDown, stage_start);
         } else {
            const amrex::Long layout_cells_this_step = cell_count();
            const amrex::Long active_cells_this_step = active_cell_count();
            const bool next_state_needs_copy = advance_all_levels<NComp, TimeMethod, EOSMethod, !CTEnabled>(time, dt);
            if constexpr (CTEnabled) {
               stage_start = profile_begin();
               diffusion = combine_diffusion_stats(diffusion, advance_ct_all_levels<DiffusionMethod>(dt));
               profile_end(ProfileStage::CT, stage_start);
            }
            if (next_state_needs_copy) {
               stage_start = profile_begin();
               for (int lev = 0; lev <= finestLevel(); ++lev) {
                  copy_next_to_state_level<NComp>(lev);
               }
               reset_barotropic_energy_all_levels<NComp, EOSMethod>();
               profile_end(ProfileStage::Update, stage_start);
            } else {
               reset_barotropic_energy_all_levels<NComp, EOSMethod>();
            }
            if constexpr (DiffusionMethod != DiffusionNone) {
               RunConfig viscosity_cfg = viscosity_diffusion_config();
               if (viscosity_enabled(viscosity_cfg)) {
                  stage_start = profile_begin();
                  diffusion = combine_diffusion_stats(
                     diffusion,
                     apply_diffusion_all_levels<DiffusionMethod, NComp>(time + dt, dt, viscosity_cfg)
                  );
                  reset_barotropic_energy_all_levels<NComp, EOSMethod>();
                  profile_end(ProfileStage::Diffusion, stage_start);
               }
               if constexpr (CTEnabled) {
                  stage_start = profile_begin();
                  diffusion = combine_diffusion_stats(
                     diffusion,
                     advance_ct_inactive_magnetic_resistivity_all_levels<DiffusionMethod>(dt)
                  );
                  reset_barotropic_energy_all_levels<NComp, EOSMethod>();
                  profile_end(ProfileStage::Diffusion, stage_start);
               }
            }
            active_cell_updates_this_step = static_cast<amrex::Real>(active_cells_this_step);
            layout_cell_updates_this_step = static_cast<amrex::Real>(layout_cells_this_step);
            stage_start = profile_begin();
            average_down_state();
            if constexpr (CTEnabled) {
               if (finestLevel() > 0) {
                  average_down_face_b();
                  sync_all_cell_b_from_faces(state_, false);
               }
            }
            reset_barotropic_energy_all_levels<NComp, EOSMethod>();
            profile_end(ProfileStage::AverageDown, stage_start);
            time += dt;
            for (int lev = 0; lev <= finestLevel(); ++lev) {
               level_steps_[lev] = step;
               level_times_[lev] = time;
               level_old_times_[lev] = time;
               level_dts_[lev] = dt;
            }
            stage_start = profile_begin();
            const bool refreshed_solid =
               refresh_solid_geometry_synced(time, true, !has_mhd_components<NComp>() && !is_mhd(cfg_));
            if (refreshed_solid) {
               reset_barotropic_energy_all_levels<NComp, EOSMethod>();
            }
            profile_end(ProfileStage::AverageDown, stage_start);
            current_step = step;
         }
         total_active_cell_updates += active_cell_updates_this_step;
         total_layout_cell_updates += layout_cell_updates_this_step;
         const bool reached_final_time = has_final_time && time >= cfg_.final_time;
         const bool reached_run_end = (step == cfg_.max_step) || reached_final_time;
         const bool write_scalars = scalar_cadence.should_record(step, reached_run_end);
         const bool emit_scalars = stream_scalars ? write_scalars : reached_final_time;
         const bool write_aux_gfs = aux_cadence.should_record(step, reached_run_end);
         const bool will_regrid = should_regrid(step, reached_final_time);

         if (will_regrid) {
            const amrex::Real regrid_start = profile_begin();
            const bool hierarchy_changed = regrid_hierarchy<SubcyclingEnabled>(time, step);
            if (cfg_.regrid_mode == "aligned" && cfg_.subcycling == "on") {
               aligned_base_regrid_tick_ = aligned_regrid_ticks_;
            }
            if (hierarchy_changed) {
               reset_barotropic_energy_all_levels<NComp, EOSMethod>();
            }
            profile_end(ProfileStage::Regrid, regrid_start);
         }

         if (!will_regrid && needs_fresh_state_boundaries_after_step(write_scalars, write_aux_gfs)) {
            stage_start = profile_begin();
            fill_all_level_boundaries<SubcyclingEnabled>(state_, time);
            profile_end(ProfileStage::BoundaryFill, stage_start);
         }
         if constexpr (!has_mhd_components<NComp>()) {
            if (project_closed_hydro_solid_amr_mass(mass0)) {
               reset_barotropic_energy_all_levels<NComp, EOSMethod>();
            }
         }
         if (cfg_.debug_check_finite || accepted_physical_check_due(current_step, false)) {
            const amrex::Real check_start = profile_begin();
            debug_check_state_all_levels<NComp, EOSMethod>("post-step accepted state", current_step, time);
            profile_end(ProfileStage::Diagnostics, check_start);
         }
         if (should_refresh_mhd_aux_fields(step, write_aux_gfs)) {
            const amrex::Real aux_start = profile_begin();
            refresh_all_mhd_aux_fields_for_output();
            profile_end(ProfileStage::Diagnostics, aux_start);
         }
         if (emit_scalars) {
            const amrex::Real now = amrex::second();
            const amrex::Real elapsed = now - last_scalar_wall;
            const amrex::Real mcells_per_s = elapsed > 0.0
               ? (total_active_cell_updates - last_scalar_active_cell_updates) / elapsed / 1.0e6
               : 0.0;
            const amrex::Real layout_mcells_per_s = elapsed > 0.0
               ? (total_layout_cell_updates - last_scalar_layout_cell_updates) / elapsed / 1.0e6
               : 0.0;
            const amrex::Real sim_time_per_s = elapsed > 0.0
               ? (time - last_scalar_sim_time) / elapsed
               : 0.0;
            const amrex::Real diagnostics_start = profile_begin();
            const auto scalars = compute_current_scalars<EOSMethod>(
               step,
               time,
               dt,
               mass0,
               energy0,
               gravity_total_energy0,
               passive_mass0,
               mcells_per_s,
               layout_mcells_per_s,
               sim_time_per_s,
               diffusion
            );
            profile_end(ProfileStage::Diagnostics, diagnostics_start);
            validate_scalar_health(scalars);
            if (amrex::ParallelDescriptor::IOProcessor()) {
               append_scalar_row(scalar_file, scalars);
               append_current_amr_level_row(amr_level_file, step, time);
            }
            wrote_scalars = true;
            last_scalar_wall = now;
            last_scalar_active_cell_updates = total_active_cell_updates;
            last_scalar_layout_cell_updates = total_layout_cell_updates;
            last_scalar_sim_time = time;
         }

         if (plot_cadence.should_record(step, reached_run_end)) {
            const amrex::Real output_start = profile_begin();
            write_current_state_plotfile(step, time);
            profile_end(ProfileStage::Output, output_start);
         }
         if (write_aux_gfs) {
            const amrex::Real output_start = profile_begin();
            write_current_aux_gfs(step, time);
            profile_end(ProfileStage::Output, output_start);
         }
         if (checkpoint_cadence.should_record(step, reached_run_end)) {
            const amrex::Real output_start = profile_begin();
            write_checkpoint(
               state_,
               level_steps_,
               level_times_,
               level_dts_,
               Geom(),
               refRatio(),
               finestLevel(),
               checkpoint_dir(),
               step,
               checkpoint_cfg,
               checkpoint_face_b(),
               mass0,
               energy0,
               aligned_regrid_ticks_,
               aligned_base_regrid_tick_,
               cfg_.checkpoint_compression
            );
            profile_end(ProfileStage::Output, output_start);
         }
      }

      const amrex::Real elapsed_wall = amrex::second() - start_wall;
      amrex::Real max_elapsed_wall = elapsed_wall;
      amrex::ParallelDescriptor::ReduceRealMax(max_elapsed_wall, amrex::ParallelDescriptor::IOProcessorNumber());
      write_profile_file(max_elapsed_wall);
      write_profile_detail_file(max_elapsed_wall);
      if (amrex::ParallelDescriptor::IOProcessor()) {
         if (!stream_scalars) {
            const auto scalars = compute_current_scalars<EOSMethod>(
               current_step,
               time,
               dt,
               mass0,
               energy0,
               gravity_total_energy0,
               passive_mass0,
               0.0,
               0.0,
               0.0,
               {}
            );
            validate_scalar_health(scalars);
         if (!wrote_scalars) {
            append_scalar_row(scalar_file, scalars);
            append_current_amr_level_row(amr_level_file, current_step, time);
         }
      }
      scalar_file.close();
      amr_level_file.close();
      const int advanced_steps = current_step - start_step;
         const amrex::Real average_mcells_s = max_elapsed_wall > 0.0
            ? total_active_cell_updates / max_elapsed_wall / 1.0e6
            : 0.0;
         const amrex::Real average_layout_mcells_s = max_elapsed_wall > 0.0
            ? total_layout_cell_updates / max_elapsed_wall / 1.0e6
            : 0.0;
         const amrex::Real average_sim_time_s = max_elapsed_wall > 0.0
            ? (time - start_sim_time) / max_elapsed_wall
            : 0.0;
         amrex::Print() << "PyFluid AMReX " << cfg_.physics << " complete: dim=" << AMREX_SPACEDIM
                        << " eos=" << cfg_.eos
                        << " recon=" << cfg_.recon
                        << (!cfg_.recon_fine.empty() ? " recon_fine=" : "")
                        << (!cfg_.recon_fine.empty() ? cfg_.recon_fine : "")
                        << (!cfg_.recon_fine.empty() ? " recon_fine_start_level=" : "")
                        << (!cfg_.recon_fine.empty() ? std::to_string(cfg_.recon_fine_start_level) : "")
                        << " riemann=" << cfg_.riemann
                        << " timestep=" << cfg_.timestep
                        << " cfl=" << cfg_.cfl
                        << " bc=" << cfg_.bc
                        << " diffusion=" << cfg_.diffusion
                        << " viscosity=" << cfg_.viscosity
                        << " resistivity=" << cfg_.resistivity
                        << " source=" << cfg_.source
                        << " gravity=" << cfg_.gravity
                        << (gravity_enabled(cfg_) ? " gravity_coupling=" : "")
                        << (gravity_enabled(cfg_) ? cfg_.gravity_coupling : "")
                        << (cfg_.min_feature_length > 0.0 ? " amr_min_feature_length=" : "")
                        << (cfg_.min_feature_length > 0.0 ? std::to_string(cfg_.min_feature_length) : "")
                        << (is_mhd(cfg_) ? " ct=" : "")
                        << (is_mhd(cfg_) ? cfg_.ct : "")
                        << " cells=" << cell_count()
                        << " steps=" << advanced_steps
                        << " final_step=" << current_step
                        << " time=" << time
                        << " average=" << average_mcells_s << " Mcells/s"
                        << " layout_average=" << average_layout_mcells_s << " layout_Mcells/s"
                        << " sim_time_per_s=" << average_sim_time_s << "\n";
         amrex::Print() << "AMR regrid attempts=" << regrid_attempt_count_
                        << " noops=" << regrid_noop_count_
                        << " modified_levels=" << regrid_modified_level_count_ << "\n";
      }
   }

   RunConfig viscosity_diffusion_config() const
   {
      RunConfig diffusion_cfg = cfg_;
      diffusion_cfg.resistivity = 0.0;
      return diffusion_cfg;
   }
