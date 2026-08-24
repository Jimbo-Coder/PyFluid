// Included inside pyfluid_amrex::AmrSolver private section.
// Top-level branch dispatch and the main synchronized/subcycled evolution loop.

   using EvolutionTimestepFunction = amrex::Real (AmrSolver::*)();
   using SingleRateAdvanceFunction = bool (AmrSolver::*)(amrex::Real, amrex::Real);
   using CtAdvanceFunction = DiffusionStats (AmrSolver::*)(amrex::Real);
   using DiffusionAdvanceFunction = DiffusionStats (
      AmrSolver::*)(amrex::Real, amrex::Real, RunConfig const&
   );
   using SubcycledAdvanceFunction = DiffusionStats (
      AmrSolver::*)(amrex::Real, amrex::Real&, amrex::Real&
   );
   using EvolutionBoundaryFillFunction = void (
      AmrSolver::*)(LevelData&, amrex::Real
   );

   struct EvolutionOperations {
      bool subcycling = false;
      EvolutionTimestepFunction compute_timestep = nullptr;
      SingleRateAdvanceFunction advance_single_rate = nullptr;
      CtAdvanceFunction advance_ct = nullptr;
      DiffusionAdvanceFunction advance_diffusion = nullptr;
      CtAdvanceFunction advance_inactive_ct_resistivity = nullptr;
      SubcycledAdvanceFunction advance_subcycled = nullptr;
      EvolutionBoundaryFillFunction fill_boundaries = nullptr;
   };

   template <int NComp, bool CTEnabled>
   void dispatch_evolve_eos()
   {
      dispatch_evolve_timestep<NComp, CTEnabled, EOSRuntime>();
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
         evolve_impl<NComp, CTEnabled, EOSMethod>(
            make_evolution_operations<NComp, CTEnabled, TimeMethod, DiffusionMethod, EOSMethod, true>()
         );
#else
         abort_disabled_build_feature("subcycling", "on");
#endif
      } else {
#if PYFLUID_AMREX_ENABLE_SUBCYCLING_OFF
         evolve_impl<NComp, CTEnabled, EOSMethod>(
            make_evolution_operations<NComp, CTEnabled, TimeMethod, DiffusionMethod, EOSMethod, false>()
         );
#else
         abort_disabled_build_feature("subcycling", "off");
#endif
      }
   }

   template <int NComp, int EOSMethod, int DiffusionMethod, bool SubcyclingEnabled>
   amrex::Real compute_evolution_timestep()
   {
      if constexpr (SubcyclingEnabled) {
         return adaptive_base_dt_for_subcycling<NComp, EOSMethod, DiffusionMethod>();
      }
      return adaptive_dt_all_levels<NComp, EOSMethod, DiffusionMethod>();
   }

   template <bool SubcyclingEnabled, int EOSMethod>
   bool regrid_hierarchy_and_project(
      amrex::Real time,
      int step,
      int lbase
   )
   {
      int effective_lbase = std::max(lbase, 0);
      if (cfg_.initial_refine_fixed_levels > 0) {
         effective_lbase = std::max(
            effective_lbase,
            std::min(cfg_.initial_refine_fixed_levels, finestLevel())
         );
      }
      const bool hierarchy_changed = regrid_hierarchy<SubcyclingEnabled>(
         time, step, effective_lbase
      );
      if (!hierarchy_changed) {
         return false;
      }
      if (spacetime_state_.active()) {
         prepare_spacetime_coupling_range(
            spacetime_state_.state(), effective_lbase, finestLevel() + 1,
            time, -1.0, 0
         );
      }
      project_relativistic_atmosphere_range<EOSMethod>(
         fluid_state_.state(), effective_lbase, finestLevel() + 1
      );
      // The base regrid owner seeded every new/remade fine old-time state
      // before this final physical projection. Keep those equal-time copies
      // consistent with the state that evolution and output will consume.
      for (int lev = effective_lbase + 1; lev <= finestLevel(); ++lev) {
         copy_level_state_to_old(lev);
      }
      return true;
   }

   template <
      int NComp,
      bool CTEnabled,
      int TimeMethod,
      int DiffusionMethod,
      int EOSMethod,
      bool SubcyclingEnabled
   >
   EvolutionOperations make_evolution_operations()
   {
      EvolutionOperations operations;
      operations.subcycling = SubcyclingEnabled;
      operations.compute_timestep =
         &AmrSolver::compute_evolution_timestep<NComp, EOSMethod, DiffusionMethod, SubcyclingEnabled>;
      operations.fill_boundaries = &AmrSolver::fill_all_level_boundaries<SubcyclingEnabled>;
      if constexpr (SubcyclingEnabled) {
         operations.advance_subcycled = &AmrSolver::advance_subcycled_hierarchy<
            NComp, CTEnabled, TimeMethod, DiffusionMethod, EOSMethod
         >;
      } else {
         operations.advance_single_rate = &AmrSolver::advance_all_levels<
            NComp, TimeMethod, EOSMethod, !CTEnabled
         >;
         if constexpr (CTEnabled) {
            operations.advance_ct = &AmrSolver::advance_ct_all_levels<DiffusionMethod>;
         }
         if constexpr (DiffusionMethod != DiffusionNone) {
            operations.advance_diffusion =
               &AmrSolver::apply_diffusion_all_levels<DiffusionMethod, NComp>;
            if constexpr (CTEnabled) {
               operations.advance_inactive_ct_resistivity =
                  &AmrSolver::advance_ct_inactive_magnetic_resistivity_all_levels<DiffusionMethod>;
            }
         }
      }
      return operations;
   }

   void abort_disabled_build_feature(char const* kind, char const* name) const
   {
      amrex::Abort(
         std::string("pyfluid.") + kind + "=" + name
         + " was selected, but that branch is disabled in this AMReX build"
      );
   }

   template <int NComp, bool CTEnabled, int EOSMethod>
#if defined(__GNUC__) || defined(__clang__)
   __attribute__((noinline))
#endif
   void evolve_impl(EvolutionOperations const& operations)
   {
      amrex::Real time = 0.0;
      amrex::Real dt = 0.0;
      if (cfg_.restart_file.empty()) {
         InitFromScratch(0.0);
         clip_multipatch_initial_hierarchy(0.0);
         materialize_configured_initial_data();
      } else {
         load_checkpoint(cfg_.restart_file, time, dt);
      }
      enforce_cell_budget("building initial AMR hierarchy");
      if (!cfg_.initial_data_solve_only) {
         validate_active_hierarchy_for_evolution();
         define_flux_registers();
      }
      enforce_solid_mask_nesting_from_fine();
      average_down_state();
      average_down_spacetime();
      if constexpr (CTEnabled) {
         if (finestLevel() > 0) {
            average_down_face_b();
            sync_all_cell_b_from_faces(fluid_state_.state(), false);
         }
      }
      reset_barotropic_energy_all_levels<NComp, EOSMethod>();
      (this->*operations.fill_boundaries)(fluid_state_.state(), time);
      fill_all_spacetime_boundaries(spacetime_state_.state(), time);
      const int start_step = level_steps_[0];
      run_evolution_preflight(time);
      debug_check_state_all_levels("initial accepted state", start_step, time, true);
      if (cfg_.initial_data_solve_only) {
         if (amrex::ParallelDescriptor::IOProcessor()) {
            std::filesystem::create_directories(cfg_.output_dir);
            write_run_parameter_file(
               (std::filesystem::path(cfg_.output_dir) / "run.in").string()
            );
         }
         write_physical_initial_data_artifact();
         report_relativistic_recovery_events(start_step, time);
         amrex::Print() << "completed initial-data-only solve and artifact write\n";
         print_relativistic_recovery_summary();
         return;
      }
      if (!cfg_.initial_data_output_file.empty()) {
         write_physical_initial_data_artifact();
      }
      if (cfg_.restart_file.empty()
          && cfg_.initial_data_perturbation != "none") {
         // Reusable physical-ID artifacts contain the equilibrium.  The
         // perturbation is an evolution-start modifier, so apply it only
         // after any requested artifact write and validate the modified state
         // independently.  Checkpoint restart already contains the live
         // perturbed state and must not apply it a second time.
         apply_configured_initial_data_perturbation();
         average_down_state();
         reset_barotropic_energy_all_levels<NComp, EOSMethod>();
         (this->*operations.fill_boundaries)(fluid_state_.state(), time);
         run_evolution_preflight(time);
         debug_check_state_all_levels(
            "perturbed initial accepted state", start_step, time, true
         );
      }

      std::ofstream scalar_file;
      std::ofstream amr_level_file;
      std::ofstream multipole_file;
      const std::filesystem::path scalar_path = std::filesystem::path(cfg_.output_dir) / "scalars.tsv";
      const std::filesystem::path amr_level_path = std::filesystem::path(cfg_.output_dir) / "amr_levels.tsv";
      const std::filesystem::path multipole_path = std::filesystem::path(cfg_.output_dir) / "multipoles.tsv";
      const auto history_last_step = [](std::filesystem::path const& path) -> std::optional<int> {
         if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) == 0) {
            return std::nullopt;
         }
         std::ifstream stream(path);
         std::string line;
         std::optional<int> last_step;
         while (std::getline(stream, line)) {
            if (line.empty() || line.front() == '#') {
               continue;
            }
            std::istringstream row(line);
            int row_step = -1;
            if (row >> row_step) {
               last_step = row_step;
            }
         }
         return last_step;
      };
      const std::optional<int> scalar_history_step = history_last_step(scalar_path);
      const std::optional<int> amr_history_step = history_last_step(amr_level_path);
      const bool radiation_enabled = cfg_.diagnostic_multipole_l_max >= 2;
      const bool azimuthal_modes_enabled = !azimuthal_mode_field_names(cfg_).empty();
      const bool matter_modes_enabled = cfg_.diagnostic_matter_mode_m_max >= 0;
      const bool multipoles_enabled = radiation_enabled
         || azimuthal_modes_enabled || matter_modes_enabled;
      const std::optional<int> multipole_history_step = multipoles_enabled
         ? history_last_step(multipole_path)
         : std::nullopt;
      if (!cfg_.restart_file.empty() && scalar_history_step.has_value() != amr_history_step.has_value()) {
         amrex::Abort("restart output contains incomplete scalar/AMR history");
      }
      if (!cfg_.restart_file.empty() && scalar_history_step.has_value()
          && *scalar_history_step != *amr_history_step) {
         amrex::Abort("restart scalar and AMR histories end at different steps");
      }
      if (!cfg_.restart_file.empty() && scalar_history_step.has_value()
          && *scalar_history_step > start_step) {
         amrex::Abort("restart output history extends beyond the checkpoint step");
      }
      if (!cfg_.restart_file.empty() && multipoles_enabled
          && multipole_history_step.has_value()
          && *multipole_history_step > start_step) {
         amrex::Abort("restart multipole history extends beyond the checkpoint step");
      }
      if (!cfg_.restart_file.empty() && multipoles_enabled
          && multipole_history_step.has_value()
          && scalar_history_step.has_value()
          && *multipole_history_step != *scalar_history_step) {
         amrex::Abort("restart scalar and multipole histories end at different steps");
      }
      const bool append_restart_history =
         !cfg_.restart_file.empty() && scalar_history_step.has_value() && amr_history_step.has_value();
      const bool append_restart_initial =
         append_restart_history && *scalar_history_step < start_step;
      const bool append_multipole_history =
         multipoles_enabled && !cfg_.restart_file.empty()
         && multipole_history_step.has_value();
      const bool append_multipole_initial =
         append_multipole_history && *multipole_history_step < start_step;
      ScalarDiagnosticSelection scalar_selection =
         scalar_diagnostic_selection(cfg_.scalar_diagnostics);
      if (radiation_enabled) {
         scalar_selection.enable_group(ScalarDiagnosticGroup::Radiation);
      }
      if (matter_modes_enabled) {
         scalar_selection.enable_group(ScalarDiagnosticGroup::Modes);
      }
      apply_scalar_diagnostic_requirements(scalar_selection, cfg_);
      if (amrex::ParallelDescriptor::IOProcessor()) {
         std::filesystem::create_directories(cfg_.output_dir);
         write_run_parameter_file((std::filesystem::path(cfg_.output_dir) / "run.in").string());
         const std::ios::openmode history_mode = std::ios::out
            | (append_restart_history ? std::ios::app : std::ios::trunc);
         scalar_file.open(scalar_path, history_mode);
         amr_level_file.open(amr_level_path, history_mode);
         if (!append_restart_history) {
            write_scalar_header(scalar_file, cfg_, start_step, time, scalar_selection);
            write_amr_level_header(amr_level_file, maxLevel(), cfg_, start_step, time);
         }
         if (multipoles_enabled) {
            const std::ios::openmode multipole_mode = std::ios::out
               | (append_multipole_history ? std::ios::app : std::ios::trunc);
            multipole_file.open(multipole_path, multipole_mode);
            if (!append_multipole_history) {
               write_multipole_header(multipole_file, start_step, time);
            }
         }
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

      auto compute_and_append_multipoles = [&](
         ScalarDiagnostics& scalars,
         int diagnostic_step,
         amrex::Real diagnostic_time,
         bool emit_record
      ) {
         if (!emit_record || !multipoles_enabled) {
            return;
         }
         std::vector<MultipoleMode> modes;
         if (radiation_enabled) {
            RadiationDiagnostics radiation =
               compute_spacetime_radiation_diagnostics(diagnostic_time);
            scalars.extraction_radius = radiation.radius;
            scalars.matter_energy_flux = radiation.matter_energy_flux;
            scalars.psi4_l2 = radiation.psi4_l2;
            modes.insert(
               modes.end(),
               std::make_move_iterator(radiation.psi4_modes.begin()),
               std::make_move_iterator(radiation.psi4_modes.end())
            );
         }
         if (azimuthal_modes_enabled) {
            MultipoleDiagnostics azimuthal =
               compute_azimuthal_surface_modes(diagnostic_time);
            modes.insert(
               modes.end(),
               std::make_move_iterator(azimuthal.modes.begin()),
               std::make_move_iterator(azimuthal.modes.end())
            );
         }
         if (matter_modes_enabled) {
            MatterModeDiagnostics matter = compute_matter_volume_modes(
               diagnostic_time
            );
            scalars.matter_mode_mass = matter.mass;
            scalars.matter_barycenter1 = matter.barycenter[0];
            scalars.matter_barycenter2 = matter.barycenter[1];
            scalars.matter_barycenter3 = matter.barycenter[2];
            modes.insert(
               modes.end(),
               std::make_move_iterator(matter.modes.begin()),
               std::make_move_iterator(matter.modes.end())
            );
         }
         if (amrex::ParallelDescriptor::IOProcessor()) {
            append_multipole_rows(
               multipole_file,
               diagnostic_step,
               diagnostic_time,
               modes
            );
         }
      };

      update_apparent_horizon(start_step, time);
      auto initial = compute_current_scalars(
         start_step, time, dt, 1.0, 1.0, 1.0, 1.0,
         std::numeric_limits<amrex::Real>::quiet_NaN(), 0.0, 0.0, 0.0, {},
         EOSMethod, scalar_selection
      );
      const bool emit_initial_history = stream_scalars
         && (!append_restart_history || append_restart_initial);
      const bool emit_initial_multipoles = stream_scalars
         && (!append_multipole_history || append_multipole_initial);
      compute_and_append_multipoles(
         initial, start_step, time, emit_initial_multipoles
      );
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
      std::vector<CheckpointFieldSet> checkpoint_extra_fields;
      if (spacetime_state_.active()) {
         checkpoint_extra_fields.push_back(
            {spacetime_state_.field_set().name, &spacetime_state_.state()}
         );
      }
      const bool has_checkpoint_reference =
         std::isfinite(checkpoint_reference_mass_) && std::isfinite(checkpoint_reference_energy_);
      const amrex::Real mass0 = has_checkpoint_reference ? checkpoint_reference_mass_ : initial.mass;
      const amrex::Real energy0 = has_checkpoint_reference ? checkpoint_reference_energy_ : initial.energy;
      const amrex::Real gravity_total_energy0 = initial.gravity_total_energy;
      const amrex::Real passive_mass0 = initial.passive_mass;
      const amrex::Real max_rho0 = std::isfinite(checkpoint_reference_max_rho_)
         ? checkpoint_reference_max_rho_
         : initial.max_rho;
      initial.max_rho_rel_change = (initial.max_rho - max_rho0) / max_rho0;
      initial.mass_rel_drift = amrex::Math::abs(initial.mass - mass0) / amrex::max(amrex::Math::abs(mass0), 1.0);
      initial.energy_rel_drift = amrex::Math::abs(initial.energy - energy0) / amrex::max(amrex::Math::abs(energy0), 1.0);
      initial.gravity_total_energy_rel_drift = 0.0;
      initial.passive_mass_rel_drift = 0.0;
      if (amrex::ParallelDescriptor::IOProcessor()) {
         if (emit_initial_history) {
            append_scalar_row(scalar_file, initial, scalar_selection);
            append_current_amr_level_row(amr_level_file, start_step, time);
            wrote_scalars = true;
         }
      }
      if (plot_cadence.should_record_initial()) {
         write_current_state_plotfile(start_step, time);
      }
      const bool write_initial_aux_gfs = aux_cadence.should_record_initial();
      if (should_refresh_initial_mhd_aux_fields(write_initial_aux_gfs)) {
         refresh_all_mhd_aux_fields_for_output();
      }
      if (write_initial_aux_gfs) {
         write_current_aux_gfs(start_step, time);
      }
      if (checkpoint_cadence.should_record_initial()) {
         write_checkpoint(
            fluid_state_.state(),
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
            checkpoint_extra_fields,
            mass0,
            energy0,
            max_rho0,
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
         reset_relativistic_recovery_events();
         amrex::Real stage_start = profile_.begin();
         dt = (this->*operations.compute_timestep)();
         profile_.end(ProfileStage::Timestep, stage_start);
         if (has_final_time) {
            dt = amrex::min(dt, cfg_.final_time - time);
         }
         validate_timestep(dt, step, time);
         DiffusionStats diffusion;
         amrex::Real active_cell_updates_this_step = 0.0;
         amrex::Real layout_cell_updates_this_step = 0.0;
         if (operations.subcycling) {
            diffusion = (this->*operations.advance_subcycled)(
               dt,
               active_cell_updates_this_step,
               layout_cell_updates_this_step
            );
            time = level_times_[0];
            current_step = level_steps_[0];
            stage_start = profile_.begin();
            const bool refreshed_solid =
               refresh_solid_geometry_synced(time, true, !has_mhd_components<NComp>() && !is_mhd(cfg_));
            if (refreshed_solid) {
               reset_barotropic_energy_all_levels<NComp, EOSMethod>();
            }
            profile_.end(ProfileStage::AverageDown, stage_start);
         } else {
            const amrex::Long layout_cells_this_step = cell_count();
            const amrex::Long active_cells_this_step = active_cell_count();
            if constexpr (CTEnabled) {
               begin_ct_step_all_levels();
            }
            const bool next_state_needs_copy =
               (this->*operations.advance_single_rate)(time, dt);
            if constexpr (CTEnabled) {
               stage_start = profile_.begin();
               diffusion = combine_diffusion_stats(
                  diffusion,
                  (this->*operations.advance_ct)(dt)
               );
               profile_.end(ProfileStage::CT, stage_start);
            }
            if (next_state_needs_copy) {
               stage_start = profile_.begin();
               for (int lev = 0; lev <= finestLevel(); ++lev) {
                  copy_next_to_state_level<NComp>(lev);
               }
               reset_barotropic_energy_all_levels<NComp, EOSMethod>();
               profile_.end(ProfileStage::Update, stage_start);
            } else {
               reset_barotropic_energy_all_levels<NComp, EOSMethod>();
            }
            if (operations.advance_diffusion != nullptr) {
               RunConfig viscosity_cfg = viscosity_diffusion_config();
               if (viscosity_enabled(viscosity_cfg)) {
                  stage_start = profile_.begin();
                  diffusion = combine_diffusion_stats(
                     diffusion,
                     (this->*operations.advance_diffusion)(time + dt, dt, viscosity_cfg)
                  );
                  reset_barotropic_energy_all_levels<NComp, EOSMethod>();
                  profile_.end(ProfileStage::Diffusion, stage_start);
               }
               if constexpr (CTEnabled) {
                  stage_start = profile_.begin();
                  diffusion = combine_diffusion_stats(
                     diffusion,
                     (this->*operations.advance_inactive_ct_resistivity)(dt)
                  );
                  reset_barotropic_energy_all_levels<NComp, EOSMethod>();
                  profile_.end(ProfileStage::Diffusion, stage_start);
               }
            }
            active_cell_updates_this_step = static_cast<amrex::Real>(active_cells_this_step);
            layout_cell_updates_this_step = static_cast<amrex::Real>(layout_cells_this_step);
            stage_start = profile_.begin();
            average_down_state();
            average_down_spacetime();
            if constexpr (CTEnabled) {
               if (finestLevel() > 0) {
                  average_down_face_b();
                  sync_all_cell_b_from_faces(fluid_state_.state(), false);
               }
            }
            if (finestLevel() > 0 || CTEnabled) {
               if (spacetime_state_.evolves()) {
                  prepare_spacetime_coupling_range(
                     spacetime_state_.state(), 0, finestLevel() + 1,
                     time + dt, -1.0, 0
                  );
               }
               // CT owns the accepted magnetic field. Revalidate the final
               // conserved state only after its face field has been synced
               // back to cell centers; atmosphere projection preserves B and
               // records any last-resort reset.
               project_relativistic_atmosphere_range<EOSMethod>(
                  fluid_state_.state(), 0, finestLevel() + 1
               );
            }
            reset_barotropic_energy_all_levels<NComp, EOSMethod>();
            profile_.end(ProfileStage::AverageDown, stage_start);
            time += dt;
            for (int lev = 0; lev <= finestLevel(); ++lev) {
               level_steps_[lev] = step;
               level_times_[lev] = time;
               level_old_times_[lev] = time;
               level_dts_[lev] = dt;
            }
            stage_start = profile_.begin();
            const bool refreshed_solid =
               refresh_solid_geometry_synced(time, true, !has_mhd_components<NComp>() && !is_mhd(cfg_));
            if (refreshed_solid) {
               reset_barotropic_energy_all_levels<NComp, EOSMethod>();
            }
            profile_.end(ProfileStage::AverageDown, stage_start);
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
            const amrex::Real regrid_start = profile_.begin();
            const bool hierarchy_changed = operations.subcycling
               ? regrid_hierarchy_and_project<true, EOSMethod>(time, step, 0)
               : regrid_hierarchy_and_project<false, EOSMethod>(time, step, 0);
            if (cfg_.regrid_mode == "aligned" && cfg_.subcycling == "on") {
               aligned_base_regrid_tick_ = aligned_regrid_ticks_;
            }
            if (hierarchy_changed) {
               reset_barotropic_energy_all_levels<NComp, EOSMethod>();
            }
            profile_.end(ProfileStage::Regrid, regrid_start);
         }

         update_apparent_horizon(step, time);

         if (!will_regrid && needs_fresh_state_boundaries_after_step(write_scalars, write_aux_gfs)) {
            stage_start = profile_.begin();
            (this->*operations.fill_boundaries)(fluid_state_.state(), time);
            profile_.end(ProfileStage::BoundaryFill, stage_start);
         }
         if constexpr (!has_mhd_components<NComp>()) {
            if (project_closed_hydro_solid_amr_mass(mass0)) {
               reset_barotropic_energy_all_levels<NComp, EOSMethod>();
            }
         }
         report_relativistic_recovery_events(current_step, time);
         if (cfg_.debug_check_finite || accepted_physical_check_due(current_step, false)) {
            const amrex::Real check_start = profile_.begin();
            debug_check_state_all_levels("post-step accepted state", current_step, time);
            profile_.end(ProfileStage::Diagnostics, check_start);
         }
         if (should_refresh_mhd_aux_fields(step, write_aux_gfs)) {
            const amrex::Real aux_start = profile_.begin();
            refresh_all_mhd_aux_fields_for_output();
            profile_.end(ProfileStage::Diagnostics, aux_start);
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
            const amrex::Real diagnostics_start = profile_.begin();
            auto scalars = compute_current_scalars(
               step,
               time,
               dt,
               mass0,
               energy0,
               gravity_total_energy0,
               passive_mass0,
               max_rho0,
               mcells_per_s,
               layout_mcells_per_s,
               sim_time_per_s,
               diffusion,
               EOSMethod,
               scalar_selection
            );
            compute_and_append_multipoles(
               scalars, step, time, multipoles_enabled
            );
            profile_.end(ProfileStage::Diagnostics, diagnostics_start);
            validate_scalar_health(scalars);
            if (amrex::ParallelDescriptor::IOProcessor()) {
               append_scalar_row(scalar_file, scalars, scalar_selection);
               append_current_amr_level_row(amr_level_file, step, time);
            }
            wrote_scalars = true;
            last_scalar_wall = now;
            last_scalar_active_cell_updates = total_active_cell_updates;
            last_scalar_layout_cell_updates = total_layout_cell_updates;
            last_scalar_sim_time = time;
         }

         if (plot_cadence.should_record(step, reached_run_end)) {
            const amrex::Real output_start = profile_.begin();
            write_current_state_plotfile(step, time);
            profile_.end(ProfileStage::Output, output_start);
         }
         if (write_aux_gfs) {
            const amrex::Real output_start = profile_.begin();
            write_current_aux_gfs(step, time);
            profile_.end(ProfileStage::Output, output_start);
         }
         if (checkpoint_cadence.should_record(step, reached_run_end)) {
            const amrex::Real output_start = profile_.begin();
            write_checkpoint(
               fluid_state_.state(),
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
               checkpoint_extra_fields,
               mass0,
               energy0,
               max_rho0,
               aligned_regrid_ticks_,
               aligned_base_regrid_tick_,
               cfg_.checkpoint_compression
            );
            profile_.end(ProfileStage::Output, output_start);
         }
      }

      const amrex::Real elapsed_wall = amrex::second() - start_wall;
      amrex::Real max_elapsed_wall = elapsed_wall;
      amrex::ParallelDescriptor::ReduceRealMax(max_elapsed_wall, amrex::ParallelDescriptor::IOProcessorNumber());
      const std::filesystem::path output_dir(cfg_.output_dir);
      profile_.write(output_dir, max_elapsed_wall);
      profile_.write_detail(output_dir, max_elapsed_wall);
      if (!stream_scalars && !wrote_scalars) {
         auto scalars = compute_current_scalars(
            current_step,
            time,
            dt,
            mass0,
            energy0,
            gravity_total_energy0,
            passive_mass0,
            max_rho0,
            0.0,
            0.0,
            0.0,
            {},
            EOSMethod,
            scalar_selection
         );
         compute_and_append_multipoles(
            scalars, current_step, time, multipoles_enabled
         );
         validate_scalar_health(scalars);
         if (amrex::ParallelDescriptor::IOProcessor()) {
            append_scalar_row(scalar_file, scalars, scalar_selection);
            append_current_amr_level_row(amr_level_file, current_step, time);
         }
         wrote_scalars = true;
      }
      if (amrex::ParallelDescriptor::IOProcessor()) {
         scalar_file.close();
         amr_level_file.close();
         multipole_file.close();
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
         print_relativistic_recovery_summary();
      }
   }

   RunConfig viscosity_diffusion_config() const
   {
      RunConfig diffusion_cfg = cfg_;
      diffusion_cfg.resistivity = 0.0;
      return diffusion_cfg;
   }
