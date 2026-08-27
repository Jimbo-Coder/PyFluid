// Included inside pyfluid_amrex::AmrSolver private section.
// RHS/flux-stage assembly for conservative updates.

   int rmhd_candidate_riemann_method() const
   {
      if (cfg_.rmhd_candidate_riemann == "llf") {
         return RiemannLLF;
      }
      if (cfg_.rmhd_candidate_riemann == "hll") {
         return RiemannHLL;
      }
      return riemann_method_;
   }

   template <int EOSMethod>
   void project_relativistic_atmosphere_range(
      LevelData& state,
      int lev_begin,
      int lev_end,
      int ghost_cells = 0
   )
   {
      if (!physical_system_is_relativistic(physics_system_)) {
         return;
      }
      const EOSParams eos = eos_params(cfg_);
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
         spacetime_state_.active(),
         "relativistic atmosphere projection requires canonical spacetime fields"
      );
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         apply_relativistic_atmosphere(
            *state[lev],
            *spacetime_coupling_.cell_spacetime()[lev],
            physics_system_,
            eos,
            recovery_atmosphere_,
            ghost_cells,
            step_relativistic_c2p_fallback_count_.dataPtr(),
            step_relativistic_atmosphere_reset_count_.dataPtr()
         );
      }
   }

   void reset_relativistic_recovery_events()
   {
      if (!physical_system_is_relativistic(physics_system_)) {
         return;
      }
      const amrex::Long zero = 0;
#ifdef AMREX_USE_GPU
      amrex::Gpu::htod_memcpy(
         step_relativistic_c2p_fallback_count_.dataPtr(),
         &zero,
         sizeof(zero)
      );
      amrex::Gpu::htod_memcpy(
         step_relativistic_atmosphere_reset_count_.dataPtr(),
         &zero,
         sizeof(zero)
      );
      amrex::Gpu::htod_memcpy(
         step_rmhd_pwl_correction_count_.dataPtr(),
         &zero,
         sizeof(zero)
      );
      amrex::Gpu::htod_memcpy(
         step_rmhd_pwc_correction_count_.dataPtr(),
         &zero,
         sizeof(zero)
      );
      amrex::Gpu::htod_memcpy(
         step_rmhd_riemann_hll_fallback_count_.dataPtr(),
         &zero,
         sizeof(zero)
      );
#else
      *step_relativistic_c2p_fallback_count_.dataPtr() = zero;
      *step_relativistic_atmosphere_reset_count_.dataPtr() = zero;
      *step_rmhd_pwl_correction_count_.dataPtr() = zero;
      *step_rmhd_pwc_correction_count_.dataPtr() = zero;
      *step_rmhd_riemann_hll_fallback_count_.dataPtr() = zero;
#endif
   }

   void report_relativistic_recovery_events(int step, amrex::Real time)
   {
      if (!physical_system_is_relativistic(physics_system_)) {
         return;
      }
      amrex::Long recovery_counts[] = {
         step_relativistic_c2p_fallback_count_.dataValue(),
         step_relativistic_atmosphere_reset_count_.dataValue(),
         step_rmhd_pwl_correction_count_.dataValue(),
         step_rmhd_pwc_correction_count_.dataValue(),
         step_rmhd_riemann_hll_fallback_count_.dataValue(),
      };
      constexpr int num_recovery_counts = static_cast<int>(
         sizeof(recovery_counts) / sizeof(recovery_counts[0])
      );
      amrex::ParallelDescriptor::ReduceLongSum(
         recovery_counts, num_recovery_counts
      );
      const amrex::Long c2p_fallback_count = recovery_counts[0];
      const amrex::Long atmosphere_reset_count = recovery_counts[1];
      const amrex::Long rmhd_pwl_correction_count = recovery_counts[2];
      const amrex::Long rmhd_pwc_correction_count = recovery_counts[3];
      const amrex::Long rmhd_riemann_hll_fallback_count = recovery_counts[4];
      relativistic_c2p_fallback_count_ += c2p_fallback_count;
      relativistic_atmosphere_reset_count_ += atmosphere_reset_count;
      rmhd_pwl_correction_count_ += rmhd_pwl_correction_count;
      rmhd_pwc_correction_count_ += rmhd_pwc_correction_count;
      rmhd_riemann_hll_fallback_count_ += rmhd_riemann_hll_fallback_count;
      if (
         c2p_fallback_count != 0 || atmosphere_reset_count != 0
         || rmhd_pwl_correction_count != 0 || rmhd_pwc_correction_count != 0
         || rmhd_riemann_hll_fallback_count != 0
      ) {
         const bool used_recovery_or_reset =
            c2p_fallback_count != 0 || atmosphere_reset_count != 0;
         amrex::Print()
            << (used_recovery_or_reset
                  ? "WARNING: relativistic recovery repair"
                  : "INFO: relativistic conservative correction")
            << " at step=" << step
            << " time=" << time
            << " c2p_fallbacks=" << c2p_fallback_count
            << " atmosphere_resets=" << atmosphere_reset_count
            << " rmhd_pwl_corrections=" << rmhd_pwl_correction_count
            << " rmhd_pwc_corrections=" << rmhd_pwc_correction_count
            << " rmhd_riemann_hll_fallbacks="
            << rmhd_riemann_hll_fallback_count
            << " cumulative_c2p_fallbacks="
            << relativistic_c2p_fallback_count_
            << " cumulative_atmosphere_resets="
            << relativistic_atmosphere_reset_count_
            << " cumulative_rmhd_pwl_corrections="
            << rmhd_pwl_correction_count_
            << " cumulative_rmhd_pwc_corrections="
            << rmhd_pwc_correction_count_
            << " cumulative_rmhd_riemann_hll_fallbacks="
            << rmhd_riemann_hll_fallback_count_ << "\n";
      }
   }

   void print_relativistic_recovery_summary() const
   {
      if (!physical_system_is_relativistic(physics_system_)) {
         return;
      }
      amrex::Print()
         << "Relativistic recovery c2p_fallbacks="
         << relativistic_c2p_fallback_count_
         << " atmosphere_resets="
         << relativistic_atmosphere_reset_count_
         << " rmhd_pwl_corrections="
         << rmhd_pwl_correction_count_
         << " rmhd_pwc_corrections="
         << rmhd_pwc_correction_count_
         << " rmhd_riemann_hll_fallbacks="
         << rmhd_riemann_hll_fallback_count_ << "\n";
   }

   template <int NComp, int EOSMethod>
   void fill_stage_primitive_cache(
      amrex::MultiFab& primitive,
      amrex::MultiFab& state,
      int lev,
      bool use_solid_mask,
      amrex::Real time
   )
   {
      const bool relativistic = physical_system_is_relativistic(physics_system_);
      const bool analytic_fixed_spacetime =
         relativistic && uses_analytic_fixed_spacetime_kernels();
      AMREX_ALWAYS_ASSERT(!relativistic || spacetime_state_.active());
      fill_primitive_cache(PrimitiveCacheLaunch{
         primitive,
         state,
         eos_params(cfg_),
         physics_system_,
         NComp,
         relativistic && !analytic_fixed_spacetime
            ? spacetime_coupling_.cell_spacetime()[lev].get()
            : nullptr,
         use_solid_mask ? primitive_needed_mask_[lev].get() : nullptr,
         use_solid_mask ? &primitive_needed_kind_[lev] : nullptr,
         &Geom(lev),
         fixed_spacetime_parameters(cfg_),
         time,
         ct_enabled(cfg_) && has_mhd_components<NComp>()
            ? &face_b_[lev]
            : nullptr,
         ct_enabled(cfg_) && has_mhd_components<NComp>()
            ? ct_face_primitive_mask_[lev].get()
            : nullptr,
      });
   }

   template <int NComp, int EOSMethod, bool SubcyclingEnabled = false>
   void compute_rhs_range(
      LevelData& input,
      LevelData const& candidate_base,
      amrex::Real reflux_weight,
      int lev_begin,
      int lev_end,
      amrex::Real rhs_time,
      amrex::Real candidate_base_weight,
      amrex::Real candidate_input_weight,
      amrex::Real candidate_increment_scale
   )
   {
      prepare_stage_sources<SubcyclingEnabled>(input, lev_begin, lev_end, rhs_time);
      amrex::Real rhs_wall_power = 0.0;
      bool rhs_has_wall_power = false;
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         const auto fluxes = flux_ptrs(lev);
         const bool use_solid_mask = solid_active(lev) && !has_mhd_components<NComp>();
         const bool use_mapped_patches = mixed_patch_system_enabled() && !has_mhd_components<NComp>();
         const auto solid_apertures = use_solid_mask ? solid_aperture_ptrs(lev) : std::array<amrex::MultiFab const*, AMREX_SPACEDIM>{};
         const int recon_method = recon_method_for_level(lev);
         const PatchGridSystem patch_grid_system = use_mapped_patches
            ? production_patch_grid_system_for_level(lev)
            : PatchGridSystem{};
         const PatchLayout patch_layout = use_mapped_patches ? production_patch_layout_for_level(lev) : PatchLayout{};
         amrex::MultiFab const* stored_spacetime =
            spacetime_state_.active() && !uses_analytic_fixed_spacetime_kernels()
            ? spacetime_coupling_.cell_spacetime()[lev].get()
            : nullptr;
         amrex::Real stage_start = profile_.begin();
         debug_check_reconstruction_input(*input[lev], lev, rhs_time, "RHS reconstruction input");
         debug_poison_mfab(*primitive_[lev]);
         fill_stage_primitive_cache<NComp, EOSMethod>(
            *primitive_[lev], *input[lev], lev, use_solid_mask, rhs_time
         );
         profile_.end(ProfileStage::PrimitiveCache, stage_start);
         if (!use_solid_mask) {
            debug_check_mfab_finite(*primitive_[lev], NComp, primitive_[lev]->nGrowVect(), "primitive cache");
         }
         const bool physical_vector_flux = !physical_system_is_relativistic(physics_system_)
            && !has_mhd_components<NComp>()
            && mesh_layout_spans_physical_space(mesh_coordinate_kind(), active_mesh_axes());
         if (physical_vector_flux) {
            transform_primitive_vectors<NComp>(
               *primitive_[lev], Geom(lev), mesh_coordinate_kind(), active_mesh_axes(), true
            );
         }
         stage_start = profile_.begin();
         debug_poison_fluxes(lev);
         if ((profile_.flux_detail_enabled() || stored_spacetime != nullptr) && !use_mapped_patches) {
            amrex::Real detail_start = profile_.begin();
            if (use_solid_mask) {
               launch_flux_direction_runtime<NComp, 0, true>(
                  fluxes,
                  *primitive_[lev],
                  cfg_,
                  recon_method,
                  riemann_method_,
                  solid_mask_[lev].get(),
                  &Geom(lev),
                  &solid_flux_kind_[lev][0],
                  rhs_time
               );
            } else {
               launch_flux_direction_runtime<NComp, 0>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr,
                  rhs_time, nullptr, nullptr, stored_spacetime, &recovery_atmosphere_,
                  step_rmhd_riemann_hll_fallback_count_.dataPtr()
               );
            }
            profile_.detail_end(ProfileDetailStage::FluxX, detail_start);
#if (AMREX_SPACEDIM >= 2)
            detail_start = profile_.begin();
            if (use_solid_mask) {
               launch_flux_direction_runtime<NComp, 1, true>(
                  fluxes,
                  *primitive_[lev],
                  cfg_,
                  recon_method,
                  riemann_method_,
                  solid_mask_[lev].get(),
                  &Geom(lev),
                  &solid_flux_kind_[lev][1],
                  rhs_time
               );
            } else {
               launch_flux_direction_runtime<NComp, 1>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr,
                  rhs_time, nullptr, nullptr, stored_spacetime, &recovery_atmosphere_,
                  step_rmhd_riemann_hll_fallback_count_.dataPtr()
               );
            }
            profile_.detail_end(ProfileDetailStage::FluxY, detail_start);
#endif
#if (AMREX_SPACEDIM == 3)
            detail_start = profile_.begin();
            if (use_solid_mask) {
               launch_flux_direction_runtime<NComp, 2, true>(
                  fluxes,
                  *primitive_[lev],
                  cfg_,
                  recon_method,
                  riemann_method_,
                  solid_mask_[lev].get(),
                  &Geom(lev),
                  &solid_flux_kind_[lev][2],
                  rhs_time
               );
            } else {
               launch_flux_direction_runtime<NComp, 2>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr,
                  rhs_time, nullptr, nullptr, stored_spacetime, &recovery_atmosphere_,
                  step_rmhd_riemann_hll_fallback_count_.dataPtr()
               );
            }
            profile_.detail_end(ProfileDetailStage::FluxZ, detail_start);
#endif
         } else {
            if (use_mapped_patches) {
               launch_fluxes_runtime<NComp, false, true>(
                  fluxes,
                  *primitive_[lev],
                  cfg_,
                  recon_method,
                  riemann_method_,
                  nullptr,
                  nullptr,
                  nullptr,
                  rhs_time,
                  &patch_grid_system,
                  &patch_layout
               );
            } else if (use_solid_mask) {
               launch_fluxes_runtime<NComp, true>(
                  fluxes,
                  *primitive_[lev],
                  cfg_,
                  recon_method,
                  riemann_method_,
                  solid_mask_[lev].get(),
                  &Geom(lev),
                  &solid_flux_kind_[lev],
                  rhs_time
               );
            } else {
               launch_fluxes_runtime<NComp>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr,
                  rhs_time, nullptr, nullptr, &recovery_atmosphere_,
                  step_rmhd_riemann_hll_fallback_count_.dataPtr()
               );
            }
         }
         profile_.end(ProfileStage::Flux, stage_start);
         debug_check_fluxes(lev, use_solid_mask, "reconstructed flux");
         if (physical_vector_flux) {
            transform_primitive_vectors<NComp>(
               *primitive_[lev], Geom(lev), mesh_coordinate_kind(), active_mesh_axes(), false
            );
         }
         if (patch_union_domain_enabled() && !has_mhd_components<NComp>()) {
            apply_patch_union_hydro_wall_fluxes<NComp>(fluxes, *primitive_[lev], Geom(lev), patch_union_domain_boxes(lev));
         }
         if (use_solid_mask && moving_wall_work_enabled_ && reflux_weight != 0.0) {
            amrex::iMultiFab valid_mask = valid_cell_mask(lev);
            const amrex::Real wall_power = solid_wall_energy_power_cutcell(
               *primitive_[lev],
               *solid_mask_[lev],
               *solid_volume_fraction_[lev],
               solid_apertures,
               valid_mask,
               Geom(lev),
               solid_geometry_evaluator(cfg_, rhs_time)
            );
            rhs_wall_power += wall_power;
            rhs_has_wall_power = true;
            cumulative_wall_work_ += reflux_weight * wall_power;
         }
         stage_start = profile_.begin();
         if (fluid_state_.rhs()[lev]) {
            debug_poison_mfab(*fluid_state_.rhs()[lev]);
         }
         if (use_solid_mask) {
            compute_rhs_from_fluxes_cutcell<NComp>(
               *fluid_state_.rhs()[lev],
               fluxes,
               *solid_mask_[lev],
               *solid_volume_fraction_[lev],
               solid_apertures,
               *primitive_[lev],
               Geom(lev),
               solid_geometry_evaluator(cfg_, rhs_time),
               &solid_valid_kind_[lev]
            );
         } else if (use_mapped_patches) {
            compute_rhs_from_fluxes_mapped_patches<NComp>(
               *fluid_state_.rhs()[lev],
               fluxes,
               patch_grid_system,
               patch_layout
            );
         } else if (physical_vector_flux) {
            compute_rhs_from_physical_fluxes_metric<NComp>(
               *fluid_state_.rhs()[lev], fluxes, Geom(lev), mesh_coordinate_kind(), active_mesh_axes()
            );
         } else if (mesh_is_curvilinear()) {
            compute_rhs_from_fluxes_metric<NComp>(
               *fluid_state_.rhs()[lev],
               fluxes,
               *primitive_[lev],
               Geom(lev),
               mesh_coordinate_kind(),
               active_mesh_axes()
            );
         } else if (cfg_.mesh_metric_divergence == "always") {
            compute_rhs_from_fluxes_mesh_divergence<NComp>(
               *fluid_state_.rhs()[lev],
               fluxes,
               Geom(lev),
               mesh_coordinate_kind(),
               active_mesh_axes()
            );
         } else {
            compute_rhs_from_fluxes<NComp>(*fluid_state_.rhs()[lev], fluxes, Geom(lev));
         }
         profile_.end(ProfileStage::Divergence, stage_start);
         if (sources_enabled(cfg_)) {
            const MassEnergy source_before = local_rhs_mass_energy_integral(lev);
            stage_start = profile_.begin();
            apply_stage_sources_to_rhs<NComp>(lev, input, fluxes, use_solid_mask, rhs_time);
            profile_.end(ProfileStage::Sources, stage_start);
            accumulate_source_change(source_before, local_rhs_mass_energy_integral(lev), reflux_weight);
         }
         if (physical_system_is_relativistic(physics_system_)) {
            stage_start = profile_.begin();
            add_relativistic_sources(RelativisticSourceLaunch{
               *fluid_state_.rhs()[lev],
               *primitive_[lev],
               Geom(lev),
               eos_params(cfg_),
               physics_system_,
               stored_spacetime,
               fixed_spacetime_parameters(cfg_),
               rhs_time,
            });
            profile_.end(ProfileStage::Sources, stage_start);
         }
         const bool rmhd_candidate_guard =
            physics_system_ == PhysicalSystem::RMHD
            && cfg_.rmhd_candidate_guard != "off";
         const bool defer_relativistic_limiter =
            spacetime_state_.evolves()
            && (
               physics_system_ == PhysicalSystem::RHD
               || rmhd_candidate_guard
            );
         if (!defer_relativistic_limiter) {
            if (physics_system_ == PhysicalSystem::RHD) {
               apply_rhd_admissibility_flux_limiter(
                  fluxes,
                  *fluid_state_.rhs()[lev],
                  *candidate_base[lev],
                  *input[lev],
                  *primitive_[lev],
                  eos_params(cfg_),
                  eos_method_,
                  Geom(lev),
                  fixed_spacetime_parameters(cfg_),
                  stored_spacetime,
                  stored_spacetime,
                  recovery_atmosphere_,
                  rhs_time,
                  candidate_base_weight,
                  candidate_input_weight,
                  candidate_increment_scale
               );
            } else if (rmhd_candidate_guard) {
               apply_rmhd_candidate_flux_limiter(
                  fluxes,
                  *fluid_state_.rhs()[lev],
                  *candidate_base[lev],
                  *input[lev],
                  *primitive_[lev],
                  eos_params(cfg_),
                  rmhd_candidate_riemann_method(),
                  Geom(lev),
                  fixed_spacetime_parameters(cfg_),
                  stored_spacetime,
                  stored_spacetime,
                  recovery_atmosphere_,
                  rhs_time,
                  candidate_base_weight,
                  candidate_input_weight,
                  candidate_increment_scale,
                  cfg_.rmhd_candidate_guard == "pwl-pwc",
                  cfg_.rmhd_candidate_dmp
                     && recon_method != ReconPWL
                     && recon_method != ReconPWC,
                  cfg_.rmhd_candidate_dmp_factor,
                  step_rmhd_pwl_correction_count_.dataPtr(),
                  step_rmhd_pwc_correction_count_.dataPtr(),
                  step_rmhd_riemann_hll_fallback_count_.dataPtr()
               );
            }
         }
         if (!defer_relativistic_limiter) {
            debug_check_fluxes(lev, use_solid_mask, "final flux");
            accumulate_physical_boundary_flux(lev, fluxes, reflux_weight);
            debug_check_rhs_level(lev, "RHS assembly");
            stage_start = profile_.begin();
            accumulate_ct_stage_emf(lev, fluxes, reflux_weight);
            add_fluxes_to_registers(lev, reflux_weight);
            profile_.end(ProfileStage::FluxRegister, stage_start);
         }
      }
      if (rhs_has_wall_power) {
         latest_wall_power_ = rhs_wall_power;
      }
   }

   template <int NComp, int EOSMethod, bool SubcyclingEnabled = false>
   void compute_evolution_rhs_range(
      EvolutionStageSlot input_slot,
      amrex::Real reflux_weight,
      int lev_begin,
      int lev_end,
      amrex::Real stage_time,
      EvolutionStageSlot candidate_base_slot,
      amrex::Real candidate_base_weight,
      amrex::Real candidate_input_weight,
      amrex::Real candidate_increment_scale,
      amrex::Real candidate_time,
      amrex::Real fine_step_start = -1.0,
      int rk_stage = 0
   )
   {
      if (spacetime_state_.active()) {
         prepare_spacetime_coupling_range(
            spacetime_state_.stage(input_slot),
            lev_begin,
            lev_end,
            stage_time,
            fine_step_start,
            rk_stage
         );
         const char* stage_context = input_slot == EvolutionStageSlot::State
            ? "relativistic state-stage input"
            : (input_slot == EvolutionStageSlot::Scratch
                  ? "relativistic scratch-stage input"
                  : "relativistic next-stage input");
         apply_stored_spacetime_fluid_boundaries_range(
            fluid_state_.stage(input_slot), lev_begin, lev_end
         );
         project_relativistic_atmosphere_range<EOSMethod>(
            fluid_state_.stage(input_slot), lev_begin, lev_end, cfg_.ng
         );
         debug_check_relativistic_stage_range(
            fluid_state_.stage(input_slot), lev_begin, lev_end, stage_context, stage_time
         );
      }
      compute_rhs_range<NComp, EOSMethod, SubcyclingEnabled>(
         fluid_state_.stage(input_slot),
         fluid_state_.stage(candidate_base_slot),
         reflux_weight,
         lev_begin,
         lev_end,
         stage_time,
         candidate_base_weight,
         candidate_input_weight,
         candidate_increment_scale
      );
      if (spacetime_state_.evolves()) {
         fill_spacetime_matter_range(lev_begin, lev_end);
         compute_spacetime_rhs_range(
            spacetime_state_.stage(input_slot),
            lev_begin,
            lev_end,
            stage_time
         );
      }
      const bool deferred_rmhd_candidate_guard =
         physics_system_ == PhysicalSystem::RMHD
         && cfg_.rmhd_candidate_guard != "off"
         && spacetime_state_.evolves();
      if (
         spacetime_state_.evolves()
         && (
            physics_system_ == PhysicalSystem::RHD
            || deferred_rmhd_candidate_guard
         )
      ) {
         LevelData candidate_vertex_spacetime(finestLevel() + 1);
         LevelData candidate_cell_spacetime(finestLevel() + 1);
         for (int lev = lev_begin; lev < lev_end; ++lev) {
            amrex::MultiFab const& input_vertex_spacetime =
               *spacetime_state_.stage(input_slot)[lev];
            amrex::MultiFab const& base_vertex_spacetime =
               *spacetime_state_.stage(candidate_base_slot)[lev];
            candidate_vertex_spacetime[lev] = std::make_unique<amrex::MultiFab>(
               input_vertex_spacetime.boxArray(),
               input_vertex_spacetime.DistributionMap(),
               input_vertex_spacetime.nComp(),
               input_vertex_spacetime.nGrowVect()
            );
            amrex::MultiFab::LinComb(
               *candidate_vertex_spacetime[lev],
               candidate_base_weight,
               base_vertex_spacetime,
               0,
               candidate_input_weight,
               input_vertex_spacetime,
               0,
               0,
               input_vertex_spacetime.nComp(),
               input_vertex_spacetime.nGrow()
            );
            amrex::MultiFab::Saxpy(
               *candidate_vertex_spacetime[lev],
               candidate_increment_scale,
               *spacetime_state_.rhs()[lev],
               0,
               0,
               input_vertex_spacetime.nComp(),
               0
            );
            project_spacetime_state(*candidate_vertex_spacetime[lev]);
         }
         const bool synchronous_levels = lev_end - lev_begin > 1;
         for (int lev = lev_begin; lev < lev_end; ++lev) {
            fill_spacetime_level_boundaries(
               candidate_vertex_spacetime,
               lev,
               candidate_time,
               synchronous_levels,
               fine_step_start,
               rk_stage
            );
         }
         for (int lev = lev_begin; lev < lev_end; ++lev) {
            amrex::MultiFab const& current_cell_spacetime =
               *spacetime_coupling_.cell_spacetime()[lev];
            candidate_cell_spacetime[lev] = std::make_unique<amrex::MultiFab>(
               current_cell_spacetime.boxArray(),
               current_cell_spacetime.DistributionMap(),
               current_cell_spacetime.nComp(),
               0
            );
            transfer_grid_field_centering<4, AMREX_SPACEDIM>(
               *candidate_cell_spacetime[lev],
               *candidate_vertex_spacetime[lev],
               Geom(lev),
               false
            );

            const auto fluxes = flux_ptrs(lev);
            if (physics_system_ == PhysicalSystem::RHD) {
               apply_rhd_admissibility_flux_limiter(
                  fluxes,
                  *fluid_state_.rhs()[lev],
                  *fluid_state_.stage(candidate_base_slot)[lev],
                  *fluid_state_.stage(input_slot)[lev],
                  *primitive_[lev],
                  eos_params(cfg_),
                  eos_method_,
                  Geom(lev),
                  fixed_spacetime_parameters(cfg_),
                  spacetime_coupling_.cell_spacetime()[lev].get(),
                  candidate_cell_spacetime[lev].get(),
                  recovery_atmosphere_,
                  stage_time,
                  candidate_base_weight,
                  candidate_input_weight,
                  candidate_increment_scale
               );
            } else {
               apply_rmhd_candidate_flux_limiter(
                  fluxes,
                  *fluid_state_.rhs()[lev],
                  *fluid_state_.stage(candidate_base_slot)[lev],
                  *fluid_state_.stage(input_slot)[lev],
                  *primitive_[lev],
                  eos_params(cfg_),
                  rmhd_candidate_riemann_method(),
                  Geom(lev),
                  fixed_spacetime_parameters(cfg_),
                  spacetime_coupling_.cell_spacetime()[lev].get(),
                  candidate_cell_spacetime[lev].get(),
                  recovery_atmosphere_,
                  stage_time,
                  candidate_base_weight,
                  candidate_input_weight,
                  candidate_increment_scale,
                  cfg_.rmhd_candidate_guard == "pwl-pwc",
                  cfg_.rmhd_candidate_dmp
                     && recon_method_for_level(lev) != ReconPWL
                     && recon_method_for_level(lev) != ReconPWC,
                  cfg_.rmhd_candidate_dmp_factor,
                  step_rmhd_pwl_correction_count_.dataPtr(),
                  step_rmhd_pwc_correction_count_.dataPtr(),
                  step_rmhd_riemann_hll_fallback_count_.dataPtr()
               );
            }
            debug_check_fluxes(lev, false, "final flux");
            accumulate_physical_boundary_flux(lev, fluxes, reflux_weight);
            debug_check_rhs_level(lev, "RHS assembly");
            const amrex::Real register_start = profile_.begin();
            accumulate_ct_stage_emf(lev, fluxes, reflux_weight);
            add_fluxes_to_registers(lev, reflux_weight);
            profile_.end(ProfileStage::FluxRegister, register_start);
         }
      }
   }

   template <int NComp, int EOSMethod>
   void compute_rhs_all(
      LevelData& input,
      amrex::Real reflux_weight,
      amrex::Real rhs_time,
      amrex::Real forward_euler_dt
   )
   {
      compute_rhs_range<NComp, EOSMethod, false>(
         input, input, reflux_weight, 0, finestLevel() + 1, rhs_time,
         0.0, 1.0, forward_euler_dt
      );
   }

   template <int NComp>
   bool can_apply_direct_flux_stages(int lev_begin, int lev_end) const
   {
      if (cfg_.mesh_metric_divergence == "always") {
         return false;
      }
      if (mesh_is_curvilinear()) {
         return false;
      }
      if (mixed_patch_system_enabled()) {
         return false;
      }
      if (sources_enabled(cfg_)) {
         return false;
      }
      if (spacetime_state_.active()) {
         return false;
      }
      if (
         physics_system_ == PhysicalSystem::RMHD
         && cfg_.rmhd_candidate_guard != "off"
      ) {
         return false;
      }
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         if (solid_active(lev) && (has_mhd_components<NComp>() || cfg_.solid_motion != "static")) {
            return false;
         }
      }
      return true;
   }

   template <int NComp, int EOSMethod>
   void compute_flux_stage_range(LevelData& input, amrex::Real reflux_weight, int lev_begin, int lev_end, amrex::Real rhs_time)
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         const auto fluxes = flux_ptrs(lev);
         const int recon_method = recon_method_for_level(lev);
         const bool use_solid_mask = solid_active(lev) && !has_mhd_components<NComp>() && cfg_.solid_motion == "static";
         amrex::Real stage_start = profile_.begin();
         debug_check_reconstruction_input(*input[lev], lev, rhs_time, "direct-flux reconstruction input");
         debug_poison_mfab(*primitive_[lev]);
         fill_stage_primitive_cache<NComp, EOSMethod>(
            *primitive_[lev], *input[lev], lev, use_solid_mask, rhs_time
         );
         profile_.end(ProfileStage::PrimitiveCache, stage_start);
         if (!use_solid_mask) {
            debug_check_mfab_finite(*primitive_[lev], NComp, primitive_[lev]->nGrowVect(), "primitive cache");
         }

         stage_start = profile_.begin();
         debug_poison_fluxes(lev);
         if (profile_.flux_detail_enabled()) {
            amrex::Real detail_start = profile_.begin();
            if (use_solid_mask) {
               launch_flux_direction_runtime<NComp, 0, true>(
                  fluxes,
                  *primitive_[lev],
                  cfg_,
                  recon_method,
                  riemann_method_,
                  solid_mask_[lev].get(),
                  &Geom(lev),
                  &solid_flux_kind_[lev][0],
                  rhs_time
               );
            } else {
               launch_flux_direction_runtime<NComp, 0>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr,
                  rhs_time, nullptr, nullptr, nullptr, &recovery_atmosphere_,
                  step_rmhd_riemann_hll_fallback_count_.dataPtr()
               );
            }
            profile_.detail_end(ProfileDetailStage::FluxX, detail_start);
#if (AMREX_SPACEDIM >= 2)
            detail_start = profile_.begin();
            if (use_solid_mask) {
               launch_flux_direction_runtime<NComp, 1, true>(
                  fluxes,
                  *primitive_[lev],
                  cfg_,
                  recon_method,
                  riemann_method_,
                  solid_mask_[lev].get(),
                  &Geom(lev),
                  &solid_flux_kind_[lev][1],
                  rhs_time
               );
            } else {
               launch_flux_direction_runtime<NComp, 1>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr,
                  rhs_time, nullptr, nullptr, nullptr, &recovery_atmosphere_,
                  step_rmhd_riemann_hll_fallback_count_.dataPtr()
               );
            }
            profile_.detail_end(ProfileDetailStage::FluxY, detail_start);
#endif
#if (AMREX_SPACEDIM == 3)
            detail_start = profile_.begin();
            if (use_solid_mask) {
               launch_flux_direction_runtime<NComp, 2, true>(
                  fluxes,
                  *primitive_[lev],
                  cfg_,
                  recon_method,
                  riemann_method_,
                  solid_mask_[lev].get(),
                  &Geom(lev),
                  &solid_flux_kind_[lev][2],
                  rhs_time
               );
            } else {
               launch_flux_direction_runtime<NComp, 2>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr,
                  rhs_time, nullptr, nullptr, nullptr, &recovery_atmosphere_,
                  step_rmhd_riemann_hll_fallback_count_.dataPtr()
               );
            }
            profile_.detail_end(ProfileDetailStage::FluxZ, detail_start);
#endif
         } else {
            if (use_solid_mask) {
               launch_fluxes_runtime<NComp, true>(
                  fluxes,
                  *primitive_[lev],
                  cfg_,
                  recon_method,
                  riemann_method_,
                  solid_mask_[lev].get(),
                  &Geom(lev),
                  &solid_flux_kind_[lev],
                  rhs_time
               );
            } else {
               launch_fluxes_runtime<NComp>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr,
                  rhs_time, nullptr, nullptr, &recovery_atmosphere_,
                  step_rmhd_riemann_hll_fallback_count_.dataPtr()
               );
            }
         }
         profile_.end(ProfileStage::Flux, stage_start);
         if (patch_union_domain_enabled() && !has_mhd_components<NComp>()) {
            apply_patch_union_hydro_wall_fluxes<NComp>(fluxes, *primitive_[lev], Geom(lev), patch_union_domain_boxes(lev));
         }
         debug_check_fluxes(lev, use_solid_mask, "direct flux stage");
         accumulate_physical_boundary_flux(lev, fluxes, reflux_weight);

         stage_start = profile_.begin();
         accumulate_ct_stage_emf(lev, fluxes, reflux_weight);
         add_fluxes_to_registers(lev, reflux_weight);
         profile_.end(ProfileStage::FluxRegister, stage_start);
      }
   }
