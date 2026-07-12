// Included inside pyfluid_amrex::AmrSolver private section.
// RHS/flux-stage assembly for conservative updates.

   template <int NComp, int EOSMethod, bool SubcyclingEnabled = false>
   void compute_rhs_range(LevelData& input, amrex::Real reflux_weight, int lev_begin, int lev_end, amrex::Real rhs_time)
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
         amrex::Real stage_start = profile_begin();
         debug_check_reconstruction_input(*input[lev], lev, rhs_time, "RHS reconstruction input");
         debug_poison_mfab(*primitive_[lev]);
         if (use_solid_mask) {
            fill_primitive_cache_masked<NComp, EOSMethod>(
               *primitive_[lev],
               *input[lev],
               *primitive_needed_mask_[lev],
               eos_params(cfg_),
               &primitive_needed_kind_[lev]
            );
         } else {
            fill_primitive_cache<NComp, EOSMethod>(*primitive_[lev], *input[lev], eos_params(cfg_));
         }
         profile_end(ProfileStage::PrimitiveCache, stage_start);
         if (!use_solid_mask) {
            debug_check_mfab_finite(*primitive_[lev], NComp, primitive_[lev]->nGrowVect(), "primitive cache");
         }
         const bool physical_vector_flux = !has_mhd_components<NComp>()
            && mesh_layout_spans_physical_space(mesh_coordinate_kind(), active_mesh_axes());
         if (physical_vector_flux) {
            transform_primitive_vectors<NComp>(
               *primitive_[lev], Geom(lev), mesh_coordinate_kind(), active_mesh_axes(), true
            );
         }
         stage_start = profile_begin();
         debug_poison_fluxes(lev);
         if (flux_profile_detail_enabled() && !use_mapped_patches) {
            amrex::Real detail_start = profile_begin();
            if (use_solid_mask) {
               fill_flux_direction_runtime<NComp, 0, EOSMethod, true>(
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
               fill_flux_direction_runtime<NComp, 0, EOSMethod>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr, rhs_time
               );
            }
            profile_detail_end(ProfileDetailStage::FluxX, detail_start);
#if (AMREX_SPACEDIM >= 2)
            detail_start = profile_begin();
            if (use_solid_mask) {
               fill_flux_direction_runtime<NComp, 1, EOSMethod, true>(
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
               fill_flux_direction_runtime<NComp, 1, EOSMethod>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr, rhs_time
               );
            }
            profile_detail_end(ProfileDetailStage::FluxY, detail_start);
#endif
#if (AMREX_SPACEDIM == 3)
            detail_start = profile_begin();
            if (use_solid_mask) {
               fill_flux_direction_runtime<NComp, 2, EOSMethod, true>(
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
               fill_flux_direction_runtime<NComp, 2, EOSMethod>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr, rhs_time
               );
            }
            profile_detail_end(ProfileDetailStage::FluxZ, detail_start);
#endif
         } else {
            if (use_mapped_patches) {
               fill_fluxes_runtime<NComp, EOSMethod, false, true>(
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
               fill_fluxes_runtime<NComp, EOSMethod, true>(
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
               fill_fluxes_runtime<NComp, EOSMethod>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr, rhs_time
               );
            }
         }
         profile_end(ProfileStage::Flux, stage_start);
         if (physical_vector_flux) {
            transform_primitive_vectors<NComp>(
               *primitive_[lev], Geom(lev), mesh_coordinate_kind(), active_mesh_axes(), false
            );
         }
         if (patch_union_domain_enabled() && !has_mhd_components<NComp>()) {
            apply_patch_union_hydro_wall_fluxes<NComp>(fluxes, *primitive_[lev], Geom(lev), patch_union_domain_boxes(lev));
         }
         debug_check_fluxes(lev, use_solid_mask, "flux fill");
         accumulate_physical_boundary_flux(lev, fluxes, reflux_weight);
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
         stage_start = profile_begin();
         if (rhs_[lev]) {
            debug_poison_mfab(*rhs_[lev]);
         }
         if (use_solid_mask) {
            compute_rhs_from_fluxes_cutcell<NComp>(
               *rhs_[lev],
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
               *rhs_[lev],
               fluxes,
               patch_grid_system,
               patch_layout
            );
         } else if (physical_vector_flux) {
            compute_rhs_from_physical_fluxes_metric<NComp>(
               *rhs_[lev], fluxes, Geom(lev), mesh_coordinate_kind(), active_mesh_axes()
            );
         } else if (mesh_is_curvilinear()) {
            compute_rhs_from_fluxes_metric<NComp>(
               *rhs_[lev],
               fluxes,
               *primitive_[lev],
               Geom(lev),
               mesh_coordinate_kind(),
               active_mesh_axes()
            );
         } else if (cfg_.mesh_metric_divergence == "always") {
            compute_rhs_from_fluxes_mesh_divergence<NComp>(
               *rhs_[lev],
               fluxes,
               Geom(lev),
               mesh_coordinate_kind(),
               active_mesh_axes()
            );
         } else {
            compute_rhs_from_fluxes<NComp>(*rhs_[lev], fluxes, Geom(lev));
         }
         profile_end(ProfileStage::Divergence, stage_start);
         if (sources_enabled(cfg_)) {
            const MassEnergy source_before = local_rhs_mass_energy_integral(lev);
            stage_start = profile_begin();
            apply_stage_sources_to_rhs<NComp>(lev, input, fluxes, use_solid_mask, rhs_time);
            profile_end(ProfileStage::Sources, stage_start);
            accumulate_source_change(source_before, local_rhs_mass_energy_integral(lev), reflux_weight);
         }
         debug_check_rhs_level(lev, "RHS assembly");
         stage_start = profile_begin();
         add_fluxes_to_registers(lev, reflux_weight);
         profile_end(ProfileStage::FluxRegister, stage_start);
      }
      if (rhs_has_wall_power) {
         latest_wall_power_ = rhs_wall_power;
      }
   }

   template <int NComp, int EOSMethod>
   void compute_rhs_all(LevelData& input, amrex::Real reflux_weight, amrex::Real rhs_time)
   {
      compute_rhs_range<NComp, EOSMethod, false>(input, reflux_weight, 0, finestLevel() + 1, rhs_time);
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
         amrex::Real stage_start = profile_begin();
         debug_check_reconstruction_input(*input[lev], lev, rhs_time, "direct-flux reconstruction input");
         debug_poison_mfab(*primitive_[lev]);
         if (use_solid_mask) {
            fill_primitive_cache_masked<NComp, EOSMethod>(
               *primitive_[lev],
               *input[lev],
               *primitive_needed_mask_[lev],
               eos_params(cfg_),
               &primitive_needed_kind_[lev]
            );
         } else {
            fill_primitive_cache<NComp, EOSMethod>(*primitive_[lev], *input[lev], eos_params(cfg_));
         }
         profile_end(ProfileStage::PrimitiveCache, stage_start);
         if (!use_solid_mask) {
            debug_check_mfab_finite(*primitive_[lev], NComp, primitive_[lev]->nGrowVect(), "primitive cache");
         }

         stage_start = profile_begin();
         debug_poison_fluxes(lev);
         if (flux_profile_detail_enabled()) {
            amrex::Real detail_start = profile_begin();
            if (use_solid_mask) {
               fill_flux_direction_runtime<NComp, 0, EOSMethod, true>(
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
               fill_flux_direction_runtime<NComp, 0, EOSMethod>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr, rhs_time
               );
            }
            profile_detail_end(ProfileDetailStage::FluxX, detail_start);
#if (AMREX_SPACEDIM >= 2)
            detail_start = profile_begin();
            if (use_solid_mask) {
               fill_flux_direction_runtime<NComp, 1, EOSMethod, true>(
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
               fill_flux_direction_runtime<NComp, 1, EOSMethod>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr, rhs_time
               );
            }
            profile_detail_end(ProfileDetailStage::FluxY, detail_start);
#endif
#if (AMREX_SPACEDIM == 3)
            detail_start = profile_begin();
            if (use_solid_mask) {
               fill_flux_direction_runtime<NComp, 2, EOSMethod, true>(
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
               fill_flux_direction_runtime<NComp, 2, EOSMethod>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr, rhs_time
               );
            }
            profile_detail_end(ProfileDetailStage::FluxZ, detail_start);
#endif
         } else {
            if (use_solid_mask) {
               fill_fluxes_runtime<NComp, EOSMethod, true>(
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
               fill_fluxes_runtime<NComp, EOSMethod>(
                  fluxes, *primitive_[lev], cfg_, recon_method, riemann_method_, nullptr, &Geom(lev), nullptr, rhs_time
               );
            }
         }
         profile_end(ProfileStage::Flux, stage_start);
         if (patch_union_domain_enabled() && !has_mhd_components<NComp>()) {
            apply_patch_union_hydro_wall_fluxes<NComp>(fluxes, *primitive_[lev], Geom(lev), patch_union_domain_boxes(lev));
         }
         debug_check_fluxes(lev, use_solid_mask, "direct flux stage");
         accumulate_physical_boundary_flux(lev, fluxes, reflux_weight);

         stage_start = profile_begin();
         add_fluxes_to_registers(lev, reflux_weight);
         profile_end(ProfileStage::FluxRegister, stage_start);
      }
   }
