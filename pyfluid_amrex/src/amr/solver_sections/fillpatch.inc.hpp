// Included inside pyfluid_amrex::AmrSolverBase protected section.
// State and face-centered magnetic-field FillPatch contracts. Only the
// subcycling-dependent orchestration remains templated here.

   bool should_regrid(int base_step, bool reached_final_time) const;
   bool should_regrid_aligned_descendants(int lev) const;
   bool needs_fresh_state_boundaries_after_step(bool write_scalars, bool write_aux_gfs) const;
   void synchronize_regridded_hierarchy(int lbase);
   int state_fillpatch_bc_type(int component, int dir, bool high_side) const;
   int face_fillpatch_bc_type(int face_dir, int boundary_dir, bool high_side) const;
   amrex::Vector<amrex::BCRec> make_state_fillpatch_bcs() const;
   amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> make_face_fillpatch_bcs() const;
   amrex::Vector<amrex::BCRec> const& fillpatch_bcs() const;
   amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> const& face_fillpatch_bcs() const;
   amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> face_b_array(FaceMagneticField& face_b) const;
   StateFillPatchFunc const& state_fillpatch_func() const;
   FaceFillPatchFunc const& face_fillpatch_func(int face_dir) const;
   bool compute_post_fillpatch_state_boundary_needed() const;
   void apply_post_fillpatch_state_boundary(amrex::MultiFab& state, amrex::Geometry const& geom, amrex::Real time) const;
   void fill_mixed_patch_state_ghosts(amrex::MultiFab& state, int lev) const;
   amrex::Interpolater* selected_state_interpolater() const;
   amrex::Interpolater* state_interpolater() const;
   bool level_refinement_ratio_is_anisotropic(int coarse_lev) const;

   void fill_new_level_from_coarse_state(int lev, amrex::Real time);
   void fill_remade_level_state(int lev, amrex::Real time, amrex::MultiFab& old_state);
   bool coarse_time_interval_contains(int coarse_lev, amrex::Real time) const;
   amrex::Real clamp_to_coarse_time_interval(int coarse_lev, amrex::Real time) const;
   bool rk4_stage_data_ready(int coarse_lev) const;
   void fill_rk4_coarse_stage_state(int coarse_lev, amrex::Real fine_step_start, int rk_stage);
   void fill_new_face_b_from_coarse(int lev, amrex::Real time);
   void fill_remade_face_b(int lev, amrex::Real time, FaceMagneticField& old_face_b);
   void initialize_ct_fields_from_coarse(int lev, amrex::Real time);
   void initialize_remade_ct_fields(int lev, amrex::Real time, FaceMagneticField& old_face_b);

   template <bool SubcyclingEnabled>
   bool regrid_hierarchy(amrex::Real time, int step, int requested_lbase = 0)
   {
      int lbase = std::max(requested_lbase, 0);
      if (cfg_.initial_refine_fixed_levels > 0) {
         lbase = std::max(lbase, std::min(cfg_.initial_refine_fixed_levels, finestLevel()));
      }
      if (lbase >= maxLevel()) {
         return false;
      }

      const bool whole_hierarchy = lbase == 0;
      const bool preserve_hydro_solid_totals = whole_hierarchy && !is_mhd(cfg_) && solid_geometry_enabled(cfg_);
      HydroConservedTotals pre_regrid_totals;
      if (preserve_hydro_solid_totals) {
         pre_regrid_totals = hydro_valid_conserved_totals();
      }
      regrid_lifecycle_.begin_attempt();
      regrid(lbase, time, false);
      const auto first_modified = regrid_lifecycle_.finish_attempt();
      if (!first_modified) {
         return false;
      }
      enforce_cell_budget("regridding AMR hierarchy");
      const int first_modified_level = *first_modified;
      define_flux_registers(first_modified_level);
      for (int lev = std::max(lbase + 1, first_modified_level); lev <= finestLevel(); ++lev) {
         const int relative_time_factor = level_time_factor(lev) / level_time_factor(lbase);
         level_steps_[lev] = SubcyclingEnabled ? step * relative_time_factor : step;
         level_times_[lev] = time;
         level_old_times_[lev] = time;
         level_dts_[lev] = SubcyclingEnabled && !local_cfl_subcycling()
                              ? (level_dts_[lbase] / static_cast<amrex::Real>(relative_time_factor))
                              : level_dts_[lbase];
      }
      synchronize_regridded_hierarchy(lbase);
      if (whole_hierarchy) {
         refresh_solid_geometry_synced(time, true, preserve_hydro_solid_totals);
      } else if (solid_geometry_enabled(cfg_)) {
         for (int lev = lbase; lev <= finestLevel(); ++lev) {
            refresh_solid_geometry_level(lev, time, false);
         }
         enforce_solid_mask_nesting_from_fine();
      }
      if (preserve_hydro_solid_totals) {
         repair_after_hierarchy_change(pre_regrid_totals);
         if (finestLevel() > 0) {
            average_down_state();
         }
      }
      for (int lev = lbase; lev <= finestLevel(); ++lev) {
         fill_level_boundaries<SubcyclingEnabled>(fluid_state_.state(), lev, time);
         fill_spacetime_level_boundaries(spacetime_state_.state(), lev, time);
      }
      for (int lev = lbase + 1; lev <= finestLevel(); ++lev) {
         copy_level_state_to_old(lev);
      }
      for (int lev = lbase; lev <= finestLevel(); ++lev) {
         invalidate_rk4_stage_cache(lev);
      }
      return true;
   }

   template <bool SubcyclingEnabled = false>
   void fill_level_boundaries(LevelData& data, int lev, amrex::Real time, amrex::Real fine_step_start = -1.0_rt,
                              int rk_stage = 0)
   {
      if (lev == 0) {
         apply_boundary(*data[0], Geom(0), time);
         fill_mixed_patch_state_ghosts(*data[0], 0);
         return;
      }

      amrex::Vector<amrex::MultiFab*> coarse_data;
      amrex::Vector<amrex::Real> coarse_time;
      if constexpr (SubcyclingEnabled) {
         if (rk_stage > 0 && rk4_stage_data_ready(lev - 1)) {
            fill_rk4_coarse_stage_state(lev - 1, fine_step_start, rk_stage);
            coarse_data.push_back(fluid_state_.rk4_coarse_stage_state()[lev - 1].get());
            coarse_time.push_back(time);
         } else if (coarse_time_interval_contains(lev - 1, time)) {
            coarse_data.push_back(fluid_state_.old_state()[lev - 1].get());
            coarse_data.push_back(fluid_state_.state()[lev - 1].get());
            coarse_time.push_back(level_old_times_[lev - 1]);
            coarse_time.push_back(level_times_[lev - 1]);
            time = clamp_to_coarse_time_interval(lev - 1, time);
         } else {
            coarse_data.push_back(data[lev - 1].get());
            coarse_time.push_back(time);
         }
      } else {
         coarse_data.push_back(data[lev - 1].get());
         coarse_time.push_back(time);
      }
      amrex::Vector<amrex::MultiFab*> fine_data(1, data[lev].get());
      amrex::Vector<amrex::Real> fine_time(1, time);
      auto const& bcs = fillpatch_bcs();
      const StateFillPatchFunc& boundary_func = state_fillpatch_func();
      StateFillPatchPhysBC coarse_bc(Geom(lev - 1), bcs, boundary_func);
      StateFillPatchPhysBC fine_bc(Geom(lev), bcs, boundary_func);
      amrex::FillPatchTwoLevels(*data[lev], data[lev]->nGrowVect(), time, coarse_data, coarse_time, fine_data, fine_time,
                                0, 0, num_state_components(), Geom(lev - 1), Geom(lev), coarse_bc, 0, fine_bc, 0,
                                refRatio(lev - 1), state_interpolater(), bcs, 0);
      apply_post_fillpatch_state_boundary(*data[lev], Geom(lev), time);
      complete_radial_origin_boundary(data, lev, time, coarse_data, coarse_time, fine_data, fine_time);
      fill_mixed_patch_state_ghosts(*data[lev], lev);
   }

   template <bool SubcyclingEnabled = false>
   void fill_level_boundaries_range(LevelData& data, int lev_begin, int lev_end, amrex::Real time,
                                    amrex::Real fine_step_start = -1.0_rt, int rk_stage = 0)
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         fill_level_boundaries<SubcyclingEnabled>(data, lev, time, fine_step_start, rk_stage);
      }
   }

   template <bool SubcyclingEnabled = false> void fill_all_level_boundaries(LevelData& data, amrex::Real time)
   {
      fill_level_boundaries_range<SubcyclingEnabled>(data, 0, finestLevel() + 1, time);
   }
