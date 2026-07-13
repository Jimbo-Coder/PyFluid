// Included inside pyfluid_amrex::AmrSolver private section.
// State and face-centered magnetic-field FillPatch, physical boundary, and average-down helpers.

   bool should_regrid(int base_step, bool reached_final_time) const
   {
      if (cfg_.regrid_int <= 0 || maxLevel() <= 0 || base_step == cfg_.max_step || reached_final_time) {
         return false;
      }
      if (cfg_.regrid_mode == "aligned" && cfg_.subcycling == "on") {
         return aligned_regrid_ticks_ - aligned_base_regrid_tick_ >= cfg_.regrid_int;
      }
      return base_step % cfg_.regrid_int == 0;
   }

   bool should_regrid_aligned_descendants(int lev) const
   {
      const bool reached_final_time =
         cfg_.final_time > 0.0 && level_times_[lev] >= cfg_.final_time;
      return cfg_.regrid_mode == "aligned"
         && cfg_.subcycling == "on"
         && !self_gravity_enabled(cfg_)
         && aligned_regrid_due_
         && lev < maxLevel()
         && (lev == 0 || !nearly_equal_time(level_times_[lev - 1], level_times_[lev]))
         && !reached_final_time;
   }

   bool needs_fresh_state_boundaries_after_step(bool write_scalars, bool write_aux_gfs) const
   {
      const bool scalar_divb_uses_cell_centered_b = write_scalars && is_mhd(cfg_) && !ct_enabled(cfg_);
      const bool aux_fields_use_derivatives = write_aux_gfs && aux_gfs_enabled(cfg_);
      return scalar_divb_uses_cell_centered_b || aux_fields_use_derivatives;
   }

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
      const bool preserve_hydro_solid_totals =
         whole_hierarchy && !is_mhd(cfg_) && solid_geometry_enabled(cfg_);
      HydroConservedTotals pre_regrid_totals;
      if (preserve_hydro_solid_totals) {
         pre_regrid_totals = hydro_valid_conserved_totals();
      }
      begin_regrid_change_tracking();
      regrid(lbase, time, false);
      if (!finish_regrid_change_tracking()) {
         return false;
      }
      enforce_cell_budget("regridding AMR hierarchy");
      const int first_modified_level = regrid_first_modified_level_;
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
      for (int fine_lev = finestLevel(); fine_lev > lbase; --fine_lev) {
         average_down_state_level(fine_lev);
      }
      if (ct_enabled(cfg_)) {
         for (int fine_lev = finestLevel(); fine_lev > lbase; --fine_lev) {
            average_down_face_b_level(fine_lev);
         }
         for (int lev = lbase; lev <= finestLevel(); ++lev) {
            sync_cell_b_from_faces(*state_[lev], face_b_[lev], Geom(lev), false);
         }
      }
      if (whole_hierarchy) {
         refresh_solid_geometry_synced(time, true, preserve_hydro_solid_totals);
      } else if (solid_geometry_enabled(cfg_)) {
         for (int lev = lbase; lev <= finestLevel(); ++lev) {
            refresh_solid_geometry_level(lev, time, false);
         }
         enforce_solid_mask_nesting_from_fine();
      }
      if (preserve_hydro_solid_totals) {
         repair_nonpositive_hydro_fluid_cells_all_levels();
         apply_hydro_valid_conserved_total_correction(pre_regrid_totals);
         if (finestLevel() > 0) {
            average_down_state();
         }
      }
      for (int lev = lbase; lev <= finestLevel(); ++lev) {
         fill_level_boundaries<SubcyclingEnabled>(state_, lev, time);
      }
      for (int lev = lbase + 1; lev <= finestLevel(); ++lev) {
         copy_level_state_to_old(lev);
      }
      for (int lev = lbase; lev <= finestLevel(); ++lev) {
         invalidate_rk4_stage_cache(lev);
      }
      return true;
   }

   int state_fillpatch_bc_type(int component, int dir, bool high_side) const
   {
      const auto is_periodic = Geom(0).isPeriodicArray();
      if (is_periodic[dir]) {
         return amrex::BCType::int_dir;
      }
      return amrex_state_bc_type(side_boundary_kind(cfg_, dir, high_side), component, dir, is_mhd(cfg_));
   }

   int face_fillpatch_bc_type(int face_dir, int boundary_dir, bool high_side) const
   {
      const auto is_periodic = Geom(0).isPeriodicArray();
      if (is_periodic[boundary_dir]) {
         return amrex::BCType::int_dir;
      }
      return amrex_face_b_bc_type(side_boundary_kind(cfg_, boundary_dir, high_side), face_dir, boundary_dir);
   }

   amrex::Vector<amrex::BCRec> make_state_fillpatch_bcs() const
   {
      amrex::Vector<amrex::BCRec> bcs(num_state_components());
      for (int component = 0; component < num_state_components(); ++component) {
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            bcs[component].setLo(dir, state_fillpatch_bc_type(component, dir, false));
            bcs[component].setHi(dir, state_fillpatch_bc_type(component, dir, true));
         }
      }
      return bcs;
   }

   amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> make_face_fillpatch_bcs() const
   {
      amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> bcs;
      for (int face_dir = 0; face_dir < AMREX_SPACEDIM; ++face_dir) {
         bcs[face_dir].resize(1);
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            bcs[face_dir][0].setLo(dir, face_fillpatch_bc_type(face_dir, dir, false));
            bcs[face_dir][0].setHi(dir, face_fillpatch_bc_type(face_dir, dir, true));
         }
      }
      return bcs;
   }

   amrex::Vector<amrex::BCRec> const& fillpatch_bcs() const
   {
      return state_fillpatch_bcs_;
   }

   amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> const& face_fillpatch_bcs() const
   {
      return face_fillpatch_bcs_;
   }

   amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> face_b_array(FaceMagneticField& face_b) const
   {
      amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> result;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         result[dir] = face_b.b[dir].get();
      }
      return result;
   }

   StateFillPatchFunc const& state_fillpatch_func() const
   {
      return state_fillpatch_func_;
   }

   FaceFillPatchFunc const& face_fillpatch_func(int face_dir) const
   {
      return face_fillpatch_func_[face_dir];
   }

   bool compute_post_fillpatch_state_boundary_needed() const
   {
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         if (
            side_boundary_kind(cfg_, dir, false) == BoundaryKind::Diode
            || side_boundary_kind(cfg_, dir, true) == BoundaryKind::Diode
            || side_boundary_kind(cfg_, dir, false) == BoundaryKind::Origin
            || side_boundary_kind(cfg_, dir, true) == BoundaryKind::Origin
            || (
               cfg_.inflow_profile == "uniform_beam"
               && (
                  side_boundary_kind(cfg_, dir, false) == BoundaryKind::Inflow
                  || side_boundary_kind(cfg_, dir, true) == BoundaryKind::Inflow
               )
            )
         ) {
            return true;
         }
      }
      return false;
   }

   void apply_post_fillpatch_state_boundary(
      amrex::MultiFab& state,
      amrex::Geometry const& geom,
      amrex::Real time
   ) const
   {
      if (post_fillpatch_state_boundary_needed_ || patch_union_domain_enabled()) {
         apply_boundary(state, geom, time);
      }
   }

   void fill_mixed_patch_state_ghosts(amrex::MultiFab& state, int lev) const
   {
      if (!mixed_patch_system_enabled()) {
         return;
      }
      fill_mapped_patch_state_ghosts(
         state,
         production_patch_grid_system_for_level(lev),
         production_patch_layout_for_level(lev)
      );
   }

   amrex::Interpolater* selected_state_interpolater() const
   {
      if (cfg_.state_interp == "conservative") {
         return &amrex::cell_cons_interp;
      }
      if (cfg_.state_interp == "protected") {
         return &amrex::protected_interp;
      }
      if (cfg_.state_interp == "quadratic") {
         return &amrex::quadratic_interp;
      }
      if (cfg_.state_interp == "quartic") {
         return &amrex::cell_quartic_interp;
      }
      if (cfg_.state_interp == "quartic-conservative") {
         return &amrex::quartic_interp;
      }
      amrex::Abort("unsupported amr.state_interp");
      return nullptr;
   }

   amrex::Interpolater* state_interpolater() const
   {
      if (mhd_enabled() || mesh_coordinate_kind() == MeshCartesian) {
         return selected_state_interpolater();
      }
      return &state_interpolater_;
   }

   bool level_refinement_ratio_is_anisotropic(int coarse_lev) const
   {
      const amrex::IntVect ratio = refRatio(coarse_lev);
      const int first = ratio[0];
      for (int dir = 1; dir < AMREX_SPACEDIM; ++dir) {
         if (ratio[dir] != first) {
            return true;
         }
      }
      return false;
   }

   void complete_radial_origin_boundary(
      LevelData& data,
      int lev,
      amrex::Real time,
      amrex::Vector<amrex::MultiFab*> const& coarse_data,
      amrex::Vector<amrex::Real> const& coarse_time,
      amrex::Vector<amrex::MultiFab*> const& fine_data,
      amrex::Vector<amrex::Real> const& fine_time
   ) const
   {
      if (lev <= 0 || !data[lev] || !data[lev - 1] || !needs_radial_antipodal_origin_boundary(boundary_data_)) {
         return;
      }

      amrex::MultiFab& fine = *data[lev];
      const amrex::IntVect ng = fine.nGrowVect();
      const int radial_dir = mesh_direction_for_axis(boundary_data_.mesh_axes, MeshAxisR);
      const int phi_dir = mesh_direction_for_axis(boundary_data_.mesh_axes, MeshAxisPhi);
      if (radial_dir < 0 || phi_dir < 0 || ng[radial_dir] <= 0) {
         return;
      }

      validate_origin_boundary_geometry(boundary_data_, Geom(lev));
      const amrex::Box fine_domain = Geom(lev).Domain();
      const int radial_lo = fine_domain.smallEnd(radial_dir);
      const int nphi = fine_domain.length(phi_dir);
      const int phi_hi = fine_domain.bigEnd(phi_dir);
      const int ncomp = fine.nComp();
      amrex::GpuArray<int, MaxStateComponents> origin_parity{};
      for (int q = 0; q < MaxStateComponents; ++q) {
         origin_parity[q] = origin_state_parity(boundary_data_, radial_dir, q);
      }

      bool needs_origin_shell = false;
      for (int box_index = 0; box_index < static_cast<int>(fine.boxArray().size()); ++box_index) {
         const amrex::Box& valid = fine.boxArray()[box_index];
         if (valid.smallEnd(radial_dir) - ng[radial_dir] < radial_lo) {
            needs_origin_shell = true;
            break;
         }
      }
      if (!needs_origin_shell) {
         return;
      }

      amrex::Box destination_shell = fine_domain;
      destination_shell.setSmall(radial_dir, radial_lo - ng[radial_dir]);
      destination_shell.setBig(radial_dir, radial_lo - 1);
      amrex::Box source_shell = fine_domain;
      source_shell.setSmall(radial_dir, radial_lo);
      source_shell.setBig(radial_dir, radial_lo + ng[radial_dir] - 1);
      const amrex::BoxArray destination_box_array(destination_shell);
      const amrex::BoxArray source_box_array(source_shell);
      const amrex::DistributionMapping destination_distribution(destination_box_array);
      amrex::MultiFab source_values(source_box_array, destination_distribution, ncomp, 0);
      if (cfg_.debug_check_finite) {
         source_values.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
      }
      auto const& bcs = fillpatch_bcs();
      const StateFillPatchFunc& boundary_func = state_fillpatch_func();
      for (amrex::MultiFab const* coarse_state : coarse_data) {
         debug_check_reconstruction_input(
            *coarse_state, lev - 1, time, "radial-origin coarse time slice"
         );
      }
      amrex::Vector<amrex::Vector<amrex::MultiFab*>> hierarchy_data(lev + 1);
      amrex::Vector<amrex::Vector<amrex::Real>> hierarchy_times(lev + 1);
      for (int source_lev = 0; source_lev <= lev; ++source_lev) {
         if (source_lev == lev) {
            hierarchy_data[source_lev] = fine_data;
            hierarchy_times[source_lev] = fine_time;
         } else if (source_lev == lev - 1) {
            hierarchy_data[source_lev] = coarse_data;
            hierarchy_times[source_lev] = coarse_time;
         } else if (coarse_time_interval_contains(source_lev, time)) {
            hierarchy_data[source_lev] = {old_state_[source_lev].get(), state_[source_lev].get()};
            hierarchy_times[source_lev] = {level_old_times_[source_lev], level_times_[source_lev]};
         } else {
            hierarchy_data[source_lev] = {state_[source_lev].get()};
            hierarchy_times[source_lev] = {level_times_[source_lev]};
         }
      }
      amrex::Vector<amrex::Geometry> hierarchy_geometry;
      amrex::Vector<StateFillPatchPhysBC> hierarchy_bc;
      amrex::Vector<amrex::IntVect> hierarchy_ratio;
      hierarchy_geometry.reserve(lev + 1);
      hierarchy_bc.reserve(lev + 1);
      hierarchy_ratio.reserve(lev);
      for (int source_lev = 0; source_lev <= lev; ++source_lev) {
         hierarchy_geometry.push_back(Geom(source_lev));
         hierarchy_bc.emplace_back(Geom(source_lev), bcs, boundary_func);
         if (source_lev < lev) {
            hierarchy_ratio.push_back(refRatio(source_lev));
         }
      }
      amrex::FillPatchNLevels(
         source_values,
         lev,
         amrex::IntVect::TheZeroVector(),
         time,
         hierarchy_data,
         hierarchy_times,
         0,
         0,
         ncomp,
         hierarchy_geometry,
         hierarchy_bc,
         0,
         hierarchy_ratio,
         state_interpolater(),
         bcs,
         0
      );
      if (cfg_.debug_check_finite && source_values.contains_nan()) {
         amrex::Abort("radial-origin hierarchy sampling produced nonfinite antipodal source values");
      }
      debug_check_reconstruction_input(
         source_values, lev, time, "radial-origin hierarchy sampling"
      );
      amrex::MultiFab origin_values(destination_box_array, destination_distribution, ncomp, 0);
      if (cfg_.debug_check_finite) {
         origin_values.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
      }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(origin_values, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& radial_ghosts = mfi.tilebox();
         const auto dst = origin_values.array(mfi);
         const auto src = source_values.const_array(mfi);
         amrex::ParallelFor(radial_ghosts, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int q) noexcept {
            int source_i = i;
            int source_j = j;
            int source_k = k;
            if (radial_dir == 0) {
               source_i = mirrored_index(i, radial_lo, radial_lo);
            }
#if (AMREX_SPACEDIM >= 2)
            if (radial_dir == 1) {
               source_j = mirrored_index(j, radial_lo, radial_lo);
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (radial_dir == 2) {
               source_k = mirrored_index(k, radial_lo, radial_lo);
            }
#endif
            if (phi_dir == 0) {
               source_i += nphi / 2;
            }
#if (AMREX_SPACEDIM >= 2)
            if (phi_dir == 1) {
               source_j += nphi / 2;
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (phi_dir == 2) {
               source_k += nphi / 2;
            }
#endif
            if (phi_dir == 0 && source_i > phi_hi) {
               source_i -= nphi;
            }
#if (AMREX_SPACEDIM >= 2)
            if (phi_dir == 1 && source_j > phi_hi) {
               source_j -= nphi;
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (phi_dir == 2 && source_k > phi_hi) {
               source_k -= nphi;
            }
#endif
            dst(i, j, k, q) = static_cast<amrex::Real>(origin_parity[q]) * src(source_i, source_j, source_k, q);
         });
      }
      }
      if (cfg_.debug_check_finite && origin_values.contains_nan()) {
         amrex::Abort("radial-origin parity mapping produced nonfinite ghost values");
      }
      debug_check_reconstruction_input(
         origin_values, lev, time, "radial-origin parity mapping"
      );
      fine.ParallelCopy(
         origin_values,
         0,
         0,
         ncomp,
         amrex::IntVect::TheZeroVector(),
         ng,
         Geom(lev).periodicity()
      );
      debug_check_reconstruction_input(
         fine, lev, time, "radial-origin shell completion"
      );
      if (
         cfg_.debug_check_finite
         && (
            fine.contains_nan(0, ncomp, ng)
            || fine.contains_inf(0, ncomp, ng)
         )
      ) {
         amrex::Abort("radial-origin copy left nonfinite fine-level state or ghost values");
      }
   }

   void fill_new_level_from_coarse_state(int lev, amrex::Real time)
   {
      if (lev == 0) {
         initialize_state(*state_[lev], Geom(lev), cfg_);
         apply_boundary(*state_[lev], Geom(lev), time);
         fill_mixed_patch_state_ghosts(*state_[lev], lev);
         return;
      }

      fill_level_boundaries(state_, lev - 1, time);
      auto const& bcs = fillpatch_bcs();
      const StateFillPatchFunc& boundary_func = state_fillpatch_func();
      StateFillPatchPhysBC coarse_bc(Geom(lev - 1), bcs, boundary_func);
      StateFillPatchPhysBC fine_bc(Geom(lev), bcs, boundary_func);
      amrex::InterpFromCoarseLevel(
         *state_[lev],
         state_[lev]->nGrowVect(),
         time,
         *state_[lev - 1],
         0,
         0,
         num_state_components(),
         Geom(lev - 1),
         Geom(lev),
         coarse_bc,
         0,
         fine_bc,
         0,
         refRatio(lev - 1),
         state_interpolater(),
         bcs,
         0
      );
      project_metric_conservative_prolongation(lev, *state_[lev - 1], *state_[lev]);
      apply_post_fillpatch_state_boundary(*state_[lev], Geom(lev), time);
      amrex::Vector<amrex::MultiFab*> origin_coarse_data(1, state_[lev - 1].get());
      amrex::Vector<amrex::MultiFab*> origin_fine_data(1, state_[lev].get());
      amrex::Vector<amrex::Real> origin_time(1, time);
      complete_radial_origin_boundary(
         state_, lev, time, origin_coarse_data, origin_time, origin_fine_data, origin_time
      );
      fill_mixed_patch_state_ghosts(*state_[lev], lev);
   }

   void fill_remade_level_state(int lev, amrex::Real time, amrex::MultiFab& old_state)
   {
      if (lev == 0) {
         state_[lev]->ParallelCopy(old_state, 0, 0, num_state_components(), 0, 0);
         apply_boundary(*state_[lev], Geom(lev), time);
         fill_mixed_patch_state_ghosts(*state_[lev], lev);
         return;
      }

      fill_level_boundaries(state_, lev - 1, time);
      apply_boundary(old_state, Geom(lev), time);
      amrex::Vector<amrex::MultiFab*> coarse_data(1, state_[lev - 1].get());
      amrex::Vector<amrex::MultiFab*> fine_data(1, &old_state);
      amrex::Vector<amrex::Real> coarse_time(1, time);
      amrex::Vector<amrex::Real> fine_time(1, time);
      auto const& bcs = fillpatch_bcs();
      const StateFillPatchFunc& boundary_func = state_fillpatch_func();
      StateFillPatchPhysBC coarse_bc(Geom(lev - 1), bcs, boundary_func);
      StateFillPatchPhysBC fine_bc(Geom(lev), bcs, boundary_func);
      amrex::FillPatchTwoLevels(
         *state_[lev],
         state_[lev]->nGrowVect(),
         time,
         coarse_data,
         coarse_time,
         fine_data,
         fine_time,
         0,
         0,
         num_state_components(),
         Geom(lev - 1),
         Geom(lev),
         coarse_bc,
         0,
         fine_bc,
         0,
         refRatio(lev - 1),
         state_interpolater(),
         bcs,
         0
      );
      project_metric_conservative_prolongation(lev, *state_[lev - 1], *state_[lev]);
      apply_post_fillpatch_state_boundary(*state_[lev], Geom(lev), time);
      amrex::Vector<amrex::MultiFab*> origin_fine_data(1, state_[lev].get());
      amrex::Vector<amrex::Real> origin_fine_time(1, time);
      complete_radial_origin_boundary(
         state_, lev, time, coarse_data, coarse_time, origin_fine_data, origin_fine_time
      );
      fill_mixed_patch_state_ghosts(*state_[lev], lev);
   }

   bool coarse_time_interval_contains(int coarse_lev, amrex::Real time) const
   {
      const amrex::Real old_time = level_old_times_[coarse_lev];
      const amrex::Real new_time = level_times_[coarse_lev];
      if (!(new_time > old_time) || !old_state_[coarse_lev]) {
         return false;
      }
      const amrex::Real scale = amrex::max(1.0, amrex::max(amrex::Math::abs(old_time), amrex::Math::abs(new_time)));
      const amrex::Real eps = 64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
      return time >= old_time - eps && time <= new_time + eps;
   }

   amrex::Real clamp_to_coarse_time_interval(int coarse_lev, amrex::Real time) const
   {
      return amrex::min(amrex::max(time, level_old_times_[coarse_lev]), level_times_[coarse_lev]);
   }

   bool rk4_stage_data_ready(int coarse_lev) const
   {
      if (!rk4_coarse_stage_state_[coarse_lev] || !old_state_[coarse_lev]) {
         return false;
      }
      for (auto const& stage : rk4_rhs_stage_) {
         if (!stage[coarse_lev]) {
            return false;
         }
      }
      return level_dts_[coarse_lev] > 0.0;
   }

   void fill_rk4_coarse_stage_state(int coarse_lev, amrex::Real fine_step_start, int rk_stage)
   {
      if (!rk4_stage_data_ready(coarse_lev)) {
         amrex::Abort("RK4 subcycling boundary fill requires stored coarse RK stages");
      }
      const amrex::Real dtc = level_dts_[coarse_lev];
      if (
         rk4_stage_cache_stage_[coarse_lev] == rk_stage
         && nearly_equal_time(rk4_stage_cache_fine_start_[coarse_lev], fine_step_start)
         && nearly_equal_time(rk4_stage_cache_time_[coarse_lev], level_times_[coarse_lev])
         && nearly_equal_time(rk4_stage_cache_dt_[coarse_lev], dtc)
      ) {
         return;
      }

      amrex::Real xsi = (fine_step_start - level_old_times_[coarse_lev]) / dtc;
      xsi = amrex::min(amrex::max(xsi, 0.0), 1.0);
      const amrex::Real r = level_dts_[coarse_lev + 1] / dtc;

      const amrex::Real xsi2 = xsi * xsi;
      const amrex::Real xsi3 = xsi2 * xsi;
      const amrex::Real b1 = xsi - 1.5 * xsi2 + (2.0 / 3.0) * xsi3;
      const amrex::Real b2 = xsi2 - (2.0 / 3.0) * xsi3;
      const amrex::Real b3 = b2;
      const amrex::Real b4 = -0.5 * xsi2 + (2.0 / 3.0) * xsi3;
      const amrex::Real c1 = 1.0 - 3.0 * xsi + 2.0 * xsi2;
      const amrex::Real c2 = 2.0 * xsi - 2.0 * xsi2;
      const amrex::Real c3 = c2;
      const amrex::Real c4 = -xsi + 2.0 * xsi2;
      const amrex::Real d1 = -3.0 + 4.0 * xsi;
      const amrex::Real d2 = 2.0 - 4.0 * xsi;
      const amrex::Real d3 = d2;
      const amrex::Real d4 = -1.0 + 4.0 * xsi;
      constexpr amrex::Real e1 = 4.0;
      constexpr amrex::Real e2 = -4.0;
      constexpr amrex::Real e3 = -4.0;
      constexpr amrex::Real e4 = 4.0;
      const int ncomp = num_state_components();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(*rk4_coarse_stage_state_[coarse_lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto out = rk4_coarse_stage_state_[coarse_lev]->array(mfi);
         const auto u0 = old_state_[coarse_lev]->const_array(mfi);
         const auto k1 = rk4_rhs_stage_[0][coarse_lev]->const_array(mfi);
         const auto k2 = rk4_rhs_stage_[1][coarse_lev]->const_array(mfi);
         const auto k3 = rk4_rhs_stage_[2][coarse_lev]->const_array(mfi);
         const auto k4 = rk4_rhs_stage_[3][coarse_lev]->const_array(mfi);
         amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            const amrex::Real kk1 = k1(i, j, k, n);
            const amrex::Real kk2 = k2(i, j, k, n);
            const amrex::Real kk3 = k3(i, j, k, n);
            const amrex::Real kk4 = k4(i, j, k, n);
            const amrex::Real uu = b1 * kk1 + b2 * kk2 + b3 * kk3 + b4 * kk4;
            amrex::Real value = u0(i, j, k, n) + dtc * uu;
            if (rk_stage == 2) {
               const amrex::Real ut = c1 * kk1 + c2 * kk2 + c3 * kk3 + c4 * kk4;
               value = u0(i, j, k, n) + dtc * (uu + 0.5 * r * ut);
            } else if (rk_stage == 3 || rk_stage == 4) {
               const amrex::Real ut = c1 * kk1 + c2 * kk2 + c3 * kk3 + c4 * kk4;
               const amrex::Real utt = d1 * kk1 + d2 * kk2 + d3 * kk3 + d4 * kk4;
               const amrex::Real uttt = e1 * kk1 + e2 * kk2 + e3 * kk3 + e4 * kk4;
               const amrex::Real r2 = r * r;
               const amrex::Real r3 = r2 * r;
               const amrex::Real at = (rk_stage == 3) ? 0.5 * r : r;
               const amrex::Real att = (rk_stage == 3) ? 0.25 * r2 : 0.5 * r2;
               const amrex::Real attt = (rk_stage == 3) ? 0.0625 * r3 : 0.125 * r3;
               const amrex::Real ak = (rk_stage == 3) ? -4.0 : 4.0;
               value = u0(i, j, k, n) + dtc * (uu + at * ut + att * utt + attt * (uttt + ak * (kk3 - kk2)));
            }
            out(i, j, k, n) = value;
         });
      }
      }
      rk4_stage_cache_stage_[coarse_lev] = rk_stage;
      rk4_stage_cache_fine_start_[coarse_lev] = fine_step_start;
      rk4_stage_cache_time_[coarse_lev] = level_times_[coarse_lev];
      rk4_stage_cache_dt_[coarse_lev] = dtc;
   }

      template <bool SubcyclingEnabled = false>
      void fill_level_boundaries(LevelData& data, int lev, amrex::Real time, amrex::Real fine_step_start = -1.0, int rk_stage = 0)
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
            coarse_data.push_back(rk4_coarse_stage_state_[lev - 1].get());
            coarse_time.push_back(time);
         } else if (coarse_time_interval_contains(lev - 1, time)) {
            coarse_data.push_back(old_state_[lev - 1].get());
            coarse_data.push_back(state_[lev - 1].get());
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
      amrex::FillPatchTwoLevels(
         *data[lev],
         data[lev]->nGrowVect(),
         time,
         coarse_data,
         coarse_time,
         fine_data,
         fine_time,
         0,
         0,
         num_state_components(),
         Geom(lev - 1),
         Geom(lev),
         coarse_bc,
         0,
         fine_bc,
         0,
         refRatio(lev - 1),
         state_interpolater(),
         bcs,
         0
         );
         apply_post_fillpatch_state_boundary(*data[lev], Geom(lev), time);
         complete_radial_origin_boundary(
            data, lev, time, coarse_data, coarse_time, fine_data, fine_time
         );
         fill_mixed_patch_state_ghosts(*data[lev], lev);
      }

   template <bool SubcyclingEnabled = false>
   void fill_level_boundaries_range(
      LevelData& data,
      int lev_begin,
      int lev_end,
      amrex::Real time,
      amrex::Real fine_step_start = -1.0,
      int rk_stage = 0
   )
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         fill_level_boundaries<SubcyclingEnabled>(data, lev, time, fine_step_start, rk_stage);
      }
   }

   template <bool SubcyclingEnabled = false>
   void fill_all_level_boundaries(LevelData& data, amrex::Real time)
   {
      fill_level_boundaries_range<SubcyclingEnabled>(data, 0, finestLevel() + 1, time);
   }

   void fill_new_face_b_from_coarse(int lev, amrex::Real time)
   {
      if (!ct_enabled(cfg_)) {
         return;
      }
      apply_boundary(*state_[lev], Geom(lev), time);
      fill_face_b_from_cell_b(face_b_[lev], *state_[lev]);
      apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
      if (lev == 0) {
         return;
      }

#if (AMREX_SPACEDIM > 1)
      apply_face_b_boundary(face_b_[lev - 1], Geom(lev - 1), boundary_data_);
      FaceMagneticField dst_fillpatch;
      FaceMagneticField coarse_fillpatch;
      FaceMagneticField* dst_face = &face_b_[lev];
      FaceMagneticField* coarse_face = &face_b_[lev - 1];
      const bool interpolate_face_fillpatch = mesh_is_curvilinear();
      if (interpolate_face_fillpatch) {
         allocate_face_magnetic_field_like(dst_fillpatch, face_b_[lev]);
         allocate_face_magnetic_field_like(coarse_fillpatch, face_b_[lev - 1]);
         fill_face_fillpatch_from_face_b(dst_fillpatch, face_b_[lev], Geom(lev), cfg_);
         fill_face_fillpatch_from_face_b(coarse_fillpatch, face_b_[lev - 1], Geom(lev - 1), cfg_);
         dst_face = &dst_fillpatch;
         coarse_face = &coarse_fillpatch;
      }
      auto dst = face_b_array(*dst_face);
      auto coarse = face_b_array(*coarse_face);
      auto const& bcs = face_fillpatch_bcs();
      amrex::Array<FaceFillPatchPhysBC, AMREX_SPACEDIM> coarse_bc;
      amrex::Array<FaceFillPatchPhysBC, AMREX_SPACEDIM> fine_bc;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         const FaceFillPatchFunc& boundary_func = face_fillpatch_func(dir);
         coarse_bc[dir].define(Geom(lev - 1), bcs[dir], boundary_func);
         fine_bc[dir].define(Geom(lev), bcs[dir], boundary_func);
      }
      amrex::InterpFromCoarseLevel(
         dst,
         amrex::IntVect(AMREX_D_DECL(cfg_.ng, cfg_.ng, cfg_.ng)),
         time,
         coarse,
         0,
         0,
         1,
         Geom(lev - 1),
         Geom(lev),
         coarse_bc,
         0,
         fine_bc,
         0,
         refRatio(lev - 1),
         &amrex::face_divfree_interp,
         bcs,
         0
      );
      if (interpolate_face_fillpatch) {
         fill_face_b_from_face_fillpatch(face_b_[lev], dst_fillpatch, Geom(lev), cfg_);
      }
      apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
#else
      (void)time;
      fill_face_b_from_cell_b(face_b_[lev], *state_[lev]);
      apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
#endif
   }

   void fill_remade_face_b(int lev, amrex::Real time, FaceMagneticField& old_face_b)
   {
      if (!ct_enabled(cfg_)) {
         return;
      }
      apply_boundary(*state_[lev], Geom(lev), time);
      fill_face_b_from_cell_b(face_b_[lev], *state_[lev]);
      apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
      if (lev == 0 && old_face_b.b[0]) {
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            face_b_[lev].b[dir]->ParallelCopy(
               *old_face_b.b[dir],
               0,
               0,
               1,
               0,
               0,
               Geom(lev).periodicity()
            );
         }
         apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
         return;
      }
      if (lev == 0 || !old_face_b.b[0]) {
         fill_new_face_b_from_coarse(lev, time);
         return;
      }

#if (AMREX_SPACEDIM > 1)
      apply_face_b_boundary(face_b_[lev - 1], Geom(lev - 1), boundary_data_);
      apply_face_b_boundary(old_face_b, Geom(lev), boundary_data_);
      FaceMagneticField dst_fillpatch;
      FaceMagneticField coarse_fillpatch;
      FaceMagneticField fine_fillpatch;
      FaceMagneticField* dst_face = &face_b_[lev];
      FaceMagneticField* coarse_face = &face_b_[lev - 1];
      FaceMagneticField* fine_face = &old_face_b;
      const bool interpolate_face_fillpatch = mesh_is_curvilinear();
      if (interpolate_face_fillpatch) {
         allocate_face_magnetic_field_like(dst_fillpatch, face_b_[lev]);
         allocate_face_magnetic_field_like(coarse_fillpatch, face_b_[lev - 1]);
         allocate_face_magnetic_field_like(fine_fillpatch, old_face_b);
         fill_face_fillpatch_from_face_b(dst_fillpatch, face_b_[lev], Geom(lev), cfg_);
         fill_face_fillpatch_from_face_b(coarse_fillpatch, face_b_[lev - 1], Geom(lev - 1), cfg_);
         fill_face_fillpatch_from_face_b(fine_fillpatch, old_face_b, Geom(lev), cfg_);
         dst_face = &dst_fillpatch;
         coarse_face = &coarse_fillpatch;
         fine_face = &fine_fillpatch;
      }
      auto dst = face_b_array(*dst_face);
      auto coarse = face_b_array(*coarse_face);
      auto fine = face_b_array(*fine_face);
      amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> coarse_data(1, coarse);
      amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> fine_data(1, fine);
      amrex::Vector<amrex::Real> coarse_time(1, time);
      amrex::Vector<amrex::Real> fine_time(1, time);
      auto const& bcs = face_fillpatch_bcs();
      amrex::Array<FaceFillPatchPhysBC, AMREX_SPACEDIM> coarse_bc;
      amrex::Array<FaceFillPatchPhysBC, AMREX_SPACEDIM> fine_bc;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         const FaceFillPatchFunc& boundary_func = face_fillpatch_func(dir);
         coarse_bc[dir].define(Geom(lev - 1), bcs[dir], boundary_func);
         fine_bc[dir].define(Geom(lev), bcs[dir], boundary_func);
      }
      amrex::FillPatchTwoLevels(
         dst,
         amrex::IntVect(AMREX_D_DECL(cfg_.ng, cfg_.ng, cfg_.ng)),
         time,
         coarse_data,
         coarse_time,
         fine_data,
         fine_time,
         0,
         0,
         1,
         Geom(lev - 1),
         Geom(lev),
         coarse_bc,
         0,
         fine_bc,
         0,
         refRatio(lev - 1),
         &amrex::face_divfree_interp,
         bcs,
         0
      );
      if (interpolate_face_fillpatch) {
         fill_face_b_from_face_fillpatch(face_b_[lev], dst_fillpatch, Geom(lev), cfg_);
      }
      apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
#else
      (void)time;
      fill_face_b_from_cell_b(face_b_[lev], *state_[lev]);
      apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
#endif
   }

   void initialize_ct_fields_from_coarse(int lev, amrex::Real time)
   {
      if (!ct_enabled(cfg_)) {
         return;
      }
      if (lev == 0) {
         initialize_ct_fields(lev, time);
         return;
      }
      fill_new_face_b_from_coarse(lev, time);
      sync_cell_b_from_faces(*state_[lev], face_b_[lev], Geom(lev), time == 0.0);
      apply_boundary(*state_[lev], Geom(lev), time);
   }

   void initialize_remade_ct_fields(int lev, amrex::Real time, FaceMagneticField& old_face_b)
   {
      if (!ct_enabled(cfg_)) {
         return;
      }
      fill_remade_face_b(lev, time, old_face_b);
      sync_cell_b_from_faces(*state_[lev], face_b_[lev], Geom(lev), time == 0.0);
      apply_boundary(*state_[lev], Geom(lev), time);
   }
