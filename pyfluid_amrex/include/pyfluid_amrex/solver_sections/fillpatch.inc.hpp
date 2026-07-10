// Included inside pyfluid_amrex::AmrSolver private section.
// State and face-centered magnetic-field FillPatch, physical boundary, and average-down helpers.

   template <bool SubcyclingEnabled>
   bool should_regrid(int step, bool reached_final_time) const
   {
      int base_step = step;
      if constexpr (SubcyclingEnabled) {
         const int finest_factor = level_time_factor(finestLevel());
         if (finest_factor > 1) {
            if (step % finest_factor != 0) {
               return false;
            }
            base_step = step / finest_factor;
         }
      }
      return cfg_.regrid_int > 0
         && maxLevel() > 0
         && base_step % cfg_.regrid_int == 0
         && step != cfg_.max_step
         && !reached_final_time;
   }

   bool needs_fresh_state_boundaries_after_step(bool write_scalars, bool write_aux_gfs) const
   {
      const bool scalar_divb_uses_cell_centered_b = write_scalars && is_mhd(cfg_) && !ct_enabled(cfg_);
      const bool aux_fields_use_derivatives = write_aux_gfs && aux_gfs_enabled(cfg_);
      return scalar_divb_uses_cell_centered_b || aux_fields_use_derivatives;
   }

   template <bool SubcyclingEnabled>
   void regrid_hierarchy(amrex::Real time, int step)
   {
      fill_all_level_boundaries<SubcyclingEnabled>(state_, time);
      if (ct_enabled(cfg_)) {
         apply_all_face_b_boundaries();
      }
      invalidate_rk4_stage_caches();
      int lbase = 0;
      if (cfg_.initial_refine_fixed_levels > 0) {
         lbase = std::min(cfg_.initial_refine_fixed_levels, finestLevel());
         if (lbase >= maxLevel()) {
            return;
         }
      }
      if (automatic_refinement_tensor()) {
         regrid_adaptive_tensor(lbase, time, false);
      } else {
         regrid(lbase, time, false);
      }
      enforce_cell_budget("regridding AMR hierarchy");
      define_flux_registers();
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         level_steps_[lev] = SubcyclingEnabled ? synced_level_step(lev, step) : step;
         level_times_[lev] = time;
         level_old_times_[lev] = time;
         level_dts_[lev] = SubcyclingEnabled && !local_cfl_subcycling()
            ? (level_dts_[0] / static_cast<amrex::Real>(level_time_factor(lev)))
            : level_dts_[0];
      }
      average_down_state();
      if (ct_enabled(cfg_)) {
         if (finestLevel() > 0) {
            average_down_face_b();
            sync_all_cell_b_from_faces(state_, true);
         }
      }
      refresh_solid_geometry_synced(time, true, !is_mhd(cfg_));
      fill_all_level_boundaries<SubcyclingEnabled>(state_, time);
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         copy_level_state_to_old(lev);
      }
      invalidate_rk4_stage_caches();
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

   void apply_post_fillpatch_state_boundary(amrex::MultiFab& state, amrex::Geometry const& geom) const
   {
      if (post_fillpatch_state_boundary_needed_ || patch_union_domain_enabled()) {
         apply_boundary(state, geom);
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

   void fill_nonfinite_physical_state_boundary(amrex::MultiFab& state, amrex::Geometry const& geom) const
   {
      const auto is_periodic = geom.isPeriodicArray();
      bool all_periodic = true;
      for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
         all_periodic = all_periodic && is_periodic[dim];
      }
      if (all_periodic) {
         return;
      }

      const amrex::Box domain = geom.Domain();
      const auto lo = amrex::lbound(domain);
      const auto hi = amrex::ubound(domain);
      const int ng = state.nGrow();
      const int ncomp = num_state_components();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto s = state.array(mfi);
         amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            if (amrex::Math::isfinite(s(i, j, k, n))) {
               return;
            }

            int ii = i;
            int jj = j;
            int kk = k;
            if (!is_periodic[0]) {
               if (i <= lo.x + ng - 1) {
                  ii = (lo.x + ng <= hi.x) ? lo.x + ng : hi.x;
               } else if (i >= hi.x - ng + 1) {
                  ii = (hi.x - ng >= lo.x) ? hi.x - ng : lo.x;
               }
            }
#if (AMREX_SPACEDIM >= 2)
            if (!is_periodic[1]) {
               if (j <= lo.y + ng - 1) {
                  jj = (lo.y + ng <= hi.y) ? lo.y + ng : hi.y;
               } else if (j >= hi.y - ng + 1) {
                  jj = (hi.y - ng >= lo.y) ? hi.y - ng : lo.y;
               }
            }
#else
            (void)jj;
#endif
#if (AMREX_SPACEDIM == 3)
            if (!is_periodic[2]) {
               if (k <= lo.z + ng - 1) {
                  kk = (lo.z + ng <= hi.z) ? lo.z + ng : hi.z;
               } else if (k >= hi.z - ng + 1) {
                  kk = (hi.z - ng >= lo.z) ? hi.z - ng : lo.z;
               }
            }
#else
            (void)kk;
#endif
            s(i, j, k, n) = s(ii, jj, kk, n);
         });
      }
      }
   }

   amrex::Interpolater* state_interpolater() const
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
         return &amrex::quartic_interp;
      }
      if (cfg_.state_interp == "quartic-conservative") {
         return &amrex::cell_quartic_interp;
      }
      return &amrex::cell_quartic_interp;
   }

   amrex::Interpolater* state_valid_regrid_interpolater() const
   {
      // Valid fine cells are finite-volume cell averages.  Regrid/prolongation
      // must preserve their cell integral; high-order FillPatch choices are
      // only for ghost data used by the evolution stencils.
      return &amrex::cell_cons_interp;
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

   void fill_fine_from_coarse_piecewise_constant(
      amrex::MultiFab& fine,
      amrex::MultiFab const& coarse,
      int coarse_lev,
      int num_components
   ) const
   {
      const amrex::IntVect ratio = refRatio(coarse_lev);
      fine.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
      amrex::BoxArray coarse_footprint = fine.boxArray();
      coarse_footprint.grow(fine.nGrowVect());
      coarse_footprint.coarsen(ratio);
      amrex::MultiFab coarse_on_fine(
         coarse_footprint,
         fine.DistributionMap(),
         num_components,
         0
      );
      coarse_on_fine.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
      coarse_on_fine.ParallelCopy(
         coarse,
         0,
         0,
         num_components,
         coarse.nGrowVect(),
         amrex::IntVect::TheZeroVector(),
         Geom(coarse_lev).periodicity()
      );
      const int ratio_0 = ratio[0];
#if (AMREX_SPACEDIM >= 2)
      const int ratio_1 = ratio[1];
#else
      constexpr int ratio_1 = 1;
#endif
#if (AMREX_SPACEDIM == 3)
      const int ratio_2 = ratio[2];
#else
      constexpr int ratio_2 = 1;
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(fine, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.growntilebox(fine.nGrowVect());
         const amrex::Box& coarse_box = coarse_on_fine[mfi].box();
         const auto coarse_lo = amrex::lbound(coarse_box);
         const auto coarse_hi = amrex::ubound(coarse_box);
         const auto src = coarse_on_fine.const_array(mfi);
         const auto dst = fine.array(mfi);
         amrex::ParallelFor(bx, num_components, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            const int coarse_i = amrex::coarsen(i, ratio_0);
            const int coarse_j = amrex::coarsen(j, ratio_1);
            const int coarse_k = amrex::coarsen(k, ratio_2);
            if (
               coarse_i < coarse_lo.x || coarse_i > coarse_hi.x
#if (AMREX_SPACEDIM >= 2)
               || coarse_j < coarse_lo.y || coarse_j > coarse_hi.y
#endif
#if (AMREX_SPACEDIM == 3)
               || coarse_k < coarse_lo.z || coarse_k > coarse_hi.z
#endif
            ) {
               return;
            }
            dst(i, j, k, n) = src(coarse_i, coarse_j, coarse_k, n);
         });
      }
      }
   }

   bool piecewise_constant_regrid_transfer_enabled(int coarse_lev) const
   {
      return mesh_is_curvilinear()
         || mixed_patch_system_enabled()
         || level_refinement_ratio_is_anisotropic(coarse_lev)
         || cfg_.metric_regrid == "always";
   }

   amrex::IntVect cumulative_refinement_ratio_to_level(int lev) const
   {
      amrex::IntVect ratio(AMREX_D_DECL(1, 1, 1));
      for (int coarse_lev = 0; coarse_lev < lev; ++coarse_lev) {
         ratio *= refRatio(coarse_lev);
      }
      return ratio;
   }

   void complete_radial_origin_boundary_from_base(LevelData& data, int lev) const
   {
      if (lev <= 0 || !data[lev] || !data[0] || !needs_radial_antipodal_origin_boundary(boundary_data_)) {
         return;
      }

      amrex::MultiFab& fine = *data[lev];
      amrex::MultiFab const& base = *data[0];
      const amrex::IntVect ng = fine.nGrowVect();
      const int radial_dir = mesh_direction_for_axis(boundary_data_.mesh_axes, MeshAxisR);
      const int phi_dir = mesh_direction_for_axis(boundary_data_.mesh_axes, MeshAxisPhi);
      if (radial_dir < 0 || phi_dir < 0 || ng[radial_dir] <= 0) {
         return;
      }

      validate_origin_boundary_geometry(boundary_data_, Geom(lev));
      const amrex::Box fine_domain = Geom(lev).Domain();
      const auto fine_lo = amrex::lbound(fine_domain);
      const auto fine_hi = amrex::ubound(fine_domain);
      const int nphi = fine_domain.length(phi_dir);
      const int ncomp = fine.nComp();
      const amrex::IntVect base_ratio = cumulative_refinement_ratio_to_level(lev);
      amrex::GpuArray<int, MaxStateComponents> origin_parity{};
      for (int q = 0; q < MaxStateComponents; ++q) {
         origin_parity[q] = origin_state_parity(boundary_data_, radial_dir, q);
      }

      amrex::MultiFab same_level_antipodal(fine.boxArray(), fine.DistributionMap(), ncomp, ng);
      same_level_antipodal.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
      amrex::IntVect antipodal_offset(AMREX_D_DECL(0, 0, 0));
      antipodal_offset.setVal(phi_dir, -nphi / 2);
      same_level_antipodal.ParallelCopy(
         fine,
         0,
         0,
         ncomp,
         ng,
         ng,
         antipodal_offset,
         Geom(lev).periodicity()
      );

      amrex::BoxArray base_source_boxes = fine.boxArray();
      base_source_boxes.grow(ng);
      base_source_boxes.shift(phi_dir, nphi / 2);
      base_source_boxes.coarsen(base_ratio);
      amrex::MultiFab base_source(base_source_boxes, fine.DistributionMap(), ncomp, 0);
      base_source.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
      base_source.ParallelCopy(
         base,
         0,
         0,
         ncomp,
         base.nGrowVect(),
         amrex::IntVect::TheZeroVector(),
         Geom(0).periodicity()
      );

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(fine); mfi.isValid(); ++mfi) {
         amrex::Box radial_ghosts = fine[mfi].box();
         radial_ghosts.setBig(radial_dir, fine_domain.smallEnd(radial_dir) - 1);
         if (!radial_ghosts.ok()) {
            continue;
         }
         const amrex::Box& base_source_box = base_source[mfi].box();
         const auto base_source_lo = amrex::lbound(base_source_box);
         const auto base_source_hi = amrex::ubound(base_source_box);
         const auto dst = fine.array(mfi);
         const auto same = same_level_antipodal.const_array(mfi);
         const auto coarse = base_source.const_array(mfi);
         amrex::ParallelFor(radial_ghosts, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int q) noexcept {
            int source_i = i;
            int source_j = j;
            int source_k = k;
            if (radial_dir == 0) {
               source_i = mirrored_index(i, fine_lo.x, fine_hi.x);
            }
#if (AMREX_SPACEDIM >= 2)
            if (radial_dir == 1) {
               source_j = mirrored_index(j, fine_lo.y, fine_hi.y);
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (radial_dir == 2) {
               source_k = mirrored_index(k, fine_lo.z, fine_hi.z);
            }
#endif
            const amrex::Real same_value = same(source_i, source_j, source_k, q);
            if (amrex::Math::isfinite(same_value)) {
               dst(i, j, k, q) = static_cast<amrex::Real>(origin_parity[q]) * same_value;
               return;
            }

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
            const int coarse_i = amrex::coarsen(source_i, base_ratio[0]);
#if (AMREX_SPACEDIM >= 2)
            const int coarse_j = amrex::coarsen(source_j, base_ratio[1]);
#else
            const int coarse_j = 0;
#endif
#if (AMREX_SPACEDIM == 3)
            const int coarse_k = amrex::coarsen(source_k, base_ratio[2]);
#else
            const int coarse_k = 0;
#endif
            if (
               coarse_i < base_source_lo.x || coarse_i > base_source_hi.x
#if (AMREX_SPACEDIM >= 2)
               || coarse_j < base_source_lo.y || coarse_j > base_source_hi.y
#endif
#if (AMREX_SPACEDIM == 3)
               || coarse_k < base_source_lo.z || coarse_k > base_source_hi.z
#endif
            ) {
               return;
            }
            const amrex::Real value = coarse(coarse_i, coarse_j, coarse_k, q);
            dst(i, j, k, q) = static_cast<amrex::Real>(origin_parity[q]) * value;
         });
      }
      }
   }

   void fill_new_level_from_coarse_state_piecewise_constant(int lev, amrex::Real time)
   {
      fill_level_boundaries(state_, lev - 1, time);
      fill_fine_from_coarse_piecewise_constant(
         *state_[lev],
         *state_[lev - 1],
         lev - 1,
         num_state_components()
      );
   }

   void fill_new_level_from_coarse_state(int lev, amrex::Real time)
   {
      if (lev == 0) {
         initialize_state(*state_[lev], Geom(lev), cfg_);
         fill_nonfinite_physical_state_boundary(*state_[lev], Geom(lev));
         apply_boundary(*state_[lev], Geom(lev));
         fill_mixed_patch_state_ghosts(*state_[lev], lev);
         return;
      }

      if (piecewise_constant_regrid_transfer_enabled(lev - 1)) {
         fill_new_level_from_coarse_state_piecewise_constant(lev, time);
         fill_nonfinite_physical_state_boundary(*state_[lev], Geom(lev));
         apply_post_fillpatch_state_boundary(*state_[lev], Geom(lev));
         complete_radial_origin_boundary_from_base(state_, lev);
         fill_mixed_patch_state_ghosts(*state_[lev], lev);
         return;
      }

      fill_level_boundaries(state_, lev - 1, time);
      auto const& bcs = fillpatch_bcs();
      const StateFillPatchFunc& boundary_func = state_fillpatch_func();
      StateFillPatchPhysBC coarse_bc(Geom(lev - 1), bcs, boundary_func);
      StateFillPatchPhysBC fine_bc(Geom(lev), bcs, boundary_func);
      if (level_refinement_ratio_is_anisotropic(lev - 1)) {
         fill_fine_from_coarse_piecewise_constant(
            *state_[lev],
            *state_[lev - 1],
            lev - 1,
            num_state_components()
         );
      } else {
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
            state_valid_regrid_interpolater(),
            bcs,
            0
         );
      }
      fill_nonfinite_physical_state_boundary(*state_[lev], Geom(lev));
      apply_post_fillpatch_state_boundary(*state_[lev], Geom(lev));
      complete_radial_origin_boundary_from_base(state_, lev);
      fill_mixed_patch_state_ghosts(*state_[lev], lev);
   }

   void fill_remade_level_state(int lev, amrex::Real time, amrex::MultiFab& old_state)
   {
      if (lev == 0) {
         state_[lev]->ParallelCopy(old_state, 0, 0, num_state_components(), 0, 0);
         fill_nonfinite_physical_state_boundary(*state_[lev], Geom(lev));
         apply_boundary(*state_[lev], Geom(lev));
         fill_mixed_patch_state_ghosts(*state_[lev], lev);
         return;
      }

      if (piecewise_constant_regrid_transfer_enabled(lev - 1)) {
         fill_new_level_from_coarse_state(lev, time);
         apply_boundary(old_state, Geom(lev));
         state_[lev]->ParallelCopy(
            old_state,
            0,
            0,
            num_state_components(),
            amrex::IntVect::TheZeroVector(),
            amrex::IntVect::TheZeroVector(),
            Geom(lev).periodicity()
         );
         fill_level_boundaries(state_, lev, time);
         fill_nonfinite_physical_state_boundary(*state_[lev], Geom(lev));
         apply_post_fillpatch_state_boundary(*state_[lev], Geom(lev));
         complete_radial_origin_boundary_from_base(state_, lev);
         fill_mixed_patch_state_ghosts(*state_[lev], lev);
         return;
      }

      fill_level_boundaries(state_, lev - 1, time);
      apply_boundary(old_state, Geom(lev));
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
         state_valid_regrid_interpolater(),
         bcs,
         0
      );
      fill_nonfinite_physical_state_boundary(*state_[lev], Geom(lev));
      apply_post_fillpatch_state_boundary(*state_[lev], Geom(lev));
      complete_radial_origin_boundary_from_base(state_, lev);
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
            apply_boundary(*data[0], Geom(0));
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
      if constexpr (!SubcyclingEnabled) {
         if (piecewise_constant_regrid_transfer_enabled(lev - 1)) {
            fill_level_boundaries<SubcyclingEnabled>(data, lev - 1, time, fine_step_start, rk_stage);
            amrex::MultiFab fine_valid(
               data[lev]->boxArray(),
               data[lev]->DistributionMap(),
               num_state_components(),
               0
            );
            fine_valid.ParallelCopy(
               *data[lev],
               0,
               0,
               num_state_components(),
               amrex::IntVect::TheZeroVector(),
               amrex::IntVect::TheZeroVector(),
               Geom(lev).periodicity()
            );
            fill_fine_from_coarse_piecewise_constant(
               *data[lev],
               *coarse_data.front(),
               lev - 1,
               num_state_components()
            );
            data[lev]->ParallelCopy(
               fine_valid,
               0,
               0,
               num_state_components(),
               amrex::IntVect::TheZeroVector(),
               amrex::IntVect::TheZeroVector(),
               Geom(lev).periodicity()
            );
            data[lev]->FillBoundary(Geom(lev).periodicity());
            apply_boundary(*data[lev], Geom(lev));
            complete_radial_origin_boundary_from_base(data, lev);
            fill_mixed_patch_state_ghosts(*data[lev], lev);
            return;
         }
      }
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
         apply_post_fillpatch_state_boundary(*data[lev], Geom(lev));
         complete_radial_origin_boundary_from_base(data, lev);
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
      apply_boundary(*state_[lev], Geom(lev));
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
      apply_boundary(*state_[lev], Geom(lev));
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
      sync_cell_b_from_faces(*state_[lev], face_b_[lev], Geom(lev), true);
      apply_boundary(*state_[lev], Geom(lev));
   }

   void initialize_remade_ct_fields(int lev, amrex::Real time, FaceMagneticField& old_face_b)
   {
      if (!ct_enabled(cfg_)) {
         return;
      }
      fill_remade_face_b(lev, time, old_face_b);
      sync_cell_b_from_faces(*state_[lev], face_b_[lev], Geom(lev), true);
      apply_boundary(*state_[lev], Geom(lev));
   }
