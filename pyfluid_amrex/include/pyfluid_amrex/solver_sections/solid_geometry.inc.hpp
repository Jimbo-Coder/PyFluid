// Included inside pyfluid_amrex::AmrSolver private section.
// Solid-geometry refresh, mask-kind tables, apertures, and moving-geometry state.

   void clear_solid_mask_kind_tables(int lev)
   {
      if (lev < 0 || lev >= static_cast<int>(solid_valid_kind_.size())) {
         return;
      }
      solid_valid_kind_[lev].clear();
      primitive_needed_kind_[lev].clear();
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         solid_flux_kind_[lev][dir].clear();
      }
   }

   void build_solid_mask_kind_tables(int lev)
   {
      clear_solid_mask_kind_tables(lev);
      if (!solid_active(lev)) {
         return;
      }

      solid_valid_kind_[lev].assign(state_[lev]->boxArray().size(), MaskKindMixed);
      primitive_needed_kind_[lev].assign(primitive_needed_mask_[lev]->boxArray().size(), MaskKindMixed);
      for (amrex::MFIter mfi(*state_[lev]); mfi.isValid(); ++mfi) {
         solid_valid_kind_[lev][mfi.index()] = classify_mask_region(*solid_mask_[lev], mfi, mfi.validbox());
      }
      for (amrex::MFIter mfi(*primitive_needed_mask_[lev]); mfi.isValid(); ++mfi) {
         primitive_needed_kind_[lev][mfi.index()] = classify_mask_region(*primitive_needed_mask_[lev], mfi, mfi.fabbox());
      }

      const amrex::Box domain = Geom(lev).Domain();
      const int recon_method = recon_method_for_level(lev);
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         solid_flux_kind_[lev][dir].assign(fluxes_[lev][dir]->boxArray().size(), MaskKindMixed);
         for (amrex::MFIter mfi(*fluxes_[lev][dir]); mfi.isValid(); ++mfi) {
            const amrex::Box cell_region = solid_flux_stencil_region(mfi.validbox(), dir, recon_method, domain);
            solid_flux_kind_[lev][dir][mfi.index()] = classify_mask_region(*solid_mask_[lev], mfi, cell_region);
         }
      }
   }

   void average_down_solid_volume_fraction_from_fine(int fine_lev)
   {
      if (fine_lev <= 0 || fine_lev > finestLevel()) {
         return;
      }
      const int coarse_lev = fine_lev - 1;
      if (!solid_volume_fraction_[coarse_lev] || !solid_volume_fraction_[fine_lev]) {
         return;
      }
      amrex::average_down(
         *solid_volume_fraction_[fine_lev],
         *solid_volume_fraction_[coarse_lev],
         Geom(fine_lev),
         Geom(coarse_lev),
         0,
         1,
         refRatio(coarse_lev)
      );
      solid_volume_fraction_[coarse_lev]->FillBoundary(Geom(coarse_lev).periodicity());
      fill_solid_mask_from_volume_fraction(*solid_mask_[coarse_lev], *solid_volume_fraction_[coarse_lev]);
      solid_mask_[coarse_lev]->FillBoundary(Geom(coarse_lev).periodicity());
   }

   void average_down_solid_aperture_from_fine(int fine_lev, int direction)
   {
      if (fine_lev <= 0 || fine_lev > finestLevel()) {
         return;
      }
      const int coarse_lev = fine_lev - 1;
      if (!solid_aperture_[direction][coarse_lev] || !solid_aperture_[direction][fine_lev]) {
         return;
      }
      const amrex::IntVect ratio = refRatio(coarse_lev);
      amrex::BoxArray coarse_ba = solid_aperture_[direction][fine_lev]->boxArray();
      coarse_ba.coarsen(ratio);
      amrex::MultiFab coarse_from_fine(
         coarse_ba,
         solid_aperture_[direction][fine_lev]->DistributionMap(),
         1,
         1
      );
      coarse_from_fine.setVal(1.0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(coarse_from_fine, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         amrex::Box bx = mfi.tilebox();
         bx.growHi(direction, 1);
         const auto coarse = coarse_from_fine.array(mfi);
         const auto fine = solid_aperture_[direction][fine_lev]->const_array(mfi);
         const int rx = ratio[0];
#if (AMREX_SPACEDIM >= 2)
         const int ry = ratio[1];
#else
         const int ry = 1;
#endif
#if (AMREX_SPACEDIM == 3)
         const int rz = ratio[2];
#else
         const int rz = 1;
#endif
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real sum = 0.0;
            int count = 0;
            if (direction == 0) {
               const int fi = i * rx;
               for (int kk = 0; kk < rz; ++kk) {
                  const int fk = k * rz + kk;
                  for (int jj = 0; jj < ry; ++jj) {
                     const int fj = j * ry + jj;
                     sum += fine(fi, fj, fk, 0);
                     ++count;
                  }
               }
            }
#if (AMREX_SPACEDIM >= 2)
            else if (direction == 1) {
               const int fj = j * ry;
               for (int kk = 0; kk < rz; ++kk) {
                  const int fk = k * rz + kk;
                  for (int ii = 0; ii < rx; ++ii) {
                     const int fi = i * rx + ii;
                     sum += fine(fi, fj, fk, 0);
                     ++count;
                  }
               }
            }
#endif
#if (AMREX_SPACEDIM == 3)
            else {
               const int fk = k * rz;
               for (int jj = 0; jj < ry; ++jj) {
                  const int fj = j * ry + jj;
                  for (int ii = 0; ii < rx; ++ii) {
                     const int fi = i * rx + ii;
                     sum += fine(fi, fj, fk, 0);
                     ++count;
                  }
               }
            }
#endif
            coarse(i, j, k, 0) = count > 0 ? sum / static_cast<amrex::Real>(count) : 1.0;
         });
      }
      }

      solid_aperture_[direction][coarse_lev]->ParallelCopy(
         coarse_from_fine,
         0,
         0,
         1,
         amrex::IntVect::TheDimensionVector(direction),
         amrex::IntVect::TheDimensionVector(direction),
         Geom(coarse_lev).periodicity()
      );
      solid_aperture_[direction][coarse_lev]->FillBoundary(Geom(coarse_lev).periodicity());
   }

   void enforce_solid_mask_nesting_from_fine()
   {
      if (!solid_geometry_enabled(cfg_) || finestLevel() <= 0) {
         return;
      }
      for (int fine_lev = finestLevel(); fine_lev > 0; --fine_lev) {
         if (!solid_mask_[fine_lev] || !solid_mask_[fine_lev - 1]) {
            continue;
         }
         const bool nested_cutcell_geometry =
            solid_volume_fraction_[fine_lev] != nullptr
            && solid_volume_fraction_[fine_lev - 1] != nullptr;
         if (nested_cutcell_geometry) {
            average_down_solid_volume_fraction_from_fine(fine_lev);
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
               average_down_solid_aperture_from_fine(fine_lev, dir);
            }
            continue;
         }
         const amrex::IntVect ratio = refRatio(fine_lev - 1);
         amrex::BoxArray coarse_ba = solid_mask_[fine_lev]->boxArray();
         coarse_ba.coarsen(ratio);
         amrex::iMultiFab coarse_from_fine(
            coarse_ba,
            solid_mask_[fine_lev]->DistributionMap(),
            1,
            0
         );
         coarse_from_fine.setVal(0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
         {
         for (amrex::MFIter mfi(coarse_from_fine, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const auto coarse = coarse_from_fine.array(mfi);
            const auto fine = solid_mask_[fine_lev]->const_array(mfi);
            const int rx = ratio[0];
#if (AMREX_SPACEDIM >= 2)
            const int ry = ratio[1];
#else
            const int ry = 1;
#endif
#if (AMREX_SPACEDIM == 3)
            const int rz = ratio[2];
#else
            const int rz = 1;
#endif
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               bool all_children_solid = true;
               for (int kk = 0; kk < rz; ++kk) {
                  const int fk = k * rz + kk;
                  for (int jj = 0; jj < ry; ++jj) {
                     const int fj = j * ry + jj;
                     for (int ii = 0; ii < rx; ++ii) {
                        const int fi = i * rx + ii;
                        all_children_solid = all_children_solid && fine(fi, fj, fk, 0) != 0;
                     }
                  }
               }
               coarse(i, j, k, 0) = all_children_solid ? 1 : 0;
            });
         }
         }

         solid_mask_[fine_lev - 1]->ParallelCopy(
            coarse_from_fine,
            0,
            0,
            1,
            amrex::IntVect(0),
            amrex::IntVect(0),
            Geom(fine_lev - 1).periodicity()
         );
         solid_mask_[fine_lev - 1]->FillBoundary(Geom(fine_lev - 1).periodicity());
      }
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         if (!solid_mask_[lev]) {
            continue;
         }
         if (primitive_needed_mask_[lev]) {
            fill_primitive_needed_mask(*primitive_needed_mask_[lev], *solid_mask_[lev], Geom(lev), cfg_.ng);
         }
         refresh_solid_activity(lev);
         build_solid_mask_kind_tables(lev);
      }
   }

   int recon_method_for_level(int lev) const
   {
      if (!cfg_.recon_fine.empty() && lev >= cfg_.recon_fine_start_level) {
         return recon_fine_method_;
      }
      return recon_method_;
   }

   void refresh_solid_activity(int lev)
   {
      solid_active_[lev] = solid_mask_[lev] != nullptr && solid_mask_[lev]->max(0) != 0;
   }

   amrex::Real solid_geometry_max_speed() const
   {
   if (!solid_geometry_changes_cell_mask(cfg_)) {
      return 0.0;
   }
   if (cfg_.solid_motion == "pulse") {
      const amrex::Real omega =
         2.0 * static_cast<amrex::Real>(3.141592653589793238462643383279502884L)
         * cfg_.solid_radius_frequency;
      amrex::Real max_rate = omega;
      if (cfg_.solid_radius_ramp_time > 0.0) {
         max_rate += static_cast<amrex::Real>(1.875) / cfg_.solid_radius_ramp_time;
      }
      return amrex::Math::abs(cfg_.solid_radius_amplitude) * max_rate;
   }
   amrex::Real speed = amrex::Math::abs(cfg_.solid_velocity_1);
#if (AMREX_SPACEDIM >= 2)
   speed = amrex::max(speed, amrex::Math::abs(cfg_.solid_velocity_2));
#endif
#if (AMREX_SPACEDIM == 3)
   speed = amrex::max(speed, amrex::Math::abs(cfg_.solid_velocity_3));
#endif
   return speed;
   }

   amrex::Real solid_geometry_motion_dt(int lev) const
   {
   const amrex::Real speed = solid_geometry_max_speed();
   if (speed <= 0.0 || lev < 0 || lev > finestLevel()) {
      return std::numeric_limits<amrex::Real>::max();
   }
   const auto dx = Geom(lev).CellSizeArray();
   amrex::Real min_dx = dx[0];
#if (AMREX_SPACEDIM >= 2)
   min_dx = amrex::min(min_dx, dx[1]);
#endif
#if (AMREX_SPACEDIM == 3)
   min_dx = amrex::min(min_dx, dx[2]);
#endif
   return 0.25 * min_dx / speed;
   }

   bool solid_mask_refresh_needed(int lev, amrex::Real time) const
   {
   if (lev < 0 || lev > finestLevel()) {
      return true;
   }
   if (lev >= static_cast<int>(solid_mask_time_.size()) || !std::isfinite(solid_mask_time_[lev])) {
      return true;
   }
   if (solid_geometry_changes_cell_mask(cfg_)) {
      return amrex::Math::abs(time - solid_mask_time_[lev]) > 0.0;
   }
   if (!solid_geometry_enabled(cfg_)) {
      return false;
   }
   const amrex::Real speed = solid_geometry_max_speed();
   if (speed <= 0.0) {
      return false;
   }
   const auto dx = Geom(lev).CellSizeArray();
   amrex::Real min_dx = dx[0];
#if (AMREX_SPACEDIM >= 2)
   min_dx = amrex::min(min_dx, dx[1]);
#endif
#if (AMREX_SPACEDIM == 3)
   min_dx = amrex::min(min_dx, dx[2]);
#endif
   const amrex::Real displacement = speed * amrex::Math::abs(time - solid_mask_time_[lev]);
   return displacement >= 0.25 * min_dx;
   }

   void allocate_solid_apertures_level(int lev)
   {
   if (!solid_mask_[lev]) {
      return;
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (!solid_aperture_[dir][lev]) {
         solid_aperture_[dir][lev] = std::make_unique<amrex::MultiFab>(
            solid_mask_[lev]->boxArray(),
            solid_mask_[lev]->DistributionMap(),
            1,
            1
         );
      }
   }
   }

   void fill_solid_apertures_level(int lev, amrex::Real time)
   {
   allocate_solid_apertures_level(lev);
   if (!solid_aperture_[0][lev]) {
      return;
   }
   fill_solid_face_aperture_output<0>(*solid_aperture_[0][lev], Geom(lev), cfg_, time);
#if (AMREX_SPACEDIM >= 2)
   fill_solid_face_aperture_output<1>(*solid_aperture_[1][lev], Geom(lev), cfg_, time);
#endif
#if (AMREX_SPACEDIM == 3)
   fill_solid_face_aperture_output<2>(*solid_aperture_[2][lev], Geom(lev), cfg_, time);
#endif
   }

   void refresh_solid_discrete_geometry_level(int lev, amrex::Real time)
   {
   if (!solid_geometry_enabled(cfg_) || lev < 0 || lev > finestLevel() || !solid_mask_[lev]) {
      return;
   }
   if (!solid_volume_fraction_[lev]) {
      solid_volume_fraction_[lev] = std::make_unique<amrex::MultiFab>(
         solid_mask_[lev]->boxArray(),
         solid_mask_[lev]->DistributionMap(),
         1,
         solid_mask_[lev]->nGrowVect()
      );
   }
   fill_solid_volume_fraction_data(*solid_volume_fraction_[lev], solid_mask_[lev].get(), Geom(lev), cfg_, time);
   fill_solid_mask_from_volume_fraction(*solid_mask_[lev], *solid_volume_fraction_[lev]);
   fill_solid_apertures_level(lev, time);
   }

   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> solid_aperture_ptrs(int lev) const
   {
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> ptrs{};
   if (lev < 0 || lev >= static_cast<int>(solid_mask_.size())) {
      return ptrs;
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      ptrs[dir] = solid_aperture_[dir][lev] ? solid_aperture_[dir][lev].get() : nullptr;
   }
   return ptrs;
   }

   void refresh_solid_geometry_level(int lev, amrex::Real time, bool remap_moving_solid)
   {
   if (!solid_geometry_enabled(cfg_) || lev < 0 || lev > finestLevel() || !solid_mask_[lev]) {
      return;
   }
   if (!solid_mask_refresh_needed(lev, time)) {
      if (solid_geometry_changes_cell_mask(cfg_)) {
         refresh_solid_discrete_geometry_level(lev, time);
      }
      return;
   }
      amrex::Real old_time = solid_mask_time_[lev];
      std::unique_ptr<amrex::iMultiFab> old_mask;
      std::unique_ptr<amrex::MultiFab> old_volume_fraction;
      if (remap_moving_solid && solid_geometry_changes_cell_mask(cfg_) && state_[lev]) {
         old_time = solid_mask_time_[lev];
         old_mask = std::make_unique<amrex::iMultiFab>(
            solid_mask_[lev]->boxArray(),
            solid_mask_[lev]->DistributionMap(),
            1,
            solid_mask_[lev]->nGrowVect()
         );
         if (cfg_.solid_motion == "pulse") {
            old_mask->ParallelCopy(
               *solid_mask_[lev],
               0,
               0,
               1,
               solid_mask_[lev]->nGrowVect(),
               old_mask->nGrowVect(),
               Geom(lev).periodicity()
            );
         } else {
            fill_solid_mask(*old_mask, Geom(lev), cfg_, old_time);
         }
         if (cfg_.solid_motion == "pulse" && solid_volume_fraction_[lev]) {
            old_volume_fraction = std::make_unique<amrex::MultiFab>(
               solid_volume_fraction_[lev]->boxArray(),
               solid_volume_fraction_[lev]->DistributionMap(),
               1,
               solid_volume_fraction_[lev]->nGrowVect()
            );
            old_volume_fraction->ParallelCopy(
               *solid_volume_fraction_[lev],
               0,
               0,
               1,
               solid_volume_fraction_[lev]->nGrowVect(),
               old_volume_fraction->nGrowVect(),
               Geom(lev).periodicity()
            );
         }
      }
   fill_solid_mask(*solid_mask_[lev], Geom(lev), cfg_, time);
   refresh_solid_discrete_geometry_level(lev, time);
      solid_mask_time_[lev] = time;
      if (old_mask && state_[lev]) {
         state_[lev]->FillBoundary(Geom(lev).periodicity());
         conservative_translate_box_mask_update(*state_[lev], *old_mask, *solid_mask_[lev], cfg_);
         if (old_volume_fraction && solid_volume_fraction_[lev]) {
            conservative_pulse_solid_volume_fraction_update(
               *state_[lev],
               *old_volume_fraction,
               *solid_volume_fraction_[lev],
               Geom(lev),
               cfg_,
               old_time,
               time
            );
         }
      }
   if (primitive_needed_mask_[lev]) {
      fill_primitive_needed_mask(*primitive_needed_mask_[lev], *solid_mask_[lev], Geom(lev), cfg_.ng);
   }
   refresh_solid_activity(lev);
   build_solid_mask_kind_tables(lev);
   }

   void refresh_solid_geometry_all_levels(amrex::Real time, bool remap_moving_solid)
   {
      if (!solid_geometry_changes_cell_mask(cfg_)) {
         return;
      }
      if (cfg_.solid_motion == "pulse") {
         refresh_pulse_solid_geometry_all_levels(time, remap_moving_solid);
         return;
      }
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         refresh_solid_geometry_level(lev, time, remap_moving_solid);
      }
      enforce_solid_mask_nesting_from_fine();
   }

   bool refresh_solid_geometry_synced(amrex::Real time, bool remap_moving_solid, bool preserve_hydro_totals)
   {
   if (!solid_geometry_changes_cell_mask(cfg_)) {
      return false;
   }

   HydroConservedTotals pre_refresh_totals;
   const bool preserve_totals = preserve_hydro_totals && !is_mhd(cfg_);
   if (preserve_totals) {
      pre_refresh_totals = hydro_valid_conserved_totals();
   }

   refresh_solid_geometry_all_levels(time, remap_moving_solid);

   if (preserve_totals) {
      repair_nonpositive_hydro_fluid_cells_all_levels();
      apply_hydro_valid_conserved_total_correction(pre_refresh_totals);
      if (finestLevel() > 0) {
         average_down_state();
      }
   } else if (finestLevel() > 0) {
      average_down_state();
   }
   return true;
   }

   void refresh_pulse_solid_geometry_all_levels(amrex::Real time, bool remap_moving_solid)
   {
      const int nlev = finestLevel() + 1;
      std::vector<std::unique_ptr<amrex::iMultiFab>> old_masks(nlev);
      std::vector<std::unique_ptr<amrex::MultiFab>> old_volume_fractions(nlev);
      std::vector<amrex::Real> old_times(nlev, time);
      std::vector<int> refreshed(nlev, 0);
      bool any_refreshed = false;

      for (int lev = 0; lev <= finestLevel(); ++lev) {
         if (!solid_geometry_enabled(cfg_) || !solid_mask_[lev]) {
            continue;
         }
         if (!solid_mask_refresh_needed(lev, time)) {
            refresh_solid_discrete_geometry_level(lev, time);
            continue;
         }
         old_times[lev] = solid_mask_time_[lev];
         refreshed[lev] = 1;
         any_refreshed = true;
         if (remap_moving_solid && state_[lev]) {
            old_masks[lev] = std::make_unique<amrex::iMultiFab>(
               solid_mask_[lev]->boxArray(),
               solid_mask_[lev]->DistributionMap(),
               1,
               solid_mask_[lev]->nGrowVect()
            );
            old_masks[lev]->ParallelCopy(
               *solid_mask_[lev],
               0,
               0,
               1,
               solid_mask_[lev]->nGrowVect(),
               old_masks[lev]->nGrowVect(),
               Geom(lev).periodicity()
            );
            if (solid_volume_fraction_[lev]) {
               old_volume_fractions[lev] = std::make_unique<amrex::MultiFab>(
                  solid_volume_fraction_[lev]->boxArray(),
                  solid_volume_fraction_[lev]->DistributionMap(),
                  1,
                  solid_volume_fraction_[lev]->nGrowVect()
               );
               old_volume_fractions[lev]->ParallelCopy(
                  *solid_volume_fraction_[lev],
                  0,
                  0,
                  1,
                  solid_volume_fraction_[lev]->nGrowVect(),
                  old_volume_fractions[lev]->nGrowVect(),
                  Geom(lev).periodicity()
               );
            }
         }
         fill_solid_mask(*solid_mask_[lev], Geom(lev), cfg_, time);
         solid_mask_time_[lev] = time;
      }

      if (!any_refreshed) {
         return;
      }
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         if (!solid_mask_[lev]) {
            continue;
         }
         refresh_solid_discrete_geometry_level(lev, time);
         if (refreshed[lev] != 0 && old_masks[lev] && state_[lev]) {
            state_[lev]->FillBoundary(Geom(lev).periodicity());
            if (old_volume_fractions[lev] && solid_volume_fraction_[lev]) {
               conservative_pulse_solid_volume_fraction_update(
                  *state_[lev],
                  *old_volume_fractions[lev],
                  *solid_volume_fraction_[lev],
                  Geom(lev),
                  cfg_,
                  old_times[lev],
                  time
               );
            }
         }
      }
      enforce_solid_mask_nesting_from_fine();
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         if (!solid_mask_[lev]) {
            continue;
         }
         if (primitive_needed_mask_[lev]) {
            fill_primitive_needed_mask(*primitive_needed_mask_[lev], *solid_mask_[lev], Geom(lev), cfg_.ng);
         }
         refresh_solid_activity(lev);
         build_solid_mask_kind_tables(lev);
      }
   }

   bool solid_active(int lev) const
   {
   return lev >= 0 && lev < static_cast<int>(solid_active_.size()) && solid_active_[lev] && solid_mask_[lev] != nullptr;
   }

   amrex::iMultiFab const* active_solid_mask_ptr(int lev) const
   {
   return solid_active(lev) ? solid_mask_[lev].get() : nullptr;
   }
