// Included inside pyfluid_amrex::AmrSolver private section.
// Coarse-fine gravity energy-source synchronization helpers.

   bool gravity_energy_source_register_needed(int fine_lev) const
   {
      return fine_lev > 0
         && fine_lev <= finestLevel()
         && uses_athena_ssprk3_gravity_energy_correction()
         && eos_method_ != EOSPolytrope;
   }

   amrex::iMultiFab coarse_fine_cell_mask(int lev) const
   {
      amrex::iMultiFab mask(
         fluid_state_.state()[lev]->boxArray(),
         fluid_state_.state()[lev]->DistributionMap(),
         1,
         0
      );
      if (lev < finestLevel()) {
         mask = amrex::makeFineMask(
            *fluid_state_.state()[lev],
            boxArray(lev + 1),
            refRatio(lev),
            Geom(lev).periodicity(),
            1,
            0
         );
      } else {
         mask.setVal(1);
      }
      return mask;
   }

   void fill_athena_ssprk3_gravity_face_energy_source(
      int lev,
      int dir,
      amrex::MultiFab& out
   ) const
   {
      const amrex::Real accel_offset = 0.5 * (
         gravity_start_acceleration_offset_[dir] + gravity_acceleration_offset_[dir]
      );
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(out, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto q = out.array(mfi);
         const auto f0 = gravity_stage_mass_flux_[0][lev][dir]->const_array(mfi);
         const auto f1 = gravity_stage_mass_flux_[1][lev][dir]->const_array(mfi);
         const auto f2 = gravity_stage_mass_flux_[2][lev][dir]->const_array(mfi);
         const auto g0 = gravity_start_gradient_[lev][dir]->const_array(mfi);
         const auto g = gravity_gradient_[lev][dir]->const_array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real mass_flux = (
               f0(i, j, k, 0) + f1(i, j, k, 0) + 4.0 * f2(i, j, k, 0)
            ) / 6.0;
            const amrex::Real acceleration = -0.5 * (g0(i, j, k, 0) + g(i, j, k, 0)) - accel_offset;
            q(i, j, k, 0) = 0.5 * mass_flux * acceleration;
         });
      }
      }
   }

   void accumulate_athena_ssprk3_gravity_energy_source_register_dir(
      int fine_lev,
      int dir,
      amrex::MultiFab const* fine_average_source,
      amrex::Real dt,
      bool include_coarse,
      bool include_fine
   )
   {
      const int coarse_lev = fine_lev - 1;
      amrex::iMultiFab mask = coarse_fine_cell_mask(coarse_lev);
      amrex::iMultiFab grown_mask(mask.boxArray(), mask.DistributionMap(), 1, 1);
      grown_mask.setVal(1);
      grown_mask.ParallelCopy(mask, 0, 0, 1, 0, 0);
      grown_mask.FillBoundary(Geom(coarse_lev).periodicity());
      const amrex::Box domain = Geom(coarse_lev).Domain();
      const auto lo = domain.smallEnd();
      const auto hi = domain.bigEnd();
      const amrex::Real accel_offset = 0.5 * (
         gravity_start_acceleration_offset_[dir] + gravity_acceleration_offset_[dir]
      );
      amrex::MultiFab& reg_mf = *gravity_energy_source_register_[fine_lev];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(reg_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto reg = reg_mf.array(mfi);
         const auto covered = grown_mask.const_array(mfi);
         const auto avg = include_fine
            ? fine_average_source->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
         const auto f0 = gravity_stage_mass_flux_[0][coarse_lev][dir]->const_array(mfi);
         const auto f1 = gravity_stage_mass_flux_[1][coarse_lev][dir]->const_array(mfi);
         const auto f2 = gravity_stage_mass_flux_[2][coarse_lev][dir]->const_array(mfi);
         const auto g0 = gravity_start_gradient_[coarse_lev][dir]->const_array(mfi);
         const auto g = gravity_gradient_[coarse_lev][dir]->const_array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (covered(i, j, k, 0) == 0) {
               return;
            }
            auto face_correction = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept -> amrex::Real {
               const amrex::Real mass_flux = (
                  f0(ii, jj, kk, 0) + f1(ii, jj, kk, 0) + 4.0 * f2(ii, jj, kk, 0)
               ) / 6.0;
               const amrex::Real acceleration = -0.5 * (g0(ii, jj, kk, 0) + g(ii, jj, kk, 0)) - accel_offset;
               const amrex::Real coarse_source = 0.5 * mass_flux * acceleration;
               const amrex::Real fine_source = include_fine ? avg(ii, jj, kk, 0) : 0.0;
               return fine_source - (include_coarse ? coarse_source : 0.0);
            };
            amrex::Real correction = 0.0;
            if (dir == 0) {
               if (i > lo[0] && covered(i - 1, j, k, 0) == 0) {
                  correction += face_correction(i, j, k);
               }
               if (i < hi[0] && covered(i + 1, j, k, 0) == 0) {
                  correction += face_correction(i + 1, j, k);
               }
            }
#if (AMREX_SPACEDIM >= 2)
            if (dir == 1) {
               if (j > lo[1] && covered(i, j - 1, k, 0) == 0) {
                  correction += face_correction(i, j, k);
               }
               if (j < hi[1] && covered(i, j + 1, k, 0) == 0) {
                  correction += face_correction(i, j + 1, k);
               }
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (dir == 2) {
               if (k > lo[2] && covered(i, j, k - 1, 0) == 0) {
                  correction += face_correction(i, j, k);
               }
               if (k < hi[2] && covered(i, j, k + 1, 0) == 0) {
                  correction += face_correction(i, j, k + 1);
               }
            }
#endif
            reg(i, j, k, 0) += dt * correction;
         });
      }
      }
   }

   void add_athena_ssprk3_gravity_energy_source_register(
      amrex::Real dt,
      int lev_begin,
      int lev_end
   )
   {
      if (!uses_athena_ssprk3_gravity_energy_correction() || finestLevel() == 0 || dt == 0.0) {
         return;
      }
      for (int fine_lev = 1; fine_lev <= finestLevel(); ++fine_lev) {
         const int coarse_lev = fine_lev - 1;
         const bool include_coarse = coarse_lev >= lev_begin && coarse_lev < lev_end;
         const bool include_fine = fine_lev >= lev_begin && fine_lev < lev_end;
         if (!include_coarse && !include_fine) {
            continue;
         }
         if (!gravity_energy_source_register_[fine_lev]) {
            continue;
         }
         std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> fine_source;
         std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> fine_average_source;
         if (include_fine) {
            amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> fine_ptrs;
            amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> average_ptrs;
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
               fine_source[dir] = std::make_unique<amrex::MultiFab>(
                  gravity_stage_mass_flux_[2][fine_lev][dir]->boxArray(),
                  gravity_stage_mass_flux_[2][fine_lev][dir]->DistributionMap(),
                  1,
                  0
               );
               fine_average_source[dir] = std::make_unique<amrex::MultiFab>(
                  gravity_stage_mass_flux_[2][coarse_lev][dir]->boxArray(),
                  gravity_stage_mass_flux_[2][coarse_lev][dir]->DistributionMap(),
                  1,
                  0
               );
               fill_athena_ssprk3_gravity_face_energy_source(fine_lev, dir, *fine_source[dir]);
               fine_average_source[dir]->setVal(0.0);
               fine_ptrs[dir] = fine_source[dir].get();
               average_ptrs[dir] = fine_average_source[dir].get();
            }
            amrex::average_down_faces(fine_ptrs, average_ptrs, refRatio(coarse_lev), 0);
         }
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            accumulate_athena_ssprk3_gravity_energy_source_register_dir(
               fine_lev,
               dir,
               include_fine ? fine_average_source[dir].get() : nullptr,
               dt,
               include_coarse,
               include_fine
            );
         }
      }
   }

   void apply_gravity_energy_source_register(int fine_lev, amrex::MultiFab& coarse_state)
   {
      apply_gravity_energy_source_register_scaled(fine_lev, coarse_state, 1.0);
   }

   void apply_gravity_energy_source_register_scaled(
      int fine_lev,
      amrex::MultiFab& coarse_state,
      amrex::Real scale
   )
   {
      if (
         fine_lev <= 0
         || fine_lev > finestLevel()
         || !gravity_energy_source_register_[fine_lev]
         || scale == 0.0
      ) {
         return;
      }
      amrex::MultiFab& reg_mf = *gravity_energy_source_register_[fine_lev];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(coarse_state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto state = coarse_state.array(mfi);
         const auto reg = reg_mf.const_array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            state(i, j, k, ENERGY) += scale * reg(i, j, k, 0);
         });
      }
      }
   }
