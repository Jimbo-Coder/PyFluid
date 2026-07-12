// Included inside pyfluid_amrex::AmrSolver private section.
// Hydro positivity repair and solid-ghost state extension.

   void fill_hydro_repair_seed_from_coarse_level(int lev, amrex::MultiFab& seed)
   {
      seed.setVal(0.0);
      if (lev <= 0 || lev > finestLevel()) {
         return;
      }
      const int ncomp = num_state_components();
      const amrex::Real time = level_times_[lev];
      fill_level_boundaries(state_, lev - 1, level_times_[lev - 1]);
      auto const& bcs = fillpatch_bcs();
      const StateFillPatchFunc& boundary_func = state_fillpatch_func();
      StateFillPatchPhysBC coarse_bc(Geom(lev - 1), bcs, boundary_func);
      StateFillPatchPhysBC fine_bc(Geom(lev), bcs, boundary_func);
      amrex::InterpFromCoarseLevel(
         seed,
         seed.nGrowVect(),
         time,
         *state_[lev - 1],
         0,
         0,
         ncomp,
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
      apply_post_fillpatch_state_boundary(seed, Geom(lev), time);
   }

   void repair_nonpositive_hydro_fluid_cells_level(int lev)
   {
      if (is_mhd(cfg_) || !solid_active(lev) || lev < 0 || lev > finestLevel()) {
         return;
      }
      const int ncomp = num_state_components();
      const EOSParams eos = eos_params(cfg_);
      const int eos_method = eos_method_code(cfg_);
      constexpr int repair_grow = SolidMotionSearchRadius;
      constexpr int max_repair_sweeps = 64;
      amrex::MultiFab source(
         state_[lev]->boxArray(),
         state_[lev]->DistributionMap(),
         ncomp,
         amrex::IntVect(repair_grow)
      );
      amrex::MultiFab repair_counts(state_[lev]->boxArray(), state_[lev]->DistributionMap(), 2, 0);
      std::unique_ptr<amrex::MultiFab> coarse_seed;
      if (lev > 0) {
         coarse_seed = std::make_unique<amrex::MultiFab>(
            state_[lev]->boxArray(),
            state_[lev]->DistributionMap(),
            ncomp,
            0
         );
         fill_hydro_repair_seed_from_coarse_level(lev, *coarse_seed);
      }

      for (int sweep = 0; sweep < max_repair_sweeps; ++sweep) {
         state_[lev]->FillBoundary(Geom(lev).periodicity());
         source.ParallelCopy(
            *state_[lev],
            0,
            0,
            ncomp,
            amrex::IntVect(0),
            amrex::IntVect(repair_grow),
            Geom(lev).periodicity()
         );
         repair_counts.setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
         {
         for (amrex::MFIter mfi(*state_[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const amrex::Box& fabbox = source[mfi].box();
            const auto flo = amrex::lbound(fabbox);
            const auto fhi = amrex::ubound(fabbox);
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);
            const auto dst = state_[lev]->array(mfi);
            const auto src = source.const_array(mfi);
            const auto solid = solid_mask_[lev]->const_array(mfi);
            const auto counts = repair_counts.array(mfi);
            const bool use_coarse_seed = coarse_seed != nullptr;
            const auto seed = use_coarse_seed ? coarse_seed->const_array(mfi) : amrex::Array4<amrex::Real const>{};
            for (int k = lo.z; k <= hi.z; ++k) {
               for (int j = lo.y; j <= hi.y; ++j) {
                  for (int i = lo.x; i <= hi.x; ++i) {
                     const amrex::Real current_pressure =
                        pressure_from_array_runtime(eos_method, src, i, j, k, eos, false);
                     bool current_valid = solid(i, j, k, 0) == 0
                        && src(i, j, k, RHO) > 0.0
                        && src(i, j, k, ENERGY) > 0.0
                        && current_pressure > 0.0
                        && amrex::Math::isfinite(current_pressure);
                     for (int n = 0; n < ncomp; ++n) {
                        current_valid = current_valid && amrex::Math::isfinite(src(i, j, k, n));
                     }
                     if (solid(i, j, k, 0) != 0 || current_valid) {
                        continue;
                     }
                     counts(i, j, k, 0) = 1.0;
                     bool found = false;
                     int donor_i = i;
                     int donor_j = j;
                     int donor_k = k;
                     for (int radius = 1; radius <= repair_grow && !found; ++radius) {
                     for (int ox = -radius; ox <= radius && !found; ++ox) {
#if (AMREX_SPACEDIM >= 2)
                        for (int oy = -radius; oy <= radius && !found; ++oy) {
#else
                        const int oy = 0;
                        {
#endif
#if (AMREX_SPACEDIM == 3)
                           for (int oz = -radius; oz <= radius && !found; ++oz) {
#else
                           const int oz = 0;
                           {
#endif
                              if (ox == 0 && oy == 0 && oz == 0) {
                                 continue;
                              }
                              if (
                                 std::abs(ox) != radius
#if (AMREX_SPACEDIM >= 2)
                                 && std::abs(oy) != radius
#endif
#if (AMREX_SPACEDIM == 3)
                                 && std::abs(oz) != radius
#endif
                              ) {
                                 continue;
                              }
                              const int ci = i + ox;
                              const int cj = j + oy;
                              const int ck = k + oz;
                              if (ci < flo.x || ci > fhi.x) {
                                 continue;
                              }
#if (AMREX_SPACEDIM >= 2)
                              if (cj < flo.y || cj > fhi.y) {
                                 continue;
                              }
#endif
#if (AMREX_SPACEDIM == 3)
                              if (ck < flo.z || ck > fhi.z) {
                                 continue;
                              }
#endif
                              bool donor_valid = solid(ci, cj, ck, 0) == 0
                                 && src(ci, cj, ck, RHO) > 0.0
                                 && src(ci, cj, ck, ENERGY) > 0.0;
                              const amrex::Real donor_pressure =
                                 pressure_from_array_runtime(eos_method, src, ci, cj, ck, eos, false);
                              donor_valid = donor_valid
                                 && donor_pressure > 0.0
                                 && amrex::Math::isfinite(donor_pressure);
                              for (int n = 0; n < ncomp; ++n) {
                                 donor_valid = donor_valid && amrex::Math::isfinite(src(ci, cj, ck, n));
                              }
                              if (donor_valid) {
                                 donor_i = ci;
                                 donor_j = cj;
                                 donor_k = ck;
                                 found = true;
                              }
#if (AMREX_SPACEDIM == 3)
                           }
#else
                           }
#endif
#if (AMREX_SPACEDIM >= 2)
                        }
#else
                        }
#endif
                     }
                     }
                     if (!found) {
                        bool seed_valid = use_coarse_seed
                           && seed(i, j, k, RHO) > 0.0
                           && seed(i, j, k, ENERGY) > 0.0;
                        const amrex::Real seed_pressure = use_coarse_seed
                           ? pressure_from_array_runtime(eos_method, seed, i, j, k, eos, false)
                           : -1.0;
                        seed_valid = seed_valid
                           && seed_pressure > 0.0
                           && amrex::Math::isfinite(seed_pressure);
                        for (int n = 0; n < ncomp; ++n) {
                           seed_valid = seed_valid && amrex::Math::isfinite(seed(i, j, k, n));
                        }
                        if (!seed_valid) {
                           continue;
                        }
                        for (int n = 0; n < ncomp; ++n) {
                           dst(i, j, k, n) = seed(i, j, k, n);
                        }
                        counts(i, j, k, 1) = 1.0;
                        continue;
                     }
                     for (int n = 0; n < ncomp; ++n) {
                        dst(i, j, k, n) = src(donor_i, donor_j, donor_k, n);
                     }
                     counts(i, j, k, 1) = 1.0;
                  }
               }
            }
         }
         }

         const amrex::Real bad = repair_counts.sum(0);
         if (bad <= 0.0) {
            state_[lev]->FillBoundary(Geom(lev).periodicity());
            return;
         }
         const amrex::Real repaired = repair_counts.sum(1);
         if (repaired <= 0.0) {
            std::ostringstream message;
            message << "AMR solid regrid produced " << bad
                    << " nonpositive fluid cells on level " << lev
                    << " at repair sweep " << sweep
                    << " without a positive neighboring hydro state";
            const std::string text = message.str();
            amrex::Abort(text.c_str());
         }
         state_[lev]->FillBoundary(Geom(lev).periodicity());
      }
      std::ostringstream message;
      message << "AMR solid regrid repair on level " << lev
              << " did not converge within " << max_repair_sweeps
              << " sweeps";
      const std::string text = message.str();
      amrex::Abort(text.c_str());
   }

   void extend_hydro_state_into_needed_solid_cells_level(int lev)
   {
      if (
         is_mhd(cfg_)
         || !solid_active(lev)
         || lev < 0
         || lev > finestLevel()
         || !primitive_needed_mask_[lev]
      ) {
         return;
      }
      const int ncomp = num_state_components();
      const EOSParams eos = eos_params(cfg_);
      const int eos_method = eos_method_code(cfg_);
      constexpr int repair_grow = SolidMotionSearchRadius;
      constexpr int max_extension_sweeps = 16;
      amrex::MultiFab source(
         state_[lev]->boxArray(),
         state_[lev]->DistributionMap(),
         ncomp,
         amrex::IntVect(repair_grow)
      );
      amrex::MultiFab counts(state_[lev]->boxArray(), state_[lev]->DistributionMap(), 2, 0);
      std::unique_ptr<amrex::MultiFab> coarse_seed;
      if (lev > 0) {
         coarse_seed = std::make_unique<amrex::MultiFab>(
            state_[lev]->boxArray(),
            state_[lev]->DistributionMap(),
            ncomp,
            0
         );
         fill_hydro_repair_seed_from_coarse_level(lev, *coarse_seed);
      }

      for (int sweep = 0; sweep < max_extension_sweeps; ++sweep) {
         state_[lev]->FillBoundary(Geom(lev).periodicity());
         source.ParallelCopy(
            *state_[lev],
            0,
            0,
            ncomp,
            amrex::IntVect(0),
            amrex::IntVect(repair_grow),
            Geom(lev).periodicity()
         );
         counts.setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
         {
         for (amrex::MFIter mfi(*state_[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const amrex::Box& fabbox = source[mfi].box();
            const auto flo = amrex::lbound(fabbox);
            const auto fhi = amrex::ubound(fabbox);
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);
            const auto dst = state_[lev]->array(mfi);
            const auto src = source.const_array(mfi);
            const auto solid = solid_mask_[lev]->const_array(mfi);
            const auto needed = primitive_needed_mask_[lev]->const_array(mfi);
            const auto out = counts.array(mfi);
            const bool use_coarse_seed = coarse_seed != nullptr;
            const auto seed = use_coarse_seed ? coarse_seed->const_array(mfi) : amrex::Array4<amrex::Real const>{};
            for (int k = lo.z; k <= hi.z; ++k) {
               for (int j = lo.y; j <= hi.y; ++j) {
                  for (int i = lo.x; i <= hi.x; ++i) {
                     if (solid(i, j, k, 0) == 0 || needed(i, j, k, 0) == 0) {
                        continue;
                     }
                     bool found = false;
                     int donor_i = i;
                     int donor_j = j;
                     int donor_k = k;
                     for (int radius = 1; radius <= repair_grow && !found; ++radius) {
                     for (int ox = -radius; ox <= radius && !found; ++ox) {
#if (AMREX_SPACEDIM >= 2)
                        for (int oy = -radius; oy <= radius && !found; ++oy) {
#else
                        const int oy = 0;
                        {
#endif
#if (AMREX_SPACEDIM == 3)
                           for (int oz = -radius; oz <= radius && !found; ++oz) {
#else
                           const int oz = 0;
                           {
#endif
                              if (ox == 0 && oy == 0 && oz == 0) {
                                 continue;
                              }
                              if (
                                 std::abs(ox) != radius
#if (AMREX_SPACEDIM >= 2)
                                 && std::abs(oy) != radius
#endif
#if (AMREX_SPACEDIM == 3)
                                 && std::abs(oz) != radius
#endif
                              ) {
                                 continue;
                              }
                              const int ci = i + ox;
                              const int cj = j + oy;
                              const int ck = k + oz;
                              if (ci < flo.x || ci > fhi.x) {
                                 continue;
                              }
#if (AMREX_SPACEDIM >= 2)
                              if (cj < flo.y || cj > fhi.y) {
                                 continue;
                              }
#endif
#if (AMREX_SPACEDIM == 3)
                              if (ck < flo.z || ck > fhi.z) {
                                 continue;
                              }
#endif
                              bool donor_valid = solid(ci, cj, ck, 0) == 0
                                 && src(ci, cj, ck, RHO) > 0.0
                                 && src(ci, cj, ck, ENERGY) > 0.0;
                              const amrex::Real donor_pressure =
                                 pressure_from_array_runtime(eos_method, src, ci, cj, ck, eos, false);
                              donor_valid = donor_valid
                                 && donor_pressure > 0.0
                                 && amrex::Math::isfinite(donor_pressure);
                              for (int n = 0; n < ncomp; ++n) {
                                 donor_valid = donor_valid && amrex::Math::isfinite(src(ci, cj, ck, n));
                              }
                              if (donor_valid) {
                                 donor_i = ci;
                                 donor_j = cj;
                                 donor_k = ck;
                                 found = true;
                              }
#if (AMREX_SPACEDIM == 3)
                           }
#else
                           }
#endif
#if (AMREX_SPACEDIM >= 2)
                        }
#else
                        }
#endif
                     }
                     }
                     if (found) {
                        for (int n = 0; n < ncomp; ++n) {
                           dst(i, j, k, n) = src(donor_i, donor_j, donor_k, n);
                        }
                        out(i, j, k, 1) = 1.0;
                        continue;
                     }
                     bool seed_valid = use_coarse_seed
                        && seed(i, j, k, RHO) > 0.0
                        && seed(i, j, k, ENERGY) > 0.0;
                     const amrex::Real seed_pressure = use_coarse_seed
                        ? pressure_from_array_runtime(eos_method, seed, i, j, k, eos, false)
                        : -1.0;
                     seed_valid = seed_valid
                        && seed_pressure > 0.0
                        && amrex::Math::isfinite(seed_pressure);
                     for (int n = 0; n < ncomp; ++n) {
                        seed_valid = seed_valid && amrex::Math::isfinite(seed(i, j, k, n));
                     }
                     if (seed_valid) {
                        for (int n = 0; n < ncomp; ++n) {
                           dst(i, j, k, n) = seed(i, j, k, n);
                        }
                        out(i, j, k, 1) = 1.0;
                        continue;
                     }
                     const amrex::Real current_pressure =
                        pressure_from_array_runtime(eos_method, src, i, j, k, eos, false);
                     bool current_valid = src(i, j, k, RHO) > 0.0
                        && src(i, j, k, ENERGY) > 0.0
                        && current_pressure > 0.0
                        && amrex::Math::isfinite(current_pressure);
                     for (int n = 0; n < ncomp; ++n) {
                        current_valid = current_valid && amrex::Math::isfinite(src(i, j, k, n));
                     }
                     if (!current_valid) {
                        out(i, j, k, 0) = 1.0;
                     }
                  }
               }
            }
         }
         }
         const amrex::Real bad = counts.sum(0);
         if (bad <= 0.0) {
            state_[lev]->FillBoundary(Geom(lev).periodicity());
            return;
         }
         const amrex::Real filled = counts.sum(1);
         if (filled <= 0.0) {
            std::ostringstream message;
            message << "AMR solid extension found " << bad
                    << " needed solid cells on level " << lev
                    << " without a positive neighboring hydro state";
            const std::string text = message.str();
            amrex::Abort(text.c_str());
         }
      }
      std::ostringstream message;
      message << "AMR solid extension on level " << lev
              << " did not converge within " << max_extension_sweeps
              << " sweeps";
      const std::string text = message.str();
      amrex::Abort(text.c_str());
   }

   void repair_nonpositive_hydro_fluid_cells_all_levels()
   {
      if (is_mhd(cfg_) || !solid_geometry_enabled(cfg_)) {
         return;
      }
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         repair_nonpositive_hydro_fluid_cells_level(lev);
      }
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         extend_hydro_state_into_needed_solid_cells_level(lev);
      }
   }
