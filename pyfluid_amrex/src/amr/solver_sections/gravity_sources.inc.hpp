// Included inside pyfluid_amrex::AmrSolver private section.
// Gravity field preparation, self-gravity energy bookkeeping, and generic source-stage hooks.

   void prepare_gravity_fields(LevelData const& input, bool fill_gradient) override
   {
      if (!gravity_enabled(cfg_)) {
         return;
      }
      if (self_gravity_enabled(cfg_)) {
         solve_gravity_potential(input, fill_gradient);
      } else {
         static_cast<void>(input);
         static_cast<void>(fill_gradient);
         ensure_fixed_gravity_fields();
      }
   }

   void reset_gravity_acceleration_offset()
   {
      for (auto& value : gravity_acceleration_offset_) {
         value = 0.0_rt;
      }
   }

public:
   void update_self_gravity_acceleration_offset(LevelData const& input)
   {
      reset_gravity_acceleration_offset();
      if (!self_gravity_enabled(cfg_)) {
         return;
      }
      std::array<amrex::Real, AMREX_SPACEDIM> force{};
      amrex::Real mass = 0.0_rt;
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         const amrex::Real cell_volume = cartesian_cell_volume(Geom(lev));
         amrex::iMultiFab mask = valid_cell_mask(lev);
         amrex::MultiFab reduction(input[lev]->boxArray(), input[lev]->DistributionMap(), AMREX_SPACEDIM + 1, 0);
         reduction.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
         {
         for (amrex::MFIter mfi(reduction, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const auto s = input[lev]->const_array(mfi);
            const auto valid = mask.const_array(mfi);
            const auto gx = gravity_fields_.gradient(lev, 0)->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
            const auto gy = gravity_fields_.gradient(lev, 1)->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
            const auto gz = gravity_fields_.gradient(lev, 2)->const_array(mfi);
#endif
            const auto out = reduction.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if (valid(i, j, k, 0) == 0) {
                  return;
               }
               const amrex::Real rho = s(i, j, k, RHO);
               const amrex::Real ax = -0.5_rt * (gx(i, j, k, 0) + gx(i + 1, j, k, 0));
               out(i, j, k, 0) = rho;
               out(i, j, k, 1) = rho * ax;
#if (AMREX_SPACEDIM >= 2)
               const amrex::Real ay = -0.5_rt * (gy(i, j, k, 0) + gy(i, j + 1, k, 0));
               out(i, j, k, 2) = rho * ay;
#endif
#if (AMREX_SPACEDIM == 3)
               const amrex::Real az = -0.5_rt * (gz(i, j, k, 0) + gz(i, j, k + 1, 0));
               out(i, j, k, 3) = rho * az;
#endif
            });
         }
         }
         mass += reduction.sum(0, true) * cell_volume;
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            force[dir] += reduction.sum(dir + 1, true) * cell_volume;
         }
      }
      std::array<amrex::Real, AMREX_SPACEDIM + 1> force_and_mass{};
      force_and_mass[0] = mass;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         force_and_mass[dir + 1] = force[dir];
      }
      amrex::ParallelDescriptor::ReduceRealSum(
         force_and_mass.data(), static_cast<int>(force_and_mass.size())
      );
      mass = force_and_mass[0];
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         force[dir] = force_and_mass[dir + 1];
      }
      if (!(mass > 0.0_rt) || !std::isfinite(mass)) {
         amrex::Abort("self-gravity source offset needs a finite positive mass");
      }
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         gravity_acceleration_offset_[dir] = force[dir] / mass;
         if (!std::isfinite(gravity_acceleration_offset_[dir])) {
            amrex::Abort("self-gravity source offset produced a non-finite acceleration");
         }
      }
   }

private:

   bool uses_athena_ssprk3_gravity_energy_correction() const override
   {
      return uses_athena_gravity_energy_correction() && timestep_method_ == TimeSSPRK3;
   }

   bool uses_athena_gravity_energy_correction() const override
   {
      return athena_gravity_coupling_enabled(cfg_);
   }

   void store_athena_ssprk3_start_gravity_range(int lev_begin, int lev_end)
   {
      if (!uses_athena_ssprk3_gravity_energy_correction()) {
         return;
      }
      gravity_start_acceleration_offset_ = gravity_acceleration_offset_;
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (!gravity_start_gradient_[lev][dir] || !gravity_fields_.gradient(lev, dir)) {
               amrex::Abort("Athena SSPRK3 gravity correction is missing start-gradient storage");
            }
            amrex::MultiFab::Copy(*gravity_start_gradient_[lev][dir], *gravity_fields_.gradient(lev, dir), 0, 0, 1, 0);
         }
      }
   }

   void store_athena_ssprk3_mass_flux_range(int stage, int lev_begin, int lev_end)
   {
      if (!uses_athena_ssprk3_gravity_energy_correction()) {
         return;
      }
      if (stage < 0 || stage >= static_cast<int>(gravity_stage_mass_flux_.size())) {
         amrex::Abort("invalid Athena SSPRK3 mass-flux stage");
      }
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (!gravity_stage_mass_flux_[stage][lev][dir] || !fluxes_[lev][dir]) {
               amrex::Abort("Athena SSPRK3 gravity correction is missing mass-flux storage");
            }
            amrex::MultiFab::Copy(*gravity_stage_mass_flux_[stage][lev][dir], *fluxes_[lev][dir], RHO, 0, 1, 0);
         }
      }
   }

   static amrex::Real athena_ssprk3_energy_time_factor(int stage, amrex::Real dt)
   {
      return time_integrator_stages<TimeSSPRK3>()[stage].output_time.apply(dt);
   }

   static amrex::Real athena_ssprk3_carried_energy_weight(int stage)
   {
      // Stage zero has no previous gravity release to carry forward.
      return stage == 0 ? 0.0_rt
         : time_integrator_stages<TimeSSPRK3>()[stage].input_weight;
   }

public:
   void apply_athena_ssprk3_energy_release_range(
      LevelData& data,
      int stage,
      amrex::Real dt,
      int lev_begin,
      int lev_end,
      bool replace_previous_release = false
   )
   {
      if (!uses_athena_ssprk3_gravity_energy_correction()) {
         return;
      }
      const amrex::Real time_factor = athena_ssprk3_energy_time_factor(stage, dt);
      const amrex::Real carried_weight = athena_ssprk3_carried_energy_weight(stage);
      const auto start_accel = gravity_start_acceleration_offset_;
      const auto accel = gravity_acceleration_offset_;
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         if (!gravity_stage_energy_release_[lev]) {
            amrex::Abort("Athena SSPRK3 gravity correction is missing energy-release storage");
         }
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (
               !gravity_start_gradient_[lev][dir]
               || !gravity_fields_.gradient(lev, dir)
               || !gravity_stage_mass_flux_[0][lev][dir]
               || (stage >= 1 && !gravity_stage_mass_flux_[1][lev][dir])
               || (stage >= 2 && !gravity_stage_mass_flux_[2][lev][dir])
            ) {
               amrex::Abort("Athena SSPRK3 gravity correction has incomplete stage storage");
            }
         }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
         {
         for (amrex::MFIter mfi(*data[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const auto state = data[lev]->array(mfi);
            const auto previous = gravity_stage_energy_release_[lev]->array(mfi);
            const auto f0x = gravity_stage_mass_flux_[0][lev][0]->const_array(mfi);
            const auto g0x = gravity_start_gradient_[lev][0]->const_array(mfi);
            const auto gx = gravity_fields_.gradient(lev, 0)->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
            const auto f0y = gravity_stage_mass_flux_[0][lev][1]->const_array(mfi);
            const auto g0y = gravity_start_gradient_[lev][1]->const_array(mfi);
            const auto gy = gravity_fields_.gradient(lev, 1)->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
            const auto f0z = gravity_stage_mass_flux_[0][lev][2]->const_array(mfi);
            const auto g0z = gravity_start_gradient_[lev][2]->const_array(mfi);
            const auto gz = gravity_fields_.gradient(lev, 2)->const_array(mfi);
#endif
            const bool use_f1 = stage >= 1;
            const bool use_f2 = stage >= 2;
            const auto f1x = use_f1 ? gravity_stage_mass_flux_[1][lev][0]->const_array(mfi) : amrex::Array4<amrex::Real const>{};
            const auto f2x = use_f2 ? gravity_stage_mass_flux_[2][lev][0]->const_array(mfi) : amrex::Array4<amrex::Real const>{};
#if (AMREX_SPACEDIM >= 2)
            const auto f1y = use_f1 ? gravity_stage_mass_flux_[1][lev][1]->const_array(mfi) : amrex::Array4<amrex::Real const>{};
            const auto f2y = use_f2 ? gravity_stage_mass_flux_[2][lev][1]->const_array(mfi) : amrex::Array4<amrex::Real const>{};
#endif
#if (AMREX_SPACEDIM == 3)
            const auto f1z = use_f1 ? gravity_stage_mass_flux_[1][lev][2]->const_array(mfi) : amrex::Array4<amrex::Real const>{};
            const auto f2z = use_f2 ? gravity_stage_mass_flux_[2][lev][2]->const_array(mfi) : amrex::Array4<amrex::Real const>{};
#endif
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               auto flux_combo = [=] AMREX_GPU_DEVICE(
                  amrex::Array4<amrex::Real const> const& f0,
                  amrex::Array4<amrex::Real const> const& f1,
                  amrex::Array4<amrex::Real const> const& f2,
                  int ii,
                  int jj,
                  int kk
               ) noexcept -> amrex::Real {
                  if (stage == 0) {
                     return f0(ii, jj, kk, 0);
                  }
                  if (stage == 1) {
                     return 0.5_rt * (f0(ii, jj, kk, 0) + f1(ii, jj, kk, 0));
                  }
                  return (f0(ii, jj, kk, 0) + f1(ii, jj, kk, 0) + 4.0_rt * f2(ii, jj, kk, 0)) / 6.0_rt;
               };
               const amrex::Real fx_r = flux_combo(f0x, f1x, f2x, i + 1, j, k);
               const amrex::Real fx_l = flux_combo(f0x, f1x, f2x, i, j, k);
               const amrex::Real gx_offset = 0.5_rt * (start_accel[0] + accel[0]);
               const amrex::Real gx_r = -0.5_rt * (g0x(i + 1, j, k, 0) + gx(i + 1, j, k, 0)) - gx_offset;
               const amrex::Real gx_l = -0.5_rt * (g0x(i, j, k, 0) + gx(i, j, k, 0)) - gx_offset;
               amrex::Real source = fx_r * gx_r + fx_l * gx_l;
#if (AMREX_SPACEDIM >= 2)
               const amrex::Real fy_r = flux_combo(f0y, f1y, f2y, i, j + 1, k);
               const amrex::Real fy_l = flux_combo(f0y, f1y, f2y, i, j, k);
               const amrex::Real gy_offset = 0.5_rt * (start_accel[1] + accel[1]);
               const amrex::Real gy_r = -0.5_rt * (g0y(i, j + 1, k, 0) + gy(i, j + 1, k, 0)) - gy_offset;
               const amrex::Real gy_l = -0.5_rt * (g0y(i, j, k, 0) + gy(i, j, k, 0)) - gy_offset;
               source += fy_r * gy_r + fy_l * gy_l;
#endif
#if (AMREX_SPACEDIM == 3)
               const amrex::Real fz_r = flux_combo(f0z, f1z, f2z, i, j, k + 1);
               const amrex::Real fz_l = flux_combo(f0z, f1z, f2z, i, j, k);
               const amrex::Real gz_offset = 0.5_rt * (start_accel[2] + accel[2]);
               const amrex::Real gz_r = -0.5_rt * (g0z(i, j, k + 1, 0) + gz(i, j, k + 1, 0)) - gz_offset;
               const amrex::Real gz_l = -0.5_rt * (g0z(i, j, k, 0) + gz(i, j, k, 0)) - gz_offset;
               source += fz_r * gz_r + fz_l * gz_l;
#endif
               const amrex::Real release = time_factor * 0.5_rt * source;
               const amrex::Real previous_weight = replace_previous_release ? 1.0_rt : carried_weight;
               state(i, j, k, ENERGY) += release - previous_weight * previous(i, j, k, 0);
               previous(i, j, k, 0) = release;
            });
         }
         }
      }
   }

private:

   template <bool SubcyclingEnabled>
   void prepare_stage_sources(LevelData& input, int lev_begin, int lev_end, amrex::Real rhs_time)
   {
      if (!gravity_enabled(cfg_)) {
         return;
      }
      // Source fields are prepared at the same stage time as the RHS. Future
      // source terms should follow this hook rather than solving or caching
      // stage-dependent auxiliaries inside timestepping kernels.
      const WallTime gravity_start = profile_.begin();
      if constexpr (SubcyclingEnabled) {
         if (uses_subcycled_self_gravity_source_state()) {
            fill_source_state_for_rhs(input, lev_begin, lev_end, rhs_time);
            solve_gravity_potential(fluid_state_.source_state(), true);
            update_self_gravity_acceleration_offset(fluid_state_.source_state());
            profile_.end(ProfileStage::Sources, gravity_start);
            return;
         }
      }
      prepare_gravity_fields(input, true);
      update_self_gravity_acceleration_offset(input);
      profile_.end(ProfileStage::Sources, gravity_start);
   }

   template <int NComp>
   void apply_stage_sources_to_rhs(
      int lev,
      LevelData& input,
      std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
      bool use_solid_mask,
      amrex::Real rhs_time
   )
   {
      std::array<amrex::MultiFab const*, AMREX_SPACEDIM> gravity_gradient{};
      std::array<amrex::MultiFab const*, AMREX_SPACEDIM> mass_fluxes{};
      SourceStageContext source_context;
      source_context.cfg = &cfg_;
      source_context.geom = &Geom(lev);
      source_context.time = rhs_time;
      source_context.solid_mask = use_solid_mask ? solid_mask_[lev].get() : nullptr;
      source_context.solid_valid_kind = use_solid_mask ? &solid_valid_kind_[lev] : nullptr;
      if (gravity_enabled(cfg_)) {
         // Grad phi is cached solver auxiliary storage. Phi itself belongs to
         // the gravity solve; RHS source application only consumes the stage
         // gradient and, for Athena coupling, the already-computed mass fluxes.
         gravity_gradient = gravity_fields_.gradient_pointers(lev);
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            mass_fluxes[dir] = fluxes[dir];
         }
         source_context.gravity_gradient = &gravity_gradient;
         source_context.gravity_acceleration_offset = &gravity_acceleration_offset_;
         source_context.mass_fluxes = &mass_fluxes;
         source_context.suppress_athena_gravity_energy_source = suppress_athena_gravity_energy_source_;
      }
      apply_sources<NComp>(*fluid_state_.rhs()[lev], *input[lev], source_context);
   }
