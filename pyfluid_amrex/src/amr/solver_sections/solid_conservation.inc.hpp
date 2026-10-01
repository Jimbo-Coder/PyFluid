// Included inside pyfluid_amrex::AmrSolver private section.
// Valid-cell conservation corrections for hydro runs with solid AMR geometry.

public:
   bool hydro_additive_total_correction_safe(std::array<amrex::Real, NVAR_HYDRO> const& density_delta) const
   {
      for (int n = 0; n < NVAR_HYDRO; ++n) {
         if (!std::isfinite(density_delta[n])) {
            return false;
         }
      }
      const EOSParams eos = eos_params(cfg_);
      const int eos_method = eos_method_code(cfg_);
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         amrex::iMultiFab mask = valid_cell_mask(lev);
         amrex::MultiFab bad(fluid_state_.state()[lev]->boxArray(), fluid_state_.state()[lev]->DistributionMap(), 1, 0);
         bad.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
         {
         for (amrex::MFIter mfi(*fluid_state_.state()[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const auto valid = mask.const_array(mfi);
            const auto s = fluid_state_.state()[lev]->const_array(mfi);
            const auto out = bad.array(mfi);
            const auto correction = density_delta;
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if (valid(i, j, k, 0) == 0) {
                  return;
               }
               const amrex::Real rho = s(i, j, k, RHO) + correction[RHO];
               const amrex::Real momx = s(i, j, k, MOM1) + correction[MOM1];
               const amrex::Real momy = s(i, j, k, MOM2) + correction[MOM2];
               const amrex::Real momz = s(i, j, k, MOM3) + correction[MOM3];
               const amrex::Real energy = s(i, j, k, ENERGY) + correction[ENERGY];
               bool ok = rho > 0.0_rt
                  && amrex::Math::isfinite(rho)
                  && amrex::Math::isfinite(momx)
                  && amrex::Math::isfinite(momy)
                  && amrex::Math::isfinite(momz)
                  && amrex::Math::isfinite(energy);
               if (ok) {
                  const amrex::Real kinetic = 0.5_rt * (momx * momx + momy * momy + momz * momz) / rho;
                  const amrex::Real pressure =
                     eos_pressure_from_density_energy_runtime(eos_method, rho, energy - kinetic, eos);
                  ok = pressure > 0.0_rt && amrex::Math::isfinite(pressure);
               }
               out(i, j, k, 0) = ok ? 0.0_rt : 1.0_rt;
            });
         }
         }
         if (bad.sum(0) > 0.0_rt) {
            return false;
         }
      }
      return true;
   }

private:

public:
   void apply_hydro_valid_conserved_scale(amrex::Real scale)
   {
      if (!(scale > 0.0_rt) || !std::isfinite(scale)) {
         return;
      }
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         amrex::iMultiFab mask = valid_cell_mask(lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
         {
         for (amrex::MFIter mfi(*fluid_state_.state()[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const auto valid = mask.const_array(mfi);
            const auto s = fluid_state_.state()[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if (valid(i, j, k, 0) == 0) {
                  return;
               }
               for (int n = 0; n < NVAR_HYDRO; ++n) {
                  s(i, j, k, n) *= scale;
               }
            });
         }
         }
      }
   }

private:

public:
   void apply_hydro_valid_conserved_delta(std::array<amrex::Real, NVAR_HYDRO> const& density_delta)
   {
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         amrex::iMultiFab mask = valid_cell_mask(lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
         {
         for (amrex::MFIter mfi(*fluid_state_.state()[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const auto valid = mask.const_array(mfi);
            const auto s = fluid_state_.state()[lev]->array(mfi);
            const auto correction = density_delta;
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if (valid(i, j, k, 0) == 0) {
                  return;
               }
               for (int n = 0; n < NVAR_HYDRO; ++n) {
                  s(i, j, k, n) += correction[n];
               }
            });
         }
         }
      }
   }

private:

public:
   bool hydro_mass_delta_safe(amrex::Real density_delta) const
   {
      if (!std::isfinite(density_delta)) {
         return false;
      }
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         amrex::iMultiFab mask = valid_cell_mask(lev);
         amrex::MultiFab bad(fluid_state_.state()[lev]->boxArray(), fluid_state_.state()[lev]->DistributionMap(), 1, 0);
         bad.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
         {
         for (amrex::MFIter mfi(*fluid_state_.state()[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const auto valid = mask.const_array(mfi);
            const auto s = fluid_state_.state()[lev]->const_array(mfi);
            const auto out = bad.array(mfi);
            const amrex::Real delta = density_delta;
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if (valid(i, j, k, 0) == 0) {
                  return;
               }
               const amrex::Real rho = s(i, j, k, RHO) + delta;
               out(i, j, k, 0) = rho > 0.0_rt && amrex::Math::isfinite(rho) ? 0.0_rt : 1.0_rt;
            });
         }
         }
         if (bad.sum(0) > 0.0_rt) {
            return false;
         }
      }
      return true;
   }

private:

public:
   void apply_hydro_valid_mass_delta(amrex::Real density_delta)
   {
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         amrex::iMultiFab mask = valid_cell_mask(lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
         {
         for (amrex::MFIter mfi(*fluid_state_.state()[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const auto valid = mask.const_array(mfi);
            const auto s = fluid_state_.state()[lev]->array(mfi);
            const amrex::Real delta = density_delta;
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if (valid(i, j, k, 0) == 0) {
                  return;
               }
               s(i, j, k, RHO) += delta;
            });
         }
         }
      }
   }

private:

   bool mass_conserving_boundary(BoundaryKind kind) const
   {
      return kind == BoundaryKind::Periodic
         || kind == BoundaryKind::Reflecting
         || kind == BoundaryKind::Symmetry;
   }

   bool closed_mass_domain() const
   {
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         if (
            !mass_conserving_boundary(side_boundary_kind(cfg_, dir, false))
            || !mass_conserving_boundary(side_boundary_kind(cfg_, dir, true))
         ) {
            return false;
         }
      }
      return true;
   }

   bool project_closed_hydro_solid_amr_mass(amrex::Real target_mass)
   {
      if (
         is_mhd(cfg_)
         || finestLevel() <= 1
         || !solid_geometry_enabled(cfg_)
         || solid_geometry_changes_cell_mask(cfg_)
         || !closed_mass_domain()
      ) {
         return false;
      }
      HydroConservedTotals current = hydro_valid_conserved_totals();
      if (current.volume <= 0.0_rt) {
         return false;
      }
      const amrex::Real delta_mass = target_mass - current.value[RHO];
      const amrex::Real scale = amrex::max(amrex::Math::abs(target_mass), 1.0_rt);
      if (amrex::Math::abs(delta_mass) <= 32.0_rt * std::numeric_limits<amrex::Real>::epsilon() * scale) {
         return false;
      }
      const amrex::Real density_delta = delta_mass / current.volume;
      if (!hydro_mass_delta_safe(density_delta)) {
         return false;
      }
      apply_hydro_valid_mass_delta(density_delta);
      return true;
   }

   void apply_hydro_valid_conserved_total_correction(HydroConservedTotals const& target)
   {
      HydroConservedTotals current = hydro_valid_conserved_totals();
      if (current.volume <= 0.0_rt) {
         amrex::Abort("cannot apply AMR solid regrid conservation correction without valid fluid volume");
      }
      std::array<amrex::Real, NVAR_HYDRO> density_delta{};
      for (int n = 0; n < NVAR_HYDRO; ++n) {
         density_delta[n] = (target.value[n] - current.value[n]) / current.volume;
      }
      if (!hydro_additive_total_correction_safe(density_delta)) {
         const amrex::Real mass_scale = target.value[RHO] / current.value[RHO];
         apply_hydro_valid_conserved_scale(mass_scale);
         HydroConservedTotals scaled = hydro_valid_conserved_totals();
         if (scaled.volume > 0.0_rt) {
            std::array<amrex::Real, NVAR_HYDRO> energy_delta{};
            energy_delta[ENERGY] = (target.value[ENERGY] - scaled.value[ENERGY]) / scaled.volume;
            if (hydro_additive_total_correction_safe(energy_delta)) {
               apply_hydro_valid_conserved_delta(energy_delta);
            }
         }
         return;
      }
      apply_hydro_valid_conserved_delta(density_delta);
   }
