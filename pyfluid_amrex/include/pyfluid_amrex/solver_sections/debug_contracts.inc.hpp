// Included inside pyfluid_amrex::AmrSolver private section.
// Development contracts: poison outputs before writes, fail early on invalid
// active fluid states, and optionally enforce accepted-state physical sanity.

   bool debug_contracts_enabled() const;
   bool accepted_physical_check_due(int step, bool force) const;
   void debug_poison_mfab(amrex::MultiFab& mf) const;
   void debug_poison_leveldata(LevelData& data, int lev_begin, int lev_end) const;
   void debug_poison_fluxes(int lev) const;
   void debug_check_mfab_finite(amrex::MultiFab const& mf, int ncomp, amrex::IntVect const& ngrow, char const* context) const;

   void debug_check_reconstruction_input(
      amrex::MultiFab const& state,
      int lev,
      amrex::Real time,
      char const* context
   ) const;
   void debug_check_valid_cell_finite(
      amrex::MultiFab const& mf,
      amrex::iMultiFab const& valid_mask,
      int ncomp,
      char const* context,
      int lev
   ) const;
   void abort_bad_physical_state(
      char const* context,
      int step,
      amrex::Real time,
      int lev,
      amrex::Long bad_count,
      amrex::Real min_rho,
      amrex::Real min_pressure,
      int first_bad_i,
      int first_bad_j,
      int first_bad_k,
      amrex::Real first_bad_rho,
      amrex::Real first_bad_pressure
   ) const;
   void debug_check_rhs_level(int lev, char const* context) const;
   void debug_check_fluxes(int lev, bool solid_masked, char const* context) const;

   template <int NComp, int EOSMethod>
   void check_active_cell_physical_state(int lev, char const* context, int step, amrex::Real time)
   {
      amrex::iMultiFab mask = valid_cell_mask(lev);
      amrex::Long active_count = 0;
      amrex::Long bad_count = 0;
      amrex::Real min_rho = std::numeric_limits<amrex::Real>::max();
      amrex::Real min_pressure = std::numeric_limits<amrex::Real>::max();
      const EOSParams eos = eos_params(cfg_);
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+:active_count,bad_count) reduction(min:min_rho,min_pressure) if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(*state_[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto lo = amrex::lbound(bx);
         const auto hi = amrex::ubound(bx);
         const auto state = state_[lev]->const_array(mfi);
         const auto valid = mask.const_array(mfi);
         for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
               for (int i = lo.x; i <= hi.x; ++i) {
                  if (valid(i, j, k, 0) == 0) {
                     continue;
                  }
                  ++active_count;
                  const amrex::Real rho = state(i, j, k, RHO);
                  amrex::Real pressure = 0.0;
                  if constexpr (has_mhd_components<NComp>()) {
                     pressure = pressure_from_conserved<EOSMethod>(load_mhd_state(state, i, j, k), eos);
                  } else {
                     pressure = pressure_from_state<EOSMethod>(state, i, j, k, eos);
                  }
                  if (!std::isfinite(rho) || !std::isfinite(pressure) || rho <= 0.0 || pressure <= 0.0) {
                     ++bad_count;
                     continue;
                  }
                  min_rho = std::min(min_rho, rho);
                  min_pressure = std::min(min_pressure, pressure);
               }
            }
         }
      }
      }
      amrex::ParallelDescriptor::ReduceLongSum(active_count);
      amrex::ParallelDescriptor::ReduceLongSum(bad_count);
      amrex::ParallelDescriptor::ReduceRealMin(min_rho);
      amrex::ParallelDescriptor::ReduceRealMin(min_pressure);
      if (active_count == 0 || bad_count == 0) {
         return;
      }

      int first_bad_i = std::numeric_limits<int>::min();
      int first_bad_j = std::numeric_limits<int>::min();
      int first_bad_k = std::numeric_limits<int>::min();
      amrex::Real first_bad_rho = std::numeric_limits<amrex::Real>::quiet_NaN();
      amrex::Real first_bad_pressure = std::numeric_limits<amrex::Real>::quiet_NaN();
      for (amrex::MFIter mfi(*state_[lev]); mfi.isValid() && first_bad_i == std::numeric_limits<int>::min(); ++mfi) {
         const amrex::Box& bx = mfi.validbox();
         const auto lo = amrex::lbound(bx);
         const auto hi = amrex::ubound(bx);
         const auto state = state_[lev]->const_array(mfi);
         const auto valid = mask.const_array(mfi);
         for (int k = lo.z; k <= hi.z && first_bad_i == std::numeric_limits<int>::min(); ++k) {
            for (int j = lo.y; j <= hi.y && first_bad_i == std::numeric_limits<int>::min(); ++j) {
               for (int i = lo.x; i <= hi.x; ++i) {
                  if (valid(i, j, k, 0) == 0) {
                     continue;
                  }
                  const amrex::Real rho = state(i, j, k, RHO);
                  amrex::Real pressure = 0.0;
                  if constexpr (has_mhd_components<NComp>()) {
                     pressure = pressure_from_conserved<EOSMethod>(load_mhd_state(state, i, j, k), eos);
                  } else {
                     pressure = pressure_from_state<EOSMethod>(state, i, j, k, eos);
                  }
                  if (!std::isfinite(rho) || !std::isfinite(pressure) || rho <= 0.0 || pressure <= 0.0) {
                     first_bad_i = i;
                     first_bad_j = j;
                     first_bad_k = k;
                     first_bad_rho = rho;
                     first_bad_pressure = pressure;
                     break;
                  }
               }
            }
         }
      }
      abort_bad_physical_state(
         context,
         step,
         time,
         lev,
         bad_count,
         min_rho,
         min_pressure,
         first_bad_i,
         first_bad_j,
         first_bad_k,
         first_bad_rho,
         first_bad_pressure
      );
   }

   template <int NComp, int EOSMethod>
   void debug_check_state_level(
      int lev,
      char const* context,
      int step,
      amrex::Real time,
      bool force_physical_check = false
   )
   {
      const bool check_physical = accepted_physical_check_due(step, force_physical_check);
      if (!cfg_.debug_check_finite && !check_physical) {
         return;
      }
      if (!state_[lev]) {
         return;
      }
      if (cfg_.debug_check_finite) {
         amrex::iMultiFab mask = valid_cell_mask(lev);
         debug_check_valid_cell_finite(*state_[lev], mask, NComp, context, lev);
      }
      if (check_physical) {
         check_active_cell_physical_state<NComp, EOSMethod>(lev, context, step, time);
      }
   }

   template <int NComp, int EOSMethod>
   void debug_check_state_all_levels(
      char const* context,
      int step,
      amrex::Real time,
      bool force_physical_check = false
   )
   {
      const bool check_physical = accepted_physical_check_due(step, force_physical_check);
      if (!cfg_.debug_check_finite && !check_physical) {
         return;
      }
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         debug_check_state_level<NComp, EOSMethod>(lev, context, step, time, force_physical_check);
      }
   }
