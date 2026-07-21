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
      amrex::Real first_bad_pressure,
      std::array<amrex::Real, NVAR_HYDRO> const& first_bad_conserved
   ) const;
   void debug_check_rhs_level(int lev, char const* context) const;
   void debug_check_fluxes(int lev, bool solid_masked, char const* context) const;

   template <bool MHD, int EOSMethod>
   void check_active_cell_physical_state_method(
      int lev,
      char const* context,
      int step,
      amrex::Real time
   );
   void check_active_cell_physical_state(
      int lev,
      char const* context,
      int step,
      amrex::Real time
   );
   void debug_check_state_level(
      int lev,
      char const* context,
      int step,
      amrex::Real time,
      bool force_physical_check = false
   );
   void debug_check_state_all_levels(
      char const* context,
      int step,
      amrex::Real time,
      bool force_physical_check = false
   );
   void debug_check_relativistic_stage_range(
      LevelData const& state,
      int lev_begin,
      int lev_end,
      char const* context,
      amrex::Real time
   );
   void run_evolution_preflight(amrex::Real time);
