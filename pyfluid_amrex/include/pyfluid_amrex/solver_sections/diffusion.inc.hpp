// Included inside pyfluid_amrex::AmrSolver private section.
// Viscosity diffusion helpers. MHD resistivity belongs to CT evolution.

   template <int DiffusionMethod, int NComp>
   DiffusionStats apply_diffusion_level(int lev, amrex::Real time, amrex::Real dt, RunConfig const& diffusion_cfg)
   {
      fill_level_boundaries<true>(state_, lev, time);
      if constexpr (DiffusionMethod == DiffusionExplicit) {
         apply_explicit_diffusion_level<NComp>(lev, time, dt, diffusion_cfg);
         return {};
      }
      DiffusionStats stats = apply_viscosity_diffusion<DiffusionMethod, NComp>(
         *state_[lev],
         *scratch_[lev],
         *rhs_[lev],
         Geom(lev),
         dt,
         diffusion_cfg
      );
      if (viscosity_enabled(diffusion_cfg)) {
         fill_level_boundaries<true>(state_, lev, time);
         fill_viscosity_fluxes_level<NComp, true>(lev, time, diffusion_cfg);
         add_fluxes_to_registers(lev, dt);
      }
      return stats;
   }

   template <int NComp>
   void apply_explicit_diffusion_level(int lev, amrex::Real time, amrex::Real dt, RunConfig const& diffusion_cfg)
   {
      if (viscosity_enabled(diffusion_cfg)) {
         apply_explicit_diffusion_level_impl<NComp, true>(lev, time, dt, diffusion_cfg);
      }
   }

   template <int NComp, bool DiffuseMomentum>
   void fill_viscosity_fluxes_level(int lev, amrex::Real time, RunConfig const& diffusion_cfg)
   {
      const auto fluxes = flux_ptrs(lev);
      const bool use_solid_mask = solid_active(lev) && !has_mhd_components<NComp>();
      fill_viscous_fluxes<NComp, DiffuseMomentum>(
         fluxes,
         *state_[lev],
         Geom(lev),
         diffusion_cfg,
         time,
         use_solid_mask ? solid_mask_[lev].get() : nullptr
      );
   }

   template <int NComp, bool DiffuseMomentum>
   void apply_explicit_diffusion_level_impl(int lev, amrex::Real time, amrex::Real dt, RunConfig const& diffusion_cfg)
   {
      const auto fluxes = flux_ptrs(lev);
      const bool use_solid_mask = solid_active(lev) && !has_mhd_components<NComp>();
      fill_viscosity_fluxes_level<NComp, DiffuseMomentum>(lev, time, diffusion_cfg);
      if (use_solid_mask) {
         compute_rhs_from_fluxes_masked<NComp>(*rhs_[lev], fluxes, *solid_mask_[lev], Geom(lev), &solid_valid_kind_[lev]);
      } else {
         compute_rhs_from_fluxes_mesh_divergence<NComp>(
            *rhs_[lev],
            fluxes,
            Geom(lev),
            mesh_coordinate_code(diffusion_cfg.mesh_coordinate),
            mesh_axes_layout(diffusion_cfg)
         );
         add_viscous_stress_connection_sources<NComp, DiffuseMomentum>(
            *rhs_[lev], *state_[lev], Geom(lev), diffusion_cfg
         );
      }
      add_fluxes_to_registers(lev, dt);
      if (use_solid_mask) {
         add_scaled_components_masked<NComp>(*state_[lev], *rhs_[lev], *solid_mask_[lev], dt, &solid_valid_kind_[lev]);
      } else {
         add_scaled_components<NComp>(*state_[lev], *rhs_[lev], dt);
      }
      apply_diffusion_boundary(*state_[lev], Geom(lev), diffusion_cfg);
   }

   template <int DiffusionMethod, int NComp>
   DiffusionStats apply_diffusion_all_levels(amrex::Real time, amrex::Real dt, RunConfig const& diffusion_cfg)
   {
      DiffusionStats combined;
      fill_all_level_boundaries<false>(state_, time);
      if constexpr (DiffusionMethod == DiffusionExplicit) {
         if (finestLevel() > 0) {
            reset_flux_registers();
         }
         for (int lev = 0; lev <= finestLevel(); ++lev) {
            apply_explicit_diffusion_level<NComp>(lev, time, dt, diffusion_cfg);
         }
         if (finestLevel() > 0) {
            for (int lev = finestLevel(); lev > 0; --lev) {
               reflux_level_to_state(lev);
            }
         }
         return combined;
      }
      if (finestLevel() > 0) {
         reset_flux_registers();
      }
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         // Coarser levels have already advanced implicitly. Refresh each fine
         // solve from that current hierarchy state rather than retaining the
         // pre-solve coarse boundary captured by fill_all_level_boundaries.
         fill_level_boundaries<true>(state_, lev, time);
         combined = combine_diffusion_stats(
            combined,
            apply_viscosity_diffusion<DiffusionMethod, NComp>(
               *state_[lev],
               *scratch_[lev],
               *rhs_[lev],
               Geom(lev),
               dt,
               diffusion_cfg
            )
         );
         if (viscosity_enabled(diffusion_cfg)) {
            fill_level_boundaries<true>(state_, lev, time);
            fill_viscosity_fluxes_level<NComp, true>(lev, time, diffusion_cfg);
            add_fluxes_to_registers(lev, dt);
         }
      }
      if (finestLevel() > 0) {
         for (int lev = finestLevel(); lev > 0; --lev) {
            reflux_level_to_state(lev);
         }
      }
      return combined;
   }
