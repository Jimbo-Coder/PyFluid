// Included inside pyfluid_amrex::AmrSolver private section.
// Constrained-transport magnetic-field advancement helpers.

   DiffusionStats advance_ct_ideal_level(int lev, amrex::Real dt)
   {
      const auto fluxes = flux_ptrs(lev);
      if (ct_flux_derived_emf_is_available(fluxes)) {
#if (AMREX_SPACEDIM >= 2)
         fill_edge_electric_field_from_mhd_fluxes(edge_e_[lev], fluxes, flux_owner_mask_ptrs(lev), Geom(lev));
#else
         fill_edge_electric_field_from_mhd_fluxes(edge_e_[lev], fluxes, Geom(lev));
#endif
         enforce_edge_electric_boundary(edge_e_[lev], Geom(lev), boundary_data_);
         apply_edge_electric_curl(face_b_[lev], edge_e_[lev], Geom(lev), dt, cfg_);
      } else {
         advance_ct_magnetic_field(face_b_[lev], edge_e_[lev], *fluid_state_.state()[lev], Geom(lev), dt, 0.0, cfg_);
      }
      add_edge_electric_to_registers(lev, dt);
      apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
      sync_cell_b_from_faces(*fluid_state_.next()[lev], face_b_[lev], Geom(lev), false);
      return {};
   }

   template <int DiffusionMethod>
   DiffusionStats advance_ct_inactive_magnetic_resistivity_level(int lev, amrex::Real dt)
   {
      DiffusionStats stats;
#if (AMREX_SPACEDIM == 2)
      if (!ct_enabled(cfg_) || !resistivity_enabled(cfg_)) {
         return stats;
      }
      if constexpr (DiffusionMethod == DiffusionImplicit) {
         return stats;
      } else if constexpr (DiffusionMethod == DiffusionExplicit) {
         apply_diffusion_boundary(*fluid_state_.state()[lev], Geom(lev), cfg_);
         amrex::MultiFab::Copy(*fluid_state_.scratch()[lev], *fluid_state_.state()[lev], 0, 0, num_state_components(), 0);
         explicit_ct_inactive_magnetic_resistivity_update(*fluid_state_.scratch()[lev], *fluid_state_.state()[lev], Geom(lev), cfg_, dt);
         apply_diffusion_boundary(*fluid_state_.scratch()[lev], Geom(lev), cfg_);
         for (int inactive_component = AMREX_SPACEDIM; inactive_component < 3; ++inactive_component) {
            const int state_component = B1 + inactive_component;
            amrex::MultiFab::Copy(*fluid_state_.state()[lev], *fluid_state_.scratch()[lev], state_component, state_component, 1, 0);
         }
         apply_diffusion_boundary(*fluid_state_.state()[lev], Geom(lev), cfg_);
      }
#else
      (void)lev;
      (void)dt;
#endif
      return stats;
   }

   template <int DiffusionMethod>
   DiffusionStats advance_ct_inactive_magnetic_resistivity_all_levels(amrex::Real dt)
   {
      DiffusionStats combined;
#if (AMREX_SPACEDIM == 2)
      if (ct_enabled(cfg_) && resistivity_enabled(cfg_)) {
         for (int lev = 0; lev <= finestLevel(); ++lev) {
            combined = combine_diffusion_stats(
               combined,
               advance_ct_inactive_magnetic_resistivity_level<DiffusionMethod>(lev, dt)
            );
         }
      }
#else
      (void)dt;
#endif
      return combined;
   }

   template <int DiffusionMethod>
   DiffusionStats advance_ct_level(int lev, amrex::Real dt)
   {
      DiffusionStats combined = advance_ct_ideal_level(lev, dt);

      if constexpr (DiffusionMethod == DiffusionNone) {
         return combined;
      }
      if (!resistivity_enabled(cfg_)) {
         return combined;
      }
      if constexpr (DiffusionMethod == DiffusionExplicit) {
         advance_ct_resistive_magnetic_field_from_faces(face_b_[lev], edge_e_[lev], Geom(lev), dt, cfg_.resistivity, cfg_);
         add_edge_electric_to_registers(lev, dt);
         apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
         sync_cell_b_from_faces(*fluid_state_.next()[lev], face_b_[lev], Geom(lev), false);
         return combined;
      }
#if (AMREX_SPACEDIM >= 2)
      combined = combine_diffusion_stats(
         combined,
         advance_ct_resistive_magnetic_field_implicit(
            face_b_[lev], edge_e_[lev], *fluid_state_.next()[lev], Geom(lev), dt, cfg_
         )
      );
      add_edge_electric_to_registers(lev, dt);
#else
      amrex::Abort("implicit resistive CT requires a 2D or 3D executable");
#endif
      apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
      sync_cell_b_from_faces(*fluid_state_.next()[lev], face_b_[lev], Geom(lev), false);
      return combined;
   }

   template <int DiffusionMethod>
   DiffusionStats advance_ct_all_levels(amrex::Real dt)
   {
      DiffusionStats combined;
      if constexpr (DiffusionMethod == DiffusionImplicit) {
         for (int lev = 0; lev <= finestLevel(); ++lev) {
            combined = combine_diffusion_stats(combined, advance_ct_level<DiffusionMethod>(lev, dt));
         }
         reflux_edge_flux_registers_to_face_b();
         for (int lev = 0; lev <= finestLevel(); ++lev) {
            sync_cell_b_from_faces(*fluid_state_.next()[lev], face_b_[lev], Geom(lev), false);
         }
         return combined;
      }
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         combined = combine_diffusion_stats(combined, advance_ct_level<DiffusionMethod>(lev, dt));
      }
      reflux_edge_flux_registers_to_face_b();
      return combined;
   }
