// Included inside pyfluid_amrex::AmrSolver private section.
// Flux-register accumulation, cut-cell weighting, and reflux application helpers.

   std::array<amrex::MultiFab*, AMREX_SPACEDIM> flux_ptrs(int lev)
   {
      std::array<amrex::MultiFab*, AMREX_SPACEDIM> ptrs;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         ptrs[dir] = fluxes_[lev][dir].get();
      }
      return ptrs;
   }

#if (AMREX_SPACEDIM >= 2)
   std::array<amrex::iMultiFab const*, AMREX_SPACEDIM> flux_owner_mask_ptrs(int lev) const
   {
      std::array<amrex::iMultiFab const*, AMREX_SPACEDIM> ptrs{};
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         ptrs[dir] = flux_owner_masks_[lev][dir].get();
      }
      return ptrs;
   }
#endif

   std::array<amrex::FArrayBox const*, AMREX_SPACEDIM> flux_fabs(int lev, amrex::MFIter const& mfi) const
   {
      std::array<amrex::FArrayBox const*, AMREX_SPACEDIM> ptrs;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         ptrs[dir] = &((*fluxes_[lev][dir])[mfi]);
      }
      return ptrs;
   }

   amrex::Real cell_volume(int lev) const
   {
      return cartesian_cell_volume(Geom(lev));
   }

   amrex::Real face_area(int lev, int dir) const
   {
      return cartesian_face_area(Geom(lev), dir);
   }

   bool cutcell_flux_register_needed(int fine_lev) const
   {
      return !is_mhd(cfg_)
         && solid_geometry_enabled(cfg_)
         && fine_lev > 0
         && fine_lev <= finestLevel()
         && solid_volume_fraction_[fine_lev - 1] != nullptr
         && solid_volume_fraction_[fine_lev] != nullptr
         && solid_aperture_[0][fine_lev - 1] != nullptr
         && solid_aperture_[0][fine_lev] != nullptr;
   }

   bool curvilinear_flux_register_needed(int fine_lev) const
   {
      return mesh_is_curvilinear()
         && fine_lev > 0
         && fine_lev <= finestLevel();
   }

   bool patch_union_flux_register_needed(int fine_lev) const
   {
      return patch_union_domain_enabled()
         && fine_lev > 0
         && fine_lev <= finestLevel();
   }

   bool mapped_patch_flux_register_needed(int fine_lev) const
   {
      return mixed_patch_system_enabled()
         && !mhd_enabled()
         && fine_lev > 0
         && fine_lev <= finestLevel();
   }

   bool weighted_flux_register_needed(int fine_lev) const
   {
      return cutcell_flux_register_needed(fine_lev)
         || curvilinear_flux_register_needed(fine_lev)
         || patch_union_flux_register_needed(fine_lev)
         || mapped_patch_flux_register_needed(fine_lev);
   }

   void fill_coarse_reflux_volume(int fine_lev)
   {
      if (!weighted_flux_register_needed(fine_lev)) {
         if (fine_lev > 0 && fine_lev < static_cast<int>(coarse_reflux_volume_.size())) {
            coarse_reflux_volume_[fine_lev].reset();
         }
         return;
      }
      const int coarse_lev = fine_lev - 1;
      bool needs_fill = false;
      if (
         !coarse_reflux_volume_[fine_lev]
         || !coarse_reflux_volume_[fine_lev]->boxArray().CellEqual(state_[coarse_lev]->boxArray())
      ) {
         coarse_reflux_volume_[fine_lev] = std::make_unique<amrex::MultiFab>(
            state_[coarse_lev]->boxArray(),
            state_[coarse_lev]->DistributionMap(),
            1,
            0
         );
         needs_fill = true;
      }
      constexpr amrex::Real alpha_floor = 1.0e-12;
      const bool use_cutcell = cutcell_flux_register_needed(fine_lev);
      const bool use_curvilinear = curvilinear_flux_register_needed(fine_lev);
      const bool use_mapped_patches = mapped_patch_flux_register_needed(fine_lev);
      if (!use_cutcell && !use_mapped_patches && !needs_fill) {
         return;
      }
      const int coordinate = mesh_coordinate_kind();
      const MeshAxes axes = active_mesh_axes();
      const auto problo = Geom(coarse_lev).ProbLoArray();
      const auto dx = Geom(coarse_lev).CellSizeArray();
      const amrex::Real cartesian_volume = cell_volume(coarse_lev);
      const PatchGridSystem patch_grid_system = use_mapped_patches
         ? production_patch_grid_system_for_level(coarse_lev)
         : PatchGridSystem{};
      const PatchLayout patch_layout = use_mapped_patches
         ? production_patch_layout_for_level(coarse_lev)
         : PatchLayout{};
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(*coarse_reflux_volume_[fine_lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const PatchDescriptor layout_patch = use_mapped_patches
            ? patch_layout_descriptor_containing_box(patch_layout, mfi.validbox())
            : PatchDescriptor{};
         const PatchGridDescriptor grid_patch = use_mapped_patches
            ? patch_grid_descriptor_for_layout_patch(patch_grid_system, layout_patch)
            : PatchGridDescriptor{};
         const auto alpha = use_cutcell
            ? solid_volume_fraction_[coarse_lev]->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
         const auto out = coarse_reflux_volume_[fine_lev]->array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real volume = cartesian_volume;
            if (use_mapped_patches) {
               const PatchCoordinate local =
                  patch_layout_cell_center_coordinate(grid_patch, layout_patch, i, j, k);
               volume = patch_cell_volume(grid_patch, local);
            } else if (use_curvilinear) {
               volume = curvilinear_cell_volume_scale(coordinate, axes, problo, dx, i, j, k);
            }
            if (use_cutcell) {
               const amrex::Real fluid_fraction = alpha(i, j, k, 0);
               volume *= (fluid_fraction > alpha_floor ? fluid_fraction : 1.0);
            }
            out(i, j, k, 0) = volume;
         });
      }
      }
   }

   bool prepare_weighted_flux_register(int fine_lev)
   {
      if (fine_lev <= 0 || fine_lev > finestLevel() || !flux_registers_[fine_lev]) {
         return false;
      }
      if (!weighted_flux_register_needed(fine_lev)) {
         flux_registers_[fine_lev]->setCrseVolume(nullptr);
         return false;
      }
      fill_coarse_reflux_volume(fine_lev);
      flux_registers_[fine_lev]->setCrseVolume(coarse_reflux_volume_[fine_lev].get());
      return true;
   }

   void fill_area_weighted_flux_fabs(
      int lev,
      amrex::MFIter const& mfi,
      std::array<amrex::FArrayBox, AMREX_SPACEDIM>& weighted
   ) const
   {
      constexpr amrex::Real alpha_floor = 1.0e-12;
      const int ncomp = num_state_components();
      const bool use_cutcell = cutcell_flux_register_needed(lev);
      const bool use_curvilinear = mesh_is_curvilinear();
      const bool use_patch_union = patch_union_domain_enabled();
      const bool use_mapped_patches = mixed_patch_system_enabled() && !mhd_enabled();
      const bool zero_ct_magnetic_fluxes = ct_enabled(cfg_);
      const int coordinate = mesh_coordinate_kind();
      const MeshAxes axes = active_mesh_axes();
      const auto problo = Geom(lev).ProbLoArray();
      const auto dx = Geom(lev).CellSizeArray();
      const auto domain = Geom(lev).Domain();
      const auto lo = domain.smallEnd();
      const auto hi = domain.bigEnd();
      const auto domain_lo = amrex::lbound(domain);
      const auto domain_hi = amrex::ubound(domain);
      const PatchDomainBoxes patch_domain = patch_union_domain_boxes(lev);
      const auto alpha = use_cutcell
         ? solid_volume_fraction_[lev]->const_array(mfi)
         : amrex::Array4<amrex::Real const>{};
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         amrex::FArrayBox const& flux = (*fluxes_[lev][dir])[mfi];
         const amrex::Box& bx = flux.box();
         weighted[dir].resize(bx, ncomp);
         const amrex::Real cartesian_area = face_area(lev, dir);
         const auto src = flux.const_array();
         const auto aperture = use_cutcell
            ? solid_aperture_[dir][lev]->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
         const auto dst = weighted[dir].array();
         amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            if (zero_ct_magnetic_fluxes && n >= B1 && n <= B3) {
               dst(i, j, k, n) = 0.0;
               return;
            }
            const amrex::Real area = use_mapped_patches
               ? 1.0
               : (
                  use_curvilinear
                     ? curvilinear_face_area_scale(dir, coordinate, axes, problo, dx, i, j, k)
                     : cartesian_area
               );
            amrex::Real open = use_cutcell ? aperture(i, j, k, 0) : 1.0;
            int li = i;
            int lj = j;
            int lk = k;
            int ri = i;
            int rj = j;
            int rk = k;
            if (dir == 0) {
               li = i - 1;
            }
#if (AMREX_SPACEDIM >= 2)
            else if (dir == 1) {
               lj = j - 1;
            }
#endif
#if (AMREX_SPACEDIM == 3)
            else {
               lk = k - 1;
            }
#endif
            bool left_in_domain = li >= lo[0] && li <= hi[0];
            bool right_in_domain = ri >= lo[0] && ri <= hi[0];
#if (AMREX_SPACEDIM >= 2)
            left_in_domain = left_in_domain && lj >= lo[1] && lj <= hi[1];
            right_in_domain = right_in_domain && rj >= lo[1] && rj <= hi[1];
#endif
#if (AMREX_SPACEDIM == 3)
            left_in_domain = left_in_domain && lk >= lo[2] && lk <= hi[2];
            right_in_domain = right_in_domain && rk >= lo[2] && rk <= hi[2];
#endif
            if (use_cutcell && left_in_domain && alpha(li, lj, lk, 0) <= alpha_floor) {
               open = 0.0;
            }
            if (use_cutcell && right_in_domain && alpha(ri, rj, rk, 0) <= alpha_floor) {
               open = 0.0;
            }
            if (use_patch_union) {
               if (!patch_domain_face_is_interior(patch_domain, domain_lo, domain_hi, dir, i, j, k)) {
                  open = 0.0;
               }
            }
            dst(i, j, k, n) = area * open * src(i, j, k, n);
         });
      }
   }

   std::array<amrex::FArrayBox const*, AMREX_SPACEDIM> area_weighted_flux_fabs(
      int lev,
      amrex::MFIter const& mfi,
      std::array<amrex::FArrayBox, AMREX_SPACEDIM>& weighted
   ) const
   {
      fill_area_weighted_flux_fabs(lev, mfi, weighted);
      std::array<amrex::FArrayBox const*, AMREX_SPACEDIM> ptrs;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         ptrs[dir] = &weighted[dir];
      }
      return ptrs;
   }

   void add_fluxes_to_registers(int lev, amrex::Real weight)
   {
      if (weight == 0.0 || finestLevel() == 0) {
         return;
      }
      if (lev < finestLevel() && flux_registers_[lev + 1]) {
         const bool weighted_register = prepare_weighted_flux_register(lev + 1);
         for (amrex::MFIter mfi(*state_[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            if (weighted_register) {
               std::array<amrex::FArrayBox, AMREX_SPACEDIM> weighted;
               flux_registers_[lev + 1]->CrseAdd(
                  mfi,
                  area_weighted_flux_fabs(lev, mfi, weighted),
                  Geom(lev).CellSize(),
                  weight,
                  amrex::RunOn::Host
               );
            } else {
               flux_registers_[lev + 1]->CrseAdd(mfi, flux_fabs(lev, mfi), Geom(lev).CellSize(), weight, amrex::RunOn::Host);
            }
         }
      }
      if (lev > 0 && flux_registers_[lev]) {
         const bool weighted_register = prepare_weighted_flux_register(lev);
         for (amrex::MFIter mfi(*state_[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            if (weighted_register) {
               std::array<amrex::FArrayBox, AMREX_SPACEDIM> weighted;
               flux_registers_[lev]->FineAdd(
                  mfi,
                  area_weighted_flux_fabs(lev, mfi, weighted),
                  Geom(lev).CellSize(),
                  weight,
                  amrex::RunOn::Host
               );
            } else {
               flux_registers_[lev]->FineAdd(mfi, flux_fabs(lev, mfi), Geom(lev).CellSize(), weight, amrex::RunOn::Host);
            }
         }
      }
   }

#if (AMREX_SPACEDIM == 3)
   amrex::Array<amrex::FArrayBox const*, 3> edge_electric_fabs(int lev, amrex::MFIter const& mfi) const
   {
      return {
         &((*edge_e_[lev].e[0])[mfi]),
         &((*edge_e_[lev].e[1])[mfi]),
         &((*edge_e_[lev].e[2])[mfi])
      };
   }
#endif

   void fill_edge_line_integral_fab(
      int lev,
      int component,
      amrex::FArrayBox const& edge_electric,
      amrex::FArrayBox& edge_line_integral
   ) const
   {
      const int ncomp = edge_electric.nComp();
      const amrex::Box& bx = edge_electric.box();
      edge_line_integral.resize(bx, ncomp);
      const int coordinate = mesh_coordinate_kind();
      const MeshAxes axes = active_mesh_axes();
      const auto problo = Geom(lev).ProbLoArray();
      const auto dx = Geom(lev).CellSizeArray();
      const auto src = edge_electric.const_array();
      const auto dst = edge_line_integral.array();
      const int edge_component = component;
      amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
         dst(i, j, k, n) = ct_edge_line_integral(
            src(i, j, k, n),
            edge_component,
            coordinate,
            axes,
            problo,
            dx,
            i,
            j,
            k
         );
      });
   }

#if (AMREX_SPACEDIM == 3)
   amrex::Array<amrex::FArrayBox const*, 3> edge_line_integral_fabs(
      int lev,
      amrex::MFIter const& mfi,
      amrex::Array<amrex::FArrayBox, 3>& edge_line
   ) const
   {
      for (int component = 0; component < 3; ++component) {
         fill_edge_line_integral_fab(lev, component, (*edge_e_[lev].e[component])[mfi], edge_line[component]);
      }
      return {&edge_line[0], &edge_line[1], &edge_line[2]};
   }
#endif

   void add_edge_electric_to_registers(int lev, amrex::Real dt)
   {
      if (!ct_enabled(cfg_) || dt == 0.0 || finestLevel() == 0) {
         return;
      }
#if (AMREX_SPACEDIM == 2)
      if (lev < finestLevel() && edge_flux_registers_[lev + 1]) {
         for (amrex::MFIter mfi(*state_[lev]); mfi.isValid(); ++mfi) {
            amrex::FArrayBox edge_line;
            fill_edge_line_integral_fab(lev, 2, (*edge_e_[lev].e[2])[mfi], edge_line);
            edge_flux_registers_[lev + 1]->CrseAdd(mfi, edge_line, dt);
         }
      }
      if (lev > 0 && edge_flux_registers_[lev]) {
         for (amrex::MFIter mfi(*state_[lev]); mfi.isValid(); ++mfi) {
            amrex::FArrayBox edge_line;
            fill_edge_line_integral_fab(lev, 2, (*edge_e_[lev].e[2])[mfi], edge_line);
            edge_flux_registers_[lev]->FineAdd(mfi, edge_line, dt);
         }
      }
#elif (AMREX_SPACEDIM == 3)
      if (lev < finestLevel() && edge_flux_registers_[lev + 1]) {
         for (amrex::MFIter mfi(*state_[lev]); mfi.isValid(); ++mfi) {
            amrex::Array<amrex::FArrayBox, 3> edge_line;
            edge_flux_registers_[lev + 1]->CrseAdd(mfi, edge_line_integral_fabs(lev, mfi, edge_line), dt);
         }
      }
      if (lev > 0 && edge_flux_registers_[lev]) {
         for (amrex::MFIter mfi(*state_[lev]); mfi.isValid(); ++mfi) {
            amrex::Array<amrex::FArrayBox, 3> edge_line;
            edge_flux_registers_[lev]->FineAdd(mfi, edge_line_integral_fabs(lev, mfi, edge_line), dt);
         }
      }
#endif
   }

   void reflux_edge_flux_registers_to_face_b()
   {
      if (!ct_enabled(cfg_)) {
         return;
      }
      for (int lev = finestLevel(); lev > 0; --lev) {
         reflux_edge_flux_register_to_face_b(lev);
      }
   }

   void reflux_edge_flux_register_to_face_b(int fine_lev)
   {
      if (!ct_enabled(cfg_)) {
         return;
      }
#if (AMREX_SPACEDIM >= 2)
      if (fine_lev > 0 && fine_lev <= finestLevel() && edge_flux_registers_[fine_lev]) {
         auto coarse_faces = face_b_array(face_b_[fine_lev - 1]);
         edge_flux_registers_[fine_lev]->Reflux(coarse_faces, mesh_coordinate_kind(), active_mesh_axes());
         apply_face_b_boundary(face_b_[fine_lev - 1], Geom(fine_lev - 1), boundary_data_);
      }
#else
      (void)fine_lev;
#endif
   }

   void reflux_next_levels()
   {
      for (int lev = finestLevel(); lev > 0; --lev) {
         reflux_level_to_next(lev);
      }
   }

   void reflux_level_to_next(int fine_lev)
   {
      if (fine_lev > 0 && fine_lev <= finestLevel() && flux_registers_[fine_lev]) {
         flux_registers_[fine_lev]->Reflux(*next_[fine_lev - 1], 0, 0, num_state_components());
         apply_gravity_energy_source_register(fine_lev, *next_[fine_lev - 1]);
      }
   }

   void reflux_level_to_state(int fine_lev)
   {
      if (fine_lev > 0 && fine_lev <= finestLevel() && flux_registers_[fine_lev]) {
         flux_registers_[fine_lev]->Reflux(*state_[fine_lev - 1], 0, 0, num_state_components());
         apply_gravity_energy_source_register(fine_lev, *state_[fine_lev - 1]);
      }
   }
