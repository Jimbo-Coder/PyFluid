// Included inside pyfluid_amrex::AmrSolver private section.
// AMR tagging criteria.

   void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int /*ngrow*/) override
   {
      if (lev < static_cast<int>(latest_auto_tagged_cells_.size())) {
         latest_auto_tagged_cells_[lev] = 0;
         for (auto& dir_counts : latest_auto_direction_tagged_cells_) {
            if (lev < static_cast<int>(dir_counts.size())) {
               dir_counts[lev] = 0;
            }
         }
         for (auto& dir_strength : latest_auto_direction_strength_) {
            if (lev < static_cast<int>(dir_strength.size())) {
               dir_strength[lev] = 0.0;
            }
         }
      }
      tags.setVal(amrex::TagBox::CLEAR);
      if (!state_[lev]) {
         return;
      }

      const bool initial_refine_time = cfg_.initial_refine != "none"
         && cfg_.initial_refine_levels > 0
         && time <= 64.0 * std::numeric_limits<amrex::Real>::epsilon();
      const bool fixed_initial_refine = cfg_.initial_refine == "box"
         && cfg_.initial_refine_fixed_levels > 0;
      const int coordinate = mesh_coordinate_kind();
      const MeshAxes axes = active_mesh_axes();
      const bool auto_refinement_resolved = auto_refinement_resolved_at_level(lev);
      const bool solid_phi_tagging_enabled = solid_geometry_enabled(cfg_)
         && !auto_refinement_resolved
         && cfg_.solid_refine_band_cells > 0.0;
      const auto problo = Geom(lev).ProbLoArray();
      const auto dx = Geom(lev).CellSizeArray();
      const SolidGeometryEvaluator solid_params = solid_geometry_evaluator(cfg_, time);
      amrex::Real solid_refine_dx = dx[0];
#if (AMREX_SPACEDIM >= 2)
      solid_refine_dx = amrex::max(solid_refine_dx, dx[1]);
#endif
#if (AMREX_SPACEDIM == 3)
      solid_refine_dx = amrex::max(solid_refine_dx, dx[2]);
#endif
      const amrex::Real solid_refine_band = cfg_.solid_refine_band_cells * solid_refine_dx;
      const amrex::Real solid_curvature_error = solid_params.curved_boundary_error(time, solid_refine_dx);
      const bool tag_solid_phi_band = solid_phi_tagging_enabled
         && (cfg_.solid_curvature_error <= 0.0 || solid_curvature_error > cfg_.solid_curvature_error);
      const bool solid_tagging = solid_active(lev) && tag_solid_phi_band;

      if (cfg_.tag_metric == "none" && !initial_refine_time && !fixed_initial_refine && !solid_tagging && !tag_solid_phi_band) {
         return;
      }
      // Error indicators use neighboring cells, including across coarse-fine
      // patch boundaries. Use the normal hierarchy FillPatch contract so AMR
      // never tags stale fine-level ghosts and refines its own box edges.
      fill_level_boundaries<false>(state_, lev, time);

      const bool mhd = is_mhd(cfg_);
      const amrex::Real lohner_threshold = cfg_.tag_threshold;
      const auto amr_min_feature_lengths = min_feature_lengths();
      const amrex::Real min_refine_radius = cfg_.min_refine_radius;
      const EOSParams eos = eos_params(cfg_);
      const int eos_method = eos_method_code(cfg_);
      const char tagval = amrex::TagBox::SET;
      const int initial_refine_levels = std::min(maxLevel(), cfg_.initial_refine_levels);
      const int fixed_initial_refine_levels = std::min(maxLevel(), cfg_.initial_refine_fixed_levels);
      const bool initial_box_refine = cfg_.initial_refine == "box"
         && (
            (initial_refine_time && lev < initial_refine_levels)
            || (lev < fixed_initial_refine_levels)
         );
      const bool initial_refine_only = cfg_.tag_metric == "none";
      const bool fixed_initial_hierarchy_only = initial_refine_time
         && cfg_.initial_refine_fixed_levels > 0;
      const int initial_box_count = cfg_.initial_refine_box_count;
      const auto initial_box_lo = cfg_.initial_refine_box_lo;
      const auto initial_box_hi = cfg_.initial_refine_box_hi;

      if (solid_tagging) {
         tag_solid_fluid_boundaries(tags, active_solid_mask_ptr(lev));
      }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(*state_[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto s = state_[lev]->const_array(mfi);
         const auto tag = tags.array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (initial_box_refine) {
               const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
               const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
               const amrex::Real z = problo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
#else
               const amrex::Real z = 0.0;
#endif
               for (int box = 0; box < initial_box_count; ++box) {
                  bool inside = x >= initial_box_lo[box * AMREX_SPACEDIM]
                     && x <= initial_box_hi[box * AMREX_SPACEDIM];
#if (AMREX_SPACEDIM == 3)
                  inside = inside
                     && y >= initial_box_lo[box * AMREX_SPACEDIM + 1]
                     && y <= initial_box_hi[box * AMREX_SPACEDIM + 1]
                     && z >= initial_box_lo[box * AMREX_SPACEDIM + 2]
                     && z <= initial_box_hi[box * AMREX_SPACEDIM + 2];
#elif (AMREX_SPACEDIM == 2)
                  inside = inside
                     && y >= initial_box_lo[box * AMREX_SPACEDIM + 1]
                     && y <= initial_box_hi[box * AMREX_SPACEDIM + 1];
#endif
                  if (inside) {
                     tag(i, j, k) = tagval;
                     break;
                  }
               }
            }
            if (tag_solid_phi_band) {
               const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
               const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
#else
               const amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
               const amrex::Real z = problo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
#else
               const amrex::Real z = 0.0;
#endif
               if (amrex::Math::abs(solid_params.phi(time, x, y, z)) <= solid_refine_band) {
                  tag(i, j, k) = tagval;
               }
            }
            if (initial_refine_only || fixed_initial_hierarchy_only || auto_refinement_resolved) {
               return;
            }
            const amrex::Real indicator =
               auto_amr_indicator_from_state(
                  s,
                  i,
                  j,
                  k,
                  eos,
                  eos_method,
                  mhd,
                  coordinate,
                  axes,
                  problo,
                  dx,
                  amr_min_feature_lengths,
                  min_refine_radius
               );
            if (indicator > lohner_threshold) {
               tag(i, j, k) = tagval;
            }
         });
      }
      }

      if (
         !initial_refine_only
         && !fixed_initial_hierarchy_only
         && !auto_refinement_resolved
         && lev < static_cast<int>(latest_auto_tagged_cells_.size())
      ) {
         const DirectionalRefinementDemand demand =
            measure_directional_refinement_demand(lev, time, false);
         latest_auto_tagged_cells_[lev] = demand.tagged_cells;
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            latest_auto_direction_tagged_cells_[dir][lev] = demand.direction_tags[dir];
            latest_auto_direction_strength_[dir][lev] = demand.direction_strength[dir];
         }
      }
   }
