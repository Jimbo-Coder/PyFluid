// Included inside pyfluid_amrex::AmrSolver private section.
// MultiFab RK stage algebra and fused update helpers.

   template <int NComp>
   void copy_plus_flux_divergence_range(
      LevelData& out,
      LevelData const& base,
      amrex::Real scale,
      int lev_begin,
      int lev_end
   )
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         if (solid_active(lev) && !has_mhd_components<NComp>() && cfg_.solid_motion == "static") {
            copy_plus_flux_divergence_components_masked<NComp>(
               *out[lev],
               *base[lev],
               flux_ptrs(lev),
               *solid_mask_[lev],
               Geom(lev),
               scale,
               &solid_valid_kind_[lev]
            );
         } else {
            copy_plus_flux_divergence_components<NComp>(*out[lev], *base[lev], flux_ptrs(lev), Geom(lev), scale);
         }
      }
   }

   template <int NComp>
   void copy_two_plus_flux_divergence_range(
      LevelData& out_a,
      LevelData& out_b,
      LevelData const& base,
      amrex::Real scale_a,
      amrex::Real scale_b,
      int lev_begin,
      int lev_end
   )
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         if (solid_active(lev) && !has_mhd_components<NComp>() && cfg_.solid_motion == "static") {
            copy_two_plus_flux_divergence_components_masked<NComp>(
               *out_a[lev],
               *out_b[lev],
               *base[lev],
               flux_ptrs(lev),
               *solid_mask_[lev],
               Geom(lev),
               scale_a,
               scale_b,
               &solid_valid_kind_[lev]
            );
         } else {
            copy_two_plus_flux_divergence_components<NComp>(
               *out_a[lev], *out_b[lev], *base[lev], flux_ptrs(lev), Geom(lev), scale_a, scale_b
            );
         }
      }
   }

   template <int NComp>
   void add_flux_divergence_and_copy_plus_flux_divergence_range(
      LevelData& accumulator,
      LevelData& scratch,
      LevelData const& base,
      amrex::Real accumulator_scale,
      amrex::Real scratch_scale,
      int lev_begin,
      int lev_end
   )
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         if (solid_active(lev) && !has_mhd_components<NComp>() && cfg_.solid_motion == "static") {
            add_flux_divergence_and_copy_plus_flux_divergence_components_masked<NComp>(
               *accumulator[lev],
               *scratch[lev],
               *base[lev],
               flux_ptrs(lev),
               *solid_mask_[lev],
               Geom(lev),
               accumulator_scale,
               scratch_scale,
               &solid_valid_kind_[lev]
            );
         } else {
            add_flux_divergence_and_copy_plus_flux_divergence_components<NComp>(
               *accumulator[lev],
               *scratch[lev],
               *base[lev],
               flux_ptrs(lev),
               Geom(lev),
               accumulator_scale,
               scratch_scale
            );
         }
      }
   }

   template <int NComp>
   void affine_combination_plus_flux_divergence_range(
      LevelData& out,
      LevelData const& base,
      LevelData const& input,
      amrex::Real input_weight,
      amrex::Real flux_scale,
      int lev_begin,
      int lev_end
   )
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         if (solid_active(lev) && !has_mhd_components<NComp>() && cfg_.solid_motion == "static") {
            affine_combination_plus_flux_divergence_components_masked<NComp>(
               *out[lev],
               *base[lev],
               *input[lev],
               flux_ptrs(lev),
               *solid_mask_[lev],
               Geom(lev),
               input_weight,
               flux_scale,
               &solid_valid_kind_[lev]
            );
         } else {
            affine_combination_plus_flux_divergence_components<NComp>(
               *out[lev],
               *base[lev],
               *input[lev],
               flux_ptrs(lev),
               Geom(lev),
               input_weight,
               flux_scale
            );
         }
      }
   }

   template <int NComp>
   void store_rk4_rhs_stage(int stage_index, int lev_begin, int lev_end)
   {
      static_assert(NComp > 0);
      fluid_state_.store_rk4_rhs_stage(stage_index, lev_begin, lev_end);
   }

   template <int NComp>
   void copy_plus_scaled_range(LevelData& out, LevelData const& base, LevelData const& increment, amrex::Real scale, int lev_begin, int lev_end)
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         if (solid_active(lev) && !has_mhd_components<NComp>()) {
            copy_plus_scaled_components_masked<NComp>(
               *out[lev],
               *base[lev],
               *increment[lev],
               *solid_mask_[lev],
               scale,
               &solid_valid_kind_[lev]
            );
         } else {
            copy_plus_scaled_components<NComp>(*out[lev], *base[lev], *increment[lev], scale);
         }
      }
   }

   template <int NComp>
   void copy_plus_scaled_all(LevelData& out, LevelData const& base, LevelData const& increment, amrex::Real scale)
   {
      copy_plus_scaled_range<NComp>(out, base, increment, scale, 0, finestLevel() + 1);
   }

   template <int NComp>
   void add_scaled_range(LevelData& out, LevelData const& increment, amrex::Real scale, int lev_begin, int lev_end)
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         if (solid_active(lev) && !has_mhd_components<NComp>()) {
            add_scaled_components_masked<NComp>(*out[lev], *increment[lev], *solid_mask_[lev], scale, &solid_valid_kind_[lev]);
         } else {
            add_scaled_components<NComp>(*out[lev], *increment[lev], scale);
         }
      }
   }

   template <int NComp>
   void add_scaled_all(LevelData& out, LevelData const& increment, amrex::Real scale)
   {
      add_scaled_range<NComp>(out, increment, scale, 0, finestLevel() + 1);
   }

   template <int NComp>
   void copy_two_plus_scaled_range(
      LevelData& out_a,
      LevelData& out_b,
      LevelData const& base,
      LevelData const& increment,
      amrex::Real scale_a,
      amrex::Real scale_b,
      int lev_begin,
      int lev_end
   )
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         if (solid_active(lev) && !has_mhd_components<NComp>()) {
            copy_two_plus_scaled_components_masked<NComp>(
               *out_a[lev],
               *out_b[lev],
               *base[lev],
               *increment[lev],
               *solid_mask_[lev],
               scale_a,
               scale_b,
               &solid_valid_kind_[lev]
            );
         } else {
            copy_two_plus_scaled_components<NComp>(*out_a[lev], *out_b[lev], *base[lev], *increment[lev], scale_a, scale_b);
         }
      }
   }

   template <int NComp>
   void copy_two_plus_scaled_all(
      LevelData& out_a,
      LevelData& out_b,
      LevelData const& base,
      LevelData const& increment,
      amrex::Real scale_a,
      amrex::Real scale_b
   )
   {
      copy_two_plus_scaled_range<NComp>(out_a, out_b, base, increment, scale_a, scale_b, 0, finestLevel() + 1);
   }

   template <int NComp>
   void add_scaled_and_copy_plus_scaled_range(
      LevelData& accumulator,
      LevelData& scratch,
      LevelData const& base,
      LevelData const& increment,
      amrex::Real accumulator_scale,
      amrex::Real scratch_scale,
      int lev_begin,
      int lev_end
   )
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         if (solid_active(lev) && !has_mhd_components<NComp>()) {
            add_scaled_and_copy_plus_scaled_components_masked<NComp>(
               *accumulator[lev],
               *scratch[lev],
               *base[lev],
               *increment[lev],
               *solid_mask_[lev],
               accumulator_scale,
               scratch_scale,
               &solid_valid_kind_[lev]
            );
         } else {
            add_scaled_and_copy_plus_scaled_components<NComp>(
               *accumulator[lev],
               *scratch[lev],
               *base[lev],
               *increment[lev],
               accumulator_scale,
               scratch_scale
            );
         }
      }
   }

   template <int NComp>
   void add_scaled_and_copy_plus_scaled_all(
      LevelData& accumulator,
      LevelData& scratch,
      LevelData const& base,
      LevelData const& increment,
      amrex::Real accumulator_scale,
      amrex::Real scratch_scale
   )
   {
      add_scaled_and_copy_plus_scaled_range<NComp>(
         accumulator,
         scratch,
         base,
         increment,
         accumulator_scale,
         scratch_scale,
         0,
         finestLevel() + 1
      );
   }

   template <int NComp>
   void affine_combination_plus_scaled_range(
      LevelData& out,
      LevelData const& base,
      LevelData const& input,
      LevelData const& increment,
      amrex::Real input_weight,
      amrex::Real increment_scale,
      int lev_begin,
      int lev_end
   )
   {
      for (int lev = lev_begin; lev < lev_end; ++lev) {
         if (solid_active(lev) && !has_mhd_components<NComp>()) {
            affine_combination_plus_scaled_components_masked<NComp>(
               *out[lev],
               *base[lev],
               *input[lev],
               *increment[lev],
               *solid_mask_[lev],
               input_weight,
               increment_scale,
               &solid_valid_kind_[lev]
            );
         } else {
            affine_combination_plus_scaled_components<NComp>(
               *out[lev],
               *base[lev],
               *input[lev],
               *increment[lev],
               input_weight,
               increment_scale
            );
         }
      }
   }

   template <int NComp>
   void affine_combination_plus_scaled_all(
      LevelData& out,
      LevelData const& base,
      LevelData const& input,
      LevelData const& increment,
      amrex::Real input_weight,
      amrex::Real increment_scale
   )
   {
      affine_combination_plus_scaled_range<NComp>(
         out,
         base,
         input,
         increment,
         input_weight,
         increment_scale,
         0,
         finestLevel() + 1
      );
   }

   template <int NComp>
   void evolution_copy_plus_scaled_range(
      EvolutionStageSlot output_slot,
      EvolutionStageSlot base_slot,
      amrex::Real scale,
      int lev_begin,
      int lev_end
   )
   {
      copy_plus_scaled_range<NComp>(
         fluid_state_.stage(output_slot),
         fluid_state_.stage(base_slot),
         fluid_state_.rhs(),
         scale,
         lev_begin,
         lev_end
      );
      if (spacetime_state_.evolves()) {
         spacetime_state_.linear_combination(
            spacetime_state_.stage(output_slot),
            1.0_rt,
            spacetime_state_.stage(base_slot),
            scale,
            spacetime_state_.rhs(),
            lev_begin,
            lev_end
         );
      }
   }

   template <int NComp>
   void evolution_affine_combination_plus_scaled_range(
      EvolutionStageSlot output_slot,
      EvolutionStageSlot base_slot,
      EvolutionStageSlot input_slot,
      amrex::Real input_weight,
      amrex::Real increment_scale,
      int lev_begin,
      int lev_end
   )
   {
      affine_combination_plus_scaled_range<NComp>(
         fluid_state_.stage(output_slot),
         fluid_state_.stage(base_slot),
         fluid_state_.stage(input_slot),
         fluid_state_.rhs(),
         input_weight,
         increment_scale,
         lev_begin,
         lev_end
      );
      if (spacetime_state_.evolves()) {
         for (int lev = lev_begin; lev < lev_end; ++lev) {
            affine_combination_plus_scaled_components(
               *spacetime_state_.stage(output_slot)[lev],
               *spacetime_state_.stage(base_slot)[lev],
               *spacetime_state_.stage(input_slot)[lev],
               *spacetime_state_.rhs()[lev],
               input_weight,
               increment_scale
            );
         }
      }
   }

   template <int NComp>
   void evolution_copy_two_plus_scaled_range(
      EvolutionStageSlot output_a_slot,
      EvolutionStageSlot output_b_slot,
      EvolutionStageSlot base_slot,
      amrex::Real scale_a,
      amrex::Real scale_b,
      int lev_begin,
      int lev_end
   )
   {
      copy_two_plus_scaled_range<NComp>(
         fluid_state_.stage(output_a_slot),
         fluid_state_.stage(output_b_slot),
         fluid_state_.stage(base_slot),
         fluid_state_.rhs(),
         scale_a,
         scale_b,
         lev_begin,
         lev_end
      );
      if (spacetime_state_.evolves()) {
         spacetime_state_.linear_combination(
            spacetime_state_.stage(output_a_slot),
            1.0_rt,
            spacetime_state_.stage(base_slot),
            scale_a,
            spacetime_state_.rhs(),
            lev_begin,
            lev_end
         );
         spacetime_state_.linear_combination(
            spacetime_state_.stage(output_b_slot),
            1.0_rt,
            spacetime_state_.stage(base_slot),
            scale_b,
            spacetime_state_.rhs(),
            lev_begin,
            lev_end
         );
      }
   }

   template <int NComp>
   void evolution_add_scaled_and_copy_plus_scaled_range(
      EvolutionStageSlot accumulator_slot,
      EvolutionStageSlot scratch_slot,
      EvolutionStageSlot base_slot,
      amrex::Real accumulator_scale,
      amrex::Real scratch_scale,
      int lev_begin,
      int lev_end
   )
   {
      add_scaled_and_copy_plus_scaled_range<NComp>(
         fluid_state_.stage(accumulator_slot),
         fluid_state_.stage(scratch_slot),
         fluid_state_.stage(base_slot),
         fluid_state_.rhs(),
         accumulator_scale,
         scratch_scale,
         lev_begin,
         lev_end
      );
      if (spacetime_state_.evolves()) {
         spacetime_state_.add_scaled(
            spacetime_state_.stage(accumulator_slot),
            accumulator_scale,
            spacetime_state_.rhs(),
            lev_begin,
            lev_end
         );
         spacetime_state_.linear_combination(
            spacetime_state_.stage(scratch_slot),
            1.0_rt,
            spacetime_state_.stage(base_slot),
            scratch_scale,
            spacetime_state_.rhs(),
            lev_begin,
            lev_end
         );
      }
   }

   template <int NComp>
   void evolution_add_scaled_range(
      EvolutionStageSlot output_slot,
      amrex::Real scale,
      int lev_begin,
      int lev_end
   )
   {
      add_scaled_range<NComp>(
         fluid_state_.stage(output_slot),
         fluid_state_.rhs(),
         scale,
         lev_begin,
         lev_end
      );
      if (spacetime_state_.evolves()) {
         spacetime_state_.add_scaled(
            spacetime_state_.stage(output_slot),
            scale,
            spacetime_state_.rhs(),
            lev_begin,
            lev_end
         );
      }
   }

   void evolution_copy_range(
      EvolutionStageSlot output_slot,
      EvolutionStageSlot input_slot,
      int lev_begin,
      int lev_end
   )
   {
      fluid_state_.copy(
         fluid_state_.stage(output_slot),
         fluid_state_.stage(input_slot),
         lev_begin,
         lev_end
      );
      if (spacetime_state_.evolves()) {
         spacetime_state_.copy(
            spacetime_state_.stage(output_slot),
            spacetime_state_.stage(input_slot),
            lev_begin,
            lev_end
         );
      }
   }
