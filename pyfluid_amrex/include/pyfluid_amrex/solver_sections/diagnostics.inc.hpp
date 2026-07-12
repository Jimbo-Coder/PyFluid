// Included inside pyfluid_amrex::AmrSolver private section.
// Symmetry factors and scalar diagnostics.

   bool symmetry_axis_active(int dir) const;
   amrex::Real symmetry_volume_factor() const;
   amrex::Real symmetry_momentum_factor(int dir, amrex::Real volume_factor) const;
   amrex::Real symmetry_angular_momentum_factor(int dir, amrex::Real volume_factor) const;

   CflLimiterLocation current_cfl_limiter_location() const
   {
      CflLimiterLocation local;
      local.frequency = -std::numeric_limits<amrex::Real>::infinity();

      const int coordinate = mesh_coordinate_kind();
      const MeshAxes axes = active_mesh_axes();
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         if (!frequency_[lev]) {
            continue;
         }
         amrex::iMultiFab mask = valid_cell_mask(lev);
         const auto problo = Geom(lev).ProbLoArray();
         const auto dx = Geom(lev).CellSizeArray();
         for (amrex::MFIter mfi(*frequency_[lev]); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            const auto f = frequency_[lev]->const_array(mfi);
            const auto valid = mask.const_array(mfi);
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);
            for (int k = lo.z; k <= hi.z; ++k) {
               for (int j = lo.y; j <= hi.y; ++j) {
                  for (int i = lo.x; i <= hi.x; ++i) {
                     if (valid(i, j, k, 0) == 0) {
                        continue;
                     }
                     const amrex::Real value = f(i, j, k, 0);
                     if (value <= local.frequency) {
                        continue;
                     }
                     const auto q = mesh_coordinate_triplet_at_location(
                        coordinate, axes, problo, dx, i, j, k, false, false, false
                     );
                     local.frequency = value;
                     local.level = lev;
                     local.i = i;
                     local.j = j;
                     local.k = k;
                     local.q1 = q[0];
                     local.q2 = q[1];
                     local.q3 = q[2];
                  }
               }
            }
         }
      }

      amrex::Real global_frequency = local.frequency;
      amrex::ParallelDescriptor::ReduceRealMax(global_frequency);
      if (!std::isfinite(global_frequency) || global_frequency <= 0.0) {
         return {};
      }

      const amrex::Real tolerance =
         64.0 * std::numeric_limits<amrex::Real>::epsilon() * std::max<amrex::Real>(1.0, std::abs(global_frequency));
      int owner_rank = (
         local.level >= 0 && std::abs(local.frequency - global_frequency) <= tolerance
      ) ? amrex::ParallelDescriptor::MyProc() : -1;
      amrex::ParallelDescriptor::ReduceIntMax(owner_rank);
      if (owner_rank < 0) {
         return {};
      }

      std::array<int, 4> int_packet = {{
         local.level,
         local.i,
         local.j,
         local.k,
      }};
      std::array<amrex::Real, 4> real_packet = {{
         global_frequency,
         local.q1,
         local.q2,
         local.q3,
      }};
      amrex::ParallelDescriptor::Bcast(int_packet.data(), int_packet.size(), owner_rank);
      amrex::ParallelDescriptor::Bcast(real_packet.data(), real_packet.size(), owner_rank);

      CflLimiterLocation result;
      result.frequency = real_packet[0];
      result.level = int_packet[0];
      result.i = int_packet[1];
      result.j = int_packet[2];
      result.k = int_packet[3];
      result.q1 = real_packet[1];
      result.q2 = real_packet[2];
      result.q3 = real_packet[3];
      return result;
   }

   template <int EOSMethod>
   ScalarDiagnostics compute_current_scalars(
      int step,
      amrex::Real time,
      amrex::Real dt,
      amrex::Real mass0,
      amrex::Real energy0,
      amrex::Real gravity_total_energy0,
      amrex::Real passive_mass0,
      amrex::Real mcells_per_s,
      amrex::Real layout_mcells_per_s,
      amrex::Real sim_time_per_s,
      DiffusionStats diffusion
   )
   {
      ScalarDiagnostics d;
      d.step = step;
      d.time = time;
      d.dt = dt;
      d.min_rho = std::numeric_limits<amrex::Real>::max();
      d.min_pressure = std::numeric_limits<amrex::Real>::max();
      amrex::Real gravity_density_offset = 0.0;
      amrex::Real gravity_energy_factor = 0.0;
      if (gravity_enabled(cfg_)) {
         prepare_gravity_fields(state_, false);
         if (self_gravity_enabled(cfg_)) {
            const DensityTotal density = valid_density_total(state_);
            gravity_density_offset = density.volume > 0.0 ? density.mass / density.volume : 0.0;
            gravity_energy_factor = 0.5;
         } else {
            gravity_energy_factor = 1.0;
         }
      }

      const int coordinate = mesh_coordinate_kind();
      const MeshAxes axes = active_mesh_axes();
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         if (is_mhd(cfg_)) {
            fill_pressure<true, EOSMethod>(*pressure_[lev], *state_[lev], eos_params(cfg_));
         } else {
            fill_pressure<false, EOSMethod>(*pressure_[lev], *state_[lev], eos_params(cfg_));
         }
         amrex::iMultiFab mask = valid_cell_mask(lev);
         amrex::MultiFab reduction(state_[lev]->boxArray(), state_[lev]->DistributionMap(), ScalarReductionNComp, 0);
         amrex::MultiFab const* gravity_phi = gravity_enabled(cfg_) ? gravity_phi_[lev].get() : nullptr;
         const int passive_component = cfg_.passive_scalars > 0 ? passive_component_begin(is_mhd(cfg_)) : -1;
         if (mixed_patch_system_enabled() && !is_mhd(cfg_)) {
            fill_scalar_reduction_fields_mapped_hydro(
               reduction,
               *state_[lev],
               *pressure_[lev],
               mask,
               production_patch_grid_system_for_level(lev),
               production_patch_layout_for_level(lev),
               gravity_phi,
               gravity_density_offset,
               gravity_energy_factor,
               passive_component
            );
         } else if (is_mhd(cfg_)) {
            fill_scalar_reduction_fields<true>(
               reduction,
               *state_[lev],
               *pressure_[lev],
               mask,
               Geom(lev),
               coordinate,
               axes,
               face_b_ptr(lev),
               nullptr,
               gravity_phi,
               gravity_density_offset,
               gravity_energy_factor,
               passive_component
            );
         } else {
            fill_scalar_reduction_fields<false>(
               reduction,
               *state_[lev],
               *pressure_[lev],
               mask,
               Geom(lev),
               coordinate,
               axes,
               nullptr,
               solid_active(lev) ? solid_volume_fraction_[lev].get() : nullptr,
               gravity_phi,
               gravity_density_offset,
               gravity_energy_factor,
               passive_component
            );
         }
         d.mass += reduction.sum(ScalarReductionMass);
         d.mom1 += reduction.sum(ScalarReductionMom1);
         d.mom2 += reduction.sum(ScalarReductionMom2);
         d.mom3 += reduction.sum(ScalarReductionMom3);
         d.energy += reduction.sum(ScalarReductionEnergy);
         const auto [rho_min, pressure_min] = valid_min_rho_pressure(*state_[lev], *pressure_[lev], mask);
         d.min_rho = amrex::min(d.min_rho, rho_min);
         d.min_pressure = amrex::min(d.min_pressure, pressure_min);
         d.magnetic_energy += reduction.sum(ScalarReductionMagneticEnergy);
         d.divb_l1 += reduction.sum(ScalarReductionDivB);
         d.angular_mom1 += reduction.sum(ScalarReductionAngularMom1);
         d.angular_mom2 += reduction.sum(ScalarReductionAngularMom2);
         d.angular_mom3 += reduction.sum(ScalarReductionAngularMom3);
         d.gravity_potential_energy += reduction.sum(ScalarReductionGravityPotentialEnergy);
         d.passive_mass += reduction.sum(ScalarReductionPassiveMass);
      }

      const amrex::Real symmetry_factor = symmetry_volume_factor();
      d.mass *= symmetry_factor;
      d.passive_mass *= symmetry_factor;
      d.mom1 *= symmetry_momentum_factor(0, symmetry_factor);
      d.mom2 *= symmetry_momentum_factor(1, symmetry_factor);
      d.mom3 *= symmetry_momentum_factor(2, symmetry_factor);
      d.angular_mom1 *= symmetry_angular_momentum_factor(0, symmetry_factor);
      d.angular_mom2 *= symmetry_angular_momentum_factor(1, symmetry_factor);
      d.angular_mom3 *= symmetry_angular_momentum_factor(2, symmetry_factor);
      d.energy *= symmetry_factor;
      d.gravity_potential_energy *= symmetry_factor;
      d.gravity_total_energy = d.energy + d.gravity_potential_energy;
      d.magnetic_energy *= symmetry_factor;
      d.divb_l1 *= symmetry_factor;
      d.mass_rel_drift = amrex::Math::abs(d.mass - mass0) / amrex::max(amrex::Math::abs(mass0), 1.0);
      d.energy_rel_drift = amrex::Math::abs(d.energy - energy0) / amrex::max(amrex::Math::abs(energy0), 1.0);
      d.passive_mass_rel_drift =
         cfg_.passive_scalars > 0
            ? amrex::Math::abs(d.passive_mass - passive_mass0) / amrex::max(amrex::Math::abs(passive_mass0), 1.0)
            : 0.0;
      const amrex::Real gravity_total_energy_scale = amrex::Math::abs(gravity_total_energy0);
      d.gravity_total_energy_rel_drift =
         amrex::Math::abs(d.gravity_total_energy - gravity_total_energy0)
         / (gravity_total_energy_scale > 0.0 ? gravity_total_energy_scale : 1.0);
      const amrex::Real energy_delta = d.energy - wall_work_reference_energy_;
      d.wall_work = cumulative_wall_work_;
      d.wall_power = latest_wall_power_;
      d.energy_minus_wall_work = energy_delta - cumulative_wall_work_;
      const amrex::Real work_scale = amrex::max(
         amrex::max(amrex::Math::abs(energy_delta), amrex::Math::abs(cumulative_wall_work_)),
         1.0
      );
      d.wall_work_rel_error = amrex::Math::abs(d.energy_minus_wall_work) / work_scale;
      d.external_budget_valid = external_conservation_budget_supported() ? 1 : 0;
      if (d.external_budget_valid) {
         d.boundary_mass_change = conservation_ledger_.boundary_change.mass;
         d.boundary_energy_change = conservation_ledger_.boundary_change.energy;
         d.source_mass_change = conservation_ledger_.source_change.mass;
         d.source_energy_change = conservation_ledger_.source_change.energy;
         amrex::ParallelDescriptor::ReduceRealSum(d.boundary_mass_change);
         amrex::ParallelDescriptor::ReduceRealSum(d.boundary_energy_change);
         amrex::ParallelDescriptor::ReduceRealSum(d.source_mass_change);
         amrex::ParallelDescriptor::ReduceRealSum(d.source_energy_change);
         d.boundary_mass_change *= symmetry_factor;
         d.boundary_energy_change *= symmetry_factor;
         d.source_mass_change *= symmetry_factor;
         d.source_energy_change *= symmetry_factor;
         d.external_mass_change = d.boundary_mass_change + d.source_mass_change;
         d.external_energy_change = d.boundary_energy_change + d.source_energy_change + d.wall_work;
         const amrex::Real measured_mass_change = d.mass - conservation_ledger_.reference.mass;
         const amrex::Real measured_energy_change = d.energy - conservation_ledger_.reference.energy;
         d.external_mass_residual = measured_mass_change - d.external_mass_change;
         d.external_energy_residual = measured_energy_change - d.external_energy_change;
         const amrex::Real mass_scale = amrex::max(
            amrex::max(amrex::Math::abs(measured_mass_change), amrex::Math::abs(d.external_mass_change)),
            1.0
         );
         const amrex::Real energy_scale = amrex::max(
            amrex::max(amrex::Math::abs(measured_energy_change), amrex::Math::abs(d.external_energy_change)),
            1.0
         );
         d.external_mass_rel_error = amrex::Math::abs(d.external_mass_residual) / mass_scale;
         d.external_energy_rel_error = amrex::Math::abs(d.external_energy_residual) / energy_scale;
      }
      d.diffusion_residual = diffusion.residual;
      d.diffusion_iterations = diffusion.iterations;
      const AmrLevelDiagnostics amr = current_amr_level_diagnostics();
      d.amr_total_cells = amr.total_cells;
      d.amr_active_finest_level = amr.active_finest_level;
      d.amr_finest_headroom = amr.finest_headroom;
      d.amr_touches_max_level = amr.touches_max_level;
      d.amr_total_budget_fraction = amr.total_budget_fraction;
      d.amr_peak_level_budget_fraction = amr.peak_level_budget_fraction;
      d.mcells_per_s = mcells_per_s;
      d.layout_mcells_per_s = layout_mcells_per_s;
      d.sim_time_per_s = sim_time_per_s;
      if (cfg_.cfl_limiter_diagnostics) {
         set_cfl_limiter_diagnostics(d, current_cfl_limiter_location());
      }
      return d;
   }
