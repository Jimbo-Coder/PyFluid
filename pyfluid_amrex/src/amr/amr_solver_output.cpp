#include "amr/amr_solver_base.hpp"
#include "pyfluid_amrex/patch_domain.hpp"
#include "pyfluid_amrex/checkpoint.hpp"
#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/initial_data.hpp"
#include "amr/output.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

void AmrSolverBase::write_current_state_plotfile(int step, amrex::Real time) const
{
   if (finestLevel() == 0) {
      write_plotfile(*fluid_state_.state()[0], Geom(0), cfg_.output_dir, step, time, is_mhd(cfg_), cfg_.passive_scalars, cfg_);
   } else {
      write_multilevel_plotfile(
         fluid_state_.state(),
         Geom(),
         level_steps_,
         refRatio(),
         finestLevel(),
         cfg_.output_dir,
         step,
         time,
         is_mhd(cfg_),
         cfg_.passive_scalars,
         cfg_
      );
   }
   // A fixed spacetime is immutable run context rather than a time-series
   // field. Preserve one grid snapshot for inspection, but do not duplicate
   // the usually much larger tensor state beside every fluid frame. Evolved
   // spacetime remains part of every state output.
   if (spacetime_state_.active()
       && (spacetime_state_.evolves() || !fixed_spacetime_plot_written_)) {
      const std::filesystem::path plot_dir =
         std::filesystem::path(cfg_.output_dir) / "stateGF";
      std::filesystem::create_directories(plot_dir);
      write_output_gf(
         finestLevel(),
         spacetime_state_.state(),
         Geom(),
         level_steps_,
         refRatio(),
         (plot_dir / spacetime_state_.field_set().name).string(),
         grid_field_component_names(spacetime_state_.field_set()),
         time,
         step,
         cfg_,
         "state"
      );
      if (!spacetime_state_.evolves()) {
         fixed_spacetime_plot_written_ = true;
      }
   }
}

void AmrSolverBase::write_debug_stage_state(
   LevelData const& state,
   std::string const& stage_name,
   amrex::Real time,
   int lev_begin,
   int lev_end
) const
{
   if (!cfg_.debug_stage_output) {
      return;
   }
   if (finestLevel() != 0 || lev_begin != 0 || lev_end != 1) {
      amrex::Abort(
         "pyfluid.debug_stage_output currently supports one uniform AMR level"
      );
   }
   const int step = level_steps_.at(0) + 1;
   const std::filesystem::path plot_dir =
      std::filesystem::path(cfg_.output_dir) / "debugStage";
   std::filesystem::create_directories(plot_dir);
   write_output_gf(
      *state[0],
      (plot_dir / (stage_name + "GF")).string(),
      Geom(0),
      state_grid_field_names(physics_system_, cfg_.passive_scalars),
      time,
      step,
      cfg_,
      "debug_stage"
   );
}

void AmrSolverBase::write_current_aux_gfs(int step, amrex::Real time)
{
   if (!aux_gfs_enabled(cfg_)) {
      return;
   }
   const bool output_gravity_phi = aux_gfs_contains(cfg_, AuxGridField::GravityPotential);
   const bool output_gravity_gradient = aux_gfs_contains(cfg_, AuxGridField::GravityGradient);
   const bool output_apparent_horizon_mask = aux_gfs_contains(
      cfg_, AuxGridField::ApparentHorizonMask
   );
   const bool output_excision_mask = aux_gfs_contains(
      cfg_, AuxGridField::ExcisionMask
   );
   if (gravity_enabled(cfg_) && (output_gravity_phi || output_gravity_gradient)) {
      prepare_gravity_fields(fluid_state_.state(), output_gravity_gradient);
   }
   if (mhd_edge_aux_requested()) {
      refresh_all_mhd_aux_fields_for_output();
   }
   if (finestLevel() == 0) {
      const auto gravity_gradient = gravity_fields_.gradient_pointers(0);
      write_aux_gf_plotfile(
         *fluid_state_.state()[0],
         spacetime_state_.active()
               && !uses_analytic_fixed_spacetime_kernels()
            ? spacetime_coupling_.cell_spacetime()[0].get()
            : nullptr,
         face_b_ptr(0),
         edge_current_density_ptr(0),
         edge_electric_field_ptr(0),
         gravity_phi_ptr(0),
         gravity_gradient,
         solid_mask_[0].get(),
         output_apparent_horizon_mask
            ? apparent_horizon_mask_ptr(0) : nullptr,
         output_excision_mask ? excision_mask_ptr(0) : nullptr,
         solid_volume_fraction_[0].get(),
         solid_aperture_ptrs(0),
         Geom(0),
         cfg_,
         step,
         time,
         uses_analytic_fixed_spacetime_kernels(),
         cfg_
      );
   } else {
      std::vector<amrex::iMultiFab const*> apparent_horizon_masks(
         finestLevel() + 1
      );
      std::vector<amrex::iMultiFab const*> excision_masks(
         finestLevel() + 1
      );
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         apparent_horizon_masks[lev] = output_apparent_horizon_mask
            ? apparent_horizon_mask_ptr(lev) : nullptr;
         excision_masks[lev] = output_excision_mask
            ? excision_mask_ptr(lev) : nullptr;
      }
      write_multilevel_aux_gf_plotfile(
         fluid_state_.state(),
         spacetime_coupling_.cell_spacetime(),
         face_b_,
         edge_current_,
         edge_e_,
         gravity_fields_,
         solid_mask_,
         apparent_horizon_masks,
         excision_masks,
         solid_volume_fraction_,
         solid_aperture_,
         Geom(),
         level_steps_,
         refRatio(),
         finestLevel(),
         cfg_,
         step,
         time,
         uses_analytic_fixed_spacetime_kernels(),
         cfg_
      );
   }
}

std::string AmrSolverBase::checkpoint_dir() const
{
   if (!cfg_.checkpoint_dir.empty()) {
      return cfg_.checkpoint_dir;
   }
   return (std::filesystem::path(cfg_.output_dir) / "checkpoints").string();
}

void AmrSolverBase::load_checkpoint(std::string const& checkpoint_name, amrex::Real& time, amrex::Real& dt)
{
   const CheckpointHeader checkpoint = read_checkpoint_header(checkpoint_name);
   if (checkpoint.finest_level > maxLevel()) {
      amrex::Abort("checkpoint has more AMR levels than this run allows");
   }
   validate_checkpoint_compatibility(
      checkpoint_name,
      cfg_,
      num_state_components(),
      Geom(0),
      refRatio(),
      checkpoint.finest_level
   );

   SetFinestLevel(checkpoint.finest_level);
   for (int lev = 0; lev <= checkpoint.finest_level; ++lev) {
      const amrex::BoxArray& ba = checkpoint.box_arrays[lev];
      const amrex::DistributionMapping default_dm(ba);
      const amrex::DistributionMapping dm =
         obstacle_weighted_distribution_mapping(lev, ba, default_dm, checkpoint.time[lev]);
      SetBoxArray(lev, ba);
      SetDistributionMap(lev, dm);
      allocate_level(lev, ba, dm, checkpoint.time[lev]);
      read_checkpoint_field(
         *fluid_state_.state()[lev],
         checkpoint_name,
         lev,
         checkpoint_state_field_name()
      );
      if (spacetime_state_.active()) {
         if (!checkpoint_field_exists(
                checkpoint_name, spacetime_state_.field_set().name
             )) {
            amrex::Abort("evolved-spacetime checkpoint is missing spacetime state");
         }
         read_checkpoint_field(
            *spacetime_state_.state()[lev],
            checkpoint_name,
            lev,
            spacetime_state_.field_set().name
         );
         fill_spacetime_level_boundaries(
            spacetime_state_.state(), lev, checkpoint.time[lev]
         );
      }
      apply_boundary(*fluid_state_.state()[lev], Geom(lev), checkpoint.time[lev]);
      refresh_stored_spacetime_fluid_boundary(lev, checkpoint.time[lev]);
      set_level_time_state(lev, checkpoint.time[lev], checkpoint.dt[lev]);
      copy_level_state_to_old(lev);
      if (ct_enabled(cfg_)) {
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            read_checkpoint_field(
               *face_b_[lev].b[dir],
               checkpoint_name,
               lev,
               checkpoint_face_b_field_name(dir)
            );
         }
         apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
         sync_cell_b_from_faces(*fluid_state_.state()[lev], face_b_[lev], Geom(lev), false);
      }
      level_steps_[lev] = checkpoint.level_steps[lev];
   }
   for (int lev = checkpoint.finest_level + 1; lev <= maxLevel(); ++lev) {
      ClearLevel(lev);
      level_steps_[lev] = 0;
   }
   if (ct_enabled(cfg_)) {
      fill_all_level_boundaries<true>(fluid_state_.state(), checkpoint.time[0]);
      for (int lev = 0; lev <= checkpoint.finest_level; ++lev) {
         FaceMagneticField checkpoint_face_b = std::move(face_b_[lev]);
         allocate_face_magnetic_field(
            face_b_[lev], checkpoint.box_arrays[lev], DistributionMap(lev), cfg_.ng
         );
         fill_remade_face_b(lev, checkpoint.time[lev], checkpoint_face_b);
         sync_cell_b_from_faces(*fluid_state_.state()[lev], face_b_[lev], Geom(lev), false);
         copy_level_state_to_old(lev);
      }
   }
   time = checkpoint.time[0];
   dt = checkpoint.dt[0];
   checkpoint_reference_mass_ = checkpoint.reference_mass;
   checkpoint_reference_energy_ = checkpoint.reference_energy;
   checkpoint_reference_max_rho_ = checkpoint.reference_max_rho;
   regrid_lifecycle_.restore_clock(
      {checkpoint.aligned_regrid_ticks, checkpoint.aligned_base_regrid_tick},
      cfg_.regrid_int
   );
}

void AmrSolverBase::apply_boundary(amrex::MultiFab& state, amrex::Geometry const& geom, amrex::Real time) const
{
   apply_configured_state_boundary(state, geom, boundary_data_, time);
   if (patch_union_domain_enabled()) {
      int lev = -1;
      for (int level = 0; level <= finestLevel(); ++level) {
         if (Geom(level).Domain() == geom.Domain()) {
            lev = level;
            break;
         }
      }
      if (lev < 0) {
         amrex::Abort("patch-union boundary fill could not identify the AMR level");
      }
      apply_patch_union_reflecting_boundary(state, geom, patch_union_domain_boxes(lev));
   }
}

void AmrSolverBase::initialize_ct_fields(int lev, amrex::Real time)
{
   if (!ct_enabled(cfg_)) {
      return;
   }
   fill_level_boundaries(fluid_state_.state(), lev, time);
   apply_boundary(*fluid_state_.state()[lev], Geom(lev), time);
   refresh_stored_spacetime_fluid_boundary(lev, time);
   bool initialized_analytic_face_b = false;
   if (cfg_.initial_data_magnetic == "zero" && time == 0.0_rt) {
      fill_uniform_face_b(face_b_[lev], 0.0_rt, 0.0_rt, 0.0_rt);
      initialized_analytic_face_b = true;
   } else if (cfg_.ic == "magnetic_blast_noise" && time == 0.0_rt) {
      initialize_magnetic_blast_noise_face_b(face_b_[lev], Geom(lev), cfg_);
      initialized_analytic_face_b = true;
   } else if (cfg_.ic == "resistive_current_sheet" && time == 0.0_rt) {
      initialize_resistive_current_sheet_face_b(face_b_[lev], Geom(lev), cfg_);
      initialized_analytic_face_b = true;
   } else if (cfg_.ic == "reconnecting_shear_layer" && time == 0.0_rt) {
      initialize_reconnecting_shear_layer_face_b(face_b_[lev], Geom(lev), cfg_);
      initialized_analytic_face_b = true;
   } else if (cfg_.ic == "orszag_tang" && time == 0.0_rt) {
      initialize_orszag_tang_face_b(face_b_[lev], Geom(lev), cfg_);
      initialized_analytic_face_b = true;
   } else if ((cfg_.ic == "kelvin_helmholtz_mhd" || cfg_.ic == "density_cloud") && time == 0.0_rt) {
      fill_uniform_face_b(face_b_[lev], cfg_.b1_0, cfg_.b2_0, cfg_.b3_0);
      initialized_analytic_face_b = true;
   } else if ((cfg_.ic == "sod_x" || cfg_.ic == "sod_plane" || cfg_.ic == "sod_periodic_plane") && time == 0.0_rt) {
      initialize_sod_plane_face_b(face_b_[lev], Geom(lev), cfg_);
      initialized_analytic_face_b = true;
   } else {
      fill_face_b_from_cell_b(face_b_[lev], *fluid_state_.state()[lev]);
   }
   apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
   sync_cell_b_from_faces(*fluid_state_.state()[lev], face_b_[lev], Geom(lev), initialized_analytic_face_b);
   apply_boundary(*fluid_state_.state()[lev], Geom(lev), time);
   refresh_stored_spacetime_fluid_boundary(lev, time);
}

FaceMagneticField const* AmrSolverBase::face_b_ptr(int lev) const
{
   if (!ct_enabled(cfg_)) {
      return nullptr;
   }
   return &face_b_[lev];
}

std::vector<FaceMagneticField> const* AmrSolverBase::checkpoint_face_b() const
{
   if (!ct_enabled(cfg_)) {
      return nullptr;
   }
   return &face_b_;
}

} // namespace pyfluid_amrex
