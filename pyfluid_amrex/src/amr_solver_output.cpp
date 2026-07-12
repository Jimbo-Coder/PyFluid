#include "pyfluid_amrex/amr_solver.hpp"
#include "pyfluid_amrex/fields.hpp"

namespace pyfluid_amrex {

void AmrSolver::write_current_state_plotfile(int step, amrex::Real time) const
{
   if (finestLevel() == 0) {
      write_plotfile(*state_[0], Geom(0), cfg_.output_dir, step, time, is_mhd(cfg_), cfg_.passive_scalars, cfg_);
   } else {
      write_multilevel_plotfile(
         state_,
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
}

void AmrSolver::write_current_aux_gfs(int step, amrex::Real time)
{
   if (!aux_gfs_enabled(cfg_)) {
      return;
   }
   const bool output_gravity_phi = aux_gfs_contains(cfg_, AuxGridField::GravityPotential);
   const bool output_gravity_gradient = aux_gfs_contains(cfg_, AuxGridField::GravityGradient);
   if (gravity_enabled(cfg_) && (output_gravity_phi || output_gravity_gradient)) {
      prepare_gravity_fields(state_, output_gravity_gradient);
   }
   if (mhd_edge_aux_requested()) {
      refresh_all_mhd_aux_fields_for_output();
   }
   if (finestLevel() == 0) {
      const auto gravity_gradient = gravity_gradient_const_ptrs(gravity_gradient_[0]);
      write_aux_gf_plotfile(
         *state_[0],
         face_b_ptr(0),
         edge_current_density_ptr(0),
         edge_electric_field_ptr(0),
         gravity_phi_ptr(0),
         gravity_gradient,
         solid_mask_[0].get(),
         solid_volume_fraction_[0].get(),
         solid_aperture_ptrs(0),
         Geom(0),
         cfg_,
         step,
         time,
         cfg_
      );
   } else {
      write_multilevel_aux_gf_plotfile(
         state_,
         face_b_,
         edge_current_,
         edge_e_,
         gravity_phi_,
         gravity_gradient_,
         solid_mask_,
         solid_volume_fraction_,
         solid_aperture_,
         Geom(),
         level_steps_,
         refRatio(),
         finestLevel(),
         cfg_,
         step,
         time,
         cfg_
      );
   }
}

std::string AmrSolver::checkpoint_dir() const
{
   if (!cfg_.checkpoint_dir.empty()) {
      return cfg_.checkpoint_dir;
   }
   return (std::filesystem::path(cfg_.output_dir) / "checkpoints").string();
}

void AmrSolver::load_checkpoint(std::string const& checkpoint_name, amrex::Real& time, amrex::Real& dt)
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
         *state_[lev],
         checkpoint_name,
         lev,
         checkpoint_state_field_name()
      );
      apply_boundary(*state_[lev], Geom(lev), checkpoint.time[lev]);
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
         sync_cell_b_from_faces(*state_[lev], face_b_[lev], Geom(lev), false);
      }
      level_steps_[lev] = checkpoint.level_steps[lev];
   }
   for (int lev = checkpoint.finest_level + 1; lev <= maxLevel(); ++lev) {
      ClearLevel(lev);
      level_steps_[lev] = 0;
   }
   time = checkpoint.time[0];
   dt = checkpoint.dt[0];
   checkpoint_reference_mass_ = checkpoint.reference_mass;
   checkpoint_reference_energy_ = checkpoint.reference_energy;
   aligned_regrid_ticks_ = checkpoint.aligned_regrid_ticks;
   aligned_regrid_due_ = false;
}

void AmrSolver::apply_boundary(amrex::MultiFab& state, amrex::Geometry const& geom, amrex::Real time) const
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

void AmrSolver::initialize_ct_fields(int lev, amrex::Real time)
{
   if (!ct_enabled(cfg_)) {
      return;
   }
   apply_boundary(*state_[lev], Geom(lev), time);
   const bool preserve_pressure = cfg_.ic == "magnetic_blast_noise";
   if (cfg_.ic == "magnetic_blast_noise") {
      initialize_magnetic_blast_noise_face_b(face_b_[lev], Geom(lev), cfg_);
   } else if ((cfg_.ic == "kelvin_helmholtz_mhd" || cfg_.ic == "density_cloud") && time == 0.0) {
      fill_uniform_face_b(face_b_[lev], cfg_.b1_0, cfg_.b2_0, cfg_.b3_0);
   } else if ((cfg_.ic == "sod_x" || cfg_.ic == "sod_plane" || cfg_.ic == "sod_periodic_plane") && time == 0.0) {
      initialize_sod_plane_face_b(face_b_[lev], Geom(lev), cfg_);
   } else {
      fill_face_b_from_cell_b(face_b_[lev], *state_[lev]);
   }
   apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
   sync_cell_b_from_faces(*state_[lev], face_b_[lev], Geom(lev), preserve_pressure);
   apply_boundary(*state_[lev], Geom(lev), time);
}

FaceMagneticField const* AmrSolver::face_b_ptr(int lev) const
{
   if (!ct_enabled(cfg_)) {
      return nullptr;
   }
   return &face_b_[lev];
}

std::vector<FaceMagneticField> const* AmrSolver::checkpoint_face_b() const
{
   if (!ct_enabled(cfg_)) {
      return nullptr;
   }
   return &face_b_;
}

} // namespace pyfluid_amrex
