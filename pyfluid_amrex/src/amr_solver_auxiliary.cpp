#include "pyfluid_amrex/amr_solver_base.hpp"

namespace pyfluid_amrex {

EdgeElectricField const* AmrSolverBase::edge_current_density_ptr(int lev) const
{
   if (!edge_current_[lev].e[0]) {
      return nullptr;
   }
   return &edge_current_[lev];
}

EdgeElectricField const* AmrSolverBase::edge_electric_field_ptr(int lev) const
{
   if (!edge_e_[lev].e[0]) {
      return nullptr;
   }
   return &edge_e_[lev];
}

amrex::MultiFab const* AmrSolverBase::gravity_phi_ptr(int lev) const
{
   if (!gravity_phi_[lev]) {
      return nullptr;
   }
   return gravity_phi_[lev].get();
}

bool AmrSolverBase::mhd_edge_aux_requested() const
{
   return is_mhd(cfg_)
      && (
         aux_gfs_contains(cfg_, AuxGridField::CurrentDensity)
         || aux_gfs_contains(cfg_, AuxGridField::ElectricField)
      );
}

bool AmrSolverBase::cached_edge_current_density_requested() const
{
   return mhd_edge_aux_requested()
      && cfg_.aux_gf_compute_int >= 0
      && (
         aux_gfs_contains(cfg_, AuxGridField::CurrentDensity)
         || (cfg_.resistivity != 0.0 && aux_gfs_contains(cfg_, AuxGridField::ElectricField))
      );
}

bool AmrSolverBase::cached_edge_electric_field_requested() const
{
   return mhd_edge_aux_requested()
      && cfg_.aux_gf_compute_int >= 0
      && aux_gfs_contains(cfg_, AuxGridField::ElectricField);
}

bool AmrSolverBase::needs_edge_current_density() const
{
   return cached_edge_current_density_requested();
}

bool AmrSolverBase::needs_edge_electric_field() const
{
   return ct_enabled(cfg_) || cached_edge_electric_field_requested();
}

bool AmrSolverBase::should_refresh_mhd_aux_fields(int step, bool write_aux_gfs) const
{
   (void)write_aux_gfs;
   if (!is_mhd(cfg_)) {
      return false;
   }
   if (cfg_.aux_gf_compute_int > 0 && step % cfg_.aux_gf_compute_int == 0 && mhd_edge_aux_requested()) {
      return true;
   }
   return false;
}

bool AmrSolverBase::should_refresh_initial_mhd_aux_fields() const
{
   if (!mhd_edge_aux_requested()) {
      return false;
   }
   return cfg_.aux_gf_compute_int == 0 || cfg_.aux_gf_int == 0;
}

void AmrSolverBase::refresh_mhd_aux_fields_for_output(int lev)
{
   if (edge_current_[lev].e[0]) {
      if (ct_enabled(cfg_)) {
         fill_edge_current_density_from_faces(edge_current_[lev], face_b_[lev], Geom(lev), cfg_);
      } else {
         fill_edge_current_density_from_cells(edge_current_[lev], *state_[lev], Geom(lev));
      }
      enforce_edge_electric_periodicity(edge_current_[lev], Geom(lev));
   }
   if (cached_edge_electric_field_requested() || (ct_enabled(cfg_) && aux_gfs_contains(cfg_, AuxGridField::ElectricField))) {
      EdgeElectricField const* edge_j = edge_current_[lev].e[0] ? &edge_current_[lev] : nullptr;
      FaceMagneticField const* face_b = ct_enabled(cfg_) ? &face_b_[lev] : nullptr;
      fill_edge_electric_field_for_output(edge_e_[lev], *state_[lev], face_b, edge_j, Geom(lev), cfg_.resistivity, cfg_);
   }
}

void AmrSolverBase::refresh_all_mhd_aux_fields_for_output()
{
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      refresh_mhd_aux_fields_for_output(lev);
   }
}

} // namespace pyfluid_amrex
