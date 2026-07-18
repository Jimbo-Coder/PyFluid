#include "amr/amr_solver_base.hpp"

namespace pyfluid_amrex {

bool AmrSolverBase::external_conservation_budget_supported() const
{
   // Gravity has a separate fluid-plus-potential energy synchronization
   // contract. Its external ledger belongs with that total-energy update.
   // A time-dependent spacetime exchanges coordinate energy and momentum
   // with the fluid. The fluid-only boundary/source ledger does not include
   // that geometric work and therefore is not a closed budget.
   return !mixed_patch_system_enabled() && !gravity_enabled(cfg_) && !spacetime_evolves(cfg_);
}

AmrSolverBase::MassEnergy AmrSolverBase::local_rhs_mass_energy_integral(int lev) const
{
   MassEnergy total;
   if (!external_conservation_budget_supported()) {
      return total;
   }
   amrex::iMultiFab valid_mask = valid_cell_mask(lev);
   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   const auto problo = Geom(lev).ProbLoArray();
   const auto dx = Geom(lev).CellSizeArray();
   const bool use_solid_volume = solid_active(lev) && solid_volume_fraction_[lev];
   amrex::Real mass = 0.0;
   amrex::Real energy = 0.0;
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+ : mass, energy) if (amrex::Gpu::notInLaunchRegion())
#endif
   {
      for (amrex::MFIter mfi(*fluid_state_.rhs()[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto lo = amrex::lbound(bx);
         const auto hi = amrex::ubound(bx);
         const auto rhs = fluid_state_.rhs()[lev]->const_array(mfi);
         const auto valid = valid_mask.const_array(mfi);
         const auto volume_fraction =
            use_solid_volume ? solid_volume_fraction_[lev]->const_array(mfi) : amrex::Array4<amrex::Real const>{};
         for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
               for (int i = lo.x; i <= hi.x; ++i) {
                  if (valid(i, j, k, 0) == 0) {
                     continue;
                  }
                  amrex::Real volume = mesh_cell_volume_for_reduction(coordinate, axes, problo, dx, i, j, k);
                  if (use_solid_volume) {
                     volume *= volume_fraction(i, j, k, 0);
                  }
                  mass += volume * rhs(i, j, k, RHO);
                  energy += volume * rhs(i, j, k, ENERGY);
               }
            }
         }
      }
   }
   total.mass = mass;
   total.energy = energy;
   return total;
}

void AmrSolverBase::accumulate_physical_boundary_flux(int lev,
                                                      std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
                                                      amrex::Real weight)
{
   if (weight == 0.0 || !external_conservation_budget_supported()) {
      return;
   }
   const auto periodic = Geom(lev).isPeriodicArray();
   bool has_physical_boundary = false;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      has_physical_boundary = has_physical_boundary || !periodic[dir];
   }
   if (!has_physical_boundary) {
      return;
   }

   amrex::iMultiFab valid_mask = valid_cell_mask(lev);
   const amrex::Box& domain = Geom(lev).Domain();
   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   const auto problo = Geom(lev).ProbLoArray();
   const auto dx = Geom(lev).CellSizeArray();
   const bool use_solid_aperture = solid_active(lev);
   amrex::Real mass_rate = 0.0;
   amrex::Real energy_rate = 0.0;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (periodic[dir]) {
         continue;
      }
      for (int side = 0; side < 2; ++side) {
         const bool high_side = side == 1;
         const int cell_index = high_side ? domain.bigEnd(dir) : domain.smallEnd(dir);
         const amrex::Real sign = high_side ? -1.0 : 1.0;
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+ : mass_rate, energy_rate) if (amrex::Gpu::notInLaunchRegion())
#endif
         {
            for (amrex::MFIter mfi(*fluid_state_.state()[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
               amrex::Box bx = mfi.tilebox() & domain;
               if (cell_index < bx.smallEnd(dir) || cell_index > bx.bigEnd(dir)) {
                  continue;
               }
               bx.setSmall(dir, cell_index);
               bx.setBig(dir, cell_index);
               const auto lo = amrex::lbound(bx);
               const auto hi = amrex::ubound(bx);
               const auto flux = fluxes[dir]->const_array(mfi);
               const auto valid = valid_mask.const_array(mfi);
               const auto aperture =
                  use_solid_aperture ? solid_aperture_[dir][lev]->const_array(mfi) : amrex::Array4<amrex::Real const>{};
               for (int k = lo.z; k <= hi.z; ++k) {
                  for (int j = lo.y; j <= hi.y; ++j) {
                     for (int i = lo.x; i <= hi.x; ++i) {
                        if (valid(i, j, k, 0) == 0) {
                           continue;
                        }
                        int fi = i;
                        int fj = j;
                        int fk = k;
                        if (high_side) {
                           if (dir == 0) {
                              ++fi;
                           }
#if (AMREX_SPACEDIM >= 2)
                           if (dir == 1) {
                              ++fj;
                           }
#endif
#if (AMREX_SPACEDIM == 3)
                           if (dir == 2) {
                              ++fk;
                           }
#endif
                        }
                        amrex::Real area = mesh_face_area_scale(dir, coordinate, axes, problo, dx, fi, fj, fk);
                        if (use_solid_aperture) {
                           area *= aperture(fi, fj, fk, 0);
                        }
                        mass_rate += sign * area * flux(fi, fj, fk, RHO);
                        energy_rate += sign * area * flux(fi, fj, fk, ENERGY);
                     }
                  }
               }
            }
         }
      }
   }
   conservation_ledger_.boundary_change.mass += weight * mass_rate;
   conservation_ledger_.boundary_change.energy += weight * energy_rate;
}

void AmrSolverBase::accumulate_source_change(MassEnergy const& before, MassEnergy const& after, amrex::Real weight)
{
   if (weight == 0.0 || !external_conservation_budget_supported()) {
      return;
   }
   conservation_ledger_.source_change.mass += weight * (after.mass - before.mass);
   conservation_ledger_.source_change.energy += weight * (after.energy - before.energy);
}

} // namespace pyfluid_amrex
