#include "pyfluid_amrex/metric_reductions.hpp"
#include "pyfluid_amrex/reduction_sum.hpp"
#include "amr/amr_solver_base.hpp"

#include "pyfluid_amrex/apparent_horizon_tracker.hpp"

#include <AMReX_Reduce.H>

namespace pyfluid_amrex {

using namespace amrex::literals;

bool AmrSolverBase::external_conservation_budget_supported() const
{
   // Gravity has a separate fluid-plus-potential energy synchronization
   // contract. Its external ledger belongs with that total-energy update.
   // Evolving geometry does not spoil the densitized baryon conservation
   // law. Keep that ledger; its energy counterpart is separately marked
   // invalid because it omits time-dependent geometric work.
   return !mixed_patch_system_enabled() && !gravity_enabled(cfg_);
}

AmrSolverBase::MassEnergy AmrSolverBase::local_state_mass_energy_integral(
   int lev,
   amrex::MultiFab const& state
) const
{
   if (!external_conservation_budget_supported()) {
      return {};
   }
   amrex::iMultiFab valid_mask = valid_cell_mask(lev);
   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   const auto problo = Geom(lev).ProbLoArray();
   const auto dx = Geom(lev).CellSizeArray();
   const bool use_solid_volume = solid_active(lev) && solid_volume_fraction_[lev];
   amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
   amrex::ReduceData<ReductionSum, ReductionSum> reduce_data(reduce_ops);
   using ReduceTuple = decltype(reduce_data)::Type;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
      for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const auto values = state.const_array(mfi);
         const auto valid = valid_mask.const_array(mfi);
         const auto volume_fraction = use_solid_volume
            ? solid_volume_fraction_[lev]->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
         reduce_ops.eval(mfi.tilebox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
               if (valid(i, j, k, 0) == 0) {
                  return {0.0_rt, 0.0_rt};
               }
               amrex::Real volume = mesh_cell_volume_for_reduction(
                  coordinate, axes, problo, dx, i, j, k
               );
               if (use_solid_volume) {
                  volume *= volume_fraction(i, j, k, 0);
               }
               return {volume * values(i, j, k, RHO),
                       volume * values(i, j, k, ENERGY)};
            });
      }
   }
   const auto total = reduce_data.value();
   return {static_cast<amrex::Real>(amrex::get<0>(total)),
           static_cast<amrex::Real>(amrex::get<1>(total))};
}

AmrSolverBase::MassEnergy AmrSolverBase::local_rhs_mass_energy_integral(int lev) const
{
   return local_state_mass_energy_integral(lev, *fluid_state_.rhs()[lev]);
}

void AmrSolverBase::accumulate_physical_boundary_flux(
   int lev,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::Real weight
)
{
   if (weight == 0.0_rt || !external_conservation_budget_supported()) {
      return;
   }
   const auto periodic = Geom(lev).isPeriodicArray();
   bool has_physical_boundary = false;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      has_physical_boundary = has_physical_boundary || !periodic[dir];
   }
   const bool has_excision_boundary =
      cfg_.spacetime_black_hole_matter == "excision";
   if (!has_physical_boundary && !has_excision_boundary) {
      return;
   }

   amrex::iMultiFab valid_mask = valid_cell_mask(lev);
   const amrex::Box& domain = Geom(lev).Domain();
   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   const auto problo = Geom(lev).ProbLoArray();
   const auto dx = Geom(lev).CellSizeArray();
   const bool use_solid_aperture = solid_active(lev);
   amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
   amrex::ReduceData<ReductionSum, ReductionSum> reduce_data(reduce_ops);
   using ReduceTuple = decltype(reduce_data)::Type;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (periodic[dir]) {
         continue;
      }
      for (int side = 0; side < 2; ++side) {
         const bool high_side = side == 1;
         const int cell_index = high_side ? domain.bigEnd(dir) : domain.smallEnd(dir);
         const amrex::Real sign = high_side ? -1.0_rt : 1.0_rt;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
         {
            for (amrex::MFIter mfi(*fluid_state_.state()[lev], amrex::TilingIfNotGPU());
                 mfi.isValid(); ++mfi) {
               amrex::Box bx = mfi.tilebox() & domain;
               if (cell_index < bx.smallEnd(dir) || cell_index > bx.bigEnd(dir)) {
                  continue;
               }
               bx.setSmall(dir, cell_index);
               bx.setBig(dir, cell_index);
               const auto flux = fluxes[dir]->const_array(mfi);
               const auto valid = valid_mask.const_array(mfi);
               const auto aperture = use_solid_aperture
                  ? solid_aperture_[dir][lev]->const_array(mfi)
                  : amrex::Array4<amrex::Real const>{};
               reduce_ops.eval(bx, reduce_data,
                  [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                     if (valid(i, j, k, 0) == 0) {
                        return {0.0_rt, 0.0_rt};
                     }
                     const int fi = i + (high_side && dir == 0);
                     const int fj = j + (high_side && dir == 1);
                     const int fk = k + (high_side && dir == 2);
                     amrex::Real area = mesh_face_area_for_reduction(
                        dir, coordinate, axes, problo, dx, fi, fj, fk
                     );
                     if (use_solid_aperture) {
                        area *= aperture(fi, fj, fk, 0);
                     }
                     return {sign * area * flux(fi, fj, fk, RHO),
                             sign * area * flux(fi, fj, fk, ENERGY)};
                  });
            }
         }
      }
   }
   if (has_excision_boundary) {
      amrex::iMultiFab const* excision =
         apparent_horizon_tracker_->excision_mask(
            lev, *fluid_state_.state()[lev], Geom(lev)
         );
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
         excision != nullptr,
         "black-hole excision conservation requires a horizon mask"
      );
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
         for (amrex::MFIter mfi(*fluid_state_.state()[lev], amrex::TilingIfNotGPU());
              mfi.isValid(); ++mfi) {
            const auto valid = valid_mask.const_array(mfi);
            const auto inactive = excision->const_array(mfi);
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> face_flux{};
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
               face_flux[dir] = fluxes[dir]->const_array(mfi);
            }
            reduce_ops.eval(mfi.tilebox(), reduce_data,
               [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                  amrex::Real mass_rate = 0.0_rt;
                  amrex::Real energy_rate = 0.0_rt;
                  if (valid(i, j, k, 0) == 0) {
                     return {mass_rate, energy_rate};
                  }
                  // Count each excision face from its retained cell. The mask
                  // includes neighboring ghosts across FAB/rank boundaries.
                  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                     const int di = dir == 0;
                     const int dj = dir == 1;
                     const int dk = dir == 2;
                     if (inactive(i - di, j - dj, k - dk, 0) != 0) {
                        const amrex::Real area = mesh_face_area_for_reduction(
                           dir, coordinate, axes, problo, dx, i, j, k
                        );
                        mass_rate += area * face_flux[dir](i, j, k, RHO);
                        energy_rate += area * face_flux[dir](i, j, k, ENERGY);
                     }
                     const int fi = i + di;
                     const int fj = j + dj;
                     const int fk = k + dk;
                     if (inactive(fi, fj, fk, 0) != 0) {
                        const amrex::Real area = mesh_face_area_for_reduction(
                           dir, coordinate, axes, problo, dx, fi, fj, fk
                        );
                        mass_rate -= area * face_flux[dir](fi, fj, fk, RHO);
                        energy_rate -= area * face_flux[dir](fi, fj, fk, ENERGY);
                     }
                  }
                  return {mass_rate, energy_rate};
               });
         }
      }
   }
   const auto rate = reduce_data.value();
   conservation_ledger_.boundary_change.mass += weight * static_cast<amrex::Real>(amrex::get<0>(rate));
   conservation_ledger_.boundary_change.energy += weight * static_cast<amrex::Real>(amrex::get<1>(rate));
}

void AmrSolverBase::accumulate_source_change(MassEnergy const& before, MassEnergy const& after, amrex::Real weight)
{
   if (weight == 0.0_rt || !external_conservation_budget_supported()) {
      return;
   }
   conservation_ledger_.source_change.mass += weight * (after.mass - before.mass);
   conservation_ledger_.source_change.energy += weight * (after.energy - before.energy);
}

} // namespace pyfluid_amrex
