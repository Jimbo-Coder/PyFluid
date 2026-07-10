#include "pyfluid_amrex/amr_solver.hpp"

namespace pyfluid_amrex {

amrex::iMultiFab AmrSolver::valid_cell_mask(int lev) const
{
   amrex::iMultiFab mask(
      state_[lev]->boxArray(),
      state_[lev]->DistributionMap(),
      1,
      0
   );
   if (lev < finestLevel()) {
      mask = amrex::makeFineMask(
         *state_[lev],
         boxArray(lev + 1),
         refRatio(lev),
         Geom(lev).periodicity(),
         1,
         0
      );
   } else {
      mask.setVal(1);
   }
   remove_solid_cells_from_valid_mask(mask, active_solid_mask_ptr(lev));
   return mask;
}

AmrSolver::HydroConservedTotals AmrSolver::hydro_valid_conserved_totals() const
{
   HydroConservedTotals totals;
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      const amrex::Real cell_volume = cartesian_cell_volume(Geom(lev));
      amrex::iMultiFab mask = valid_cell_mask(lev);
      amrex::MultiFab reduction(state_[lev]->boxArray(), state_[lev]->DistributionMap(), NVAR_HYDRO + 1, 0);
      reduction.setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(reduction, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto s = state_[lev]->const_array(mfi);
         const auto valid = mask.const_array(mfi);
         const bool use_volume_fraction = solid_active(lev) && solid_volume_fraction_[lev] != nullptr;
         const auto vf = use_volume_fraction ? solid_volume_fraction_[lev]->const_array(mfi) : amrex::Array4<amrex::Real const>{};
         const auto out = reduction.array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (valid(i, j, k, 0) == 0) {
               return;
            }
            const amrex::Real weight = use_volume_fraction ? vf(i, j, k, 0) : 1.0;
            for (int n = 0; n < NVAR_HYDRO; ++n) {
               out(i, j, k, n) = weight * s(i, j, k, n);
            }
            out(i, j, k, NVAR_HYDRO) = weight;
         });
      }
      }
      for (int n = 0; n < NVAR_HYDRO; ++n) {
         totals.value[n] += reduction.sum(n) * cell_volume;
      }
      totals.volume += reduction.sum(NVAR_HYDRO) * cell_volume;
   }
   return totals;
}

} // namespace pyfluid_amrex
