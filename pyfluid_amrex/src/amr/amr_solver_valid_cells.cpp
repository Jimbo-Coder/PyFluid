#include "amr/amr_solver_base.hpp"

#include <algorithm>
#include <array>

#include "pyfluid_amrex/apparent_horizon_tracker.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

amrex::iMultiFab AmrSolverBase::valid_cell_mask(int lev) const
{
   amrex::iMultiFab mask(
      fluid_state_.state()[lev]->boxArray(),
      fluid_state_.state()[lev]->DistributionMap(),
      1,
      0
   );
   if (lev < finestLevel()) {
      mask = amrex::makeFineMask(
         *fluid_state_.state()[lev],
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
   if (cfg_.spacetime_black_hole_matter == "excision") {
      amrex::iMultiFab const* excision =
         apparent_horizon_tracker_->excision_mask(
            lev, *fluid_state_.state()[lev], Geom(lev)
         );
      // Solve-only ID summaries and artifact bookkeeping run before the
      // mandatory step-zero horizon search.  They describe the complete
      // materialized ID and therefore retain ordinary hierarchy ownership in
      // this pre-surface phase.  Startup finds the horizon before preflight or
      // evolution; a failed search remains fatal for an excision run.
      if (excision == nullptr) {
         return mask;
      }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(mask, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box& box = mfi.tilebox();
         const auto valid = mask.array(mfi);
         const auto inactive = excision->const_array(mfi);
         amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if (inactive(i, j, k, 0) != 0) {
                  valid(i, j, k, 0) = 0;
               }
            }
         );
      }
   }
   return mask;
}

AmrSolverBase::HydroConservedTotals AmrSolverBase::hydro_valid_conserved_totals() const
{
   HydroConservedTotals totals;
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      const amrex::Real cell_volume = cartesian_cell_volume(Geom(lev));
      amrex::iMultiFab mask = valid_cell_mask(lev);
      amrex::MultiFab reduction(fluid_state_.state()[lev]->boxArray(), fluid_state_.state()[lev]->DistributionMap(), NVAR_HYDRO + 1, 0);
      reduction.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(reduction, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto s = fluid_state_.state()[lev]->const_array(mfi);
         const auto valid = mask.const_array(mfi);
         const bool use_volume_fraction = solid_active(lev) && solid_volume_fraction_[lev] != nullptr;
         const auto vf = use_volume_fraction ? solid_volume_fraction_[lev]->const_array(mfi) : amrex::Array4<amrex::Real const>{};
         const auto out = reduction.array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (valid(i, j, k, 0) == 0) {
               return;
            }
            const amrex::Real weight = use_volume_fraction ? vf(i, j, k, 0) : 1.0_rt;
            for (int n = 0; n < NVAR_HYDRO; ++n) {
               out(i, j, k, n) = weight * s(i, j, k, n);
            }
            out(i, j, k, NVAR_HYDRO) = weight;
         });
      }
      }
      for (int n = 0; n < NVAR_HYDRO; ++n) {
         totals.value[n] += reduction.sum(n, true) * cell_volume;
      }
      totals.volume += reduction.sum(NVAR_HYDRO, true) * cell_volume;
   }
   std::array<amrex::Real, NVAR_HYDRO + 1> reduced_totals{};
   std::copy(totals.value.begin(), totals.value.end(), reduced_totals.begin());
   reduced_totals[NVAR_HYDRO] = totals.volume;
   amrex::ParallelDescriptor::ReduceRealSum(
      reduced_totals.data(), static_cast<int>(reduced_totals.size())
   );
   std::copy_n(reduced_totals.begin(), NVAR_HYDRO, totals.value.begin());
   totals.volume = reduced_totals[NVAR_HYDRO];
   return totals;
}

} // namespace pyfluid_amrex
