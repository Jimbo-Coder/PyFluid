#include "pyfluid_amrex/amr_solver.hpp"

#include <AMReX_BoxIterator.H>

namespace pyfluid_amrex {

bool AmrSolver::debug_contracts_enabled() const
{
   return cfg_.debug_poison || cfg_.debug_check_finite || cfg_.debug_check_physical || cfg_.physical_check_int > 0;
}

bool AmrSolver::accepted_physical_check_due(int step, bool force) const
{
   if (cfg_.debug_check_physical || force) {
      return true;
   }
   if (cfg_.physical_check_int <= 0) {
      return false;
   }
   return step % cfg_.physical_check_int == 0;
}

void AmrSolver::debug_poison_mfab(amrex::MultiFab& mf) const
{
   if (!cfg_.debug_poison) {
      return;
   }
   mf.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
}

void AmrSolver::debug_poison_leveldata(LevelData& data, int lev_begin, int lev_end) const
{
   if (!cfg_.debug_poison) {
      return;
   }
   for (int lev = lev_begin; lev < lev_end; ++lev) {
      if (data[lev]) {
         debug_poison_mfab(*data[lev]);
      }
   }
}

void AmrSolver::debug_poison_fluxes(int lev) const
{
   if (!cfg_.debug_poison) {
      return;
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (fluxes_[lev][dir]) {
         debug_poison_mfab(*fluxes_[lev][dir]);
      }
   }
}

void AmrSolver::debug_check_mfab_finite(
   amrex::MultiFab const& mf,
   int ncomp,
   amrex::IntVect const& ngrow,
   char const* context
) const
{
   if (!cfg_.debug_check_finite) {
      return;
   }
   if (mf.is_finite(0, ncomp, ngrow, false)) {
      return;
   }
   int first_i = std::numeric_limits<int>::min();
   int first_j = std::numeric_limits<int>::min();
   int first_k = std::numeric_limits<int>::min();
   int first_n = -1;
   amrex::Real first_value = std::numeric_limits<amrex::Real>::quiet_NaN();
   for (amrex::MFIter mfi(mf); mfi.isValid() && first_n < 0; ++mfi) {
      amrex::Box bx = amrex::grow(mfi.validbox(), ngrow);
      bx &= mfi.fabbox();
      const auto values = mf.const_array(mfi);
      for (amrex::BoxIterator bit(bx); bit.ok() && first_n < 0; ++bit) {
         const amrex::IntVect cell = bit();
         const int i = cell[0];
         int j = 0;
         int k = 0;
#if (AMREX_SPACEDIM >= 2)
         j = cell[1];
#endif
#if (AMREX_SPACEDIM == 3)
         k = cell[2];
#endif
         for (int n = 0; n < ncomp; ++n) {
            const amrex::Real value = values(i, j, k, n);
            if (!std::isfinite(value)) {
               first_i = i;
               first_j = j;
               first_k = k;
               first_n = n;
               first_value = value;
               break;
            }
         }
      }
   }
   std::ostringstream message;
   message << "Debug finite contract failed in " << context
           << ": detected NaN/Inf in " << ncomp << " component(s)"
           << "; first_bad=(" << first_i << "," << first_j << "," << first_k
           << ") component=" << first_n << " value=" << first_value << ".";
   const std::string text = message.str();
   amrex::Abort(text.c_str());
}

void AmrSolver::debug_check_valid_cell_finite(
   amrex::MultiFab const& mf,
   amrex::iMultiFab const& valid_mask,
   int ncomp,
   char const* context,
   int lev
) const
{
   if (!cfg_.debug_check_finite) {
      return;
   }
   amrex::Long bad_count = 0;
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+:bad_count) if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto lo = amrex::lbound(bx);
      const auto hi = amrex::ubound(bx);
      const auto values = mf.const_array(mfi);
      const auto mask = valid_mask.const_array(mfi);
      for (int k = lo.z; k <= hi.z; ++k) {
         for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
               if (mask(i, j, k, 0) == 0) {
                  continue;
               }
               for (int n = 0; n < ncomp; ++n) {
                  if (!std::isfinite(values(i, j, k, n))) {
                     ++bad_count;
                  }
               }
            }
         }
      }
   }
   }
   amrex::ParallelDescriptor::ReduceLongSum(bad_count);
   if (bad_count == 0) {
      return;
   }
   std::ostringstream message;
   message << "Debug finite contract failed in " << context
           << " on level " << lev
           << ": " << bad_count << " invalid active-cell value(s).";
   const std::string text = message.str();
   amrex::Abort(text.c_str());
}

void AmrSolver::abort_bad_physical_state(
   char const* context,
   int step,
   amrex::Real time,
   int lev,
   amrex::Long bad_count,
   amrex::Real min_rho,
   amrex::Real min_pressure,
   int first_bad_i,
   int first_bad_j,
   int first_bad_k,
   amrex::Real first_bad_rho,
   amrex::Real first_bad_pressure
) const
{
   std::ostringstream message;
   message << "Physical-state check failed in " << context
           << " at step " << step
           << " time " << time
           << " on level " << lev
           << ": bad_active_cells=" << bad_count
           << " min_rho=" << min_rho
           << " min_pressure=" << min_pressure
           << " first_local_bad=(" << first_bad_i << "," << first_bad_j << "," << first_bad_k << ")"
           << " first_local_rho=" << first_bad_rho
           << " first_local_pressure=" << first_bad_pressure
           << ".";
   const std::string text = message.str();
   amrex::Abort(text.c_str());
}

void AmrSolver::debug_check_rhs_level(int lev, char const* context) const
{
   if (!cfg_.debug_check_finite || !rhs_[lev]) {
      return;
   }
   amrex::iMultiFab mask = valid_cell_mask(lev);
   debug_check_valid_cell_finite(*rhs_[lev], mask, num_state_components(), context, lev);
}

void AmrSolver::debug_check_fluxes(int lev, bool solid_masked, char const* context) const
{
   if (!cfg_.debug_check_finite || solid_masked) {
      return;
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (fluxes_[lev][dir]) {
         std::ostringstream label;
         label << context << " level " << lev << " dir " << dir;
         debug_check_mfab_finite(*fluxes_[lev][dir], num_state_components(), amrex::IntVect(0), label.str().c_str());
      }
   }
}

} // namespace pyfluid_amrex
