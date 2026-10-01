#include "amr/amr_solver_base.hpp"

#include <AMReX_ParallelDescriptor.H>

#include <cmath>

#include "pyfluid_amrex/gravity.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

AmrSolverBase::DensityTotal AmrSolverBase::valid_density_total(LevelData const& data) const
{
   DensityTotal total;
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      const amrex::Real cell_volume = cartesian_cell_volume(Geom(lev));
      amrex::iMultiFab mask = valid_cell_mask(lev);
      amrex::MultiFab reduction(data[lev]->boxArray(), data[lev]->DistributionMap(), 2, 0);
      reduction.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
         for (amrex::MFIter mfi(reduction, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const auto s = data[lev]->const_array(mfi);
            const auto valid = mask.const_array(mfi);
            const auto out = reduction.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if (valid(i, j, k, 0) == 0) {
                  return;
               }
               out(i, j, k, 0) = s(i, j, k, RHO);
               out(i, j, k, 1) = 1.0_rt;
            });
         }
      }
      total.mass += reduction.sum(0, true) * cell_volume;
      total.volume += reduction.sum(1, true) * cell_volume;
   }
   std::array<amrex::Real, 2> density_totals{total.mass, total.volume};
   amrex::ParallelDescriptor::ReduceRealSum(
      density_totals.data(), static_cast<int>(density_totals.size())
   );
   total.mass = density_totals[0];
   total.volume = density_totals[1];
   return total;
}

void AmrSolverBase::solve_gravity_potential(LevelData const& input, bool fill_gradient)
{
   if (!self_gravity_enabled(cfg_)) {
      return;
   }
   const DensityTotal density = valid_density_total(input);
   if (!(density.volume > 0.0_rt) || !std::isfinite(density.mass)) {
      amrex::Abort("gravity solve needs a finite positive valid volume");
   }
   const amrex::Real mean_density = density.mass / density.volume;
   const int nlev = finestLevel() + 1;
   amrex::Vector<amrex::Geometry> geom(nlev);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> rhs_storage(nlev);
   amrex::Vector<amrex::MultiFab const*> rhs(nlev);
   for (int lev = 0; lev < nlev; ++lev) {
      geom[lev] = Geom(lev);
      rhs_storage[lev] = std::make_unique<amrex::MultiFab>(boxArray(lev), DistributionMap(lev), 1, 0);
      amrex::iMultiFab mask = valid_cell_mask(lev);
      fill_gravity_poisson_rhs(*rhs_storage[lev], *input[lev], mask, mean_density, cfg_);
      rhs[lev] = rhs_storage[lev].get();
   }
   gravity_fields_.solve_poisson(geom, rhs, cfg_, fill_gradient);
}

void AmrSolverBase::ensure_fixed_gravity_fields()
{
   if (!fixed_gravity_enabled(cfg_)) {
      return;
   }
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      gravity_fields_.ensure_fixed_level(lev, Geom(lev), cfg_);
   }
}

} // namespace pyfluid_amrex
