#include "pyfluid_amrex/gravity_fields.hpp"

#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>

#include "pyfluid_amrex/gravity.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

void GravityFields::resize(int num_levels)
{
   potentials_.resize(num_levels);
   gradients_.resize(num_levels);
   fixed_valid_.resize(num_levels, false);
}

void GravityFields::allocate_level(
   int level, amrex::BoxArray const& boxes,
   amrex::DistributionMapping const& distribution
)
{
   potentials_[level] = std::make_unique<amrex::MultiFab>(boxes, distribution, 1, 1);
   potentials_[level]->setVal(0.0_rt);
   allocate_gravity_gradient(gradients_[level], boxes, distribution);
   fixed_valid_[level] = false;
}

void GravityFields::clear_level(int level)
{
   potentials_[level].reset();
   for (auto& component : gradients_[level]) {
      component.reset();
   }
   fixed_valid_[level] = false;
}

void GravityFields::ensure_fixed_level(
   int level, amrex::Geometry const& geometry, RunConfig const& config
)
{
   bool gradient_ready = true;
   for (auto const& component : gradients_[level]) {
      gradient_ready = gradient_ready && component != nullptr;
   }
   if (fixed_valid_[level] && potentials_[level] && gradient_ready) {
      return;
   }
   if (!potentials_[level]) {
      amrex::Abort("fixed gravity requested before gravity storage exists");
   }
   fill_fixed_gravity_potential_and_gradient(
      *potentials_[level], gradients_[level], geometry, config
   );
   potentials_[level]->FillBoundary(geometry.periodicity());
   fixed_valid_[level] = true;
}

void GravityFields::solve_poisson(
   amrex::Vector<amrex::Geometry> const& geom,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   RunConfig const& config, bool fill_gradient
)
{
   const int nlev = static_cast<int>(geom.size());
   amrex::Vector<amrex::BoxArray> grids(nlev);
   amrex::Vector<amrex::DistributionMapping> dmap(nlev);
   amrex::Vector<amrex::MultiFab*> solution(nlev);
   for (int lev = 0; lev < nlev; ++lev) {
      if (!potentials_[lev]) {
         amrex::Abort("gravity potential storage is missing on an active AMR level");
      }
      grids[lev] = potentials_[lev]->boxArray();
      dmap[lev] = potentials_[lev]->DistributionMap();
      solution[lev] = potentials_[lev].get();
      if (fill_gradient) {
         allocate_gravity_gradient(gradients_[lev], grids[lev], dmap[lev]);
      }
   }

   amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc{};
   amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> hibc{};
   const auto periodic = geom[0].isPeriodicArray();
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (periodic[dir]) {
         lobc[dir] = amrex::LinOpBCType::Periodic;
         hibc[dir] = amrex::LinOpBCType::Periodic;
      } else {
         lobc[dir] = amrex::LinOpBCType::Neumann;
         hibc[dir] = amrex::LinOpBCType::Neumann;
      }
   }
   amrex::LPInfo info;
   amrex::MLPoisson poisson(geom, grids, dmap, info);
   poisson.setMaxOrder(config.gravity_max_order);
   poisson.setDomainBC(lobc, hibc);
   for (int lev = 0; lev < nlev; ++lev) {
      poisson.setLevelBC(lev, nullptr);
   }

   amrex::MLMG mlmg(poisson);
   mlmg.setMaxIter(config.gravity_max_iterations);
   mlmg.setMaxFmgIter(config.gravity_max_fmg_iterations);
   mlmg.setVerbose(config.gravity_verbose);
   mlmg.setBottomVerbose(config.gravity_verbose);
   const amrex::Real residual = mlmg.solve(solution, rhs, config.gravity_tolerance, 0.0_rt);
   if (!std::isfinite(residual)) {
      amrex::Abort("gravity Poisson solve produced a non-finite residual");
   }
   amrex::FillBoundary(solution, geom[0].periodicity());
   if (fill_gradient) {
      amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> grad_ptrs(nlev);
      for (int lev = 0; lev < nlev; ++lev) {
         grad_ptrs[lev] = gravity_gradient_ptrs(gradients_[lev]);
      }
      mlmg.getGradSolution(grad_ptrs);
   }
}

} // namespace pyfluid_amrex
