#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <memory>
#include <vector>

namespace pyfluid_amrex {

struct RunConfig;
using GravityGradient = std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>;

// Owns gravity field allocation and refresh. Stage energy/flux registers remain
// part of the integrator; the Poisson RHS is supplied by the hierarchy driver.
class GravityFields {
public:
   void resize(int num_levels);
   void allocate_level(int level, amrex::BoxArray const& boxes,
                       amrex::DistributionMapping const& distribution);
   void clear_level(int level);
   void ensure_fixed_level(int level, amrex::Geometry const& geometry,
                           RunConfig const& config);
   void solve_poisson(amrex::Vector<amrex::Geometry> const& geometry,
                      amrex::Vector<amrex::MultiFab const*> const& rhs,
                      RunConfig const& config, bool fill_gradient);

   amrex::MultiFab const* potential(int level) const noexcept
   {
      return potentials_[level].get();
   }
   amrex::MultiFab const* gradient(int level, int direction) const noexcept
   {
      return gradients_[level][direction].get();
   }
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM>
   gradient_pointers(int level) const noexcept
   {
      std::array<amrex::MultiFab const*, AMREX_SPACEDIM> pointers{};
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         pointers[direction] = gradient(level, direction);
      }
      return pointers;
   }

private:
   std::vector<std::unique_ptr<amrex::MultiFab>> potentials_;
   std::vector<GravityGradient> gradients_;
   std::vector<bool> fixed_valid_;
};

} // namespace pyfluid_amrex
