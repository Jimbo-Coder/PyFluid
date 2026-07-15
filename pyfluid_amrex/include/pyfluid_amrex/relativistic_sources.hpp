#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic_fluid.hpp"
#include "pyfluid_amrex/rhd_system.hpp"
#include "pyfluid_amrex/rmhd_system.hpp"

namespace pyfluid_amrex {

template <PhysicalSystem System>
inline void add_fixed_spacetime_sources(
   amrex::MultiFab& rhs,
   const amrex::MultiFab& primitive,
   const amrex::Geometry& geometry,
   const FixedSpacetimeParameters& spacetime,
   const EOSParams& eos,
   amrex::Real time
)
{
   static_assert(System == PhysicalSystem::RHD || System == PhysicalSystem::RMHD);
   if (!has_fixed_spacetime_sources(spacetime)) {
      return;
   }
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto cached = primitive.const_array(mfi);
      const auto source_rhs = rhs.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const FixedSpacetimeGeometry local_geometry =
            fixed_spacetime_geometry_at_cell<AMREX_SPACEDIM>(
               spacetime, lower, spacing, i, j, k, time
            );
         if constexpr (System == PhysicalSystem::RHD) {
            const RHDConserved source = generated::rhd_geometric_source(
               rhd_primitive_at(cached, i, j, k, eos),
               local_geometry.metric,
               local_geometry.derivatives
            );
            source_rhs(i, j, k, RHO) += source.dens;
            source_rhs(i, j, k, MOM1) += source.mom1;
            source_rhs(i, j, k, MOM2) += source.mom2;
            source_rhs(i, j, k, MOM3) += source.mom3;
            source_rhs(i, j, k, ENERGY) += source.tau;
         } else {
            const RMHDConserved source = generated::rmhd_geometric_source(
               rmhd_primitive_at(cached, i, j, k, eos),
               local_geometry.metric,
               local_geometry.derivatives
            );
            source_rhs(i, j, k, RHO) += source.dens;
            source_rhs(i, j, k, MOM1) += source.mom1;
            source_rhs(i, j, k, MOM2) += source.mom2;
            source_rhs(i, j, k, MOM3) += source.mom3;
            source_rhs(i, j, k, ENERGY) += source.tau;
            source_rhs(i, j, k, B1) += source.b1;
            source_rhs(i, j, k, B2) += source.b2;
            source_rhs(i, j, k, B3) += source.b3;
         }
      });
   }
}

} // namespace pyfluid_amrex
