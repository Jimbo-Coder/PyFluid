#pragma once

#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/eulerian_matter_access.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/sources.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {

template <PhysicalSystem System, int EOSMethod>
inline void fill_eulerian_matter_cells(
   amrex::MultiFab& cell_matter,
   amrex::MultiFab const& primitive,
   amrex::MultiFab const& cell_spacetime,
   EOSParams const& eos
)
{
   static_assert(System == PhysicalSystem::RHD || System == PhysicalSystem::RMHD);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(cell_matter, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto primitive_array = primitive.const_array(mfi);
      const auto spacetime_array = cell_spacetime.const_array(mfi);
      const auto matter_array = cell_matter.array(mfi);
      amrex::ParallelFor(box, [spacetime_array, matter_array, primitive_array, eos] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(
            load_z4c_state(spacetime_array, i, j, k)
         );
         if constexpr (System == PhysicalSystem::RHD) {
            store_eulerian_matter(
               matter_array,
               i,
               j,
               k,
               generated::rhd_eulerian_matter(
                  rhd_primitive_at<EOSMethod>(primitive_array, i, j, k, eos), metric
               )
            );
         } else {
            store_eulerian_matter(
               matter_array,
               i,
               j,
               k,
               generated::rmhd_eulerian_matter(
                  rmhd_primitive_at(primitive_array, i, j, k, eos), metric
               )
            );
         }
      });
   }
}

} // namespace pyfluid_amrex
