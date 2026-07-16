#pragma once

#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/kernels_generated/differential_ops.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic_fluid.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/rhd_system.hpp"
#include "pyfluid_amrex/rmhd_system.hpp"
#include "pyfluid_amrex/z4c_pointwise.hpp"

namespace pyfluid_amrex {

template <typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_eulerian_matter(
   Array const& output,
   int i,
   int j,
   int k,
   EulerianMatter const& matter
) noexcept
{
   output(i, j, k, EULERIAN_MATTER_RHO) = matter.rho;
   output(i, j, k, EULERIAN_MATTER_MOMENTUM1) = matter.momentum1;
   output(i, j, k, EULERIAN_MATTER_MOMENTUM2) = matter.momentum2;
   output(i, j, k, EULERIAN_MATTER_MOMENTUM3) = matter.momentum3;
   output(i, j, k, EULERIAN_MATTER_STRESS11) = matter.stress11;
   output(i, j, k, EULERIAN_MATTER_STRESS12) = matter.stress12;
   output(i, j, k, EULERIAN_MATTER_STRESS13) = matter.stress13;
   output(i, j, k, EULERIAN_MATTER_STRESS22) = matter.stress22;
   output(i, j, k, EULERIAN_MATTER_STRESS23) = matter.stress23;
   output(i, j, k, EULERIAN_MATTER_STRESS33) = matter.stress33;
}

template <typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
EulerianMatter load_eulerian_matter(
   Array const& input,
   int i,
   int j,
   int k
) noexcept
{
   return {
      input(i, j, k, EULERIAN_MATTER_RHO),
      input(i, j, k, EULERIAN_MATTER_MOMENTUM1),
      input(i, j, k, EULERIAN_MATTER_MOMENTUM2),
      input(i, j, k, EULERIAN_MATTER_MOMENTUM3),
      input(i, j, k, EULERIAN_MATTER_STRESS11),
      input(i, j, k, EULERIAN_MATTER_STRESS12),
      input(i, j, k, EULERIAN_MATTER_STRESS13),
      input(i, j, k, EULERIAN_MATTER_STRESS22),
      input(i, j, k, EULERIAN_MATTER_STRESS23),
      input(i, j, k, EULERIAN_MATTER_STRESS33),
   };
}

template <int Order, int Dim>
inline void transfer_z4c_vertex_to_cell(
   amrex::MultiFab& cell_state,
   amrex::MultiFab const& vertex_state
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(cell_state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.growntilebox();
      const auto source = vertex_state.const_array(mfi);
      const auto destination = cell_state.array(mfi);
      amrex::ParallelFor(
         box,
         static_cast<int>(Z4C_NUM_COMPONENTS),
         [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
            destination(i, j, k, component) = generated::vertex_to_cell<
               Order,
               Dim
            >(source, i, j, k, component);
         }
      );
   }
}

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
      const amrex::Box box = mfi.growntilebox();
      const auto primitive_array = primitive.const_array(mfi);
      const auto spacetime_array = cell_spacetime.const_array(mfi);
      const auto matter_array = cell_matter.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
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

template <int Order, int Dim>
inline void transfer_matter_cell_to_vertex(
   amrex::MultiFab& vertex_matter,
   amrex::MultiFab const& cell_matter
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(vertex_matter, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto source = cell_matter.const_array(mfi);
      const auto destination = vertex_matter.array(mfi);
      amrex::ParallelFor(
         box,
         static_cast<int>(EULERIAN_MATTER_NUM_COMPONENTS),
         [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
            destination(i, j, k, component) = generated::cell_to_vertex<
               Order,
               Dim
            >(source, i, j, k, component);
         }
      );
   }
}

} // namespace pyfluid_amrex
