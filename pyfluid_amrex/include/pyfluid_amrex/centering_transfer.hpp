#pragma once

#include <AMReX.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/kernels_generated/differential_ops.hpp"

namespace pyfluid_amrex {

inline int centering_mask(amrex::IndexType const& index_type) noexcept
{
   int mask = 0;
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      if (index_type.nodeCentered(direction)) mask |= 1 << direction;
   }
   return mask;
}

template <int Order, int Dim>
inline void transfer_grid_field_centering(
   amrex::MultiFab& destination,
   amrex::MultiFab const& source,
   bool include_destination_ghosts = false
)
{
   static_assert(Order == 2 || Order == 4 || Order == 6 || Order == 8);
   static_assert(Dim == AMREX_SPACEDIM);
   if (destination.nComp() != source.nComp()) {
      amrex::Abort("centering transfer requires matching component counts");
   }
   amrex::BoxArray expected_boxes = source.boxArray();
   expected_boxes.convert(destination.ixType());
   if (destination.boxArray() != expected_boxes) {
      amrex::Abort("centering transfer requires the same underlying cell layout");
   }

   const int source_mask = centering_mask(source.ixType());
   const int destination_mask = centering_mask(destination.ixType());
   for (int direction = 0; direction < Dim; ++direction) {
      const int required_ghosts =
         (include_destination_ghosts ? destination.nGrowVect()[direction] : 0)
         + ((((source_mask ^ destination_mask) & (1 << direction)) != 0)
               ? Order / 2
               : 0);
      if (source.nGrowVect()[direction] < required_ghosts) {
         amrex::Abort("centering transfer source has insufficient ghost cells");
      }
   }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(destination, amrex::TilingIfNotGPU());
        mfi.isValid();
        ++mfi) {
      const amrex::Box box = include_destination_ghosts
         ? mfi.growntilebox()
         : mfi.tilebox();
      const auto input = source.const_array(mfi);
      const auto output = destination.array(mfi);
      amrex::ParallelFor(
         box,
         destination.nComp(),
         [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
            output(i, j, k, component) =
               generated::topology_centering_transfer<Order, Dim>(
                  input,
                  i,
                  j,
                  k,
                  component,
                  source_mask,
                  destination_mask
               );
         }
      );
   }
}

inline void transfer_grid_field_centering(
   amrex::MultiFab& destination,
   amrex::MultiFab const& source,
   int order,
   bool include_destination_ghosts = false
)
{
   switch (order) {
   case 2:
      transfer_grid_field_centering<2, AMREX_SPACEDIM>(
         destination, source, include_destination_ghosts
      );
      return;
   case 4:
      transfer_grid_field_centering<4, AMREX_SPACEDIM>(
         destination, source, include_destination_ghosts
      );
      return;
   case 6:
      transfer_grid_field_centering<6, AMREX_SPACEDIM>(
         destination, source, include_destination_ghosts
      );
      return;
   case 8:
      transfer_grid_field_centering<8, AMREX_SPACEDIM>(
         destination, source, include_destination_ghosts
      );
      return;
   default:
      amrex::Abort("centering transfer order must be 2, 4, 6, or 8");
   }
}

} // namespace pyfluid_amrex
