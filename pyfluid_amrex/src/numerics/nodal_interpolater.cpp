#include "pyfluid_amrex/nodal_interpolater.hpp"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

namespace pyfluid_amrex {

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 4> cubic_lagrange_weights(amrex::Real position) noexcept
{
   return {{
      -position * (position - 1.0) * (position - 2.0) / 6.0,
      (position + 1.0) * (position - 1.0) * (position - 2.0) / 2.0,
      -(position + 1.0) * position * (position - 2.0) / 2.0,
      (position + 1.0) * position * (position - 1.0) / 6.0,
   }};
}

FourthOrderNodalInterpolater interpolater;

} // namespace

amrex::Box FourthOrderNodalInterpolater::CoarseBox(
   amrex::Box const& fine,
   int ratio
)
{
   return CoarseBox(fine, amrex::IntVect(ratio));
}

amrex::Box FourthOrderNodalInterpolater::CoarseBox(
   amrex::Box const& fine,
   amrex::IntVect const& ratio
)
{
   amrex::Box coarse = amrex::coarsen(fine, ratio);
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      if (ratio[direction] > 1) {
         coarse.growLo(direction, 1);
         coarse.growHi(direction, 2);
      }
   }
   return coarse;
}

void FourthOrderNodalInterpolater::interp(
   amrex::MultiFab const& coarse,
   int coarse_component,
   amrex::MultiFab& fine,
   int fine_component,
   int num_components,
   amrex::IntVect const& num_ghost,
   amrex::Geometry const& coarse_geometry,
   amrex::Geometry const& fine_geometry,
   amrex::Box const& destination_domain,
   amrex::IntVect const& ratio,
   amrex::Vector<amrex::BCRec> const& boundary_conditions,
   int boundary_component
)
{
   amrex::ignore_unused(
      coarse_geometry,
      fine_geometry,
      boundary_conditions,
      boundary_component
   );
   auto const& coarse_arrays = coarse.const_arrays();
   auto const& fine_arrays = fine.arrays();
   amrex::ParallelFor(
      fine,
      num_ghost,
      num_components,
      [=] AMREX_GPU_DEVICE(int box_index, int i, int j, int k, int n) noexcept {
         if (!destination_domain.contains(i, j, k)) {
            return;
         }

         const int coarse_i = amrex::coarsen(i, ratio[0]);
         const amrex::Real offset_i = static_cast<amrex::Real>(
            i - coarse_i * ratio[0]
         ) / static_cast<amrex::Real>(ratio[0]);
         const auto weight_i = cubic_lagrange_weights(offset_i);
#if (AMREX_SPACEDIM >= 2)
         const int coarse_j = amrex::coarsen(j, ratio[1]);
         const amrex::Real offset_j = static_cast<amrex::Real>(
            j - coarse_j * ratio[1]
         ) / static_cast<amrex::Real>(ratio[1]);
         const auto weight_j = cubic_lagrange_weights(offset_j);
#else
         const int coarse_j = 0;
#endif
#if (AMREX_SPACEDIM == 3)
         const int coarse_k = amrex::coarsen(k, ratio[2]);
         const amrex::Real offset_k = static_cast<amrex::Real>(
            k - coarse_k * ratio[2]
         ) / static_cast<amrex::Real>(ratio[2]);
         const auto weight_k = cubic_lagrange_weights(offset_k);
#else
         const int coarse_k = 0;
#endif

         amrex::Real value = 0.0;
         for (int di = 0; di < 4; ++di) {
#if (AMREX_SPACEDIM >= 2)
            for (int dj = 0; dj < 4; ++dj) {
#if (AMREX_SPACEDIM == 3)
               for (int dk = 0; dk < 4; ++dk) {
                  value += weight_i[di] * weight_j[dj] * weight_k[dk]
                     * coarse_arrays[box_index](
                        coarse_i + di - 1,
                        coarse_j + dj - 1,
                        coarse_k + dk - 1,
                        coarse_component + n
                     );
               }
#else
               value += weight_i[di] * weight_j[dj]
                  * coarse_arrays[box_index](
                     coarse_i + di - 1,
                     coarse_j + dj - 1,
                     0,
                     coarse_component + n
                  );
#endif
            }
#else
            value += weight_i[di] * coarse_arrays[box_index](
               coarse_i + di - 1,
               0,
               0,
               coarse_component + n
            );
#endif
         }
         fine_arrays[box_index](i, j, k, fine_component + n) = value;
      }
   );
}

amrex::MFInterpolater* fourth_order_nodal_interpolater() noexcept
{
   return &interpolater;
}

} // namespace pyfluid_amrex
