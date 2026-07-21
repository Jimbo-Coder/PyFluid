#include "pyfluid_amrex/nodal_interpolater.hpp"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

namespace pyfluid_amrex {

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 4> cubic_lagrange_weights(
   amrex::Real target,
   int first_index
) noexcept
{
   amrex::GpuArray<amrex::Real, 4> weights{};
   for (int point = 0; point < 4; ++point) {
      const amrex::Real node = static_cast<amrex::Real>(first_index + point);
      amrex::Real weight = 1.0;
      for (int other = 0; other < 4; ++other) {
         if (other == point) {
            continue;
         }
         const amrex::Real other_node = static_cast<amrex::Real>(
            first_index + other
         );
         weight *= (target - other_node) / (node - other_node);
      }
      weights[point] = weight;
   }
   return weights;
}

NodalInterpolater interpolater;

} // namespace

amrex::Box NodalInterpolater::CoarseBox(
   amrex::Box const& fine,
   int ratio
)
{
   return CoarseBox(fine, amrex::IntVect(ratio));
}

amrex::Box NodalInterpolater::CoarseBox(
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

void NodalInterpolater::interp(
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
      fine_geometry,
      boundary_conditions,
      boundary_component
   );
   const amrex::Box coarse_domain = amrex::convert(
      coarse_geometry.Domain(), coarse.ixType()
   );
   const auto coarse_lower = coarse_domain.loVect3d();
   const auto coarse_upper = coarse_domain.hiVect3d();
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
         const int offset_index_i = i - coarse_i * ratio[0];
         const bool interpolate_i = ratio[0] > 1 && offset_index_i != 0;
         const amrex::Real offset_i = static_cast<amrex::Real>(offset_index_i)
            / static_cast<amrex::Real>(ratio[0]);
         const int count_i = interpolate_i ? 4 : 1;
         const int first_i = interpolate_i
            ? amrex::max(
               coarse_lower[0],
               amrex::min(coarse_i - 1, coarse_upper[0] - 3)
            )
            : coarse_i;
         auto weight_i = amrex::GpuArray<amrex::Real, 4>{{1.0, 0.0, 0.0, 0.0}};
         if (interpolate_i) {
            weight_i = cubic_lagrange_weights(coarse_i + offset_i, first_i);
         }
#if (AMREX_SPACEDIM >= 2)
         const int coarse_j = amrex::coarsen(j, ratio[1]);
         const int offset_index_j = j - coarse_j * ratio[1];
         const bool interpolate_j = ratio[1] > 1 && offset_index_j != 0;
         const amrex::Real offset_j = static_cast<amrex::Real>(offset_index_j)
            / static_cast<amrex::Real>(ratio[1]);
         const int count_j = interpolate_j ? 4 : 1;
         const int first_j = interpolate_j
            ? amrex::max(
               coarse_lower[1],
               amrex::min(coarse_j - 1, coarse_upper[1] - 3)
            )
            : coarse_j;
         auto weight_j = amrex::GpuArray<amrex::Real, 4>{{1.0, 0.0, 0.0, 0.0}};
         if (interpolate_j) {
            weight_j = cubic_lagrange_weights(coarse_j + offset_j, first_j);
         }
#else
         const int coarse_j = 0;
         const int count_j = 1;
         const int first_j = 0;
         const amrex::GpuArray<amrex::Real, 4> weight_j{{1.0, 0.0, 0.0, 0.0}};
#endif
#if (AMREX_SPACEDIM == 3)
         const int coarse_k = amrex::coarsen(k, ratio[2]);
         const int offset_index_k = k - coarse_k * ratio[2];
         const bool interpolate_k = ratio[2] > 1 && offset_index_k != 0;
         const amrex::Real offset_k = static_cast<amrex::Real>(offset_index_k)
            / static_cast<amrex::Real>(ratio[2]);
         const int count_k = interpolate_k ? 4 : 1;
         const int first_k = interpolate_k
            ? amrex::max(
               coarse_lower[2],
               amrex::min(coarse_k - 1, coarse_upper[2] - 3)
            )
            : coarse_k;
         auto weight_k = amrex::GpuArray<amrex::Real, 4>{{1.0, 0.0, 0.0, 0.0}};
         if (interpolate_k) {
            weight_k = cubic_lagrange_weights(coarse_k + offset_k, first_k);
         }
#else
         const int coarse_k = 0;
         const int count_k = 1;
         const int first_k = 0;
         const amrex::GpuArray<amrex::Real, 4> weight_k{{1.0, 0.0, 0.0, 0.0}};
#endif

         amrex::Real value = 0.0;
         for (int di = 0; di < count_i; ++di) {
#if (AMREX_SPACEDIM >= 2)
            for (int dj = 0; dj < count_j; ++dj) {
#if (AMREX_SPACEDIM == 3)
               for (int dk = 0; dk < count_k; ++dk) {
                  value += weight_i[di] * weight_j[dj] * weight_k[dk]
                     * coarse_arrays[box_index](
                        first_i + di,
                        first_j + dj,
                        first_k + dk,
                        coarse_component + n
                     );
               }
#else
               value += weight_i[di] * weight_j[dj]
                  * coarse_arrays[box_index](
                     first_i + di,
                     first_j + dj,
                     0,
                     coarse_component + n
                  );
#endif
            }
#else
            value += weight_i[di] * coarse_arrays[box_index](
               first_i + di,
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

amrex::MFInterpolater* nodal_interpolater() noexcept
{
   return &interpolater;
}

} // namespace pyfluid_amrex
