#include "pyfluid_amrex/nodal_interpolater.hpp"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>
#include "pyfluid_amrex/interpolation_weights.hpp"
#include "pyfluid_amrex/spacetime_excision.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

NodalInterpolater::NodalInterpolater(
   SphericalExcisionRegion region, int spatial_order
) : excision_region_(region), spatial_order_(spatial_order)
{
   if (!is_supported_spatial_order(spatial_order_)) {
      amrex::Abort("nodal interpolation spatial order is unsupported");
   }
}

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
         const int ordinary_lo = spatial_order_ / 2 - 1;
         const int ordinary_hi = spatial_order_ / 2;
         const int excision_width = spatial_order_ - 1;
         coarse.growLo(
            direction, excision_region_.enabled() ? excision_width : ordinary_lo
         );
         coarse.growHi(
            direction, excision_region_.enabled() ? excision_width : ordinary_hi
         );
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
   const auto source_grid = excision_grid_view(excision_region_, coarse_geometry, coarse.ixType());
   const auto destination_grid = excision_grid_view(excision_region_, fine_geometry, fine.ixType());
   const int spatial_order = spatial_order_;
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
         const int available_i = coarse_upper[0] - coarse_lower[0] + 1;
         const int count_i = interpolate_i ? amrex::min(spatial_order, available_i) : 1;
         int first_i = interpolate_i
            ? amrex::max(
               coarse_lower[0],
               amrex::min(coarse_i - count_i / 2 + 1, coarse_upper[0] - count_i + 1)
            )
            : coarse_i;
         auto weight_i = amrex::GpuArray<amrex::Real, maximum_spatial_order>{};
         weight_i[0] = 1.0_rt;
         if (interpolate_i) {
            weight_i = uniform_lagrange_weights<maximum_spatial_order>(
               coarse_i + offset_i, first_i, count_i
            );
         }
#if (AMREX_SPACEDIM >= 2)
         const int coarse_j = amrex::coarsen(j, ratio[1]);
         const int offset_index_j = j - coarse_j * ratio[1];
         const bool interpolate_j = ratio[1] > 1 && offset_index_j != 0;
         const amrex::Real offset_j = static_cast<amrex::Real>(offset_index_j)
            / static_cast<amrex::Real>(ratio[1]);
         const int available_j = coarse_upper[1] - coarse_lower[1] + 1;
         const int count_j = interpolate_j ? amrex::min(spatial_order, available_j) : 1;
         int first_j = interpolate_j
            ? amrex::max(
               coarse_lower[1],
               amrex::min(coarse_j - count_j / 2 + 1, coarse_upper[1] - count_j + 1)
            )
            : coarse_j;
         auto weight_j = amrex::GpuArray<amrex::Real, maximum_spatial_order>{};
         weight_j[0] = 1.0_rt;
         if (interpolate_j) {
            weight_j = uniform_lagrange_weights<maximum_spatial_order>(
               coarse_j + offset_j, first_j, count_j
            );
         }
#else
         const int coarse_j = 0;
         const int count_j = 1;
         const int first_j = 0;
         const amrex::GpuArray<amrex::Real, maximum_spatial_order> weight_j{{1.0_rt}};
#endif
#if (AMREX_SPACEDIM == 3)
         const int coarse_k = amrex::coarsen(k, ratio[2]);
         const int offset_index_k = k - coarse_k * ratio[2];
         const bool interpolate_k = ratio[2] > 1 && offset_index_k != 0;
         const amrex::Real offset_k = static_cast<amrex::Real>(offset_index_k)
            / static_cast<amrex::Real>(ratio[2]);
         const int available_k = coarse_upper[2] - coarse_lower[2] + 1;
         const int count_k = interpolate_k ? amrex::min(spatial_order, available_k) : 1;
         int first_k = interpolate_k
            ? amrex::max(
               coarse_lower[2],
               amrex::min(coarse_k - count_k / 2 + 1, coarse_upper[2] - count_k + 1)
            )
            : coarse_k;
         auto weight_k = amrex::GpuArray<amrex::Real, maximum_spatial_order>{};
         weight_k[0] = 1.0_rt;
         if (interpolate_k) {
            weight_k = uniform_lagrange_weights<maximum_spatial_order>(
               coarse_k + offset_k, first_k, count_k
            );
         }
#else
         const int coarse_k = 0;
         const int count_k = 1;
         const int first_k = 0;
         const amrex::GpuArray<amrex::Real, maximum_spatial_order> weight_k{{1.0_rt}};
#endif

         if (source_grid.region.enabled() && destination_grid(i, j, k, 0)) {
            amrex::GpuArray<int, 3> first{{first_i, first_j, first_k}};
            const amrex::GpuArray<int, 3> count{{count_i, count_j, count_k}};
            source_grid.restrict_stencil(destination_grid.position(i, j, k), first, count);
            first_i = first[0];
            if (interpolate_i) weight_i = uniform_lagrange_weights<maximum_spatial_order>(
               coarse_i + offset_i, first_i, count_i);
#if (AMREX_SPACEDIM >= 2)
            first_j = first[1];
            if (interpolate_j) weight_j = uniform_lagrange_weights<maximum_spatial_order>(
               coarse_j + offset_j, first_j, count_j);
#endif
#if (AMREX_SPACEDIM == 3)
            first_k = first[2];
            if (interpolate_k) weight_k = uniform_lagrange_weights<maximum_spatial_order>(
               coarse_k + offset_k, first_k, count_k);
#endif
         }
         amrex::Real value = 0.0_rt;
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

amrex::MFInterpolater* nodal_interpolater(int spatial_order) noexcept
{
   return dispatch_spatial_order(spatial_order, [](auto order_tag) -> amrex::MFInterpolater* {
      static NodalInterpolater interpolater({}, decltype(order_tag)::value);
      return &interpolater;
   });
}

} // namespace pyfluid_amrex
