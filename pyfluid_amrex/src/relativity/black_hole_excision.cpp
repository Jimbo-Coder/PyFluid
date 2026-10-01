#include "pyfluid_amrex/black_hole_excision.hpp"

#include <AMReX_GpuLaunch.H>

#include <utility>

#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/spacetime_excision.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"
#include "relativity/black_hole_excision_kernels.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;
namespace {

struct BlackHoleExcisionStoredMetric {
   amrex::Array4<amrex::Real const> spacetime;
   ExcisionGridView excision{};

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric cell(int i, int j, int k) const noexcept
   {
      return generated::z4c_spacetime_metric(
         load_z4c_state(spacetime, i, j, k)
      );
   }

   template <int Direction>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric face(int i, int j, int k) const noexcept
   {
      return stored_spacetime_metric_at_face<4, Direction>(
         spacetime, i, j, k, excision
      );
   }
};

struct BlackHoleExcisionFixedMetric {
   FixedSpacetimeParameters spacetime;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lower;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> spacing;
   amrex::Real time = 0.0_rt;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric cell(int i, int j, int k) const noexcept
   {
      return fixed_spacetime_geometry_at_cell<AMREX_SPACEDIM>(
         spacetime, lower, spacing, i, j, k, time
      ).metric;
   }

   template <int Direction>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric face(int i, int j, int k) const noexcept
   {
      return fixed_spacetime_geometry_at_face<Direction, AMREX_SPACEDIM>(
         spacetime, lower, spacing, i, j, k, time
      ).metric;
   }
};

// Named host factories keep CUDA launch template arguments addressable.
struct BlackHoleExcisionStoredMetricFactory {
   amrex::MultiFab const& spacetime;
   ExcisionGridView excision;

   BlackHoleExcisionStoredMetric operator()(amrex::MFIter const& mfi) const
   {
      return {spacetime.const_array(mfi), excision};
   }
};

struct BlackHoleExcisionFixedMetricFactory {
   BlackHoleExcisionFixedMetric metric;

   BlackHoleExcisionFixedMetric operator()(amrex::MFIter const&) const
   {
      return metric;
   }
};

template <int Direction, typename MetricFactory>
void enforce_black_hole_excision_flux_direction(
   amrex::MultiFab& flux,
   amrex::MultiFab const& primitive,
   amrex::iMultiFab const& mask,
   PhysicalSystem physical_system,
   EOSParams const& eos,
   MetricFactory&& metric_factory
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto cached = primitive.const_array(mfi);
      const auto excised = mask.const_array(mfi);
      const auto values = flux.array(mfi);
      const auto metric = metric_factory(mfi);
      amrex::ParallelFor(
         box,
         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            bool replaced = false;
            if (physical_system == PhysicalSystem::RHD) {
               const RHDConserved replacement =
                  black_hole_excision_rhd_flux<Direction>(
                     cached, excised, i, j, k, eos, metric, replaced
                  );
               if (replaced) {
                  store_rhd_conserved(values, i, j, k, replacement);
               }
            } else {
               const RMHDConserved replacement =
                  black_hole_excision_rmhd_flux<Direction>(
                     cached, excised, i, j, k, eos, metric, replaced
                  );
               if (replaced) {
                  store_rmhd_conserved(values, i, j, k, replacement);
               }
            }
         }
      );
   }
}

} // namespace

void enforce_black_hole_excision_fluxes(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   amrex::iMultiFab const& mask,
   PhysicalSystem physical_system,
   EOSParams const& eos,
   amrex::Geometry const& geometry,
   FixedSpacetimeParameters const& fixed_spacetime,
   amrex::MultiFab const* cell_spacetime,
   amrex::Real time,
   SphericalExcisionRegion const& excision_region
)
{
   AMREX_ALWAYS_ASSERT(
      physical_system == PhysicalSystem::RHD
      || physical_system == PhysicalSystem::RMHD
   );
   auto apply = [&](auto&& metric_factory) {
      enforce_black_hole_excision_flux_direction<0>(
         *fluxes[0], primitive, mask, physical_system, eos, metric_factory
      );
#if (AMREX_SPACEDIM >= 2)
      enforce_black_hole_excision_flux_direction<1>(
         *fluxes[1], primitive, mask, physical_system, eos, metric_factory
      );
#endif
#if (AMREX_SPACEDIM == 3)
      enforce_black_hole_excision_flux_direction<2>(
         *fluxes[2], primitive, mask, physical_system, eos, metric_factory
      );
#endif
   };
   if (cell_spacetime != nullptr) {
      apply(BlackHoleExcisionStoredMetricFactory{
         *cell_spacetime,
         excision_grid_view(excision_region, geometry, primitive.ixType())
      });
   } else {
      apply(BlackHoleExcisionFixedMetricFactory{{
         fixed_spacetime, geometry.ProbLoArray(), geometry.CellSizeArray(), time
      }});
   }
}

void zero_black_hole_excision_rhs(
   amrex::MultiFab& rhs,
   amrex::iMultiFab const& mask
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto values = rhs.array(mfi);
      const auto excised = mask.const_array(mfi);
      const int num_components = rhs.nComp();
      amrex::ParallelFor(
         box,
         num_components,
         [=] AMREX_GPU_DEVICE(
            int i, int j, int k, int component
         ) noexcept {
            if (excised(i, j, k, 0) != 0) {
               values(i, j, k, component) = 0.0_rt;
            }
         }
      );
   }
}

} // namespace pyfluid_amrex
