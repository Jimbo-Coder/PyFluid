#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/fccz4_state_access.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/sources.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "pyfluid_amrex/reference_metric_spacetime.hpp"
#include "pyfluid_amrex/z4c_derivatives.hpp"
#include "pyfluid_amrex/spacetime_excision.hpp"
#include "pyfluid_amrex/z4c_excision_derivatives.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

template <PhysicalSystem System, int EOSMethod>
inline void add_fixed_spacetime_sources(
   amrex::MultiFab& rhs,
   const amrex::MultiFab& primitive,
   const amrex::Geometry& geometry,
   const FixedSpacetimeParameters& spacetime,
   const EOSParams& eos,
   amrex::Real time,
   int coordinate,
   MeshAxes const& axes
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
      amrex::ParallelFor(box,
         [spacetime, coordinate, axes, lower, spacing, time, cached, eos, source_rhs]
         AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if constexpr (System == PhysicalSystem::RHD) {
            const ReferenceMetricSpacetimeGeometry local_geometry =
               reference_metric_fixed_spacetime_geometry_at_cell(
                  spacetime,
                  coordinate,
                  axes,
                  lower,
                  spacing,
                  i,
                  j,
                  k,
                  time
               );
            const RHDConserved source = generated::rhd_geometric_source(
               rhd_primitive_at<EOSMethod>(cached, i, j, k, eos),
               local_geometry.metric,
               local_geometry.reference_derivatives
            );
            source_rhs(i, j, k, RHO) += source.dens;
            source_rhs(i, j, k, MOM1) += source.mom1;
            source_rhs(i, j, k, MOM2) += source.mom2;
            source_rhs(i, j, k, MOM3) += source.mom3;
            source_rhs(i, j, k, ENERGY) += source.tau;
         } else {
            const FixedSpacetimeGeometry local_geometry =
               fixed_spacetime_geometry_at_cell<AMREX_SPACEDIM>(
                  spacetime, lower, spacing, i, j, k, time
               );
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

template <PhysicalSystem System, int EOSMethod, int Order = 4>
inline void add_stored_spacetime_sources(
   amrex::MultiFab& rhs,
   amrex::MultiFab const& primitive,
   amrex::MultiFab const& cell_spacetime,
   amrex::Geometry const& geometry,
   EOSParams const& eos,
   SphericalExcisionRegion const& excision_region = {},
   int coordinate = MeshCartesian,
   MeshAxes const& axes = mesh_default_axes_layout(MeshCartesian)
)
{
   static_assert(System == PhysicalSystem::RHD || System == PhysicalSystem::RMHD);
   const auto excision = excision_grid_view(excision_region, geometry, cell_spacetime.ixType());
   const ReferenceChartGeometry reference_chart(geometry, coordinate, axes);
   const auto spacing = geometry.CellSizeArray();
   const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
      1.0_rt / spacing[0],
#if (AMREX_SPACEDIM >= 2)
      1.0_rt / spacing[1],
#else
      1.0_rt,
#endif
#if (AMREX_SPACEDIM == 3)
      1.0_rt / spacing[2],
#else
      1.0_rt,
#endif
   }};
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto cached = primitive.const_array(mfi);
      const auto spacetime = cell_spacetime.const_array(mfi);
      const auto source_rhs = rhs.array(mfi);
      amrex::ParallelFor(box,
         [excision, spacetime, inv_spacing, coordinate, reference_chart,
          cached, eos, source_rhs]
         AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (excision.region.enabled() && !excision(i, j, k, 0)) return;
         const Z4cState z4c_state = load_z4c_state(spacetime, i, j, k);
         const Z4cSpatialDerivatives z4c_derivatives =
            excision.region.enabled()
            ? gather_z4c_spatial_derivatives<Order, AMREX_SPACEDIM>(spacetime, i, j, k,
               inv_spacing, z4c_state, ExcisionZ4cDerivativeOperator<Order>{
                  excision.derivative_stencil(i, j, k, Order / 2 + 1)})
            : gather_z4c_spatial_derivatives<Order, AMREX_SPACEDIM>(spacetime, i, j, k,
               inv_spacing, z4c_state);
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(z4c_state);
         const auto geometric_derivatives = coordinate == MeshCartesian
            ? z4c_derivatives
            : reference_spacetime_derivatives(spacetime, i, j, k, z4c_derivatives,
                 reference_chart.at(i, j, k, false));
         const SpacetimeMetricDerivatives metric_derivatives =
            generated::z4c_spacetime_metric_derivatives(z4c_state, geometric_derivatives);
         if constexpr (System == PhysicalSystem::RHD) {
            const RHDConserved source = generated::rhd_geometric_source(
               rhd_primitive_at<EOSMethod>(cached, i, j, k, eos),
               metric,
               metric_derivatives
            );
            source_rhs(i, j, k, RHO) += source.dens;
            source_rhs(i, j, k, MOM1) += source.mom1;
            source_rhs(i, j, k, MOM2) += source.mom2;
            source_rhs(i, j, k, MOM3) += source.mom3;
            source_rhs(i, j, k, ENERGY) += source.tau;
         } else {
            const RMHDConserved source = generated::rmhd_geometric_source(
               rmhd_primitive_at(cached, i, j, k, eos),
               metric,
               metric_derivatives
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
