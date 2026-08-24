#pragma once

#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/boundary.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool metric_boundary_reflects(int kind) noexcept
{
   return kind == static_cast<int>(BoundaryKind::Reflecting)
      || kind == static_cast<int>(BoundaryKind::Symmetry);
}

template <typename Primitive>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void reflect_relativistic_velocity(
   Primitive& primitive,
   SpacetimeMetric const& source_metric,
   SpacetimeMetric const& ghost_metric,
   bool reflect1,
   bool reflect2,
   bool reflect3
) noexcept
{
   SpatialFourVelocity z = spatial_four_velocity_from_eulerian(
      primitive.v1, primitive.v2, primitive.v3, source_metric
   );
   if (reflect1) z.z1 = -z.z1;
   if (reflect2) z.z2 = -z.z2;
   if (reflect3) z.z3 = -z.z3;
   const EulerianVelocity velocity =
      eulerian_velocity_from_spatial_four_velocity(z, ghost_metric);
   primitive.v1 = velocity.v1;
   primitive.v2 = velocity.v2;
   primitive.v3 = velocity.v3;
}

struct FixedSpacetimeCellMetricAccessor {
   FixedSpacetimeParameters spacetime;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lower;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> spacing;
   amrex::Real time = 0.0;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric operator()(int i, int j, int k) const noexcept
   {
      return fixed_spacetime_geometry_at_cell<AMREX_SPACEDIM>(
         spacetime, lower, spacing, i, j, k, time
      ).metric;
   }
};

struct StoredSpacetimeCellMetricAccessor {
   amrex::Array4<amrex::Real const> spacetime;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   SpacetimeMetric operator()(int i, int j, int k) const noexcept
   {
      return generated::z4c_spacetime_metric(
         load_z4c_state(spacetime, i, j, k)
      );
   }
};

template <PhysicalSystem System, int EOSMethod, typename MetricAccessorFactory>
inline void apply_metric_state_boundary(
   amrex::MultiFab& state,
   const amrex::Geometry& geometry,
   const BoundaryData& boundary,
   const EOSParams& eos,
   MetricAccessorFactory&& metric_accessor_factory
)
{
   static_assert(System == PhysicalSystem::RHD || System == PhysicalSystem::RMHD);
   if (boundary_data_all_periodic(boundary)) {
      return;
   }

   const amrex::Box domain = geometry.Domain();
   const auto lo = amrex::lbound(domain);
   const auto hi = amrex::ubound(domain);
   const auto periodic = geometry.isPeriodicArray();
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   const int ng = state.nGrow();
   const int diode = static_cast<int>(BoundaryKind::Diode);
   const int symmetry = static_cast<int>(BoundaryKind::Symmetry);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      amrex::Box grown = mfi.growntilebox(ng);
      grown &= state[mfi].box();
      const amrex::BoxList ghost_boxes = amrex::boxDiff(grown, domain);
      const auto values = state.array(mfi);
      const auto source_values = state.const_array(mfi);
      const auto metric_at = metric_accessor_factory(mfi);
      for (const amrex::Box& box : ghost_boxes) {
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool xlo = !periodic[0] && i < lo.x;
            const bool xhi = !periodic[0] && i > hi.x;
#if (AMREX_SPACEDIM >= 2)
            const bool ylo = !periodic[1] && j < lo.y;
            const bool yhi = !periodic[1] && j > hi.y;
#else
            const bool ylo = false;
            const bool yhi = false;
#endif
#if (AMREX_SPACEDIM == 3)
            const bool zlo = !periodic[2] && k < lo.z;
            const bool zhi = !periodic[2] && k > hi.z;
#else
            const bool zlo = false;
            const bool zhi = false;
#endif
            if (!(xlo || xhi || ylo || yhi || zlo || zhi)) {
               return;
            }

            const int x_kind = xlo ? boundary.lo_kind[0]
               : (xhi ? boundary.hi_kind[0] : -1);
#if (AMREX_SPACEDIM >= 2)
            const int y_kind = ylo ? boundary.lo_kind[1]
               : (yhi ? boundary.hi_kind[1] : -1);
#else
            const int y_kind = -1;
#endif
#if (AMREX_SPACEDIM == 3)
            const int z_kind = zlo ? boundary.lo_kind[2]
               : (zhi ? boundary.hi_kind[2] : -1);
#else
            const int z_kind = -1;
#endif
            const bool x_reflect = metric_boundary_reflects(x_kind);
            const bool y_reflect = metric_boundary_reflects(y_kind);
            const bool z_reflect = metric_boundary_reflects(z_kind);
            const int source_i = periodic[0] ? i
               : (x_reflect ? mirrored_index(i, lo.x, hi.x)
                            : clamp_index(i, lo.x, hi.x));
#if (AMREX_SPACEDIM >= 2)
            const int source_j = periodic[1] ? j
               : (y_reflect ? mirrored_index(j, lo.y, hi.y)
                            : clamp_index(j, lo.y, hi.y));
#else
            const int source_j = 0;
#endif
#if (AMREX_SPACEDIM == 3)
            const int source_k = periodic[2] ? k
               : (z_reflect ? mirrored_index(k, lo.z, hi.z)
                            : clamp_index(k, lo.z, hi.z));
#else
            const int source_k = 0;
#endif
            const SpacetimeMetric source_metric = metric_at(
               source_i, source_j, source_k
            );
            const SpacetimeMetric ghost_metric = metric_at(i, j, k);
            const int inflow_direction = cell_inflow_boundary_dir(
               boundary, domain, i, j, k
            );
            const amrex::Real position1 = lower[0]
               + (static_cast<amrex::Real>(i) + 0.5) * spacing[0];
            amrex::Real position2 = 0.0;
            amrex::Real position3 = 0.0;
#if (AMREX_SPACEDIM >= 2)
            position2 = lower[1]
               + (static_cast<amrex::Real>(j) + 0.5) * spacing[1];
#endif
#if (AMREX_SPACEDIM == 3)
            position3 = lower[2]
               + (static_cast<amrex::Real>(k) + 0.5) * spacing[2];
#endif
            const amrex::Real inflow_weight = inflow_direction >= 0
               ? inflow_profile_weight(
                  boundary, inflow_direction, position1, position2, position3
               )
               : 0.0;
            const amrex::Real inflow_rho = boundary.inflow_background_rho
               + inflow_weight
                  * (boundary.inflow_rho - boundary.inflow_background_rho);
            const amrex::Real inflow_pressure = boundary.inflow_background_pressure
               + inflow_weight
                  * (boundary.inflow_pressure - boundary.inflow_background_pressure);
            const amrex::Real inflow_v1 = boundary.inflow_background_v1
               + inflow_weight
                  * (boundary.inflow_v1 - boundary.inflow_background_v1);
            const amrex::Real inflow_v2 = boundary.inflow_background_v2
               + inflow_weight
                  * (boundary.inflow_v2 - boundary.inflow_background_v2);
            const amrex::Real inflow_v3 = boundary.inflow_background_v3
               + inflow_weight
                  * (boundary.inflow_v3 - boundary.inflow_background_v3);

            if constexpr (System == PhysicalSystem::RHD) {
               RHDPrimitive primitive{};
               if (inflow_direction >= 0) {
                  primitive = {
                     inflow_rho,
                     relativistic_specific_internal_energy_from_pressure<EOSMethod>(
                        inflow_rho, inflow_pressure, eos
                     ),
                     inflow_pressure,
                     inflow_v1,
                     inflow_v2,
                     inflow_v3,
                  };
               } else {
                  primitive = recover_rhd<EOSMethod>(
                     load_rhd_conserved(source_values, source_i, source_j, source_k),
                     source_metric,
                     eos
                  ).primitive;
                  reflect_relativistic_velocity(
                     primitive,
                     source_metric,
                     ghost_metric,
                     x_reflect,
                     y_reflect,
                     z_reflect
                  );
               }
               if ((xlo && x_kind == diode && primitive.v1 > 0.0)
                   || (xhi && x_kind == diode && primitive.v1 < 0.0)) {
                  primitive.v1 = 0.0;
               }
               if ((ylo && y_kind == diode && primitive.v2 > 0.0)
                   || (yhi && y_kind == diode && primitive.v2 < 0.0)) {
                  primitive.v2 = 0.0;
               }
               if ((zlo && z_kind == diode && primitive.v3 > 0.0)
                   || (zhi && z_kind == diode && primitive.v3 < 0.0)) {
                  primitive.v3 = 0.0;
               }
               store_rhd_conserved(
                  values,
                  i,
                  j,
                  k,
                  generated::rhd_primitive_to_conserved(
                     primitive, ghost_metric
                  )
               );
            } else {
               RMHDPrimitive primitive{};
               if (inflow_direction >= 0) {
                  primitive = {
                     inflow_rho,
                     relativistic_specific_internal_energy_from_pressure<
                        EOSMethod
                     >(inflow_rho, inflow_pressure, eos),
                     inflow_pressure,
                     inflow_v1,
                     inflow_v2,
                     inflow_v3,
                     boundary.inflow_background_b1
                        + inflow_weight * (
                           boundary.inflow_b1 - boundary.inflow_background_b1
                        ),
                     boundary.inflow_background_b2
                        + inflow_weight * (
                           boundary.inflow_b2 - boundary.inflow_background_b2
                        ),
                     boundary.inflow_background_b3
                        + inflow_weight * (
                           boundary.inflow_b3 - boundary.inflow_background_b3
                        ),
                  };
               } else {
                  primitive = recover_srmhd(
                     load_rmhd_conserved(source_values, source_i, source_j, source_k),
                     source_metric,
                     eos
                  ).primitive;
                  reflect_relativistic_velocity(
                     primitive,
                     source_metric,
                     ghost_metric,
                     x_reflect,
                     y_reflect,
                     z_reflect
                  );
               }
               if (x_kind == symmetry) {
                  primitive.b2 = -primitive.b2;
                  primitive.b3 = -primitive.b3;
               }
               if (y_kind == symmetry) {
                  primitive.b1 = -primitive.b1;
                  primitive.b3 = -primitive.b3;
               }
               if (z_kind == symmetry) {
                  primitive.b1 = -primitive.b1;
                  primitive.b2 = -primitive.b2;
               }
               if ((xlo && x_kind == diode && primitive.v1 > 0.0)
                   || (xhi && x_kind == diode && primitive.v1 < 0.0)) {
                  primitive.v1 = 0.0;
               }
               if ((ylo && y_kind == diode && primitive.v2 > 0.0)
                   || (yhi && y_kind == diode && primitive.v2 < 0.0)) {
                  primitive.v2 = 0.0;
               }
               if ((zlo && z_kind == diode && primitive.v3 > 0.0)
                   || (zhi && z_kind == diode && primitive.v3 < 0.0)) {
                  primitive.v3 = 0.0;
               }
               store_rmhd_conserved(
                  values,
                  i,
                  j,
                  k,
                  generated::rmhd_primitive_to_conserved(
                     primitive, ghost_metric
                  )
               );
            }
         });
      }
   }
}

template <PhysicalSystem System, int EOSMethod>
inline void apply_fixed_metric_state_boundary(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   BoundaryData const& boundary,
   FixedSpacetimeParameters const& spacetime,
   EOSParams const& eos,
   amrex::Real time
)
{
   if (spacetime.metric == FixedSpacetimeMetric::Minkowski) {
      return;
   }
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   apply_metric_state_boundary<System, EOSMethod>(
      state,
      geometry,
      boundary,
      eos,
      [=](amrex::MFIter const&) {
         return FixedSpacetimeCellMetricAccessor{
            spacetime, lower, spacing, time
         };
      }
   );
}

template <PhysicalSystem System, int EOSMethod>
inline void apply_stored_metric_state_boundary(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   BoundaryData const& boundary,
   EOSParams const& eos,
   amrex::MultiFab const& cell_spacetime
)
{
   apply_metric_state_boundary<System, EOSMethod>(
      state,
      geometry,
      boundary,
      eos,
      [&](amrex::MFIter const& mfi) {
         return StoredSpacetimeCellMetricAccessor{
            cell_spacetime.const_array(mfi)
         };
      }
   );
}

} // namespace pyfluid_amrex
