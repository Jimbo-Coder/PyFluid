#pragma once

#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/mask_regions.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/z4c_pointwise.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_rhd_primitive_cache_cell(
   amrex::Array4<amrex::Real const> const& state,
   amrex::Array4<amrex::Real> const& primitive,
   int i,
   int j,
   int k,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const RHDRecoveryResult recovered = recover_rhd<EOSMethod>(
      load_rhd_conserved(state, i, j, k), metric, eos
   );
   const amrex::Real invalid = std::numeric_limits<amrex::Real>::quiet_NaN();
   const RHDPrimitive& value = recovered.primitive;
   const bool success = recovered.attempt.status == C2PStatus::Success;
   primitive(i, j, k, RHO) = success ? value.rho : invalid;
   primitive(i, j, k, MOM1) = success ? value.v1 : invalid;
   primitive(i, j, k, MOM2) = success ? value.v2 : invalid;
   primitive(i, j, k, MOM3) = success ? value.v3 : invalid;
   primitive(i, j, k, ENERGY) = success ? value.pressure : invalid;
}

template <int EOSMethod>
inline void fill_rhd_primitive_cache(
   amrex::MultiFab& primitive,
   const amrex::MultiFab& state,
   const EOSParams& eos,
   const amrex::Geometry& geometry,
   const FixedSpacetimeParameters& spacetime,
   amrex::Real time
)
{
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.growntilebox();
      const auto conserved = state.const_array(mfi);
      const auto cached = primitive.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const FixedSpacetimeGeometry local_geometry =
            fixed_spacetime_geometry_at_cell<AMREX_SPACEDIM>(
               spacetime, lower, spacing, i, j, k, time
            );
         store_rhd_primitive_cache_cell<EOSMethod>(
            conserved, cached, i, j, k, eos, local_geometry.metric
         );
      });
   }
}

template <int EOSMethod>
inline void fill_rhd_primitive_cache(
   amrex::MultiFab& primitive,
   const amrex::MultiFab& state,
   const EOSParams& eos,
   const amrex::MultiFab& cell_spacetime
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.growntilebox();
      const auto conserved = state.const_array(mfi);
      const auto cached = primitive.array(mfi);
      const auto spacetime = cell_spacetime.const_array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         store_rhd_primitive_cache_cell<EOSMethod>(
            conserved,
            cached,
            i,
            j,
            k,
            eos,
            generated::z4c_spacetime_metric(load_z4c_state(spacetime, i, j, k))
         );
      });
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_rmhd_primitive_cache_cell(
   amrex::Array4<amrex::Real const> const& state,
   amrex::Array4<amrex::Real> const& primitive,
   int i,
   int j,
   int k,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const RMHDRecoveryResult recovered = recover_srmhd(
      load_rmhd_conserved(state, i, j, k), metric, eos.gamma
   );
   const amrex::Real invalid = std::numeric_limits<amrex::Real>::quiet_NaN();
   const RMHDPrimitive& value = recovered.primitive;
   const bool success = recovered.succeeded();
   primitive(i, j, k, RHO) = success ? value.rho : invalid;
   primitive(i, j, k, MOM1) = success ? value.v1 : invalid;
   primitive(i, j, k, MOM2) = success ? value.v2 : invalid;
   primitive(i, j, k, MOM3) = success ? value.v3 : invalid;
   primitive(i, j, k, ENERGY) = success ? value.pressure : invalid;
   primitive(i, j, k, B1) = success ? value.b1 : invalid;
   primitive(i, j, k, B2) = success ? value.b2 : invalid;
   primitive(i, j, k, B3) = success ? value.b3 : invalid;
}

inline void fill_rmhd_primitive_cache(
   amrex::MultiFab& primitive,
   const amrex::MultiFab& state,
   const EOSParams& eos,
   const amrex::Geometry& geometry,
   const FixedSpacetimeParameters& spacetime,
   amrex::Real time
)
{
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.growntilebox();
      const auto conserved = state.const_array(mfi);
      const auto cached = primitive.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const FixedSpacetimeGeometry local_geometry =
            fixed_spacetime_geometry_at_cell<AMREX_SPACEDIM>(
               spacetime, lower, spacing, i, j, k, time
            );
         store_rmhd_primitive_cache_cell(
            conserved, cached, i, j, k, eos, local_geometry.metric
         );
      });
   }
}

inline void fill_rmhd_primitive_cache(
   amrex::MultiFab& primitive,
   const amrex::MultiFab& state,
   const EOSParams& eos,
   const amrex::MultiFab& cell_spacetime
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.growntilebox();
      const auto conserved = state.const_array(mfi);
      const auto cached = primitive.array(mfi);
      const auto spacetime = cell_spacetime.const_array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         store_rmhd_primitive_cache_cell(
            conserved,
            cached,
            i,
            j,
            k,
            eos,
            generated::z4c_spacetime_metric(load_z4c_state(spacetime, i, j, k))
         );
      });
   }
}

template <int NComp, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_primitive_cache_cell(
   amrex::Array4<amrex::Real const> const& state,
   amrex::Array4<amrex::Real> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& eos
) noexcept
{
   if constexpr (has_mhd_components<NComp>()) {
      const MHDPrimitive q = primitive_from_conserved<EOSMethod>(load_mhd_state(state, i, j, k), eos);
      primitive(i, j, k, RHO) = q.rho;
      primitive(i, j, k, MOM1) = q.v1;
      primitive(i, j, k, MOM2) = q.v2;
      primitive(i, j, k, MOM3) = q.v3;
      primitive(i, j, k, ENERGY) = q.pressure;
      primitive(i, j, k, B1) = q.b1;
      primitive(i, j, k, B2) = q.b2;
      primitive(i, j, k, B3) = q.b3;
   } else {
      const HydroPrimitive q = primitive_from_conserved<EOSMethod>(load_hydro_state(state, i, j, k), eos);
      primitive(i, j, k, RHO) = q.rho;
      primitive(i, j, k, MOM1) = q.v1;
      primitive(i, j, k, MOM2) = q.v2;
      primitive(i, j, k, MOM3) = q.v3;
      primitive(i, j, k, ENERGY) = q.pressure;
   }
   for (int q = base_component_count<NComp>(); q < NComp; ++q) {
      primitive(i, j, k, q) =
         state(i, j, k, q) / amrex::max(state(i, j, k, RHO), std::numeric_limits<amrex::Real>::min());
   }
}

template <int NComp, int EOSMethod>
inline void fill_primitive_cache(amrex::MultiFab& primitive, amrex::MultiFab const& state, EOSParams const& eos)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(primitive); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.fabbox();
      const auto s = state.const_array(mfi);
      const auto p = primitive.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         store_primitive_cache_cell<NComp, EOSMethod>(s, p, i, j, k, eos);
      });
   }
   }
}

inline void fill_base_primitive_cache_runtime(
   amrex::MultiFab& primitive,
   const amrex::MultiFab& state,
   const EOSParams& eos,
   int eos_method,
   PhysicalSystem system,
   const amrex::Geometry& geometry,
   const FixedSpacetimeParameters& spacetime,
   amrex::Real time
)
{
   if (system == PhysicalSystem::RHD) {
      switch (eos_method) {
      case EOSPolytrope:
         fill_rhd_primitive_cache<EOSPolytrope>(primitive, state, eos, geometry, spacetime, time);
         return;
      case EOSHybridPolytrope:
         fill_rhd_primitive_cache<EOSHybridPolytrope>(primitive, state, eos, geometry, spacetime, time);
         return;
      default:
         fill_rhd_primitive_cache<EOSIdealGas>(primitive, state, eos, geometry, spacetime, time);
         return;
      }
   }
   if (system == PhysicalSystem::RMHD) {
      fill_rmhd_primitive_cache(primitive, state, eos, geometry, spacetime, time);
      return;
   }
   const bool mhd = system == PhysicalSystem::MHD;
   switch (eos_method) {
   case EOSPolytrope:
      if (mhd) {
         fill_primitive_cache<NVAR_MHD, EOSPolytrope>(primitive, state, eos);
      } else {
         fill_primitive_cache<NVAR_HYDRO, EOSPolytrope>(primitive, state, eos);
      }
      return;
   case EOSHybridPolytrope:
      if (mhd) {
         fill_primitive_cache<NVAR_MHD, EOSHybridPolytrope>(primitive, state, eos);
      } else {
         fill_primitive_cache<NVAR_HYDRO, EOSHybridPolytrope>(primitive, state, eos);
      }
      return;
   case EOSIdealGas:
      if (mhd) {
         fill_primitive_cache<NVAR_MHD, EOSIdealGas>(primitive, state, eos);
      } else {
         fill_primitive_cache<NVAR_HYDRO, EOSIdealGas>(primitive, state, eos);
      }
      return;
   default:
      amrex::Abort("invalid EOS method in primitive-cache dispatch");
   }
}

template <int NComp>
inline void transform_primitive_vectors(
   amrex::MultiFab& primitive,
   amrex::Geometry const& geometry,
   int coordinate,
   MeshAxes const& axes,
   bool to_cartesian
)
{
   if (coordinate == MeshCartesian) {
      return;
   }
   const auto problo = geometry.ProbLoArray();
   const auto dx = geometry.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(primitive); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.fabbox();
      const auto values = primitive.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const auto q = mesh_coordinate_triplet_at_location(
            coordinate, axes, problo, dx, i, j, k, false, false, false
         );
         StateVector3 velocity{values(i, j, k, MOM1), values(i, j, k, MOM2), values(i, j, k, MOM3)};
         velocity = to_cartesian
            ? mesh_basis_vector_to_cartesian(coordinate, axes, q, velocity)
            : cartesian_vector_to_mesh_basis(coordinate, axes, q, velocity);
         values(i, j, k, MOM1) = velocity.x;
         values(i, j, k, MOM2) = velocity.y;
         values(i, j, k, MOM3) = velocity.z;
         if constexpr (has_mhd_components<NComp>()) {
            StateVector3 magnetic{values(i, j, k, B1), values(i, j, k, B2), values(i, j, k, B3)};
            magnetic = to_cartesian
               ? mesh_basis_vector_to_cartesian(coordinate, axes, q, magnetic)
               : cartesian_vector_to_mesh_basis(coordinate, axes, q, magnetic);
            values(i, j, k, B1) = magnetic.x;
            values(i, j, k, B2) = magnetic.y;
            values(i, j, k, B3) = magnetic.z;
         }
      });
   }
   }
}

template <int NComp, int EOSMethod>
inline void fill_primitive_cache_masked(
   amrex::MultiFab& primitive,
   amrex::MultiFab const& state,
   amrex::iMultiFab const& primitive_needed,
   EOSParams const& eos,
   amrex::Vector<int> const* primitive_needed_kind = nullptr
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(primitive); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.fabbox();
      const auto s = state.const_array(mfi);
      const auto needed = primitive_needed.const_array(mfi);
      const auto p = primitive.array(mfi);
      const int kind = cached_mask_kind(primitive_needed_kind, mfi, primitive_needed, bx);
      if (kind == MaskKindZero) {
         continue;
      }
      if (kind == MaskKindNonzero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            store_primitive_cache_cell<NComp, EOSMethod>(s, p, i, j, k, eos);
         });
         continue;
      }
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (needed(i, j, k, 0) == 0) {
            return;
         }
         store_primitive_cache_cell<NComp, EOSMethod>(s, p, i, j, k, eos);
      });
   }
   }
}

} // namespace pyfluid_amrex
