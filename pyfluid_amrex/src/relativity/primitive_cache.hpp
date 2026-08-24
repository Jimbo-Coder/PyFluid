#pragma once

#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/ct_fields.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/mask_regions.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "relativity/relativistic_atmosphere.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"
#include "pyfluid_amrex/state.hpp"

#include <type_traits>

namespace pyfluid_amrex {

template <typename MagneticState>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void replace_cell_magnetic_from_faces(
   MagneticState& value,
   amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& face,
   amrex::GpuArray<int, AMREX_SPACEDIM> const& active,
   int i,
   int j,
   int k
) noexcept
{
   if (active[0] != 0) {
      value.b1 = 0.5 * (face[0](i, j, k, 0) + face[0](i + 1, j, k, 0));
   }
#if (AMREX_SPACEDIM >= 2)
   if (active[1] != 0) {
      value.b2 = 0.5 * (face[1](i, j, k, 0) + face[1](i, j + 1, k, 0));
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (active[2] != 0) {
      value.b3 = 0.5 * (face[2](i, j, k, 0) + face[2](i, j, k + 1, 0));
   }
#endif
}

inline amrex::GpuArray<int, AMREX_SPACEDIM> active_face_magnetic_axes(
   amrex::Geometry const& geometry
)
{
   amrex::GpuArray<int, AMREX_SPACEDIM> active{};
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      active[direction] = geometry.Domain().length(direction) > 1 ? 1 : 0;
   }
   return active;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_rhd_primitive_cache_cell(
   amrex::Array4<amrex::Real> const& state,
   amrex::Array4<amrex::Real> const& primitive,
   int i,
   int j,
   int k,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const RHDRecoveryResult recovered =
      recover_relativistic_primitive_with_atmosphere<PhysicalSystem::RHD, EOSMethod>(
         state, i, j, k, eos, metric
      );
   const amrex::Real invalid = std::numeric_limits<amrex::Real>::quiet_NaN();
   const RHDPrimitive& value = recovered.primitive;
   const bool success = recovered.attempt.succeeded();
   primitive(i, j, k, RHO) = success ? value.rho : invalid;
   primitive(i, j, k, MOM1) = success ? value.v1 : invalid;
   primitive(i, j, k, MOM2) = success ? value.v2 : invalid;
   primitive(i, j, k, MOM3) = success ? value.v3 : invalid;
   primitive(i, j, k, ENERGY) = success ? value.pressure : invalid;
}

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
   const bool success = recovered.attempt.succeeded();
   primitive(i, j, k, RHO) = success ? value.rho : invalid;
   primitive(i, j, k, MOM1) = success ? value.v1 : invalid;
   primitive(i, j, k, MOM2) = success ? value.v2 : invalid;
   primitive(i, j, k, MOM3) = success ? value.v3 : invalid;
   primitive(i, j, k, ENERGY) = success ? value.pressure : invalid;
}

template <int EOSMethod, typename StateMultiFab>
inline void fill_rhd_primitive_cache(
   amrex::MultiFab& primitive,
   StateMultiFab& state,
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
      const auto conserved = [&]() {
         if constexpr (std::is_const_v<StateMultiFab>) {
            return state.const_array(mfi);
         } else {
            return state.array(mfi);
         }
      }();
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

template <int EOSMethod, typename StateMultiFab>
inline void fill_rhd_primitive_cache(
   amrex::MultiFab& primitive,
   StateMultiFab& state,
   const EOSParams& eos,
   const amrex::MultiFab& cell_spacetime
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.growntilebox();
      const auto conserved = [&]() {
         if constexpr (std::is_const_v<StateMultiFab>) {
            return state.const_array(mfi);
         } else {
            return state.array(mfi);
         }
      }();
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
   amrex::Array4<amrex::Real> const& state,
   amrex::Array4<amrex::Real> const& primitive,
   int i,
   int j,
   int k,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const RMHDRecoveryResult recovered =
      recover_relativistic_primitive_with_atmosphere<PhysicalSystem::RMHD, EOSRuntime>(
         state, i, j, k, eos, metric
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_rmhd_primitive_cache_cell(
   RMHDConserved const& conserved,
   amrex::Array4<amrex::Real> const& primitive,
   int i,
   int j,
   int k,
   const EOSParams& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const RMHDRecoveryResult recovered = recover_srmhd(conserved, metric, eos);
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
      load_rmhd_conserved(state, i, j, k), metric, eos
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

template <typename StateMultiFab>
inline void fill_rmhd_primitive_cache(
   amrex::MultiFab& primitive,
   StateMultiFab& state,
   const EOSParams& eos,
   const amrex::Geometry& geometry,
   const FixedSpacetimeParameters& spacetime,
   amrex::Real time,
   FaceMagneticField const* canonical_face_b = nullptr,
   amrex::iMultiFab const* canonical_face_b_cell_mask = nullptr
)
{
   AMREX_ALWAYS_ASSERT(
      (canonical_face_b == nullptr) == (canonical_face_b_cell_mask == nullptr)
   );
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.growntilebox();
      const auto conserved = [&]() {
         if constexpr (std::is_const_v<StateMultiFab>) {
            return state.const_array(mfi);
         } else {
            return state.array(mfi);
         }
      }();
      const auto cached = primitive.array(mfi);
      amrex::Array4<int const> canonical_cell{};
      if (canonical_face_b_cell_mask != nullptr) {
         canonical_cell = canonical_face_b_cell_mask->const_array(mfi);
      }
      amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> face{};
      if (canonical_face_b != nullptr) {
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            face[direction] = canonical_face_b->b[direction]->const_array(mfi);
         }
      }
      const auto active = active_face_magnetic_axes(geometry);
      const bool use_canonical_face_b = canonical_face_b != nullptr;
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const FixedSpacetimeGeometry local_geometry =
            fixed_spacetime_geometry_at_cell<AMREX_SPACEDIM>(
               spacetime, lower, spacing, i, j, k, time
            );
         if (use_canonical_face_b && canonical_cell(i, j, k, 0) != 0) {
            RMHDConserved local_conserved = load_rmhd_conserved(conserved, i, j, k);
            replace_cell_magnetic_from_faces(
               local_conserved, face, active, i, j, k
            );
            store_rmhd_primitive_cache_cell(
               local_conserved, cached, i, j, k, eos, local_geometry.metric
            );
         } else {
            store_rmhd_primitive_cache_cell(
               conserved, cached, i, j, k, eos, local_geometry.metric
            );
         }
      });
   }
}

template <typename StateMultiFab>
inline void fill_rmhd_primitive_cache(
   amrex::MultiFab& primitive,
   StateMultiFab& state,
   const EOSParams& eos,
   const amrex::MultiFab& cell_spacetime,
   amrex::Geometry const* geometry = nullptr,
   FaceMagneticField const* canonical_face_b = nullptr,
   amrex::iMultiFab const* canonical_face_b_cell_mask = nullptr
)
{
   AMREX_ALWAYS_ASSERT(canonical_face_b == nullptr || geometry != nullptr);
   AMREX_ALWAYS_ASSERT(
      (canonical_face_b == nullptr) == (canonical_face_b_cell_mask == nullptr)
   );
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.growntilebox();
      const auto conserved = [&]() {
         if constexpr (std::is_const_v<StateMultiFab>) {
            return state.const_array(mfi);
         } else {
            return state.array(mfi);
         }
      }();
      const auto cached = primitive.array(mfi);
      const auto spacetime = cell_spacetime.const_array(mfi);
      amrex::Array4<int const> canonical_cell{};
      if (canonical_face_b_cell_mask != nullptr) {
         canonical_cell = canonical_face_b_cell_mask->const_array(mfi);
      }
      amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> face{};
      if (canonical_face_b != nullptr) {
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            face[direction] = canonical_face_b->b[direction]->const_array(mfi);
         }
      }
      const auto active = canonical_face_b != nullptr
         ? active_face_magnetic_axes(*geometry)
         : amrex::GpuArray<int, AMREX_SPACEDIM>{};
      const bool use_canonical_face_b = canonical_face_b != nullptr;
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(
            load_z4c_state(spacetime, i, j, k)
         );
         if (use_canonical_face_b && canonical_cell(i, j, k, 0) != 0) {
            RMHDConserved local_conserved = load_rmhd_conserved(conserved, i, j, k);
            replace_cell_magnetic_from_faces(
               local_conserved, face, active, i, j, k
            );
            store_rmhd_primitive_cache_cell(
               local_conserved, cached, i, j, k, eos, metric
            );
         } else {
            store_rmhd_primitive_cache_cell(
               conserved, cached, i, j, k, eos, metric
            );
         }
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

template <int NComp, int EOSMethod>
inline void fill_primitive_cache_from_ct_faces(
   amrex::MultiFab& primitive,
   amrex::MultiFab const& state,
   EOSParams const& eos,
   FaceMagneticField const& canonical_face_b,
   amrex::iMultiFab const& canonical_face_b_cell_mask,
   amrex::Geometry const& geometry
)
{
   static_assert(has_mhd_components<NComp>());
   const auto active = active_face_magnetic_axes(geometry);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(primitive); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.fabbox();
      const auto conserved = state.const_array(mfi);
      const auto cached = primitive.array(mfi);
      const auto canonical_cell = canonical_face_b_cell_mask.const_array(mfi);
      amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> face{};
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         face[direction] = canonical_face_b.b[direction]->const_array(mfi);
      }
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         MHDState local_conserved = load_mhd_state(conserved, i, j, k);
         if (canonical_cell(i, j, k, 0) != 0) {
            replace_cell_magnetic_from_faces(
               local_conserved, face, active, i, j, k
            );
         }
         const MHDPrimitive value =
            primitive_from_conserved<EOSMethod>(local_conserved, eos);
         cached(i, j, k, RHO) = value.rho;
         cached(i, j, k, MOM1) = value.v1;
         cached(i, j, k, MOM2) = value.v2;
         cached(i, j, k, MOM3) = value.v3;
         cached(i, j, k, ENERGY) = value.pressure;
         cached(i, j, k, B1) = value.b1;
         cached(i, j, k, B2) = value.b2;
         cached(i, j, k, B3) = value.b3;
         for (int component = base_component_count<NComp>();
              component < NComp;
              ++component) {
            cached(i, j, k, component) = conserved(i, j, k, component)
               / amrex::max(
                  conserved(i, j, k, RHO),
                  std::numeric_limits<amrex::Real>::min()
               );
         }
      });
   }
   }
}

template <int EOSMethod, typename StateMultiFab>
inline void fill_base_primitive_cache(
   amrex::MultiFab& primitive,
   StateMultiFab& state,
   const EOSParams& eos,
   PhysicalSystem system,
   const amrex::Geometry& geometry,
   const FixedSpacetimeParameters& spacetime,
   amrex::Real time,
   FaceMagneticField const* canonical_face_b = nullptr,
   amrex::iMultiFab const* canonical_face_b_cell_mask = nullptr
)
{
   if (system == PhysicalSystem::RHD) {
      fill_rhd_primitive_cache<EOSMethod>(
         primitive, state, eos, geometry, spacetime, time
      );
      return;
   }
   if (system == PhysicalSystem::RMHD) {
      fill_rmhd_primitive_cache(
         primitive,
         state,
         eos,
         geometry,
         spacetime,
         time,
         canonical_face_b,
         canonical_face_b_cell_mask
      );
      return;
   }
   const bool mhd = system == PhysicalSystem::MHD;
   if (mhd) {
      if (canonical_face_b != nullptr) {
         fill_primitive_cache_from_ct_faces<NVAR_MHD, EOSMethod>(
            primitive,
            state,
            eos,
            *canonical_face_b,
            *canonical_face_b_cell_mask,
            geometry
         );
      } else {
         fill_primitive_cache<NVAR_MHD, EOSMethod>(primitive, state, eos);
      }
   } else {
      fill_primitive_cache<NVAR_HYDRO, EOSMethod>(primitive, state, eos);
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
