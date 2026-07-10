#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <limits>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/diffusion.hpp"
#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/mask_regions.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/patch_grid.hpp"
#include "pyfluid_amrex/patch_layout_builder.hpp"
#include "pyfluid_amrex/patch_metrics.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hydro_signal_frequency_cell(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   EOSParams const& eos
) noexcept
{
   const amrex::Real rho = state(i, j, k, RHO);
   const amrex::Real pressure = pressure_from_state<EOSMethod>(state, i, j, k, eos);
   const amrex::Real cs = sound_speed_from_primitive<EOSMethod>(rho, pressure, eos);
   const StateVector3 velocity = load_velocity_vector3(state, i, j, k);
   amrex::Real value = (amrex::Math::abs(velocity.x) + cs) / dx[0];
#if (AMREX_SPACEDIM >= 2)
   value += (amrex::Math::abs(velocity.y) + cs) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
   value += (amrex::Math::abs(velocity.z) + cs) / dx[2];
#endif
   return value;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hydro_signal_frequency_cell_metric(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int coordinate,
   MeshAxes const& axes,
   EOSParams const& eos
) noexcept
{
   const amrex::Real rho = state(i, j, k, RHO);
   const amrex::Real pressure = pressure_from_state<EOSMethod>(state, i, j, k, eos);
   const amrex::Real cs = sound_speed_from_primitive<EOSMethod>(rho, pressure, eos);
   const StateVector3 velocity = load_velocity_vector3(state, i, j, k);
   amrex::Real value =
      (amrex::Math::abs(velocity.x) + cs)
      / mesh_face_gradient_spacing(coordinate, axes, problo, dx, 0, i, j, k);
#if (AMREX_SPACEDIM >= 2)
   value +=
      (amrex::Math::abs(velocity.y) + cs)
      / mesh_face_gradient_spacing(coordinate, axes, problo, dx, 1, i, j, k);
#endif
#if (AMREX_SPACEDIM == 3)
   value +=
      (amrex::Math::abs(velocity.z) + cs)
      / mesh_face_gradient_spacing(coordinate, axes, problo, dx, 2, i, j, k);
#endif
   return value;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_hydro_signal_frequency_cell(
   amrex::Array4<amrex::Real const> const& state,
   amrex::Array4<amrex::Real> const& frequency,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   EOSParams const& eos
) noexcept
{
   const amrex::Real value = hydro_signal_frequency_cell<EOSMethod>(state, i, j, k, dx, eos);
   frequency(i, j, k, 0) = value;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mhd_signal_frequency_cell(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   EOSParams const& eos
) noexcept
{
   const MHDPrimitive primitive = primitive_from_conserved<EOSMethod>(load_mhd_state(state, i, j, k), eos);
   amrex::Real value = (amrex::Math::abs(primitive.v1) + mhd_fast_speed_from_primitive<EOSMethod>(primitive, 0, eos)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
   value += (amrex::Math::abs(primitive.v2) + mhd_fast_speed_from_primitive<EOSMethod>(primitive, 1, eos)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
   value += (amrex::Math::abs(primitive.v3) + mhd_fast_speed_from_primitive<EOSMethod>(primitive, 2, eos)) / dx[2];
#endif
   return value;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mhd_signal_frequency_cell_metric(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int coordinate,
   MeshAxes const& axes,
   EOSParams const& eos
) noexcept
{
   const MHDPrimitive primitive = primitive_from_conserved<EOSMethod>(load_mhd_state(state, i, j, k), eos);
   amrex::Real value =
      (amrex::Math::abs(primitive.v1) + mhd_fast_speed_from_primitive<EOSMethod>(primitive, 0, eos))
      / mesh_face_gradient_spacing(coordinate, axes, problo, dx, 0, i, j, k);
#if (AMREX_SPACEDIM >= 2)
   value +=
      (amrex::Math::abs(primitive.v2) + mhd_fast_speed_from_primitive<EOSMethod>(primitive, 1, eos))
      / mesh_face_gradient_spacing(coordinate, axes, problo, dx, 1, i, j, k);
#endif
#if (AMREX_SPACEDIM == 3)
   value +=
      (amrex::Math::abs(primitive.v3) + mhd_fast_speed_from_primitive<EOSMethod>(primitive, 2, eos))
      / mesh_face_gradient_spacing(coordinate, axes, problo, dx, 2, i, j, k);
#endif
   return value;
}

template <int EOSMethod>
inline void fill_signal_frequency(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   EOSParams const& eos
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto f = frequency.array(mfi);
      if (curvilinear_mesh_coordinate(coordinate)) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            f(i, j, k, 0) = hydro_signal_frequency_cell_metric<EOSMethod>(s, i, j, k, problo, dx, coordinate, axes, eos);
         });
      } else {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            f(i, j, k, 0) = hydro_signal_frequency_cell<EOSMethod>(s, i, j, k, dx, eos);
         });
      }
   }
   }
}

template <int EOSMethod>
inline void fill_signal_frequency_mapped_hydro(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout,
   EOSParams const& eos
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      PatchDescriptor const& layout_patch_ref = patch_layout_descriptor_containing_box(layout, mfi.validbox());
      PatchGridDescriptor const& grid_patch_ref = patch_grid_descriptor_for_layout_patch(grid_system, layout_patch_ref);
      const PatchDescriptor layout_patch = layout_patch_ref;
      const PatchGridDescriptor grid_patch = grid_patch_ref;
      const auto s = state.const_array(mfi);
      const auto f = frequency.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const StateVector3 velocity = load_velocity_vector3(s, i, j, k);
         const amrex::Real rho = s(i, j, k, RHO);
         const amrex::Real pressure = pressure_from_state<EOSMethod>(s, i, j, k, eos);
         const amrex::Real cs = sound_speed_from_primitive<EOSMethod>(rho, pressure, eos);
         const PhysicalVector physical_velocity = physical_vector(velocity.x, velocity.y, velocity.z);
         const PatchCoordinate center =
            patch_layout_cell_center_coordinate(grid_patch, layout_patch, i, j, k);
         const amrex::Real volume = patch_cell_volume(grid_patch, center);
         if (volume <= 0.0 || !amrex::Math::isfinite(volume)) {
            f(i, j, k, 0) = 0.0;
            return;
         }
         amrex::Real frequency_value = 0.0;
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const PatchCoordinate face_lo =
               patch_layout_face_coordinate(grid_patch, layout_patch, dir, i, j, k);
            amrex::IntVect hi_cell(AMREX_D_DECL(i, j, k));
            hi_cell[dir] += 1;
            const int hi_i = hi_cell[0];
#if (AMREX_SPACEDIM >= 2)
            const int hi_j = hi_cell[1];
#else
            const int hi_j = 0;
#endif
#if (AMREX_SPACEDIM == 3)
            const int hi_k = hi_cell[2];
#else
            const int hi_k = 0;
#endif
               const PatchCoordinate face_hi =
                  patch_layout_face_coordinate(grid_patch, layout_patch, dir, hi_i, hi_j, hi_k);
               const PatchFaceMetric metric_lo = patch_face_metric(grid_patch, face_lo, dir);
               const PatchFaceMetric metric_hi = patch_face_metric(grid_patch, face_hi, dir);
            const amrex::Real signal_lo = amrex::Math::abs(dot(physical_velocity, metric_lo.unit_normal)) + cs;
            const amrex::Real signal_hi = amrex::Math::abs(dot(physical_velocity, metric_hi.unit_normal)) + cs;
            frequency_value += 0.5 * (
               metric_lo.area * signal_lo
               + metric_hi.area * signal_hi
            ) / volume;
         }
         f(i, j, k, 0) = frequency_value;
      });
   }
   }
}

template <int EOSMethod>
inline void fill_signal_frequency_masked(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   EOSParams const& eos,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto solid = solid_mask.const_array(mfi);
      const auto f = frequency.array(mfi);
      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
      if (kind == MaskKindNonzero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            f(i, j, k, 0) = 0.0;
         });
         continue;
      }
      if (kind == MaskKindZero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            store_hydro_signal_frequency_cell<EOSMethod>(s, f, i, j, k, dx, eos);
         });
         continue;
      }
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (solid(i, j, k, 0) != 0) {
            f(i, j, k, 0) = 0.0;
            return;
         }
         store_hydro_signal_frequency_cell<EOSMethod>(s, f, i, j, k, dx, eos);
      });
   }
   }
}

template <int EOSMethod>
inline void fill_mhd_signal_frequency(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   EOSParams const& eos
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto f = frequency.array(mfi);
      if (curvilinear_mesh_coordinate(coordinate)) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            f(i, j, k, 0) = mhd_signal_frequency_cell_metric<EOSMethod>(s, i, j, k, problo, dx, coordinate, axes, eos);
         });
      } else {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            f(i, j, k, 0) = mhd_signal_frequency_cell<EOSMethod>(s, i, j, k, dx, eos);
         });
      }
   }
   }
}

template <int EOSMethod>
inline void fill_mhd_signal_frequency_masked(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   EOSParams const& eos,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto solid = solid_mask.const_array(mfi);
      const auto f = frequency.array(mfi);
      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
      if (kind == MaskKindNonzero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            f(i, j, k, 0) = 0.0;
         });
         continue;
      }
      if (kind == MaskKindZero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            f(i, j, k, 0) = mhd_signal_frequency_cell<EOSMethod>(s, i, j, k, dx, eos);
         });
         continue;
      }
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (solid(i, j, k, 0) != 0) {
            f(i, j, k, 0) = 0.0;
            return;
         }
         f(i, j, k, 0) = mhd_signal_frequency_cell<EOSMethod>(s, i, j, k, dx, eos);
      });
   }
   }
}

template <int EOSMethod>
inline amrex::Real max_hydro_signal_frequency_masked(
   amrex::MultiFab const& state,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   EOSParams const& eos,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
   const bool curvilinear = curvilinear_mesh_coordinate(coordinate);
   amrex::Real max_frequency = 0.0;
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(max:max_frequency) if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
      if (kind == MaskKindNonzero) {
         continue;
      }
      const auto s = state.const_array(mfi);
      const auto solid = solid_mask.const_array(mfi);
      const auto lo = bx.smallEnd();
      const auto hi = bx.bigEnd();
#if (AMREX_SPACEDIM == 1)
      constexpr int jlo = 0;
      constexpr int jhi = 0;
      constexpr int klo = 0;
      constexpr int khi = 0;
#elif (AMREX_SPACEDIM == 2)
      const int jlo = lo[1];
      const int jhi = hi[1];
      constexpr int klo = 0;
      constexpr int khi = 0;
#else
      const int jlo = lo[1];
      const int jhi = hi[1];
      const int klo = lo[2];
      const int khi = hi[2];
#endif
      for (int k = klo; k <= khi; ++k) {
            for (int j = jlo; j <= jhi; ++j) {
               for (int i = lo[0]; i <= hi[0]; ++i) {
                  if (kind == MaskKindMixed && solid(i, j, k, 0) != 0) {
                     continue;
                  }
                  const amrex::Real frequency = curvilinear
                     ? hydro_signal_frequency_cell_metric<EOSMethod>(s, i, j, k, problo, dx, coordinate, axes, eos)
                     : hydro_signal_frequency_cell<EOSMethod>(s, i, j, k, dx, eos);
                  max_frequency = amrex::max(max_frequency, frequency);
            }
         }
      }
   }
   }
   amrex::ParallelDescriptor::ReduceRealMax(max_frequency);
   return max_frequency;
}

template <int EOSMethod>
inline amrex::Real max_mhd_signal_frequency_masked(
   amrex::MultiFab const& state,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   EOSParams const& eos,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
   const bool curvilinear = curvilinear_mesh_coordinate(coordinate);
   amrex::Real max_frequency = 0.0;
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(max:max_frequency) if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const int kind = cached_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
      if (kind == MaskKindNonzero) {
         continue;
      }
      const auto s = state.const_array(mfi);
      const auto solid = solid_mask.const_array(mfi);
      const auto lo = bx.smallEnd();
      const auto hi = bx.bigEnd();
#if (AMREX_SPACEDIM == 1)
      constexpr int jlo = 0;
      constexpr int jhi = 0;
      constexpr int klo = 0;
      constexpr int khi = 0;
#elif (AMREX_SPACEDIM == 2)
      const int jlo = lo[1];
      const int jhi = hi[1];
      constexpr int klo = 0;
      constexpr int khi = 0;
#else
      const int jlo = lo[1];
      const int jhi = hi[1];
      const int klo = lo[2];
      const int khi = hi[2];
#endif
      for (int k = klo; k <= khi; ++k) {
         for (int j = jlo; j <= jhi; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
               if (kind == MaskKindMixed && solid(i, j, k, 0) != 0) {
                  continue;
               }
               const amrex::Real frequency = curvilinear
                  ? mhd_signal_frequency_cell_metric<EOSMethod>(s, i, j, k, problo, dx, coordinate, axes, eos)
                  : mhd_signal_frequency_cell<EOSMethod>(s, i, j, k, dx, eos);
               max_frequency = amrex::max(max_frequency, frequency);
            }
         }
      }
   }
   }
   amrex::ParallelDescriptor::ReduceRealMax(max_frequency);
   return max_frequency;
}

template <int EOSMethod>
inline amrex::Real max_hydro_cutcell_signal_frequency_masked(
   amrex::MultiFab const& state,
   amrex::iMultiFab const& solid_mask,
   amrex::MultiFab const& volume_fraction,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& apertures,
   amrex::Geometry const& geom,
   EOSParams const& eos,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   constexpr amrex::Real alpha_floor = 1.0e-12;
   const auto dx = geom.CellSizeArray();
   const amrex::Box& domain = geom.Domain();
   const auto dlo = domain.smallEnd();
   const auto dhi = domain.bigEnd();
   amrex::Real max_frequency = 0.0;
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(max:max_frequency) if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
      if (kind == MaskKindNonzero) {
         continue;
      }
      const auto s = state.const_array(mfi);
      const auto lo = bx.smallEnd();
      const auto hi = bx.bigEnd();
#if (AMREX_SPACEDIM == 1)
      constexpr int jlo = 0;
      constexpr int jhi = 0;
      constexpr int klo = 0;
      constexpr int khi = 0;
#elif (AMREX_SPACEDIM == 2)
      const int jlo = lo[1];
      const int jhi = hi[1];
      constexpr int klo = 0;
      constexpr int khi = 0;
#else
      const int jlo = lo[1];
      const int jhi = hi[1];
      const int klo = lo[2];
      const int khi = hi[2];
#endif
      if (kind == MaskKindZero) {
         for (int k = klo; k <= khi; ++k) {
            for (int j = jlo; j <= jhi; ++j) {
               for (int i = lo[0]; i <= hi[0]; ++i) {
                  max_frequency = amrex::max(max_frequency, hydro_signal_frequency_cell<EOSMethod>(s, i, j, k, dx, eos));
               }
            }
         }
         continue;
      }
      const auto solid = solid_mask.const_array(mfi);
      const auto alpha = volume_fraction.const_array(mfi);
      const auto ax = apertures[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto ay = apertures[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto az = apertures[2]->const_array(mfi);
#endif
      for (int k = klo; k <= khi; ++k) {
         for (int j = jlo; j <= jhi; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
               const amrex::Real fluid_fraction = alpha(i, j, k, 0);
               if ((kind == MaskKindMixed && solid(i, j, k, 0) != 0) || fluid_fraction <= alpha_floor) {
                  continue;
               }
               const StateVector3 velocity = load_velocity_vector3(s, i, j, k);
               const amrex::Real rho = s(i, j, k, RHO);
               const amrex::Real pressure = pressure_from_state<EOSMethod>(s, i, j, k, eos);
               const amrex::Real cs = sound_speed_from_primitive<EOSMethod>(rho, pressure, eos);

               amrex::Real ax_l = ax(i, j, k, 0);
               if (i > dlo[0] && alpha(i - 1, j, k, 0) <= alpha_floor) {
                  ax_l = 0.0;
               }
               amrex::Real ax_r = ax(i + 1, j, k, 0);
               if (i < dhi[0] && alpha(i + 1, j, k, 0) <= alpha_floor) {
                  ax_r = 0.0;
               }
               amrex::Real frequency =
                  0.5 * (ax_l + ax_r) * (amrex::Math::abs(velocity.x) + cs) / dx[0];
#if (AMREX_SPACEDIM >= 2)
               amrex::Real ay_l = ay(i, j, k, 0);
               if (j > dlo[1] && alpha(i, j - 1, k, 0) <= alpha_floor) {
                  ay_l = 0.0;
               }
               amrex::Real ay_r = ay(i, j + 1, k, 0);
               if (j < dhi[1] && alpha(i, j + 1, k, 0) <= alpha_floor) {
                  ay_r = 0.0;
               }
               frequency +=
                  0.5 * (ay_l + ay_r) * (amrex::Math::abs(velocity.y) + cs) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
               amrex::Real az_l = az(i, j, k, 0);
               if (k > dlo[2] && alpha(i, j, k - 1, 0) <= alpha_floor) {
                  az_l = 0.0;
               }
               amrex::Real az_r = az(i, j, k + 1, 0);
               if (k < dhi[2] && alpha(i, j, k + 1, 0) <= alpha_floor) {
                  az_r = 0.0;
               }
               frequency +=
                  0.5 * (az_l + az_r) * (amrex::Math::abs(velocity.z) + cs) / dx[2];
#endif
               max_frequency = amrex::max(max_frequency, frequency / fluid_fraction);
            }
         }
      }
   }
   }
   amrex::ParallelDescriptor::ReduceRealMax(max_frequency);
   return max_frequency;
}

template <int NComp, int EOSMethod, int DiffusionMethod>
inline amrex::Real adaptive_dt(amrex::MultiFab& frequency, amrex::MultiFab const& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const EOSParams eos = eos_params(cfg);
   if constexpr (has_mhd_components<NComp>()) {
      fill_mhd_signal_frequency<EOSMethod>(frequency, state, geom, cfg, eos);
   } else {
      fill_signal_frequency<EOSMethod>(frequency, state, geom, cfg, eos);
   }
   const amrex::Real max_frequency = frequency.max(0);
   const amrex::Real advective_dt = cfg.cfl / max_frequency;
   return amrex::min(advective_dt, diffusion_dt_limit<DiffusionMethod>(geom, cfg));
}

template <int NComp, int EOSMethod, int DiffusionMethod>
inline amrex::Real adaptive_dt_mapped_hydro(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   RunConfig const& cfg,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout
)
{
   static_assert(!has_mhd_components<NComp>(), "mapped patch timestep helper is hydro-only");
   if constexpr (DiffusionMethod != DiffusionNone) {
      amrex::ignore_unused(frequency, state, cfg, grid_system, layout);
      amrex::Abort("mapped patch hydro diffusion timestep support is not implemented");
      return std::numeric_limits<amrex::Real>::max();
   } else {
      const EOSParams eos = eos_params(cfg);
      fill_signal_frequency_mapped_hydro<EOSMethod>(frequency, state, grid_system, layout, eos);
      const amrex::Real max_frequency = frequency.max(0);
      if (max_frequency <= 0.0) {
         return std::numeric_limits<amrex::Real>::max();
      }
      return cfg.cfl / max_frequency;
   }
}

template <int NComp, int EOSMethod, int DiffusionMethod>
inline amrex::Real adaptive_dt_masked(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const EOSParams eos = eos_params(cfg);
   (void)frequency;
   amrex::Real max_frequency = 0.0;
   if constexpr (has_mhd_components<NComp>()) {
      max_frequency = max_mhd_signal_frequency_masked<EOSMethod>(state, solid_mask, geom, cfg, eos, solid_valid_kind);
   } else {
      max_frequency = max_hydro_signal_frequency_masked<EOSMethod>(state, solid_mask, geom, cfg, eos, solid_valid_kind);
   }
   if (max_frequency <= 0.0) {
      return std::numeric_limits<amrex::Real>::max();
   }
   const amrex::Real advective_dt = cfg.cfl / max_frequency;
   return amrex::min(advective_dt, diffusion_dt_limit<DiffusionMethod>(geom, cfg));
}

template <int NComp, int EOSMethod, int DiffusionMethod>
inline amrex::Real adaptive_dt_cutcell_masked(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::iMultiFab const& solid_mask,
   amrex::MultiFab const& volume_fraction,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& apertures,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const EOSParams eos = eos_params(cfg);
   (void)frequency;
   amrex::Real max_frequency = 0.0;
   if constexpr (!has_mhd_components<NComp>()) {
      max_frequency = max_hydro_cutcell_signal_frequency_masked<EOSMethod>(
         state,
         solid_mask,
         volume_fraction,
         apertures,
         geom,
         eos,
         solid_valid_kind
      );
   } else {
      max_frequency = max_mhd_signal_frequency_masked<EOSMethod>(state, solid_mask, geom, cfg, eos, solid_valid_kind);
   }
   if (max_frequency <= 0.0) {
      return std::numeric_limits<amrex::Real>::max();
   }
   const amrex::Real advective_dt = cfg.cfl / max_frequency;
   return amrex::min(advective_dt, diffusion_dt_limit<DiffusionMethod>(geom, cfg));
}

} // namespace pyfluid_amrex
