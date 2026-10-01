#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

#include <array>

#include "pyfluid_amrex/mask_regions.hpp"
#include "pyfluid_amrex/time_integration.hpp"

namespace pyfluid_amrex {

template <int NComp>
inline void add_flux_divergence_to_state(
   amrex::MultiFab& state,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::Geometry const& geom,
   amrex::Real scale
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto fz = fluxes[2]->const_array(mfi);
#endif
      const auto s = state.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         for (int q = 0; q < NComp; ++q) {
            amrex::Real value = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
            value -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
            value -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
            s(i, j, k, q) += scale * value;
         }
      });
   }
   }
}

template <int NComp>
inline void add_flux_divergence_to_state_masked(
   amrex::MultiFab& state,
   amrex::iMultiFab const& solid_mask,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::Geometry const& geom,
   amrex::Real scale,
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
      const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto fz = fluxes[2]->const_array(mfi);
#endif
      const auto solid = solid_mask.const_array(mfi);
      const auto s = state.array(mfi);
      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
      if (kind == MaskKindNonzero) {
         continue;
      }
      if (kind == MaskKindZero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            for (int q = 0; q < NComp; ++q) {
               amrex::Real value = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
               value -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
               value -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
               s(i, j, k, q) += scale * value;
            }
         });
         continue;
      }
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (solid(i, j, k, 0) != 0) {
            return;
         }
         for (int q = 0; q < NComp; ++q) {
            amrex::Real value = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
            value -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
            value -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
            s(i, j, k, q) += scale * value;
         }
      });
   }
   }
}

template <int NComp>
inline void copy_plus_flux_divergence_components(
   amrex::MultiFab& out,
   amrex::MultiFab const& base,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::Geometry const& geom,
   amrex::Real scale
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(base, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto fz = fluxes[2]->const_array(mfi);
#endif
      const auto o = out.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         for (int q = 0; q < NComp; ++q) {
            amrex::Real rhs = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
            rhs -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
            rhs -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
            o(i, j, k, q) = b(i, j, k, q) + scale * rhs;
         }
      });
   }
   }
}

template <int NComp>
inline void copy_two_plus_flux_divergence_components(
   amrex::MultiFab& out_a,
   amrex::MultiFab& out_b,
   amrex::MultiFab const& base,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::Geometry const& geom,
   amrex::Real scale_a,
   amrex::Real scale_b
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(base, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto fz = fluxes[2]->const_array(mfi);
#endif
      const auto a = out_a.array(mfi);
      const auto o = out_b.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         for (int q = 0; q < NComp; ++q) {
            const amrex::Real value = b(i, j, k, q);
            amrex::Real rhs = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
            rhs -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
            rhs -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
            a(i, j, k, q) = value + scale_a * rhs;
            o(i, j, k, q) = value + scale_b * rhs;
         }
      });
   }
   }
}

template <int NComp>
inline void add_flux_divergence_and_copy_plus_flux_divergence_components(
   amrex::MultiFab& accumulator,
   amrex::MultiFab& scratch,
   amrex::MultiFab const& base,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::Geometry const& geom,
   amrex::Real accumulator_scale,
   amrex::Real scratch_scale
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(accumulator, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto fz = fluxes[2]->const_array(mfi);
#endif
      const auto acc = accumulator.array(mfi);
      const auto s = scratch.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         for (int q = 0; q < NComp; ++q) {
            amrex::Real rhs = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
            rhs -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
            rhs -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
            acc(i, j, k, q) += accumulator_scale * rhs;
            s(i, j, k, q) = b(i, j, k, q) + scratch_scale * rhs;
         }
      });
   }
   }
}

template <int NComp>
inline void affine_combination_plus_flux_divergence_components(
   amrex::MultiFab& out,
   amrex::MultiFab const& base,
   amrex::MultiFab const& input,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::Geometry const& geom,
   amrex::Real input_weight,
   amrex::Real flux_scale
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
      for (amrex::MFIter mfi(out, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto b = base.const_array(mfi);
         const auto in = input.const_array(mfi);
         const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
         const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
         const auto fz = fluxes[2]->const_array(mfi);
#endif
         const auto o = out.array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            for (int q = 0; q < NComp; ++q) {
               amrex::Real rhs = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
               rhs -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
               rhs -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
               o(i, j, k, q) = affine_state_update(
                  b(i, j, k, q), in(i, j, k, q), input_weight, flux_scale * rhs);
            }
         });
      }
   }
}

template <int NComp>
inline void copy_plus_flux_divergence_components_masked(
   amrex::MultiFab& out,
   amrex::MultiFab const& base,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   amrex::Real scale,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(base, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto fz = fluxes[2]->const_array(mfi);
#endif
      const auto solid = solid_mask.const_array(mfi);
      const auto o = out.array(mfi);
      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
      if (kind == MaskKindNonzero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            for (int q = 0; q < NComp; ++q) {
               o(i, j, k, q) = b(i, j, k, q);
            }
         });
         continue;
      }
      if (kind == MaskKindZero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            for (int q = 0; q < NComp; ++q) {
               amrex::Real rhs = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
               rhs -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
               rhs -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
               o(i, j, k, q) = b(i, j, k, q) + scale * rhs;
            }
         });
         continue;
      }
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (solid(i, j, k, 0) != 0) {
            for (int q = 0; q < NComp; ++q) {
               o(i, j, k, q) = b(i, j, k, q);
            }
            return;
         }
         for (int q = 0; q < NComp; ++q) {
            amrex::Real rhs = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
            rhs -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
            rhs -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
            o(i, j, k, q) = b(i, j, k, q) + scale * rhs;
         }
      });
   }
   }
}

template <int NComp>
inline void copy_two_plus_flux_divergence_components_masked(
   amrex::MultiFab& out_a,
   amrex::MultiFab& out_b,
   amrex::MultiFab const& base,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   amrex::Real scale_a,
   amrex::Real scale_b,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(base, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto fz = fluxes[2]->const_array(mfi);
#endif
      const auto solid = solid_mask.const_array(mfi);
      const auto a = out_a.array(mfi);
      const auto o = out_b.array(mfi);
      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
      if (kind == MaskKindNonzero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            for (int q = 0; q < NComp; ++q) {
               const amrex::Real value = b(i, j, k, q);
               a(i, j, k, q) = value;
               o(i, j, k, q) = value;
            }
         });
         continue;
      }
      if (kind == MaskKindZero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            for (int q = 0; q < NComp; ++q) {
               const amrex::Real value = b(i, j, k, q);
               amrex::Real rhs = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
               rhs -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
               rhs -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
               a(i, j, k, q) = value + scale_a * rhs;
               o(i, j, k, q) = value + scale_b * rhs;
            }
         });
         continue;
      }
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const bool is_solid = solid(i, j, k, 0) != 0;
         for (int q = 0; q < NComp; ++q) {
            const amrex::Real value = b(i, j, k, q);
            if (is_solid) {
               a(i, j, k, q) = value;
               o(i, j, k, q) = value;
            } else {
               amrex::Real rhs = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
               rhs -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
               rhs -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
               a(i, j, k, q) = value + scale_a * rhs;
               o(i, j, k, q) = value + scale_b * rhs;
            }
         }
      });
   }
   }
}

template <int NComp>
inline void add_flux_divergence_and_copy_plus_flux_divergence_components_masked(
   amrex::MultiFab& accumulator,
   amrex::MultiFab& scratch,
   amrex::MultiFab const& base,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   amrex::Real accumulator_scale,
   amrex::Real scratch_scale,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(accumulator, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto fz = fluxes[2]->const_array(mfi);
#endif
      const auto solid = solid_mask.const_array(mfi);
      const auto acc = accumulator.array(mfi);
      const auto s = scratch.array(mfi);
      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
      if (kind == MaskKindNonzero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            for (int q = 0; q < NComp; ++q) {
               s(i, j, k, q) = b(i, j, k, q);
            }
         });
         continue;
      }
      if (kind == MaskKindZero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            for (int q = 0; q < NComp; ++q) {
               amrex::Real rhs = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
               rhs -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
               rhs -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
               acc(i, j, k, q) += accumulator_scale * rhs;
               s(i, j, k, q) = b(i, j, k, q) + scratch_scale * rhs;
            }
         });
         continue;
      }
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const bool is_solid = solid(i, j, k, 0) != 0;
         for (int q = 0; q < NComp; ++q) {
            if (is_solid) {
               s(i, j, k, q) = b(i, j, k, q);
            } else {
               amrex::Real rhs = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
               rhs -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
               rhs -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
               acc(i, j, k, q) += accumulator_scale * rhs;
               s(i, j, k, q) = b(i, j, k, q) + scratch_scale * rhs;
            }
         }
      });
   }
   }
}

template <int NComp>
inline void affine_combination_plus_flux_divergence_components_masked(
   amrex::MultiFab& out,
   amrex::MultiFab const& base,
   amrex::MultiFab const& input,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   amrex::Real input_weight,
   amrex::Real flux_scale,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
      for (amrex::MFIter mfi(out, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto b = base.const_array(mfi);
         const auto in = input.const_array(mfi);
         const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
         const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
         const auto fz = fluxes[2]->const_array(mfi);
#endif
         const auto solid = solid_mask.const_array(mfi);
         const auto o = out.array(mfi);
         const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
         if (kind == MaskKindNonzero) {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               for (int q = 0; q < NComp; ++q) {
                  o(i, j, k, q) = b(i, j, k, q);
               }
            });
            continue;
         }
         if (kind == MaskKindZero) {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               for (int q = 0; q < NComp; ++q) {
                  amrex::Real rhs = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
                  rhs -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
                  rhs -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
                  o(i, j, k, q) = affine_state_update(
                     b(i, j, k, q), in(i, j, k, q), input_weight, flux_scale * rhs);
               }
            });
            continue;
         }
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            for (int q = 0; q < NComp; ++q) {
               if (solid(i, j, k, 0) != 0) {
                  o(i, j, k, q) = b(i, j, k, q);
               } else {
                  amrex::Real rhs = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
                  rhs -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
                  rhs -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
                  o(i, j, k, q) = affine_state_update(
                     b(i, j, k, q), in(i, j, k, q), input_weight, flux_scale * rhs);
               }
            }
         });
      }
   }
}



} // namespace pyfluid_amrex
