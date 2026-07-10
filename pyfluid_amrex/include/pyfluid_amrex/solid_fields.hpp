#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_TagBox.H>

#include "pyfluid_amrex/solid_geometry.hpp"

namespace pyfluid_amrex {

inline void fill_solid_phi_output(
   amrex::MultiFab& output,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Real time
)
{
   output.setVal(1.0e300);
   if (!solid_geometry_enabled(cfg)) {
      return;
   }
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const SolidGeometryEvaluator params = solid_geometry_evaluator(cfg, time);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(output, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.growntilebox(output.nGrowVect());
      const auto out = output.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
#else
         const amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real z = problo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
#else
         const amrex::Real z = 0.0;
#endif
         out(i, j, k, 0) = params.phi(time, x, y, z);
      });
   }
   }
}

inline void fill_primitive_needed_mask(
   amrex::iMultiFab& needed_mask,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   int buffer
)
{
   needed_mask.setVal(0);
   const amrex::Box domain = geom.Domain();
   const auto dlo = amrex::lbound(domain);
   const auto dhi = amrex::ubound(domain);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(needed_mask); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.fabbox();
      const auto lo = bx.smallEnd();
      const auto hi = bx.bigEnd();
      const auto solid = solid_mask.const_array(mfi);
      const auto needed = needed_mask.array(mfi);
      const int ng = buffer;
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (solid(i, j, k, 0) == 0) {
            needed(i, j, k, 0) = 1;
            return;
         }
         for (int dk = -ng; dk <= ng; ++dk) {
#if (AMREX_SPACEDIM == 3)
            const int kk = k + dk;
            if (kk < lo[2] || kk > hi[2] || kk < dlo.z || kk > dhi.z) {
               continue;
            }
#else
            const int kk = k;
            if (dk != 0) {
               continue;
            }
#endif
            for (int dj = -ng; dj <= ng; ++dj) {
#if (AMREX_SPACEDIM >= 2)
               const int jj = j + dj;
               if (jj < lo[1] || jj > hi[1] || jj < dlo.y || jj > dhi.y) {
                  continue;
               }
#else
               const int jj = j;
               if (dj != 0) {
                  continue;
               }
#endif
               for (int di = -ng; di <= ng; ++di) {
                  const int ii = i + di;
                  if (ii < lo[0] || ii > hi[0] || ii < dlo.x || ii > dhi.x) {
                     continue;
                  }
                  if (solid(ii, jj, kk, 0) == 0) {
                     needed(i, j, k, 0) = 1;
                     return;
                  }
               }
            }
         }
      });
   }
   }
}

inline void fill_solid_mask_output(amrex::MultiFab& output, amrex::iMultiFab const* solid_mask)
{
   output.setVal(0.0);
   if (solid_mask == nullptr) {
      return;
   }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(output, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.growntilebox(output.nGrowVect());
      const auto mask = solid_mask->const_array(mfi);
      const auto out = output.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         out(i, j, k, 0) = mask(i, j, k, 0) != 0 ? 1.0 : 0.0;
      });
   }
   }
}

inline void fill_solid_volume_fraction_output(amrex::MultiFab& output, amrex::iMultiFab const* solid_mask)
{
   output.setVal(1.0);
   if (solid_mask == nullptr) {
      return;
   }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(output, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto mask = solid_mask->const_array(mfi);
      const auto out = output.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         out(i, j, k, 0) = mask(i, j, k, 0) != 0 ? 0.0 : 1.0;
      });
   }
   }
}

inline void fill_solid_volume_fraction_output(
   amrex::MultiFab& output,
   amrex::iMultiFab const* solid_mask,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Real time
)
{
   const int samples_per_dim = cfg.solid_volume_fraction_samples_per_dim;
   if (samples_per_dim <= 1 || !solid_geometry_enabled(cfg)) {
      fill_solid_volume_fraction_output(output, solid_mask);
      return;
   }
   output.setVal(1.0);
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const SolidGeometryEvaluator params = solid_geometry_evaluator(cfg, time);
#if (AMREX_SPACEDIM >= 2)
   const amrex::Real dy = dx[1];
#else
   const amrex::Real dy = 1.0;
#endif
#if (AMREX_SPACEDIM >= 2)
   const amrex::Real ylo = problo[1];
#else
   const amrex::Real ylo = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
   const amrex::Real dz = dx[2];
   const amrex::Real zlo = problo[2];
#else
   const amrex::Real dz = 1.0;
   const amrex::Real zlo = 0.0;
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(output, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.growntilebox(output.nGrowVect());
      const auto out = output.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real x0 = problo[0] + static_cast<amrex::Real>(i) * dx[0];
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real y0 = ylo + static_cast<amrex::Real>(j) * dy;
#else
         const amrex::Real y0 = ylo;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real z0 = zlo + static_cast<amrex::Real>(k) * dz;
#else
         const amrex::Real z0 = zlo;
#endif
         out(i, j, k, 0) = sampled_cell_fluid_fraction(params, time, x0, y0, z0, dx[0], dy, dz, samples_per_dim);
      });
   }
   }
}

inline void fill_solid_volume_fraction_data(
   amrex::MultiFab& volume_fraction,
   amrex::iMultiFab const* solid_mask,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Real time
)
{
   fill_solid_volume_fraction_output(volume_fraction, solid_mask, geom, cfg, time);
}

inline void fill_solid_mask_from_volume_fraction(
   amrex::iMultiFab& solid_mask,
   amrex::MultiFab const& volume_fraction,
   amrex::Real fluid_fraction_floor = 1.0e-12
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(solid_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.growntilebox(solid_mask.nGrowVect());
      const auto vf = volume_fraction.const_array(mfi);
      const auto mask = solid_mask.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         mask(i, j, k, 0) = vf(i, j, k, 0) <= fluid_fraction_floor ? 1 : 0;
      });
   }
   }
}

inline void fill_solid_swept_volume_output(
   amrex::MultiFab& output,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Real reference_time,
   amrex::Real time
)
{
   output.setVal(0.0);
   if (!solid_geometry_enabled(cfg)) {
      return;
   }
   const int samples_per_dim = cfg.solid_volume_fraction_samples_per_dim;
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const SolidGeometryEvaluator params = solid_geometry_evaluator(cfg, time);
#if (AMREX_SPACEDIM >= 2)
   const amrex::Real dy = dx[1];
   const amrex::Real ylo = problo[1];
#else
   const amrex::Real dy = 1.0;
   const amrex::Real ylo = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
   const amrex::Real dz = dx[2];
   const amrex::Real zlo = problo[2];
#else
   const amrex::Real dz = 1.0;
   const amrex::Real zlo = 0.0;
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(output, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto out = output.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real x0 = problo[0] + static_cast<amrex::Real>(i) * dx[0];
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real y0 = ylo + static_cast<amrex::Real>(j) * dy;
#else
         const amrex::Real y0 = ylo;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real z0 = zlo + static_cast<amrex::Real>(k) * dz;
#else
         const amrex::Real z0 = zlo;
#endif
         const amrex::Real old_fluid =
            sampled_cell_fluid_fraction(params, reference_time, x0, y0, z0, dx[0], dy, dz, samples_per_dim);
         const amrex::Real new_fluid =
            sampled_cell_fluid_fraction(params, time, x0, y0, z0, dx[0], dy, dz, samples_per_dim);
         out(i, j, k, 0) = old_fluid - new_fluid;
      });
   }
   }
}

template <int Direction>
inline void fill_solid_face_aperture_output(
   amrex::MultiFab& output,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Real time
)
{
   output.setVal(1.0);
   if (!solid_geometry_enabled(cfg)) {
      return;
   }
   const int samples_per_dim = cfg.solid_volume_fraction_samples_per_dim;
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const SolidGeometryEvaluator params = solid_geometry_evaluator(cfg, time);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(output, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.growntilebox(output.nGrowVect());
      const auto out = output.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         out(i, j, k, 0) = sampled_face_fluid_fraction<Direction>(params, time, i, j, k, problo, dx, samples_per_dim);
      });
   }
   }
}

inline void remove_solid_cells_from_valid_mask(amrex::iMultiFab& valid_mask, amrex::iMultiFab const* solid_mask)
{
   if (solid_mask == nullptr) {
      return;
   }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(valid_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto valid = valid_mask.array(mfi);
      const auto solid = solid_mask->const_array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (solid(i, j, k, 0) != 0) {
            valid(i, j, k, 0) = 0;
         }
      });
   }
   }
}

inline void tag_solid_fluid_boundaries(amrex::TagBoxArray& tags, amrex::iMultiFab const* solid_mask)
{
   if (solid_mask == nullptr) {
      return;
   }
   constexpr char tagval = amrex::TagBox::SET;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(*solid_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto solid = solid_mask->const_array(mfi);
      const auto tag = tags.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (solid(i, j, k, 0) != 0) {
            return;
         }
         bool adjacent_to_solid =
            solid(i - 1, j, k, 0) != 0 || solid(i + 1, j, k, 0) != 0;
#if (AMREX_SPACEDIM >= 2)
         adjacent_to_solid = adjacent_to_solid
            || solid(i, j - 1, k, 0) != 0 || solid(i, j + 1, k, 0) != 0;
#endif
#if (AMREX_SPACEDIM == 3)
         adjacent_to_solid = adjacent_to_solid
            || solid(i, j, k - 1, 0) != 0 || solid(i, j, k + 1, 0) != 0;
#endif
         if (adjacent_to_solid) {
            tag(i, j, k) = tagval;
         }
      });
   }
   }
}

} // namespace pyfluid_amrex
