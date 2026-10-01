#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

#include <cmath>

#include "pyfluid_amrex/solid_geometry.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

inline amrex::Real count_mask_transition(
   amrex::iMultiFab const& old_mask,
   amrex::iMultiFab const& new_mask,
   int old_value,
   int new_value
)
{
   amrex::MultiFab flags(old_mask.boxArray(), old_mask.DistributionMap(), 1, 0);
   flags.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(flags, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto old_solid = old_mask.const_array(mfi);
      const auto new_solid = new_mask.const_array(mfi);
      const auto out = flags.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const bool old_matches = old_value == 0 ? old_solid(i, j, k, 0) == 0 : old_solid(i, j, k, 0) != 0;
         const bool new_matches = new_value == 0 ? new_solid(i, j, k, 0) == 0 : new_solid(i, j, k, 0) != 0;
         out(i, j, k, 0) = old_matches && new_matches ? 1.0_rt : 0.0_rt;
      });
   }
   }
   return flags.sum(0);
}

inline amrex::Real count_translated_box_receivers(
   amrex::iMultiFab const& old_mask,
   amrex::iMultiFab const& new_mask,
   RunConfig const& cfg
)
{
   const int axis = solid_translation_axis(cfg);
   const int sign = solid_translation_sign(cfg);
   if (axis < 0 || sign == 0) {
      return 0.0_rt;
   }
   amrex::MultiFab flags(old_mask.boxArray(), old_mask.DistributionMap(), 1, 0);
   flags.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(flags, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const amrex::Box& fabbox = old_mask[mfi].box();
      const auto lo = fabbox.smallEnd();
      const auto hi = fabbox.bigEnd();
      const auto old_solid = old_mask.const_array(mfi);
      const auto new_solid = new_mask.const_array(mfi);
      const auto out = flags.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (new_solid(i, j, k, 0) != 0) {
            return;
         }
         int si = i;
         int sj = j;
         int sk = k;
         if (axis == 0) {
            si -= sign;
         }
#if (AMREX_SPACEDIM >= 2)
         if (axis == 1) {
            sj -= sign;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         if (axis == 2) {
            sk -= sign;
         }
#endif
         if (si < lo[0] || si > hi[0]) {
            return;
         }
#if (AMREX_SPACEDIM >= 2)
         if (sj < lo[1] || sj > hi[1]) {
            return;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         if (sk < lo[2] || sk > hi[2]) {
            return;
         }
#endif
         const bool source_was_fluid = old_solid(si, sj, sk, 0) == 0;
         const bool source_is_solid = new_solid(si, sj, sk, 0) != 0;
         out(i, j, k, 0) = source_was_fluid && source_is_solid ? 1.0_rt : 0.0_rt;
      });
   }
   }
   return flags.sum(0);
}

inline void conservative_translate_box_mask_update(
   amrex::MultiFab& state,
   amrex::iMultiFab const& old_mask,
   amrex::iMultiFab const& new_mask,
   RunConfig const& cfg
)
{
   if (cfg.solid_geometry != "boxes" || cfg.solid_motion != "translate") {
      return;
   }
   const int axis = solid_translation_axis(cfg);
   const int sign = solid_translation_sign(cfg);
   if (axis < 0 || sign == 0) {
      amrex::Abort("pyfluid.solid_motion = translate requires exactly one nonzero solid_velocity component");
   }

   const amrex::Real newly_uncovered = count_mask_transition(old_mask, new_mask, 1, 0);
   if (newly_uncovered > 0.0_rt) {
      amrex::Abort(
         "translated box uncovered fluid cells. Internal translating solids need swept-volume/cut-cell support "
         "before they can conserve mass, momentum, and energy."
      );
   }
   const amrex::Real newly_covered = count_mask_transition(old_mask, new_mask, 0, 1);
   if (newly_covered <= 0.0_rt) {
      return;
   }
   const amrex::Real receivers = count_translated_box_receivers(old_mask, new_mask, cfg);
   if (receivers + 0.5_rt < newly_covered) {
      amrex::Abort(
         "translated box swept fluid cells without a receiving fluid cell in the direction of motion. "
         "Use a smaller timestep or keep the piston away from the outflow boundary."
      );
   }

   const int ncomp = state.nComp();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const amrex::Box& fabbox = old_mask[mfi].box();
      const auto lo = fabbox.smallEnd();
      const auto hi = fabbox.bigEnd();
      const auto s = state.array(mfi);
      const auto old_solid = old_mask.const_array(mfi);
      const auto new_solid = new_mask.const_array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (new_solid(i, j, k, 0) != 0) {
            return;
         }
         int si = i;
         int sj = j;
         int sk = k;
         if (axis == 0) {
            si -= sign;
         }
#if (AMREX_SPACEDIM >= 2)
         if (axis == 1) {
            sj -= sign;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         if (axis == 2) {
            sk -= sign;
         }
#endif
         if (si < lo[0] || si > hi[0]) {
            return;
         }
#if (AMREX_SPACEDIM >= 2)
         if (sj < lo[1] || sj > hi[1]) {
            return;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         if (sk < lo[2] || sk > hi[2]) {
            return;
         }
#endif
         const bool source_was_fluid = old_solid(si, sj, sk, 0) == 0;
         const bool source_is_solid = new_solid(si, sj, sk, 0) != 0;
         if (!source_was_fluid || !source_is_solid) {
            return;
         }
         for (int n = 0; n < ncomp; ++n) {
            s(i, j, k, n) += s(si, sj, sk, n);
         }
      });
   }
   }
}


template <typename MaskArray>
inline amrex::Real outward_fluid_shell_weight_sum(
   MaskArray const& old_solid,
   MaskArray const& new_solid,
   amrex::Box const& fabbox,
   amrex::Box const& domain,
   int i,
   int j,
   int k,
   int radius,
   bool require_old_fluid,
   amrex::Real nx,
   amrex::Real ny,
   amrex::Real nz,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx
)
{
   const auto flo = amrex::lbound(fabbox);
   const auto fhi = amrex::ubound(fabbox);
   const auto dlo = amrex::lbound(domain);
   const auto dhi = amrex::ubound(domain);
   amrex::Real sum = 0.0_rt;
   for (int ox = -radius; ox <= radius; ++ox) {
#if (AMREX_SPACEDIM >= 2)
   for (int oy = -radius; oy <= radius; ++oy) {
#else
   const int oy = 0;
   {
#endif
#if (AMREX_SPACEDIM == 3)
   for (int oz = -radius; oz <= radius; ++oz) {
#else
   const int oz = 0;
   {
#endif
      if (
         std::abs(ox) != radius
#if (AMREX_SPACEDIM >= 2)
         && std::abs(oy) != radius
#endif
#if (AMREX_SPACEDIM == 3)
         && std::abs(oz) != radius
#endif
      ) {
         continue;
      }
      const int ci = i + ox;
      const int cj = j + oy;
      const int ck = k + oz;
      if (ci < flo.x || ci > fhi.x || ci < dlo.x || ci > dhi.x) {
         continue;
      }
#if (AMREX_SPACEDIM >= 2)
      if (cj < flo.y || cj > fhi.y || cj < dlo.y || cj > dhi.y) {
         continue;
      }
#else
      (void)cj;
#endif
#if (AMREX_SPACEDIM == 3)
      if (ck < flo.z || ck > fhi.z || ck < dlo.z || ck > dhi.z) {
         continue;
      }
#else
      (void)ck;
#endif
      const bool was_fluid = old_solid(ci, cj, ck, 0) == 0;
      const bool is_fluid = new_solid(ci, cj, ck, 0) == 0;
      if (is_fluid && (!require_old_fluid || was_fluid)) {
         const amrex::Real vx = static_cast<amrex::Real>(ox) * dx[0];
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real vy = static_cast<amrex::Real>(oy) * dx[1];
#else
         const amrex::Real vy = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real vz = static_cast<amrex::Real>(oz) * dx[2];
#else
         const amrex::Real vz = 0.0_rt;
#endif
         const amrex::Real projection = vx * nx + vy * ny + vz * nz;
         if (projection > 0.0_rt) {
            const amrex::Real dist2 = squared_norm(StateVector3{vx, vy, vz});
            sum += projection / amrex::max(dist2, dx[0] * dx[0]);
         }
      }
#if (AMREX_SPACEDIM == 3)
   }
#else
   }
#endif
#if (AMREX_SPACEDIM >= 2)
   }
#else
   }
#endif
   }
   return sum;
}

template <typename MaskArray, typename StateArray, typename DeltaArray>
inline void distribute_to_outward_fluid_shell(
   StateArray const& state,
   DeltaArray const& delta,
   MaskArray const& old_solid,
   MaskArray const& new_solid,
   amrex::Box const& fabbox,
   amrex::Box const& domain,
   int i,
   int j,
   int k,
   int radius,
   bool require_old_fluid,
   amrex::Real nx,
   amrex::Real ny,
   amrex::Real nz,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real inv_weight_sum,
   int ncomp
)
{
   const auto flo = amrex::lbound(fabbox);
   const auto fhi = amrex::ubound(fabbox);
   const auto dlo = amrex::lbound(domain);
   const auto dhi = amrex::ubound(domain);
   for (int ox = -radius; ox <= radius; ++ox) {
#if (AMREX_SPACEDIM >= 2)
   for (int oy = -radius; oy <= radius; ++oy) {
#else
   const int oy = 0;
   {
#endif
#if (AMREX_SPACEDIM == 3)
   for (int oz = -radius; oz <= radius; ++oz) {
#else
   const int oz = 0;
   {
#endif
      if (
         std::abs(ox) != radius
#if (AMREX_SPACEDIM >= 2)
         && std::abs(oy) != radius
#endif
#if (AMREX_SPACEDIM == 3)
         && std::abs(oz) != radius
#endif
      ) {
         continue;
      }
      const int ci = i + ox;
      const int cj = j + oy;
      const int ck = k + oz;
      if (ci < flo.x || ci > fhi.x || ci < dlo.x || ci > dhi.x) {
         continue;
      }
#if (AMREX_SPACEDIM >= 2)
      if (cj < flo.y || cj > fhi.y || cj < dlo.y || cj > dhi.y) {
         continue;
      }
#endif
#if (AMREX_SPACEDIM == 3)
      if (ck < flo.z || ck > fhi.z || ck < dlo.z || ck > dhi.z) {
         continue;
      }
#endif
      const bool was_fluid = old_solid(ci, cj, ck, 0) == 0;
      const bool is_fluid = new_solid(ci, cj, ck, 0) == 0;
      if (is_fluid && (!require_old_fluid || was_fluid)) {
         const amrex::Real vx = static_cast<amrex::Real>(ox) * dx[0];
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real vy = static_cast<amrex::Real>(oy) * dx[1];
#else
         const amrex::Real vy = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real vz = static_cast<amrex::Real>(oz) * dx[2];
#else
         const amrex::Real vz = 0.0_rt;
#endif
         const amrex::Real projection = vx * nx + vy * ny + vz * nz;
         if (projection > 0.0_rt) {
            const amrex::Real dist2 = squared_norm(StateVector3{vx, vy, vz});
            const amrex::Real weight = projection / amrex::max(dist2, dx[0] * dx[0]) * inv_weight_sum;
            for (int n = 0; n < ncomp; ++n) {
               delta(ci, cj, ck, n) += weight * state(i, j, k, n);
            }
         }
      }
#if (AMREX_SPACEDIM == 3)
   }
#else
   }
#endif
#if (AMREX_SPACEDIM >= 2)
   }
#else
   }
#endif
   }
}

template <typename MaskArray, typename StateArray, typename DeltaArray>
inline void fill_from_outward_fluid_shell(
   StateArray const& state,
   DeltaArray const& delta,
   MaskArray const& old_solid,
   MaskArray const& new_solid,
   amrex::Box const& fabbox,
   amrex::Box const& domain,
   int i,
   int j,
   int k,
   int radius,
   bool require_old_fluid,
   amrex::Real nx,
   amrex::Real ny,
   amrex::Real nz,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real inv_weight_sum,
   int ncomp
)
{
   const auto flo = amrex::lbound(fabbox);
   const auto fhi = amrex::ubound(fabbox);
   const auto dlo = amrex::lbound(domain);
   const auto dhi = amrex::ubound(domain);
   for (int ox = -radius; ox <= radius; ++ox) {
#if (AMREX_SPACEDIM >= 2)
   for (int oy = -radius; oy <= radius; ++oy) {
#else
   const int oy = 0;
   {
#endif
#if (AMREX_SPACEDIM == 3)
   for (int oz = -radius; oz <= radius; ++oz) {
#else
   const int oz = 0;
   {
#endif
      if (
         std::abs(ox) != radius
#if (AMREX_SPACEDIM >= 2)
         && std::abs(oy) != radius
#endif
#if (AMREX_SPACEDIM == 3)
         && std::abs(oz) != radius
#endif
      ) {
         continue;
      }
      const int ci = i + ox;
      const int cj = j + oy;
      const int ck = k + oz;
      if (ci < flo.x || ci > fhi.x || ci < dlo.x || ci > dhi.x) {
         continue;
      }
#if (AMREX_SPACEDIM >= 2)
      if (cj < flo.y || cj > fhi.y || cj < dlo.y || cj > dhi.y) {
         continue;
      }
#endif
#if (AMREX_SPACEDIM == 3)
      if (ck < flo.z || ck > fhi.z || ck < dlo.z || ck > dhi.z) {
         continue;
      }
#endif
      const bool was_fluid = old_solid(ci, cj, ck, 0) == 0;
      const bool is_fluid = new_solid(ci, cj, ck, 0) == 0;
      if (is_fluid && (!require_old_fluid || was_fluid)) {
         const amrex::Real vx = static_cast<amrex::Real>(ox) * dx[0];
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real vy = static_cast<amrex::Real>(oy) * dx[1];
#else
         const amrex::Real vy = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real vz = static_cast<amrex::Real>(oz) * dx[2];
#else
         const amrex::Real vz = 0.0_rt;
#endif
	         const amrex::Real projection = vx * nx + vy * ny + vz * nz;
	         if (projection > 0.0_rt) {
	            const amrex::Real dist2 = squared_norm(StateVector3{vx, vy, vz});
	            const amrex::Real weight = projection / amrex::max(dist2, dx[0] * dx[0]) * inv_weight_sum;
	            for (int n = 0; n < ncomp; ++n) {
	               const amrex::Real contribution = weight * state(ci, cj, ck, n);
	               delta(i, j, k, n) += contribution;
	               delta(ci, cj, ck, n) -= contribution;
	            }
	         }
	      }
#if (AMREX_SPACEDIM == 3)
   }
#else
   }
#endif
#if (AMREX_SPACEDIM >= 2)
   }
#else
   }
#endif
   }
}

inline void conservative_pulse_solid_mask_update(
   amrex::MultiFab& state,
   amrex::iMultiFab const& old_mask,
   amrex::iMultiFab const& new_mask,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Real old_time,
   amrex::Real new_time
)
{
   if (!solid_geometry_enabled(cfg) || cfg.solid_motion != "pulse") {
      return;
   }
   const amrex::Real changed =
      count_mask_transition(old_mask, new_mask, 0, 1)
      + count_mask_transition(old_mask, new_mask, 1, 0);
   if (changed <= 0.0_rt) {
      return;
   }

   const int ncomp = state.nComp();
   amrex::MultiFab delta(state.boxArray(), state.DistributionMap(), ncomp, SolidMotionSearchRadius);
   delta.setVal(0.0_rt);

   const amrex::Box domain = geom.Domain();
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const SolidGeometryEvaluator params = solid_geometry_evaluator(cfg, new_time);

   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const amrex::Box& fabbox = old_mask[mfi].box();
      const auto lo = amrex::lbound(bx);
      const auto hi = amrex::ubound(bx);
      const auto s = state.array(mfi);
      const auto old_solid = old_mask.const_array(mfi);
      const auto new_solid = new_mask.const_array(mfi);
      const auto d = delta.array(mfi);
      for (int k = lo.z; k <= hi.z; ++k) {
         for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
               if (old_solid(i, j, k, 0) == new_solid(i, j, k, 0)) {
                  continue;
               }
               const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5_rt) * dx[0];
#if (AMREX_SPACEDIM >= 2)
               const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5_rt) * dx[1];
#else
               const amrex::Real y = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
               const amrex::Real z = problo[2] + (static_cast<amrex::Real>(k) + 0.5_rt) * dx[2];
#else
               const amrex::Real z = 0.0_rt;
#endif
	               amrex::Real nx = 0.0_rt;
	               amrex::Real ny = 0.0_rt;
	               amrex::Real nz = 0.0_rt;
	               params.normal_at(new_time, x, y, z, dx, nx, ny, nz);

               if (old_solid(i, j, k, 0) == 0 && new_solid(i, j, k, 0) != 0) {
                  int receiver_radius = 0;
                  amrex::Real weight_sum = 0.0_rt;
                  for (int radius = 1; radius <= SolidMotionSearchRadius; ++radius) {
                     weight_sum = outward_fluid_shell_weight_sum(
                        old_solid,
                        new_solid,
                        fabbox,
                        domain,
                        i,
                        j,
                        k,
                        radius,
                        false,
                        nx,
                        ny,
                        nz,
                        dx
                     );
                     if (weight_sum > 0.0_rt) {
                        receiver_radius = radius;
                        break;
                     }
                  }
	                  if (receiver_radius == 0) {
	                     amrex::Abort(
	                        "pulsing solid covered a fluid cell without an outward receiving fluid cell. "
	                        "Reduce the timestep or keep the pulsing obstacle away from boundaries."
	                     );
                  }
                  distribute_to_outward_fluid_shell(
                     s,
                     d,
                     old_solid,
                     new_solid,
                     fabbox,
                     domain,
                     i,
                     j,
                     k,
                     receiver_radius,
                     false,
                     nx,
                     ny,
                     nz,
                     dx,
                     1.0_rt / weight_sum,
                     ncomp
                  );
               } else {
                  int donor_radius = 0;
                  amrex::Real weight_sum = 0.0_rt;
                  for (int radius = 1; radius <= SolidMotionSearchRadius; ++radius) {
                     weight_sum = outward_fluid_shell_weight_sum(
                        old_solid,
                        new_solid,
                        fabbox,
                        domain,
                        i,
                        j,
                        k,
                        radius,
                        true,
                        nx,
                        ny,
                        nz,
                        dx
                     );
                     if (weight_sum > 0.0_rt) {
                        donor_radius = radius;
                        break;
                     }
                  }
	                  if (donor_radius == 0) {
	                     amrex::Abort(
	                        "pulsing solid uncovered a cell without an outward fluid donor. "
	                        "Reduce the timestep or keep the pulsing obstacle away from boundaries."
	                     );
                  }
                  fill_from_outward_fluid_shell(
                     s,
                     d,
                     old_solid,
                     new_solid,
                     fabbox,
                     domain,
                     i,
                     j,
                     k,
                     donor_radius,
                     true,
                     nx,
                     ny,
                     nz,
                     dx,
                     1.0_rt / weight_sum,
                     ncomp
                  );
               }
            }
         }
      }
   }

   delta.SumBoundary(geom.periodicity());
   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const auto lo = amrex::lbound(bx);
      const auto hi = amrex::ubound(bx);
      const auto s = state.array(mfi);
      const auto old_solid = old_mask.const_array(mfi);
      const auto new_solid = new_mask.const_array(mfi);
      const auto d = delta.const_array(mfi);
      for (int k = lo.z; k <= hi.z; ++k) {
         for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
               const bool became_fluid = old_solid(i, j, k, 0) != 0 && new_solid(i, j, k, 0) == 0;
               const bool became_solid = old_solid(i, j, k, 0) == 0 && new_solid(i, j, k, 0) != 0;
               for (int n = 0; n < ncomp; ++n) {
                  if (became_fluid || became_solid) {
                     s(i, j, k, n) = 0.0_rt;
                  }
                  s(i, j, k, n) += d(i, j, k, n);
               }
            }
         }
      }
   }
   state.FillBoundary(geom.periodicity());
   (void)old_time;
}

template <typename FractionArray>
inline amrex::Real outward_fluid_fraction_shell_weight_sum(
   FractionArray const& old_fraction,
   FractionArray const& new_fraction,
   amrex::Box const& fabbox,
   amrex::Box const& domain,
   int i,
   int j,
   int k,
   int radius,
   bool require_old_fluid,
   amrex::Real nx,
   amrex::Real ny,
   amrex::Real nz,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx
)
{
   constexpr amrex::Real eps = 1.0e-14_rt;
   const auto flo = amrex::lbound(fabbox);
   const auto fhi = amrex::ubound(fabbox);
   const auto dlo = amrex::lbound(domain);
   const auto dhi = amrex::ubound(domain);
   amrex::Real sum = 0.0_rt;
   for (int ox = -radius; ox <= radius; ++ox) {
#if (AMREX_SPACEDIM >= 2)
   for (int oy = -radius; oy <= radius; ++oy) {
#else
   const int oy = 0;
   {
#endif
#if (AMREX_SPACEDIM == 3)
   for (int oz = -radius; oz <= radius; ++oz) {
#else
   const int oz = 0;
   {
#endif
      if (
         std::abs(ox) != radius
#if (AMREX_SPACEDIM >= 2)
         && std::abs(oy) != radius
#endif
#if (AMREX_SPACEDIM == 3)
         && std::abs(oz) != radius
#endif
      ) {
         continue;
      }
      const int ci = i + ox;
      const int cj = j + oy;
      const int ck = k + oz;
      if (ci < flo.x || ci > fhi.x || ci < dlo.x || ci > dhi.x) {
         continue;
      }
#if (AMREX_SPACEDIM >= 2)
      if (cj < flo.y || cj > fhi.y || cj < dlo.y || cj > dhi.y) {
         continue;
      }
#endif
#if (AMREX_SPACEDIM == 3)
      if (ck < flo.z || ck > fhi.z || ck < dlo.z || ck > dhi.z) {
         continue;
      }
#endif
      const bool was_fluid = old_fraction(ci, cj, ck, 0) > eps;
      const bool is_fluid = new_fraction(ci, cj, ck, 0) > eps;
      if (is_fluid && (!require_old_fluid || was_fluid)) {
         const amrex::Real vx = static_cast<amrex::Real>(ox) * dx[0];
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real vy = static_cast<amrex::Real>(oy) * dx[1];
#else
         const amrex::Real vy = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real vz = static_cast<amrex::Real>(oz) * dx[2];
#else
         const amrex::Real vz = 0.0_rt;
#endif
         const amrex::Real projection = vx * nx + vy * ny + vz * nz;
         if (projection > 0.0_rt) {
            const amrex::Real dist2 = squared_norm(StateVector3{vx, vy, vz});
            sum += projection / amrex::max(dist2, dx[0] * dx[0]);
         }
      }
#if (AMREX_SPACEDIM == 3)
   }
#else
   }
#endif
#if (AMREX_SPACEDIM >= 2)
   }
#else
   }
#endif
   }
   return sum;
}

template <typename StateArray, typename DeltaArray, typename FractionArray>
inline void distribute_fraction_to_outward_fluid_shell(
   StateArray const& state,
   DeltaArray const& delta,
   FractionArray const& old_fraction,
   FractionArray const& new_fraction,
   amrex::Box const& fabbox,
   amrex::Box const& domain,
   int i,
   int j,
   int k,
   int radius,
   bool require_old_fluid,
   amrex::Real nx,
   amrex::Real ny,
   amrex::Real nz,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real inv_weight_sum,
   amrex::Real fluid_fraction,
   int ncomp
)
{
   constexpr amrex::Real eps = 1.0e-14_rt;
   const auto flo = amrex::lbound(fabbox);
   const auto fhi = amrex::ubound(fabbox);
   const auto dlo = amrex::lbound(domain);
   const auto dhi = amrex::ubound(domain);
   for (int ox = -radius; ox <= radius; ++ox) {
#if (AMREX_SPACEDIM >= 2)
   for (int oy = -radius; oy <= radius; ++oy) {
#else
   const int oy = 0;
   {
#endif
#if (AMREX_SPACEDIM == 3)
   for (int oz = -radius; oz <= radius; ++oz) {
#else
   const int oz = 0;
   {
#endif
      if (
         std::abs(ox) != radius
#if (AMREX_SPACEDIM >= 2)
         && std::abs(oy) != radius
#endif
#if (AMREX_SPACEDIM == 3)
         && std::abs(oz) != radius
#endif
      ) {
         continue;
      }
      const int ci = i + ox;
      const int cj = j + oy;
      const int ck = k + oz;
      if (ci < flo.x || ci > fhi.x || ci < dlo.x || ci > dhi.x) {
         continue;
      }
#if (AMREX_SPACEDIM >= 2)
      if (cj < flo.y || cj > fhi.y || cj < dlo.y || cj > dhi.y) {
         continue;
      }
#endif
#if (AMREX_SPACEDIM == 3)
      if (ck < flo.z || ck > fhi.z || ck < dlo.z || ck > dhi.z) {
         continue;
      }
#endif
      const bool was_fluid = old_fraction(ci, cj, ck, 0) > eps;
      const bool is_fluid = new_fraction(ci, cj, ck, 0) > eps;
      if (is_fluid && (!require_old_fluid || was_fluid)) {
         const amrex::Real vx = static_cast<amrex::Real>(ox) * dx[0];
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real vy = static_cast<amrex::Real>(oy) * dx[1];
#else
         const amrex::Real vy = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real vz = static_cast<amrex::Real>(oz) * dx[2];
#else
         const amrex::Real vz = 0.0_rt;
#endif
         const amrex::Real projection = vx * nx + vy * ny + vz * nz;
         if (projection > 0.0_rt) {
            const amrex::Real dist2 = squared_norm(StateVector3{vx, vy, vz});
            const amrex::Real weight = projection / amrex::max(dist2, dx[0] * dx[0]) * inv_weight_sum;
            for (int n = 0; n < ncomp; ++n) {
               const amrex::Real amount = weight * fluid_fraction * state(i, j, k, n);
               delta(i, j, k, n) -= amount;
               delta(ci, cj, ck, n) += amount;
            }
         }
      }
#if (AMREX_SPACEDIM == 3)
   }
#else
   }
#endif
#if (AMREX_SPACEDIM >= 2)
   }
#else
   }
#endif
   }
}

template <typename StateArray, typename DeltaArray, typename FractionArray>
inline void fill_fraction_from_outward_fluid_shell(
   StateArray const& state,
   DeltaArray const& delta,
   FractionArray const& old_fraction,
   FractionArray const& new_fraction,
   amrex::Box const& fabbox,
   amrex::Box const& domain,
   int i,
   int j,
   int k,
   int radius,
   bool require_old_fluid,
   amrex::Real nx,
   amrex::Real ny,
   amrex::Real nz,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real inv_weight_sum,
   amrex::Real fluid_fraction,
   int ncomp
)
{
   constexpr amrex::Real eps = 1.0e-14_rt;
   const auto flo = amrex::lbound(fabbox);
   const auto fhi = amrex::ubound(fabbox);
   const auto dlo = amrex::lbound(domain);
   const auto dhi = amrex::ubound(domain);
   for (int ox = -radius; ox <= radius; ++ox) {
#if (AMREX_SPACEDIM >= 2)
   for (int oy = -radius; oy <= radius; ++oy) {
#else
   const int oy = 0;
   {
#endif
#if (AMREX_SPACEDIM == 3)
   for (int oz = -radius; oz <= radius; ++oz) {
#else
   const int oz = 0;
   {
#endif
      if (
         std::abs(ox) != radius
#if (AMREX_SPACEDIM >= 2)
         && std::abs(oy) != radius
#endif
#if (AMREX_SPACEDIM == 3)
         && std::abs(oz) != radius
#endif
      ) {
         continue;
      }
      const int ci = i + ox;
      const int cj = j + oy;
      const int ck = k + oz;
      if (ci < flo.x || ci > fhi.x || ci < dlo.x || ci > dhi.x) {
         continue;
      }
#if (AMREX_SPACEDIM >= 2)
      if (cj < flo.y || cj > fhi.y || cj < dlo.y || cj > dhi.y) {
         continue;
      }
#endif
#if (AMREX_SPACEDIM == 3)
      if (ck < flo.z || ck > fhi.z || ck < dlo.z || ck > dhi.z) {
         continue;
      }
#endif
      const bool was_fluid = old_fraction(ci, cj, ck, 0) > eps;
      const bool is_fluid = new_fraction(ci, cj, ck, 0) > eps;
      if (is_fluid && (!require_old_fluid || was_fluid)) {
         const amrex::Real vx = static_cast<amrex::Real>(ox) * dx[0];
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real vy = static_cast<amrex::Real>(oy) * dx[1];
#else
         const amrex::Real vy = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real vz = static_cast<amrex::Real>(oz) * dx[2];
#else
         const amrex::Real vz = 0.0_rt;
#endif
         const amrex::Real projection = vx * nx + vy * ny + vz * nz;
         if (projection > 0.0_rt) {
            const amrex::Real dist2 = squared_norm(StateVector3{vx, vy, vz});
            const amrex::Real weight = projection / amrex::max(dist2, dx[0] * dx[0]) * inv_weight_sum;
            for (int n = 0; n < ncomp; ++n) {
               const amrex::Real amount = weight * fluid_fraction * state(ci, cj, ck, n);
               delta(i, j, k, n) += amount;
               delta(ci, cj, ck, n) -= amount;
            }
         }
      }
#if (AMREX_SPACEDIM == 3)
   }
#else
   }
#endif
#if (AMREX_SPACEDIM >= 2)
   }
#else
   }
#endif
   }
}

inline void conservative_pulse_solid_volume_fraction_update(
   amrex::MultiFab& state,
   amrex::MultiFab const& old_volume_fraction,
   amrex::MultiFab const& new_volume_fraction,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Real old_time,
   amrex::Real new_time
)
{
   if (!solid_geometry_enabled(cfg) || cfg.solid_motion != "pulse") {
      return;
   }

   constexpr amrex::Real eps = 1.0e-14_rt;
   const int ncomp = state.nComp();
   amrex::MultiFab delta(state.boxArray(), state.DistributionMap(), ncomp, SolidMotionSearchRadius);
   delta.setVal(0.0_rt);

   const amrex::Box domain = geom.Domain();
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const SolidGeometryEvaluator params = solid_geometry_evaluator(cfg, new_time);

   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const amrex::Box& fabbox = old_volume_fraction[mfi].box();
      const auto lo = amrex::lbound(bx);
      const auto hi = amrex::ubound(bx);
      const auto s = state.array(mfi);
      const auto old_fraction = old_volume_fraction.const_array(mfi);
      const auto new_fraction = new_volume_fraction.const_array(mfi);
      const auto d = delta.array(mfi);
      for (int k = lo.z; k <= hi.z; ++k) {
         for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
               const amrex::Real old_alpha = old_fraction(i, j, k, 0);
               const amrex::Real new_alpha = new_fraction(i, j, k, 0);
               const amrex::Real change = new_alpha - old_alpha;
               if (std::abs(change) <= eps) {
                  continue;
               }
               const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5_rt) * dx[0];
#if (AMREX_SPACEDIM >= 2)
               const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5_rt) * dx[1];
#else
               const amrex::Real y = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
               const amrex::Real z = problo[2] + (static_cast<amrex::Real>(k) + 0.5_rt) * dx[2];
#else
               const amrex::Real z = 0.0_rt;
#endif
               amrex::Real nx = 0.0_rt;
               amrex::Real ny = 0.0_rt;
               amrex::Real nz = 0.0_rt;
               params.normal_at(new_time, x, y, z, dx, nx, ny, nz);
               if (change < 0.0_rt) {
                  const amrex::Real lost = -change;
                  int receiver_radius = 0;
                  amrex::Real weight_sum = 0.0_rt;
                  for (int radius = 1; radius <= SolidMotionSearchRadius; ++radius) {
                     weight_sum = outward_fluid_fraction_shell_weight_sum(
                        old_fraction,
                        new_fraction,
                        fabbox,
                        domain,
                        i,
                        j,
                        k,
                        radius,
                        false,
                        nx,
                        ny,
                        nz,
                        dx
                     );
                     if (weight_sum > 0.0_rt) {
                        receiver_radius = radius;
                        break;
                     }
                  }
                  if (receiver_radius == 0) {
                     amrex::Abort(
                        "pulsing solid volume fraction decreased without an outward receiving fluid cell. "
                        "Reduce the timestep or keep the pulsing obstacle away from boundaries."
                     );
                  }
                  distribute_fraction_to_outward_fluid_shell(
                     s,
                     d,
                     old_fraction,
                     new_fraction,
                     fabbox,
                     domain,
                     i,
                     j,
                     k,
                     receiver_radius,
                     false,
                     nx,
                     ny,
                     nz,
                     dx,
                     1.0_rt / weight_sum,
                     lost,
                     ncomp
                  );
               } else {
                  int donor_radius = 0;
                  amrex::Real weight_sum = 0.0_rt;
                  for (int radius = 1; radius <= SolidMotionSearchRadius; ++radius) {
                     weight_sum = outward_fluid_fraction_shell_weight_sum(
                        old_fraction,
                        new_fraction,
                        fabbox,
                        domain,
                        i,
                        j,
                        k,
                        radius,
                        true,
                        nx,
                        ny,
                        nz,
                        dx
                     );
                     if (weight_sum > 0.0_rt) {
                        donor_radius = radius;
                        break;
                     }
                  }
                  if (donor_radius == 0) {
                     amrex::Abort(
                        "pulsing solid volume fraction increased without an outward fluid donor. "
                        "Reduce the timestep or keep the pulsing obstacle away from boundaries."
                     );
                  }
                  fill_fraction_from_outward_fluid_shell(
                     s,
                     d,
                     old_fraction,
                     new_fraction,
                     fabbox,
                     domain,
                     i,
                     j,
                     k,
                     donor_radius,
                     true,
                     nx,
                     ny,
                     nz,
                     dx,
                     1.0_rt / weight_sum,
                     change,
                     ncomp
                  );
               }
            }
         }
      }
   }

   delta.SumBoundary(geom.periodicity());
   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const auto lo = amrex::lbound(bx);
      const auto hi = amrex::ubound(bx);
      const auto s = state.array(mfi);
      const auto old_fraction = old_volume_fraction.const_array(mfi);
      const auto new_fraction = new_volume_fraction.const_array(mfi);
      const auto d = delta.const_array(mfi);
      for (int k = lo.z; k <= hi.z; ++k) {
         for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
               const amrex::Real old_alpha = old_fraction(i, j, k, 0);
               const amrex::Real new_alpha = new_fraction(i, j, k, 0);
               for (int n = 0; n < ncomp; ++n) {
                  const amrex::Real extensive = old_alpha * s(i, j, k, n) + d(i, j, k, n);
                  s(i, j, k, n) = new_alpha > eps ? extensive / new_alpha : 0.0_rt;
               }
            }
         }
      }
   }
   state.FillBoundary(geom.periodicity());
   (void)old_time;
}



} // namespace pyfluid_amrex
