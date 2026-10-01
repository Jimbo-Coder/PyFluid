#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_TagBox.H>

#include <cmath>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/physical_geometry.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

static constexpr int SolidMotionStatic = 0;
static constexpr int SolidMotionTranslate = 1;
static constexpr int SolidMotionRotate = 2;
static constexpr int SolidMotionPulse = 3;
static constexpr int SolidMotionSearchRadius = 4;

inline bool solid_geometry_enabled(RunConfig const& cfg)
{
   return (cfg.solid_geometry == "boxes" && cfg.solid_box_count > 0)
      || cfg.solid_geometry == "wedge"
      || cfg.solid_geometry == "circle";
}

inline bool solid_geometry_changes_cell_mask(RunConfig const& cfg)
{
   return solid_geometry_enabled(cfg) && (cfg.solid_motion == "translate" || cfg.solid_motion == "pulse");
}

inline int solid_translation_axis(RunConfig const& cfg)
{
   const amrex::Real eps = 0.0_rt;
   int axis = -1;
   int count = 0;
   if (std::abs(cfg.solid_velocity_1) > eps) {
      axis = 0;
      ++count;
   }
#if (AMREX_SPACEDIM >= 2)
   if (std::abs(cfg.solid_velocity_2) > eps) {
      axis = 1;
      ++count;
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (std::abs(cfg.solid_velocity_3) > eps) {
      axis = 2;
      ++count;
   }
#endif
   return count == 1 ? axis : -1;
}

inline int solid_translation_sign(RunConfig const& cfg)
{
   const int axis = solid_translation_axis(cfg);
   if (axis == 0) {
      return cfg.solid_velocity_1 > 0.0_rt ? 1 : -1;
   }
#if (AMREX_SPACEDIM >= 2)
   if (axis == 1) {
      return cfg.solid_velocity_2 > 0.0_rt ? 1 : -1;
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (axis == 2) {
      return cfg.solid_velocity_3 > 0.0_rt ? 1 : -1;
   }
#endif
   return 0;
}

struct SolidGeometryEvaluator {
   bool boxes_enabled = false;
   bool wedge_enabled = false;
   bool circle_enabled = false;
   int motion = SolidMotionStatic;
   amrex::Real time = 0.0_rt;
   int box_count = 0;
   amrex::Real box_lo[MaxSolidBoxValues]{};
   amrex::Real box_hi[MaxSolidBoxValues]{};
   amrex::Real wedge_x0 = 0.0_rt;
   amrex::Real wedge_y0 = 0.0_rt;
   amrex::Real wedge_slope = 0.0_rt;
   amrex::Real circle_center[3]{};
   amrex::Real circle_radius = 0.0_rt;
   amrex::Real circle_radius_amplitude = 0.0_rt;
   amrex::Real circle_radius_frequency = 0.0_rt;
   amrex::Real circle_radius_phase = 0.0_rt;
   amrex::Real circle_radius_ramp_time = 0.0_rt;
   int volume_fraction_samples_per_dim = 1;
   amrex::Real velocity[3]{};
   amrex::Real angular_velocity[3]{};

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real circle_radius_envelope(amrex::Real query_time) const noexcept
   {
      if (circle_radius_ramp_time <= 0.0_rt) {
         return 1.0_rt;
      }
      if (query_time <= 0.0_rt) {
         return 0.0_rt;
      }
      if (query_time >= circle_radius_ramp_time) {
         return 1.0_rt;
      }
      const amrex::Real u = query_time / circle_radius_ramp_time;
      return u * u * u * (10.0_rt + u * (-15.0_rt + 6.0_rt * u));
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real circle_radius_envelope_velocity(amrex::Real query_time) const noexcept
   {
      if (circle_radius_ramp_time <= 0.0_rt || query_time <= 0.0_rt || query_time >= circle_radius_ramp_time) {
         return 0.0_rt;
      }
      const amrex::Real u = query_time / circle_radius_ramp_time;
      return 30.0_rt * u * u * (1.0_rt - u) * (1.0_rt - u) / circle_radius_ramp_time;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real translation_shift(amrex::Real query_time, int dir) const noexcept
   {
      return motion == SolidMotionTranslate ? velocity[dir] * query_time : 0.0_rt;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real circle_radius_at(amrex::Real query_time) const noexcept
   {
      if (motion != SolidMotionPulse) {
         return circle_radius;
      }
      const amrex::Real phase = circle_radius_phase
         + constants::two_pi
         * circle_radius_frequency
         * query_time;
      return circle_radius + circle_radius_amplitude * circle_radius_envelope(query_time) * std::sin(phase);
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real circle_radius_velocity_at(amrex::Real query_time) const noexcept
   {
      if (motion != SolidMotionPulse) {
         return 0.0_rt;
      }
      const amrex::Real omega =
         constants::two_pi
         * circle_radius_frequency;
      const amrex::Real phase = circle_radius_phase + omega * query_time;
      const amrex::Real envelope = circle_radius_envelope(query_time);
      const amrex::Real envelope_velocity = circle_radius_envelope_velocity(query_time);
      return circle_radius_amplitude
         * (envelope_velocity * std::sin(phase) + envelope * omega * std::cos(phase));
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real center_component_at(amrex::Real query_time, int dir) const noexcept
   {
      return circle_center[dir] + translation_shift(query_time, dir);
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real box_phi_at(amrex::Real query_time, int box, amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
   {
      const int base = box * AMREX_SPACEDIM;
      const amrex::Real xlo = box_lo[base] + translation_shift(query_time, 0);
      const amrex::Real xhi = box_hi[base] + translation_shift(query_time, 0);
      const amrex::Real dx = x < xlo ? xlo - x : (x > xhi ? x - xhi : 0.0_rt);
      amrex::Real outside2 = dx * dx;
      amrex::Real inside = amrex::min(x - xlo, xhi - x);
#if (AMREX_SPACEDIM >= 2)
      const amrex::Real ylo = box_lo[base + 1] + translation_shift(query_time, 1);
      const amrex::Real yhi = box_hi[base + 1] + translation_shift(query_time, 1);
      const amrex::Real dy = y < ylo ? ylo - y : (y > yhi ? y - yhi : 0.0_rt);
      outside2 += dy * dy;
      inside = amrex::min(inside, amrex::min(y - ylo, yhi - y));
#endif
#if (AMREX_SPACEDIM == 3)
      const amrex::Real zlo = box_lo[base + 2] + translation_shift(query_time, 2);
      const amrex::Real zhi = box_hi[base + 2] + translation_shift(query_time, 2);
      const amrex::Real dz = z < zlo ? zlo - z : (z > zhi ? z - zhi : 0.0_rt);
      outside2 += dz * dz;
      inside = amrex::min(inside, amrex::min(z - zlo, zhi - z));
#else
      (void)z;
#endif
      if (outside2 > 0.0_rt) {
         return std::sqrt(outside2);
      }
      return -inside;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real circle_phi_at(amrex::Real query_time, amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
   {
      const amrex::Real dx = x - center_component_at(query_time, 0);
#if (AMREX_SPACEDIM >= 2)
      const amrex::Real dy = y - center_component_at(query_time, 1);
#else
      const amrex::Real dy = 0.0_rt;
      (void)y;
#endif
#if (AMREX_SPACEDIM == 3)
      const amrex::Real dz = z - center_component_at(query_time, 2);
#else
      const amrex::Real dz = 0.0_rt;
      (void)z;
#endif
      return cartesian_radius(dx, dy, dz) - circle_radius_at(query_time);
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real wedge_phi_at(amrex::Real query_time, amrex::Real x, amrex::Real y) const noexcept
   {
      const amrex::Real shifted_x0 = wedge_x0 + translation_shift(query_time, 0);
      const amrex::Real shifted_y0 = wedge_y0 + translation_shift(query_time, 1);
      if (x < shifted_x0) {
         const amrex::Real dx = shifted_x0 - x;
         const amrex::Real dy = y < shifted_y0 ? shifted_y0 - y : 0.0_rt;
         return cartesian_cylindrical_radius(dx, dy);
      }
      const amrex::Real wall_y = shifted_y0 + wedge_slope * (x - shifted_x0);
      return (y - wall_y) / std::sqrt(1.0_rt + wedge_slope * wedge_slope);
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real phi(amrex::Real query_time, amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
   {
      amrex::Real value = 1.0e300_rt;
      if (boxes_enabled) {
         for (int n = 0; n < box_count; ++n) {
            value = amrex::min(value, box_phi_at(query_time, n, x, y, z));
         }
      }
      if (circle_enabled) {
         value = amrex::min(value, circle_phi_at(query_time, x, y, z));
      }
      if (wedge_enabled) {
#if (AMREX_SPACEDIM >= 2)
         value = amrex::min(value, wedge_phi_at(query_time, x, y));
#endif
      }
      return value;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real curved_boundary_error(amrex::Real query_time, amrex::Real cell_width) const noexcept
   {
#if (AMREX_SPACEDIM >= 2)
      amrex::Real error = 0.0_rt;
      const amrex::Real radius = circle_radius_at(query_time);
      if (circle_enabled && radius > 0.0_rt) {
         amrex::Real cell_diagonal = cell_width * std::sqrt(2.0_rt);
#if (AMREX_SPACEDIM == 3)
         cell_diagonal = cell_width * std::sqrt(3.0_rt);
#endif
         const amrex::Real half_chord = 0.5_rt * cell_diagonal;
         const amrex::Real sagitta = half_chord >= radius
            ? radius
            : radius - std::sqrt(radius * radius - half_chord * half_chord);
         error = amrex::max(error, sagitta);
      }
      return error;
#else
      (void)query_time;
      (void)cell_width;
      return 0.0_rt;
#endif
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   bool contains(amrex::Real query_time, amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
   {
      return phi(query_time, x, y, z) <= 0.0_rt;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   void normal_at(
      amrex::Real query_time,
      amrex::Real x,
      amrex::Real y,
      amrex::Real z,
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
      amrex::Real& nx,
      amrex::Real& ny,
      amrex::Real& nz
   ) const noexcept
   {
      const amrex::Real hx = 0.25_rt * dx[0];
      amrex::Real gx = (phi(query_time, x + hx, y, z) - phi(query_time, x - hx, y, z)) / (2.0_rt * hx);
#if (AMREX_SPACEDIM >= 2)
      const amrex::Real hy = 0.25_rt * dx[1];
      amrex::Real gy = (phi(query_time, x, y + hy, z) - phi(query_time, x, y - hy, z)) / (2.0_rt * hy);
#else
      amrex::Real gy = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
      const amrex::Real hz = 0.25_rt * dx[2];
      amrex::Real gz = (phi(query_time, x, y, z + hz) - phi(query_time, x, y, z - hz)) / (2.0_rt * hz);
#else
      amrex::Real gz = 0.0_rt;
#endif
      const amrex::Real norm = std::sqrt(gx * gx + gy * gy + gz * gz);
      if (norm > 0.0_rt && amrex::Math::isfinite(norm)) {
         nx = gx / norm;
         ny = gy / norm;
         nz = gz / norm;
      } else {
         nx = 1.0_rt;
         ny = 0.0_rt;
         nz = 0.0_rt;
      }
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real velocity_component(amrex::Real query_time, int dir, amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
   {
      (void)x;
      (void)y;
      (void)z;
      if (motion == SolidMotionTranslate) {
         return velocity[dir];
      }
      if (motion == SolidMotionRotate) {
         const amrex::Real rx = x - center_component_at(query_time, 0);
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real ry = y - center_component_at(query_time, 1);
#else
         const amrex::Real ry = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real rz = z - center_component_at(query_time, 2);
#else
         const amrex::Real rz = 0.0_rt;
#endif
         const amrex::Real vx = angular_velocity[1] * rz - angular_velocity[2] * ry;
         const amrex::Real vy = angular_velocity[2] * rx - angular_velocity[0] * rz;
         const amrex::Real vz = angular_velocity[0] * ry - angular_velocity[1] * rx;
         return dir == 0 ? vx : (dir == 1 ? vy : vz);
      }
      if (motion == SolidMotionPulse) {
         const amrex::Real rx = x - center_component_at(query_time, 0);
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real ry = y - center_component_at(query_time, 1);
#else
         const amrex::Real ry = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real rz = z - center_component_at(query_time, 2);
#else
         const amrex::Real rz = 0.0_rt;
#endif
         const amrex::Real radius = cartesian_radius(rx, ry, rz);
         if (radius <= 0.0_rt) {
            return 0.0_rt;
         }
         const amrex::Real nx = rx / radius;
         const amrex::Real ny = ry / radius;
         const amrex::Real nz = rz / radius;
         return circle_radius_velocity_at(query_time) * (dir == 0 ? nx : (dir == 1 ? ny : nz));
      }
      return 0.0_rt;
   }

};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sampled_cell_fluid_fraction(
   SolidGeometryEvaluator const& params,
   amrex::Real query_time,
   amrex::Real x0,
   amrex::Real y0,
   amrex::Real z0,
   amrex::Real dx,
   amrex::Real dy,
   amrex::Real dz,
   int samples_per_dim
) noexcept
{
   const amrex::Real inv_samples = 1.0_rt / static_cast<amrex::Real>(samples_per_dim);
   amrex::Real inv_total = inv_samples;
#if (AMREX_SPACEDIM >= 2)
   inv_total *= inv_samples;
#else
   (void)y0;
   (void)dy;
#endif
#if (AMREX_SPACEDIM == 3)
   inv_total *= inv_samples;
#else
   (void)z0;
   (void)dz;
#endif
   amrex::Real fluid_samples = 0.0_rt;
   for (int si = 0; si < samples_per_dim; ++si) {
      const amrex::Real x = x0 + (static_cast<amrex::Real>(si) + 0.5_rt) * inv_samples * dx;
#if (AMREX_SPACEDIM >= 2)
      for (int sj = 0; sj < samples_per_dim; ++sj) {
         const amrex::Real y = y0 + (static_cast<amrex::Real>(sj) + 0.5_rt) * inv_samples * dy;
#else
      {
         const amrex::Real y = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
         for (int sk = 0; sk < samples_per_dim; ++sk) {
            const amrex::Real z = z0 + (static_cast<amrex::Real>(sk) + 0.5_rt) * inv_samples * dz;
#else
         {
            const amrex::Real z = 0.0_rt;
#endif
            if (!params.contains(query_time, x, y, z)) {
               fluid_samples += 1.0_rt;
            }
         }
      }
   }
   return fluid_samples * inv_total;
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sampled_face_fluid_fraction(
   SolidGeometryEvaluator const& params,
   amrex::Real query_time,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int samples_per_dim
) noexcept
{
   const amrex::Real inv_samples = 1.0_rt / static_cast<amrex::Real>(samples_per_dim);
   constexpr int nx = Direction == 0 ? 1 : 0;
#if (AMREX_SPACEDIM >= 2)
   constexpr int ny = Direction == 1 ? 1 : 0;
#endif
#if (AMREX_SPACEDIM == 3)
   constexpr int nz = Direction == 2 ? 1 : 0;
#endif
   const int sx_count = nx == 1 ? 1 : samples_per_dim;
#if (AMREX_SPACEDIM >= 2)
   const int sy_count = ny == 1 ? 1 : samples_per_dim;
#else
   constexpr int sy_count = 1;
#endif
#if (AMREX_SPACEDIM == 3)
   const int sz_count = nz == 1 ? 1 : samples_per_dim;
#else
   constexpr int sz_count = 1;
#endif
   const amrex::Real inv_total = 1.0_rt / static_cast<amrex::Real>(sx_count * sy_count * sz_count);
   amrex::Real fluid_samples = 0.0_rt;
   for (int si = 0; si < sx_count; ++si) {
      const amrex::Real x = Direction == 0
         ? problo[0] + static_cast<amrex::Real>(i) * dx[0]
         : problo[0] + (static_cast<amrex::Real>(i) + (static_cast<amrex::Real>(si) + 0.5_rt) * inv_samples) * dx[0];
#if (AMREX_SPACEDIM >= 2)
      for (int sj = 0; sj < sy_count; ++sj) {
         const amrex::Real y = Direction == 1
            ? problo[1] + static_cast<amrex::Real>(j) * dx[1]
            : problo[1] + (static_cast<amrex::Real>(j) + (static_cast<amrex::Real>(sj) + 0.5_rt) * inv_samples) * dx[1];
#else
      {
         const amrex::Real y = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
         for (int sk = 0; sk < sz_count; ++sk) {
            const amrex::Real z = Direction == 2
               ? problo[2] + static_cast<amrex::Real>(k) * dx[2]
               : problo[2] + (static_cast<amrex::Real>(k) + (static_cast<amrex::Real>(sk) + 0.5_rt) * inv_samples) * dx[2];
#else
         {
            const amrex::Real z = 0.0_rt;
#endif
            if (!params.contains(query_time, x, y, z)) {
               fluid_samples += 1.0_rt;
            }
         }
      }
   }
   return fluid_samples * inv_total;
}

inline SolidGeometryEvaluator solid_geometry_evaluator(RunConfig const& cfg, amrex::Real time)
{
   SolidGeometryEvaluator p;
   if (!solid_geometry_enabled(cfg)) {
      return p;
   }
   p.time = time;
   if (cfg.solid_motion == "translate") {
      p.motion = SolidMotionTranslate;
   } else if (cfg.solid_motion == "rotate") {
      p.motion = SolidMotionRotate;
   } else if (cfg.solid_motion == "pulse") {
      p.motion = SolidMotionPulse;
   } else {
      p.motion = SolidMotionStatic;
   }
   p.velocity[0] = cfg.solid_velocity_1;
   p.velocity[1] = cfg.solid_velocity_2;
   p.velocity[2] = cfg.solid_velocity_3;
   p.angular_velocity[0] = cfg.solid_angular_velocity_1;
   p.angular_velocity[1] = cfg.solid_angular_velocity_2;
   p.angular_velocity[2] = cfg.solid_angular_velocity_3;
   p.boxes_enabled = cfg.solid_geometry == "boxes";
   p.wedge_enabled = cfg.solid_geometry == "wedge";
   p.circle_enabled = cfg.solid_geometry == "circle";
   p.box_count = cfg.solid_box_count;
   for (int n = 0; n < cfg.solid_box_count * AMREX_SPACEDIM; ++n) {
      p.box_lo[n] = cfg.solid_box_lo[n];
      p.box_hi[n] = cfg.solid_box_hi[n];
   }
   p.wedge_x0 = cfg.solid_wedge_x0;
   p.wedge_y0 = cfg.solid_wedge_y0;
   p.wedge_slope = std::tan(cfg.solid_wedge_angle_deg * constants::radians_per_degree);
   p.circle_center[0] = cfg.solid_center_x;
   p.circle_center[1] = cfg.solid_center_y;
   p.circle_center[2] = cfg.solid_center_z;
   p.circle_radius = cfg.solid_radius;
   p.circle_radius_amplitude = cfg.solid_radius_amplitude;
   p.circle_radius_frequency = cfg.solid_radius_frequency;
   p.circle_radius_phase = cfg.solid_radius_phase;
   p.circle_radius_ramp_time = cfg.solid_radius_ramp_time;
   p.volume_fraction_samples_per_dim = cfg.solid_volume_fraction_samples_per_dim;
   return p;
}

inline void fill_solid_mask(amrex::iMultiFab& mask, amrex::Geometry const& geom, RunConfig const& cfg, amrex::Real time = 0.0_rt)
{
   mask.setVal(0);
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
   for (amrex::MFIter mfi(mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.fabbox();
      const auto m = mask.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
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
         m(i, j, k, 0) = params.contains(time, x, y, z) ? 1 : 0;
      });
   }
   }
}



} // namespace pyfluid_amrex
