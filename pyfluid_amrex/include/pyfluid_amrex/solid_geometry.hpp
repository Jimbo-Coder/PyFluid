#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_TagBox.H>

#include <cmath>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/solid_geometry_point.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

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
