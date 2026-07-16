#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include <cmath>
#include <string>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/initial_data/sampling.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real passive_phase_coordinate(amrex::Real q, amrex::Real lo, amrex::Real hi, int is_periodic) noexcept
{
   const amrex::Real width = hi - lo;
   return is_periodic != 0 && width > 0.0 ? (q - lo) / width : q;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real passive_profile_value(
   int profile,
   InitialDataSamplePoint const& sample,
   amrex::Real value,
   amrex::Real background,
   amrex::Real amplitude,
   amrex::Real left,
   amrex::Real right,
   amrex::Real sigma,
   PatchCoordinate const& center,
   amrex::Real shock_x0,
   amrex::Real wave_number,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi,
   amrex::GpuArray<int, AMREX_SPACEDIM> const& periodic
) noexcept
{
   constexpr int ProfileUniform = 1;
   constexpr int ProfileGaussian = 2;
   constexpr int ProfileSinWave = 3;
   constexpr int ProfileStepX = 4;
   if (profile == ProfileUniform) {
      return value;
   }
   if (profile == ProfileGaussian) {
      const amrex::Real r2 = initial_data_physical_distance2(
         mesh_coordinate,
         mesh_axes,
         sample.mesh,
         sample.physical,
         center,
         problo,
         probhi,
         periodic
      );
      return background + amplitude * std::exp(-0.5 * r2 / (sigma * sigma));
   }
   if (profile == ProfileSinWave) {
      const amrex::Real two_pi = constants::two_pi;
      amrex::Real phase = 0.0;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         phase += passive_phase_coordinate(
            patch_coordinate_component(sample.mesh, dir),
            problo[dir],
            probhi[dir],
            periodic[dir]
         );
      }
      return background + amplitude * std::sin(two_pi * wave_number * phase);
   }
   if (profile == ProfileStepX) {
      return patch_coordinate_component(sample.mesh, 0) < shock_x0 ? left : right;
   }
   return 0.0;
}

inline int passive_profile_code(std::string const& name)
{
   if (name == "uniform") {
      return 1;
   }
   if (name == "gaussian") {
      return 2;
   }
   if (name == "sin_wave") {
      return 3;
   }
   if (name == "step_x") {
      return 4;
   }
   return 0;
}

inline amrex::Real passive_coordinate_or_default(amrex::Real value, amrex::Real fallback)
{
   return std::isfinite(value) ? value : fallback;
}

inline void initialize_passive_scalars(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   if (cfg.passive_scalars == 0) {
      return;
   }
   const int passive_begin = passive_component_begin(is_mhd(cfg));
   if (state.nComp() < passive_begin + cfg.passive_scalars) {
      amrex::Abort("passive scalar initialization needs a state vector with passive components");
   }
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const auto periodic = geom.isPeriodicArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const int profile = passive_profile_code(cfg.passive_profile);
   const amrex::Real sigma = cfg.passive_sigma > 0.0 ? cfg.passive_sigma : cfg.bump_sigma;
   const amrex::Real x0 = passive_coordinate_or_default(cfg.passive_x0, cfg.bump_x0);
   const amrex::Real y0 = passive_coordinate_or_default(cfg.passive_y0, cfg.bump_y0);
   const amrex::Real z0 = passive_coordinate_or_default(cfg.passive_z0, cfg.bump_z0);
   const PatchCoordinate center_mesh = make_patch_coordinate(x0, y0, z0);
   const PatchCoordinate center = mesh_coordinate_to_physical_point(mesh_coordinate, mesh_axes, center_mesh);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real value = cfg.passive_value;
      const amrex::Real background = cfg.passive_background;
      const amrex::Real amplitude = cfg.passive_amplitude;
      const amrex::Real left = cfg.passive_left;
      const amrex::Real right = cfg.passive_right;
      const amrex::Real shock_x0 = cfg.shock_x0;
      const amrex::Real wave_number = cfg.wave_number;
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const InitialDataSamplePoint sample = cell_center_initial_data_sample_point(
            mesh_coordinate,
            mesh_axes,
            problo,
            dx,
            i,
            j,
            k
         );
         const amrex::Real dye = passive_profile_value(
            profile,
            sample,
            value,
            background,
            amplitude,
            left,
            right,
            sigma,
            center,
            shock_x0,
            wave_number,
            mesh_coordinate,
            mesh_axes,
            problo,
            probhi,
            periodic
         );
         arr(i, j, k, passive_begin) = arr(i, j, k, RHO) * dye;
      });
   }
   }
}

} // namespace pyfluid_amrex
