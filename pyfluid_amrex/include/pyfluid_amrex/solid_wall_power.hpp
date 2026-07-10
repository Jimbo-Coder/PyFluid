#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

#include <array>

#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/solid_geometry.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

template <int Direction>
inline amrex::Real solid_wall_energy_power_direction(
   amrex::MultiFab const& flux,
   amrex::iMultiFab const& solid_mask,
   amrex::iMultiFab const& valid_cell_mask,
   amrex::Geometry const& geom
)
{
   const amrex::Real area = cartesian_face_area<Direction>(geom);

   amrex::MultiFab contribution(flux.boxArray(), flux.DistributionMap(), 1, 0);
   contribution.setVal(0.0);
   const amrex::Box& domain = geom.Domain();
   const auto lo = domain.smallEnd();
   const auto hi = domain.bigEnd();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(contribution, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto f = flux.const_array(mfi);
      const auto solid = solid_mask.const_array(mfi);
      const auto valid = valid_cell_mask.const_array(mfi);
      const auto out = contribution.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         int li = i;
         int lj = j;
         int lk = k;
         int ri = i;
         int rj = j;
         int rk = k;
         if constexpr (Direction == 0) {
            li = i - 1;
         }
#if (AMREX_SPACEDIM >= 2)
         else if constexpr (Direction == 1) {
            lj = j - 1;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         else {
            lk = k - 1;
         }
#endif

         bool left_in_domain = li >= lo[0] && li <= hi[0];
         bool right_in_domain = ri >= lo[0] && ri <= hi[0];
#if (AMREX_SPACEDIM >= 2)
         left_in_domain = left_in_domain && lj >= lo[1] && lj <= hi[1];
         right_in_domain = right_in_domain && rj >= lo[1] && rj <= hi[1];
#endif
#if (AMREX_SPACEDIM == 3)
         left_in_domain = left_in_domain && lk >= lo[2] && lk <= hi[2];
         right_in_domain = right_in_domain && rk >= lo[2] && rk <= hi[2];
#endif
         const bool left_solid = left_in_domain && solid(li, lj, lk, 0) != 0;
         const bool right_solid = right_in_domain && solid(ri, rj, rk, 0) != 0;
         const bool left_fluid = left_in_domain && !left_solid;
         const bool right_fluid = right_in_domain && !right_solid;
         const bool left_valid_fluid = left_fluid && valid(li, lj, lk, 0) != 0;
         const bool right_valid_fluid = right_fluid && valid(ri, rj, rk, 0) != 0;
         if (left_solid && right_valid_fluid) {
            out(i, j, k, 0) = f(i, j, k, ENERGY) * area;
         } else if (left_valid_fluid && right_solid) {
            out(i, j, k, 0) = -f(i, j, k, ENERGY) * area;
         } else {
            out(i, j, k, 0) = 0.0;
         }
      });
   }
   }
   return contribution.sum(0);
}

inline amrex::Real solid_wall_energy_power(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::iMultiFab const& solid_mask,
   amrex::iMultiFab const& valid_cell_mask,
   amrex::Geometry const& geom
)
{
   amrex::Real power = solid_wall_energy_power_direction<0>(*fluxes[0], solid_mask, valid_cell_mask, geom);
#if (AMREX_SPACEDIM >= 2)
   power += solid_wall_energy_power_direction<1>(*fluxes[1], solid_mask, valid_cell_mask, geom);
#endif
#if (AMREX_SPACEDIM == 3)
   power += solid_wall_energy_power_direction<2>(*fluxes[2], solid_mask, valid_cell_mask, geom);
#endif
   return power;
}

inline amrex::Real solid_wall_energy_power_cutcell(
   amrex::MultiFab const& primitive,
   amrex::iMultiFab const& solid_mask,
   amrex::MultiFab const& volume_fraction,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& apertures,
   amrex::iMultiFab const& valid_cell_mask,
   amrex::Geometry const& geom,
   SolidGeometryEvaluator const& solid_params
)
{
   if (
      solid_params.motion != SolidMotionTranslate
      && solid_params.motion != SolidMotionPulse
   ) {
      return 0.0;
   }
   constexpr amrex::Real alpha_floor = 1.0e-12;
   const auto dx = geom.CellSizeArray();
   const CartesianUniformMetric metric = make_cartesian_uniform_metric(dx);
   const amrex::Real area_x = metric.face_area[0];
#if (AMREX_SPACEDIM >= 2)
   const amrex::Real area_y = metric.face_area[1];
#endif
#if (AMREX_SPACEDIM == 3)
   const amrex::Real area_z = metric.face_area[2];
#endif
   const auto problo = geom.ProbLoArray();
   const amrex::Box& domain = geom.Domain();
   const auto lo = domain.smallEnd();
   const auto hi = domain.bigEnd();
   amrex::MultiFab contribution(primitive.boxArray(), primitive.DistributionMap(), 1, 0);
   contribution.setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(contribution, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto p = primitive.const_array(mfi);
      const auto solid = solid_mask.const_array(mfi);
      const auto valid = valid_cell_mask.const_array(mfi);
      const auto alpha = volume_fraction.const_array(mfi);
      const auto ax = apertures[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto ay = apertures[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto az = apertures[2]->const_array(mfi);
#endif
      const auto out = contribution.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (valid(i, j, k, 0) == 0 || solid(i, j, k, 0) != 0 || alpha(i, j, k, 0) <= alpha_floor) {
            out(i, j, k, 0) = 0.0;
            return;
         }
         amrex::Real ax_l = ax(i, j, k, 0);
         if (i > lo[0] && alpha(i - 1, j, k, 0) <= alpha_floor) {
            ax_l = 0.0;
         }
         amrex::Real ax_r = ax(i + 1, j, k, 0);
         if (i < hi[0] && alpha(i + 1, j, k, 0) <= alpha_floor) {
            ax_r = 0.0;
         }
         const amrex::Real sx = (ax_r - ax_l) * area_x;
#if (AMREX_SPACEDIM >= 2)
         amrex::Real ay_l = ay(i, j, k, 0);
         if (j > lo[1] && alpha(i, j - 1, k, 0) <= alpha_floor) {
            ay_l = 0.0;
         }
         amrex::Real ay_r = ay(i, j + 1, k, 0);
         if (j < hi[1] && alpha(i, j + 1, k, 0) <= alpha_floor) {
            ay_r = 0.0;
         }
         const amrex::Real sy = (ay_r - ay_l) * area_y;
#else
         const amrex::Real sy = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
         amrex::Real az_l = az(i, j, k, 0);
         if (k > lo[2] && alpha(i, j, k - 1, 0) <= alpha_floor) {
            az_l = 0.0;
         }
         amrex::Real az_r = az(i, j, k + 1, 0);
         if (k < hi[2] && alpha(i, j, k + 1, 0) <= alpha_floor) {
            az_r = 0.0;
         }
         const amrex::Real sz = (az_r - az_l) * area_z;
#else
         const amrex::Real sz = 0.0;
#endif
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
         const amrex::Real wall_vx = solid_params.velocity_component(solid_params.time, 0, x, y, z);
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real wall_vy = solid_params.velocity_component(solid_params.time, 1, x, y, z);
#else
         const amrex::Real wall_vy = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real wall_vz = solid_params.velocity_component(solid_params.time, 2, x, y, z);
#else
         const amrex::Real wall_vz = 0.0;
#endif
         out(i, j, k, 0) = p(i, j, k, ENERGY) * (wall_vx * sx + wall_vy * sy + wall_vz * sz);
      });
   }
   }
   return contribution.sum(0);
}


} // namespace pyfluid_amrex
