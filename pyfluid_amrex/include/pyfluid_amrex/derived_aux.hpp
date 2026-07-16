#pragma once

#include <AMReX_Gpu.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <cmath>

#include "pyfluid_amrex/amr_indicator.hpp"
#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/kernels_generated/vector_ops.hpp"
#include "pyfluid_amrex/primitive_cache.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real velocity_component(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int component
) noexcept
{
   return state(i, j, k, MOM1 + component) / state(i, j, k, RHO);
}

inline void fill_vorticity_output(
   amrex::MultiFab& vorticity,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(vorticity, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto w = vorticity.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real dvz_dx =
            (velocity_component(s, i + 1, j, k, 2) - velocity_component(s, i - 1, j, k, 2)) / (2.0 * dx[0]);
         const amrex::Real dvy_dx =
            (velocity_component(s, i + 1, j, k, 1) - velocity_component(s, i - 1, j, k, 1)) / (2.0 * dx[0]);
         amrex::Real dvz_dy = 0.0;
         amrex::Real dvx_dy = 0.0;
         amrex::Real dvy_dz = 0.0;
         amrex::Real dvx_dz = 0.0;
#if (AMREX_SPACEDIM >= 2)
         dvz_dy =
            (velocity_component(s, i, j + 1, k, 2) - velocity_component(s, i, j - 1, k, 2)) / (2.0 * dx[1]);
         dvx_dy =
            (velocity_component(s, i, j + 1, k, 0) - velocity_component(s, i, j - 1, k, 0)) / (2.0 * dx[1]);
#endif
#if (AMREX_SPACEDIM == 3)
         dvy_dz =
            (velocity_component(s, i, j, k + 1, 1) - velocity_component(s, i, j, k - 1, 1)) / (2.0 * dx[2]);
         dvx_dz =
            (velocity_component(s, i, j, k + 1, 0) - velocity_component(s, i, j, k - 1, 0)) / (2.0 * dx[2]);
#endif
         w(i, j, k, 0) = generated::curl_x_cartesian(dvz_dy, dvy_dz);
         w(i, j, k, 1) = generated::curl_y_cartesian(dvx_dz, dvz_dx);
         w(i, j, k, 2) = generated::curl_z_cartesian(dvy_dx, dvx_dy);
      });
   }
   }
}

inline void fill_mach_number_output(
   amrex::MultiFab& mach,
   amrex::MultiFab const& state,
   amrex::Geometry const& geometry,
   amrex::Real time,
   RunConfig const& cfg
)
{
   const EOSParams eos = eos_params(cfg);
   const int eos_method = eos_method_code(cfg);
   const PhysicalSystem system = physical_system(cfg);
   amrex::MultiFab primitive(
      state.boxArray(), state.DistributionMap(), state.nComp(), state.nGrowVect()
   );
   fill_base_primitive_cache_runtime(
      primitive, state, eos, eos_method, system,
      geometry, fixed_spacetime_parameters(cfg), time
   );
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(mach, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto values = primitive.const_array(mfi);
      const auto out = mach.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real rho = values(i, j, k, RHO);
         const StateVector3 velocity = {
            values(i, j, k, MOM1),
            values(i, j, k, MOM2),
            values(i, j, k, MOM3),
         };
         const amrex::Real p = values(i, j, k, ENERGY);
         const amrex::Real cs = fluid_sound_speed_from_primitive_runtime(
            system, eos_method, rho, p, eos
         );
         out(i, j, k, 0) = std::sqrt(squared_norm(velocity)) / amrex::max(cs, 1.0e-300);
      });
   }
   }
}

inline void fill_plasma_beta_output(
   amrex::MultiFab& plasma_beta,
   amrex::MultiFab const& state,
   amrex::Geometry const& geometry,
   amrex::Real time,
   RunConfig const& cfg
)
{
   const EOSParams eos = eos_params(cfg);
   const int eos_method = eos_method_code(cfg);
   const PhysicalSystem system = physical_system(cfg);
   amrex::MultiFab primitive(
      state.boxArray(), state.DistributionMap(), state.nComp(), state.nGrowVect()
   );
   fill_base_primitive_cache_runtime(
      primitive, state, eos, eos_method, system,
      geometry, fixed_spacetime_parameters(cfg), time
   );
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(plasma_beta, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto values = primitive.const_array(mfi);
      const auto out = plasma_beta.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         out(i, j, k, 0) = generated::plasma_beta_from_pressure(
            values(i, j, k, ENERGY),
            1.0e-300,
            values(i, j, k, B1),
            values(i, j, k, B2),
            values(i, j, k, B3)
         );
      });
   }
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real entropy_proxy_gamma(int eos_method, EOSParams const& eos) noexcept
{
   if (eos_method == EOSIdealGas) {
      return eos.gamma;
   }
   return eos.segment_gamma[0];
}

inline void fill_entropy_proxy_output(
   amrex::MultiFab& entropy,
   amrex::MultiFab const& state,
   amrex::Geometry const& geometry,
   amrex::Real time,
   RunConfig const& cfg
)
{
   const EOSParams eos = eos_params(cfg);
   const int eos_method = eos_method_code(cfg);
   const PhysicalSystem system = physical_system(cfg);
   amrex::MultiFab primitive(
      state.boxArray(), state.DistributionMap(), state.nComp(), state.nGrowVect()
   );
   fill_base_primitive_cache_runtime(
      primitive, state, eos, eos_method, system,
      geometry, fixed_spacetime_parameters(cfg), time
   );
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(entropy, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto values = primitive.const_array(mfi);
      const auto out = entropy.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real rho = values(i, j, k, RHO);
         const amrex::Real p = values(i, j, k, ENERGY);
         out(i, j, k, 0) = p / std::pow(amrex::max(rho, 1.0e-300), entropy_proxy_gamma(eos_method, eos));
      });
   }
   }
}

inline void fill_amr_tag_metric_output(
   amrex::MultiFab& metric,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   amrex::Real time,
   RunConfig const& cfg
)
{
   const EOSParams eos = eos_params(cfg);
   const int eos_method = eos_method_code(cfg);
   const PhysicalSystem system = physical_system(cfg);
   amrex::MultiFab primitive(
      state.boxArray(), state.DistributionMap(), state.nComp(), state.nGrowVect()
   );
   fill_base_primitive_cache_runtime(
      primitive, state, eos, eos_method, system,
      geom, fixed_spacetime_parameters(cfg), time
   );
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const auto min_feature_lengths = configured_min_feature_lengths(cfg);
   const amrex::Real min_refine_radius = cfg.min_refine_radius;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(metric, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto values = primitive.const_array(mfi);
      const auto out = metric.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         out(i, j, k, 0) = auto_amr_indicator_from_primitive(
            values,
            i,
            j,
            k,
            eos,
            eos_method,
            system,
            mesh_coordinate,
            mesh_axes,
            problo,
            dx,
            min_feature_lengths,
            min_refine_radius
         );
      });
   }
   }
}

} // namespace pyfluid_amrex
