#pragma once

#include <AMReX_BC_TYPES.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/rhs.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

using GravityGradient = std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>;

inline amrex::Real gravity_poisson_factor(RunConfig const& cfg)
{
   constexpr amrex::Real pi = 3.141592653589793238462643383279502884;
   return 4.0 * pi * cfg.gravity_constant;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real fixed_gravity_potential(RunConfig const& cfg, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
   if (cfg.gravity_fixed_potential == "harmonic") {
      const amrex::Real omega2 = cfg.gravity_omega * cfg.gravity_omega;
      const amrex::Real dx = x - cfg.gravity_center_x;
      const amrex::Real dy = y - cfg.gravity_center_y;
      const amrex::Real dz = z - cfg.gravity_center_z;
      return 0.5 * omega2 * (dx * dx + dy * dy + dz * dz);
   }
   return -(cfg.gravity_a1 * x + cfg.gravity_a2 * y + cfg.gravity_a3 * z);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real fixed_gravity_gradient_component(
   RunConfig const& cfg,
   int dir,
   amrex::Real x,
   amrex::Real y,
   amrex::Real z
) noexcept
{
   if (cfg.gravity_fixed_potential == "harmonic") {
      const amrex::Real omega2 = cfg.gravity_omega * cfg.gravity_omega;
      if (dir == 0) {
         return omega2 * (x - cfg.gravity_center_x);
      }
      if (dir == 1) {
         return omega2 * (y - cfg.gravity_center_y);
      }
      return omega2 * (z - cfg.gravity_center_z);
   }
   if (dir == 0) {
      return -cfg.gravity_a1;
   }
   if (dir == 1) {
      return -cfg.gravity_a2;
   }
   return -cfg.gravity_a3;
}

inline void allocate_gravity_gradient(
   GravityGradient& gradient,
   amrex::BoxArray const& cell_ba,
   amrex::DistributionMapping const& dm
)
{
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const amrex::BoxArray face_ba = amrex::convert(cell_ba, amrex::IntVect::TheDimensionVector(dir));
      if (!gradient[dir] || !gradient[dir]->boxArray().CellEqual(face_ba)) {
         gradient[dir] = std::make_unique<amrex::MultiFab>(face_ba, dm, 1, 0);
      }
      gradient[dir]->setVal(0.0);
   }
}

inline void fill_fixed_gravity_potential_and_gradient(
   amrex::MultiFab& phi,
   GravityGradient& gradient,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
   allocate_gravity_gradient(gradient, phi.boxArray(), phi.DistributionMap());
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
      for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto out = phi.array(mfi);
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
            out(i, j, k, 0) = fixed_gravity_potential(cfg, x, y, z);
         });
      }
   }

   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::MultiFab& component = *gradient[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
         for (amrex::MFIter mfi(component, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const auto out = component.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
               amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
#else
               amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
               amrex::Real z = problo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
#else
               amrex::Real z = 0.0;
#endif
               if (dir == 0) {
                  x = problo[0] + static_cast<amrex::Real>(i) * dx[0];
               }
#if (AMREX_SPACEDIM >= 2)
               if (dir == 1) {
                  y = problo[1] + static_cast<amrex::Real>(j) * dx[1];
               }
#endif
#if (AMREX_SPACEDIM == 3)
               if (dir == 2) {
                  z = problo[2] + static_cast<amrex::Real>(k) * dx[2];
               }
#endif
               out(i, j, k, 0) = fixed_gravity_gradient_component(cfg, dir, x, y, z);
            });
         }
      }
   }
}

inline amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> gravity_gradient_ptrs(GravityGradient& gradient)
{
   amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> ptrs{};
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      ptrs[dir] = gradient[dir].get();
   }
   return ptrs;
}

inline std::array<amrex::MultiFab const*, AMREX_SPACEDIM> gravity_gradient_const_ptrs(GravityGradient const& gradient)
{
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> ptrs{};
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      ptrs[dir] = gradient[dir].get();
   }
   return ptrs;
}

inline void fill_cell_centered_gravity_gradient(
   amrex::MultiFab& output,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& gravity_gradient
)
{
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (gravity_gradient[dir] == nullptr) {
         amrex::Abort("grad_phi output requested before gravity gradient was computed");
      }
   }
   output.setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
      for (amrex::MFIter mfi(output, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto out = output.array(mfi);
         const auto grad_x = gravity_gradient[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
         const auto grad_y = gravity_gradient[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
         const auto grad_z = gravity_gradient[2]->const_array(mfi);
#endif
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            out(i, j, k, 0) = 0.5 * (grad_x(i, j, k, 0) + grad_x(i + 1, j, k, 0));
#if (AMREX_SPACEDIM >= 2)
            out(i, j, k, 1) = 0.5 * (grad_y(i, j, k, 0) + grad_y(i, j + 1, k, 0));
#endif
#if (AMREX_SPACEDIM == 3)
            out(i, j, k, 2) = 0.5 * (grad_z(i, j, k, 0) + grad_z(i, j, k + 1, 0));
#endif
         });
      }
   }
}

inline void fill_gravity_poisson_rhs(
   amrex::MultiFab& rhs,
   amrex::MultiFab const& state,
   amrex::iMultiFab const& valid_mask,
   amrex::Real mean_density,
   RunConfig const& cfg
)
{
   const amrex::Real factor = gravity_poisson_factor(cfg);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
      for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto out = rhs.array(mfi);
         const auto s = state.const_array(mfi);
         const auto valid = valid_mask.const_array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            out(i, j, k, 0) = valid(i, j, k, 0) != 0
               ? factor * (s(i, j, k, RHO) - mean_density)
               : 0.0;
         });
      }
   }
}

template <int NComp, bool AddEnergySource = true>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void add_gravity_source_cell(
   amrex::Array4<amrex::Real> const& rhs,
   amrex::Array4<amrex::Real const> const& state,
   amrex::Array4<amrex::Real const> const& grad_x,
#if (AMREX_SPACEDIM >= 2)
   amrex::Array4<amrex::Real const> const& grad_y,
#endif
#if (AMREX_SPACEDIM == 3)
   amrex::Array4<amrex::Real const> const& grad_z,
#endif
   std::array<amrex::Real, AMREX_SPACEDIM> const& acceleration_offset,
   int i,
   int j,
   int k
) noexcept
{
   const amrex::Real gx = -0.5 * (grad_x(i, j, k, 0) + grad_x(i + 1, j, k, 0)) - acceleration_offset[0];
#if (AMREX_SPACEDIM >= 2)
   const amrex::Real gy = -0.5 * (grad_y(i, j, k, 0) + grad_y(i, j + 1, k, 0)) - acceleration_offset[1];
#else
   constexpr amrex::Real gy = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
   const amrex::Real gz = -0.5 * (grad_z(i, j, k, 0) + grad_z(i, j, k + 1, 0)) - acceleration_offset[2];
#else
   constexpr amrex::Real gz = 0.0;
#endif
   const amrex::Real rho = state(i, j, k, RHO);
   const StateVector3 momentum = load_momentum_vector3(state, i, j, k);
   const StateVector3 acceleration = {gx, gy, gz};
   rhs(i, j, k, MOM1) += rho * gx;
#if (AMREX_SPACEDIM >= 2)
   rhs(i, j, k, MOM2) += rho * gy;
#endif
#if (AMREX_SPACEDIM == 3)
   rhs(i, j, k, MOM3) += rho * gz;
#endif
   if constexpr (AddEnergySource) {
      rhs(i, j, k, ENERGY) += dot(momentum, acceleration);
   }
   (void)NComp;
}

template <int NComp>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void add_athena_gravity_energy_source_cell(
   amrex::Array4<amrex::Real> const& rhs,
   std::array<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& flux,
   amrex::Array4<amrex::Real const> const& grad_x,
#if (AMREX_SPACEDIM >= 2)
   amrex::Array4<amrex::Real const> const& grad_y,
#endif
#if (AMREX_SPACEDIM == 3)
   amrex::Array4<amrex::Real const> const& grad_z,
#endif
   std::array<amrex::Real, AMREX_SPACEDIM> const& acceleration_offset,
   int i,
   int j,
   int k
) noexcept
{
   amrex::Real source =
      flux[0](i + 1, j, k, RHO) * (-grad_x(i + 1, j, k, 0) - acceleration_offset[0])
      + flux[0](i, j, k, RHO) * (-grad_x(i, j, k, 0) - acceleration_offset[0]);
#if (AMREX_SPACEDIM >= 2)
   source +=
      flux[1](i, j + 1, k, RHO) * (-grad_y(i, j + 1, k, 0) - acceleration_offset[1])
      + flux[1](i, j, k, RHO) * (-grad_y(i, j, k, 0) - acceleration_offset[1]);
#endif
#if (AMREX_SPACEDIM == 3)
   source +=
      flux[2](i, j, k + 1, RHO) * (-grad_z(i, j, k + 1, 0) - acceleration_offset[2])
      + flux[2](i, j, k, RHO) * (-grad_z(i, j, k, 0) - acceleration_offset[2]);
#endif
   rhs(i, j, k, ENERGY) += 0.5 * source;
   (void)NComp;
}

template <int NComp>
inline void apply_athena_gravity_energy_source(
   amrex::MultiFab& rhs,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& fluxes,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& gravity_gradient,
   std::array<amrex::Real, AMREX_SPACEDIM> const& acceleration_offset,
   amrex::iMultiFab const* solid_mask = nullptr,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (fluxes[dir] == nullptr || gravity_gradient[dir] == nullptr) {
         amrex::Abort("Athena gravity coupling requires hydro/MHD mass fluxes and face gravity");
      }
   }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
      for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto r = rhs.array(mfi);
         std::array<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> flux{};
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            flux[dir] = fluxes[dir]->const_array(mfi);
         }
         const auto grad_x = gravity_gradient[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
         const auto grad_y = gravity_gradient[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
         const auto grad_z = gravity_gradient[2]->const_array(mfi);
#endif
         const auto accel = acceleration_offset;

         if (solid_mask == nullptr) {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               add_athena_gravity_energy_source_cell<NComp>(
                  r,
                  flux,
                  grad_x,
#if (AMREX_SPACEDIM >= 2)
                  grad_y,
#endif
#if (AMREX_SPACEDIM == 3)
                  grad_z,
#endif
                  accel,
                  i,
                  j,
                  k
               );
            });
            continue;
         }

         const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, *solid_mask, bx);
         if (kind == MaskKindNonzero) {
            continue;
         }
         if (kind == MaskKindZero) {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               add_athena_gravity_energy_source_cell<NComp>(
                  r,
                  flux,
                  grad_x,
#if (AMREX_SPACEDIM >= 2)
                  grad_y,
#endif
#if (AMREX_SPACEDIM == 3)
                  grad_z,
#endif
                  accel,
                  i,
                  j,
                  k
               );
            });
            continue;
         }

         const auto solid = solid_mask->const_array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (solid(i, j, k, 0) == 0) {
               add_athena_gravity_energy_source_cell<NComp>(
                  r,
                  flux,
                  grad_x,
#if (AMREX_SPACEDIM >= 2)
                  grad_y,
#endif
#if (AMREX_SPACEDIM == 3)
                  grad_z,
#endif
                  accel,
                  i,
                  j,
                  k
               );
            }
         });
      }
   }
}

template <int NComp>
inline void apply_gravity_source(
   amrex::MultiFab& rhs,
   amrex::MultiFab const& state,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& gravity_gradient,
   std::array<amrex::Real, AMREX_SPACEDIM> const& acceleration_offset,
   amrex::iMultiFab const* solid_mask = nullptr,
   amrex::Vector<int> const* solid_valid_kind = nullptr,
   bool use_athena_energy_source = false,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const* fluxes = nullptr,
   bool suppress_energy_source = false
)
{
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (gravity_gradient[dir] == nullptr) {
         amrex::Abort("gravity source requested before gravity gradient was computed");
      }
   }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
      for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto r = rhs.array(mfi);
         const auto s = state.const_array(mfi);
         const auto grad_x = gravity_gradient[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
         const auto grad_y = gravity_gradient[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
         const auto grad_z = gravity_gradient[2]->const_array(mfi);
#endif
         const auto accel = acceleration_offset;
         if (solid_mask == nullptr) {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if (use_athena_energy_source || suppress_energy_source) {
                  add_gravity_source_cell<NComp, false>(
                     r,
                     s,
                     grad_x,
#if (AMREX_SPACEDIM >= 2)
                     grad_y,
#endif
#if (AMREX_SPACEDIM == 3)
                     grad_z,
#endif
                     accel,
                     i,
                     j,
                     k
                  );
               } else {
                  add_gravity_source_cell<NComp>(
                     r,
                     s,
                     grad_x,
#if (AMREX_SPACEDIM >= 2)
                     grad_y,
#endif
#if (AMREX_SPACEDIM == 3)
                     grad_z,
#endif
                     accel,
                     i,
                     j,
                     k
                  );
               }
            });
            continue;
         }

         const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, *solid_mask, bx);
         if (kind == MaskKindNonzero) {
            continue;
         }
         if (kind == MaskKindZero) {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if (use_athena_energy_source || suppress_energy_source) {
                  add_gravity_source_cell<NComp, false>(
                     r,
                     s,
                     grad_x,
#if (AMREX_SPACEDIM >= 2)
                     grad_y,
#endif
#if (AMREX_SPACEDIM == 3)
                     grad_z,
#endif
                     accel,
                     i,
                     j,
                     k
                  );
               } else {
                  add_gravity_source_cell<NComp>(
                     r,
                     s,
                     grad_x,
#if (AMREX_SPACEDIM >= 2)
                     grad_y,
#endif
#if (AMREX_SPACEDIM == 3)
                     grad_z,
#endif
                     accel,
                     i,
                     j,
                     k
                  );
               }
            });
            continue;
         }

         const auto solid = solid_mask->const_array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (solid(i, j, k, 0) == 0) {
               if (use_athena_energy_source || suppress_energy_source) {
                  add_gravity_source_cell<NComp, false>(
                     r,
                     s,
                     grad_x,
#if (AMREX_SPACEDIM >= 2)
                     grad_y,
#endif
#if (AMREX_SPACEDIM == 3)
                     grad_z,
#endif
                     accel,
                     i,
                     j,
                     k
                  );
               } else {
                  add_gravity_source_cell<NComp>(
                     r,
                     s,
                     grad_x,
#if (AMREX_SPACEDIM >= 2)
                     grad_y,
#endif
#if (AMREX_SPACEDIM == 3)
                     grad_z,
#endif
                     accel,
                     i,
                     j,
                     k
                  );
               }
            }
         });
      }
   }
   if (use_athena_energy_source) {
      if (fluxes == nullptr) {
         amrex::Abort("Athena gravity coupling requested without mass fluxes");
      }
      apply_athena_gravity_energy_source<NComp>(
         rhs,
         *fluxes,
         gravity_gradient,
         acceleration_offset,
         solid_mask,
         solid_valid_kind
      );
   }
}

} // namespace pyfluid_amrex
