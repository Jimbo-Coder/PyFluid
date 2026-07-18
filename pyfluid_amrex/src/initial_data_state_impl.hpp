#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <cmath>
#include <string>

#include "pyfluid_amrex/boundary.hpp"
#include "pyfluid_amrex/characteristic_basis.hpp"
#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/initial_data/common.hpp"
#include "pyfluid_amrex/initial_data/passive_scalars.hpp"
#include "pyfluid_amrex/initial_data/rhd.hpp"
#include "pyfluid_amrex/initial_data/sampling.hpp"
#include "pyfluid_amrex/kernels_generated/characteristic_transforms.hpp"
#include "pyfluid_amrex/patch_grid_config.hpp"
#include "pyfluid_amrex/patch_metrics.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

inline void initialize_uniform(amrex::MultiFab& state, RunConfig const& cfg)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho0 = cfg.rho0;
      const amrex::Real pressure0 = cfg.pressure0;
      const amrex::Real v1_0 = cfg.v1_0;
      const amrex::Real v2_0 = cfg.v2_0;
      const amrex::Real v3_0 = cfg.v3_0;
      const amrex::Real b1_0 = cfg.b1_0;
      const amrex::Real b2_0 = cfg.b2_0;
      const amrex::Real b3_0 = cfg.b3_0;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);
      const bool mhd = state.nComp() >= NVAR_MHD;
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         arr(i, j, k, RHO) = rho0;
         arr(i, j, k, MOM1) = rho0 * v1_0;
         arr(i, j, k, MOM2) = rho0 * v2_0;
         arr(i, j, k, MOM3) = rho0 * v3_0;
         if (mhd) {
            arr(i, j, k, ENERGY) = total_mhd_energy_runtime(eos_method, rho0, v1_0, v2_0, v3_0, pressure0, b1_0, b2_0, b3_0, eos);
            arr(i, j, k, B1) = b1_0;
            arr(i, j, k, B2) = b2_0;
            arr(i, j, k, B3) = b3_0;
         } else {
            arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho0, v1_0, v2_0, v3_0, pressure0, eos);
         }
      });
   }
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate cartesian_velocity_to_mesh_basis(
   int mesh_coordinate,
   MeshAxes const& axes,
   PatchCoordinate const& mesh,
   amrex::Real vx,
   amrex::Real vy,
   amrex::Real vz
) noexcept
{
   PatchCoordinate velocity{};
   if (mesh_coordinate == MeshCartesian) {
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         const int axis = mesh_axis(axes, dir);
         amrex::Real value = 0.0;
         if (axis == MeshAxisX) {
            value = vx;
         } else if (axis == MeshAxisY) {
            value = vy;
         } else if (axis == MeshAxisZ) {
            value = vz;
         }
         set_patch_coordinate_component(velocity, dir, value);
      }
      return velocity;
   }

   if (mesh_coordinate == MeshCylindrical) {
      const amrex::Real phi = initial_data_coordinate_for_axis(axes, mesh, MeshAxisPhi, 0.0);
      const amrex::Real cphi = std::cos(phi);
      const amrex::Real sphi = std::sin(phi);
      const amrex::Real vr = vx * cphi + vy * sphi;
      const amrex::Real vphi = -vx * sphi + vy * cphi;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         const int axis = mesh_axis(axes, dir);
         amrex::Real value = 0.0;
         if (axis == MeshAxisR) {
            value = vr;
         } else if (axis == MeshAxisPhi) {
            value = vphi;
         } else if (axis == MeshAxisZ) {
            value = vz;
         }
         set_patch_coordinate_component(velocity, dir, value);
      }
      return velocity;
   }

   if (mesh_coordinate == MeshSpherical) {
      const amrex::Real theta = initial_data_coordinate_for_axis(
         axes,
         mesh,
         MeshAxisTheta,
         constants::half_pi
      );
      const amrex::Real phi = initial_data_coordinate_for_axis(axes, mesh, MeshAxisPhi, 0.0);
      const amrex::Real ctheta = std::cos(theta);
      const amrex::Real stheta = std::sin(theta);
      const amrex::Real cphi = std::cos(phi);
      const amrex::Real sphi = std::sin(phi);
      const amrex::Real vr = vx * stheta * cphi + vy * stheta * sphi + vz * ctheta;
      const amrex::Real vtheta = vx * ctheta * cphi + vy * ctheta * sphi - vz * stheta;
      const amrex::Real vphi = -vx * sphi + vy * cphi;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         const int axis = mesh_axis(axes, dir);
         amrex::Real value = 0.0;
         if (axis == MeshAxisR) {
            value = vr;
         } else if (axis == MeshAxisTheta) {
            value = vtheta;
         } else if (axis == MeshAxisPhi) {
            value = vphi;
         }
         set_patch_coordinate_component(velocity, dir, value);
      }
      return velocity;
   }

   return velocity;
}

inline void initialize_cartesian_advection(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho0 = cfg.rho0;
      const amrex::Real pressure0 = cfg.pressure0;
      const amrex::Real vx = cfg.v1_0;
      const amrex::Real vy = cfg.v2_0;
      const amrex::Real vz = cfg.v3_0;
      const amrex::Real b1_0 = cfg.b1_0;
      const amrex::Real b2_0 = cfg.b2_0;
      const amrex::Real b3_0 = cfg.b3_0;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);
      const bool mhd = state.nComp() >= NVAR_MHD;
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
         const PatchCoordinate velocity =
            cartesian_velocity_to_mesh_basis(mesh_coordinate, mesh_axes, sample.mesh, vx, vy, vz);
         const amrex::Real v1 = velocity.q1;
         const amrex::Real v2 = velocity.q2;
         const amrex::Real v3 = velocity.q3;
         arr(i, j, k, RHO) = rho0;
         arr(i, j, k, MOM1) = rho0 * v1;
         arr(i, j, k, MOM2) = rho0 * v2;
         arr(i, j, k, MOM3) = rho0 * v3;
         if (mhd) {
            arr(i, j, k, ENERGY) =
               total_mhd_energy_runtime(eos_method, rho0, v1, v2, v3, pressure0, b1_0, b2_0, b3_0, eos);
            arr(i, j, k, B1) = b1_0;
            arr(i, j, k, B2) = b2_0;
            arr(i, j, k, B3) = b3_0;
         } else {
            arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho0, v1, v2, v3, pressure0, eos);
         }
      });
   }
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real magnetic_blast_noise_ax(amrex::Real, amrex::Real y, amrex::Real z, amrex::Real a0, amrex::Real k1) noexcept
{
   return a0 * std::sin(k1 * y) * std::sin(2.0 * k1 * z);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real magnetic_blast_noise_ay(amrex::Real x, amrex::Real, amrex::Real z, amrex::Real a0, amrex::Real k1) noexcept
{
   return 0.7 * a0 * std::sin(2.0 * k1 * z) * std::sin(k1 * x);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real magnetic_blast_noise_az(amrex::Real x, amrex::Real y, amrex::Real, amrex::Real a0, amrex::Real k1) noexcept
{
   return 0.5 * a0 * std::sin(k1 * x) * std::sin(2.0 * k1 * y);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real magnetic_blast_noise_bx(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real a0, amrex::Real k1) noexcept
{
   return a0 * k1 * std::sin(k1 * x) * std::cos(2.0 * k1 * y)
      - 1.4 * a0 * k1 * std::cos(2.0 * k1 * z) * std::sin(k1 * x);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real magnetic_blast_noise_by(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real a0, amrex::Real k1) noexcept
{
   return 2.0 * a0 * k1 * std::sin(k1 * y) * std::cos(2.0 * k1 * z)
      - 0.5 * a0 * k1 * std::cos(k1 * x) * std::sin(2.0 * k1 * y);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real magnetic_blast_noise_bz(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real a0, amrex::Real k1) noexcept
{
   return 0.7 * a0 * k1 * std::sin(2.0 * k1 * z) * std::cos(k1 * x)
      - a0 * k1 * std::cos(k1 * y) * std::sin(2.0 * k1 * z);
}

inline void initialize_sine_wave(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const amrex::Real two_pi = constants::two_pi;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho0 = cfg.rho0;
      const amrex::Real pressure0 = cfg.pressure0;
      const amrex::Real amplitude = cfg.amplitude;
      const amrex::Real wave_number = cfg.wave_number;
      const amrex::Real wave_mode_1 = cfg.wave_mode_1;
      const amrex::Real wave_mode_2 = cfg.wave_mode_2;
      const amrex::Real wave_mode_3 = cfg.wave_mode_3;
      const amrex::Real v1_0 = cfg.v1_0;
      const amrex::Real v2_0 = cfg.v2_0;
      const amrex::Real v3_0 = cfg.v3_0;
      const amrex::Real b1_0 = cfg.b1_0;
      const amrex::Real b2_0 = cfg.b2_0;
      const amrex::Real b3_0 = cfg.b3_0;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);
      const bool mhd = state.nComp() >= NVAR_MHD;

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
         const amrex::Real phase = two_pi * wave_number
            * wave_phase_coordinate(sample.mesh, problo, probhi, wave_mode_1, wave_mode_2, wave_mode_3);
         const amrex::Real rho = rho0 + amplitude * std::sin(phase);
         arr(i, j, k, RHO) = rho;
         arr(i, j, k, MOM1) = rho * v1_0;
         arr(i, j, k, MOM2) = rho * v2_0;
         arr(i, j, k, MOM3) = rho * v3_0;
         if (mhd) {
            arr(i, j, k, ENERGY) = total_mhd_energy_runtime(eos_method, rho, v1_0, v2_0, v3_0, pressure0, b1_0, b2_0, b3_0, eos);
            arr(i, j, k, B1) = b1_0;
            arr(i, j, k, B2) = b2_0;
            arr(i, j, k, B3) = b3_0;
         } else {
            arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho, v1_0, v2_0, v3_0, pressure0, eos);
         }
      });
   }
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real reconstruction_medley_profile(amrex::Real phase) noexcept
{
   phase -= std::floor(phase);
   const int segment = amrex::min(static_cast<int>(4.0 * phase), 3);
   const amrex::Real local = 4.0 * phase - static_cast<amrex::Real>(segment);
   if (segment == 0) {
      const amrex::Real sine = std::sin(constants::pi * local);
      return sine * sine;
   }
   if (segment == 1) {
      return local >= 0.2 && local <= 0.8 ? 1.0 : 0.0;
   }
   if (segment == 2) {
      return 1.0 - std::abs(2.0 * local - 1.0);
   }

   constexpr amrex::Real sigma = 0.12;
   const amrex::Real z = (local - 0.5) / sigma;
   const amrex::Real edge = std::exp(-0.5 * (0.5 / sigma) * (0.5 / sigma));
   return amrex::max((std::exp(-0.5 * z * z) - edge) / (1.0 - edge), 0.0);
}

inline void initialize_advection_medley(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const amrex::Real domain_length = probhi[0] - problo[0];
   const amrex::Real rho0 = cfg.rho0;
   const amrex::Real pressure0 = cfg.pressure0;
   const amrex::Real amplitude = cfg.amplitude;
   const amrex::Real velocity = cfg.v1_0;
   const EOSParams eos = eos_params(cfg);
   const int eos_method = eos_method_code(cfg);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         amrex::ignore_unused(j, k);
         const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
         const amrex::Real rho = rho0 + amplitude * reconstruction_medley_profile((x - problo[0]) / domain_length);
         arr(i, j, k, RHO) = rho;
         arr(i, j, k, MOM1) = rho * velocity;
         arr(i, j, k, MOM2) = 0.0;
         arr(i, j, k, MOM3) = 0.0;
         arr(i, j, k, ENERGY) = total_hydro_energy_runtime(
            eos_method,
            rho,
            velocity,
            0.0,
            0.0,
            pressure0,
            eos
         );
      });
   }
   }
}

inline void initialize_characteristic_medley(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const amrex::Real domain_length = probhi[0] - problo[0];
   const amrex::Real rho0 = cfg.rho0;
   const amrex::Real pressure0 = cfg.pressure0;
   const amrex::Real amplitude = cfg.amplitude;
   const EOSParams eos = eos_params(cfg);
   const int eos_method = eos_method_code(cfg);
   const amrex::Real sound_speed = sound_speed_from_primitive_runtime(eos_method, rho0, pressure0, eos);
   const amrex::Real sound2 = sound_speed * sound_speed;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         amrex::ignore_unused(j, k);
         const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
         const amrex::Real phase = (x - problo[0]) / domain_length;
         const amrex::Real sound_minus = amplitude * reconstruction_medley_profile(phase);
         const amrex::Real entropy = amplitude * reconstruction_medley_profile(phase + 1.0 / 3.0);
         const amrex::Real sound_plus = amplitude * reconstruction_medley_profile(phase + 2.0 / 3.0);
         const amrex::Real rho = rho0 + sound_minus + entropy + sound_plus;
         const amrex::Real velocity = sound_speed * (sound_plus - sound_minus) / rho0;
         const amrex::Real pressure = pressure0 + sound2 * (sound_minus + sound_plus);
         arr(i, j, k, RHO) = rho;
         arr(i, j, k, MOM1) = rho * velocity;
         arr(i, j, k, MOM2) = 0.0;
         arr(i, j, k, MOM3) = 0.0;
         arr(i, j, k, ENERGY) = total_hydro_energy_runtime(
            eos_method,
            rho,
            velocity,
            0.0,
            0.0,
            pressure,
            eos
         );
      });
   }
   }
}

inline void initialize_mhd_characteristic_medley(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const amrex::Real domain_length = probhi[0] - problo[0];
   const amrex::Real rho0 = cfg.rho0;
   const amrex::Real pressure0 = cfg.pressure0;
   const amrex::Real amplitude = cfg.amplitude;
   const amrex::Real b1_0 = cfg.b1_0;
   const amrex::Real b2_0 = cfg.b2_0;
   const amrex::Real b3_0 = cfg.b3_0;
   const EOSParams eos = eos_params(cfg);
   const MHDCharacteristicBasis basis = mhd_characteristic_basis<EOSIdealGas>(
      rho0,
      pressure0,
      b1_0,
      b2_0,
      b3_0,
      eos
   );

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         amrex::ignore_unused(j, k);
         const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
         const amrex::Real phase = (x - problo[0]) / domain_length;
         const amrex::Real phase_step = 1.0 / 7.0;
         const MHDCharacteristics waves{
            amplitude * reconstruction_medley_profile(phase),
            amplitude * reconstruction_medley_profile(phase + phase_step),
            amplitude * reconstruction_medley_profile(phase + 2.0 * phase_step),
            amplitude * reconstruction_medley_profile(phase + 3.0 * phase_step),
            amplitude * reconstruction_medley_profile(phase + 4.0 * phase_step),
            amplitude * reconstruction_medley_profile(phase + 5.0 * phase_step),
            amplitude * reconstruction_medley_profile(phase + 6.0 * phase_step),
         };
         const MHDReducedPrimitive perturbation =
            generated::mhd_reduced_from_characteristics(waves, basis);
         const amrex::Real rho = rho0 + perturbation.rho;
         const amrex::Real v1 = perturbation.vn;
         const amrex::Real v2 = perturbation.vt1;
         const amrex::Real v3 = perturbation.vt2;
         const amrex::Real pressure = pressure0 + perturbation.pressure;
         const amrex::Real b2 = b2_0 + perturbation.bt1;
         const amrex::Real b3 = b3_0 + perturbation.bt2;
         arr(i, j, k, RHO) = rho;
         arr(i, j, k, MOM1) = rho * v1;
         arr(i, j, k, MOM2) = rho * v2;
         arr(i, j, k, MOM3) = rho * v3;
         arr(i, j, k, ENERGY) = total_mhd_energy<EOSIdealGas>(
            rho,
            v1,
            v2,
            v3,
            pressure,
            b1_0,
            b2,
            b3,
            eos
         );
         arr(i, j, k, B1) = b1_0;
         arr(i, j, k, B2) = b2;
         arr(i, j, k, B3) = b3;
      });
   }
   }
}

inline void initialize_linear_wave_x(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const amrex::Real two_pi = constants::two_pi;
   const bool mhd = state.nComp() >= NVAR_MHD;
   const EOSParams eos = eos_params(cfg);
   const int eos_method = eos_method_code(cfg);
   const amrex::Real rho0 = cfg.rho0;
   const amrex::Real pressure0 = base_pressure_runtime(eos_method, rho0, cfg.pressure0, eos);
   const amrex::Real cs = sound_speed_from_primitive_runtime(eos_method, rho0, pressure0, eos);
   const amrex::Real sound2 = cs * cs;
   const amrex::Real amplitude = cfg.amplitude;
   const amrex::Real wave_number = cfg.wave_number;
   const amrex::Real v1_0 = cfg.v1_0;
   const amrex::Real v2_0 = cfg.v2_0;
   const amrex::Real v3_0 = cfg.v3_0;
   const amrex::Real b1_0 = cfg.b1_0;
   const amrex::Real b2_0 = cfg.b2_0;
   const amrex::Real b3_0 = cfg.b3_0;
   const bool family_acoustic_plus = cfg.wave_family == "acoustic_plus";
   const bool family_acoustic_minus = cfg.wave_family == "acoustic_minus";
   const bool family_entropy = cfg.wave_family == "entropy";
   const bool family_shear_y = cfg.wave_family == "shear_y";
   const bool family_shear_z = cfg.wave_family == "shear_z";
   const bool family_fast_plus = cfg.wave_family == "fast_plus";
   const bool family_fast_minus = cfg.wave_family == "fast_minus";
   const bool family_alfven_plus = cfg.wave_family == "alfven_plus";
   const bool family_alfven_minus = cfg.wave_family == "alfven_minus";
   const bool family_slow_plus = cfg.wave_family == "slow_plus";
   const bool family_slow_minus = cfg.wave_family == "slow_minus";

   amrex::Real mhd_fast = 0.0;
   amrex::Real mhd_slow = 0.0;
   amrex::Real mhd_alfven = 0.0;
   amrex::Real mhd_denominator = 1.0;
   amrex::Real alfven_ey = 1.0;
   amrex::Real alfven_ez = 0.0;
   if (mhd) {
      const amrex::Real va2 = (b1_0 * b1_0 + b2_0 * b2_0 + b3_0 * b3_0) / rho0;
      const amrex::Real can2 = b1_0 * b1_0 / rho0;
      const amrex::Real disc = amrex::max((sound2 + va2) * (sound2 + va2) - 4.0 * sound2 * can2, 0.0);
      mhd_fast = std::sqrt(0.5 * (sound2 + va2 + std::sqrt(disc)));
      mhd_slow = std::sqrt(amrex::max(0.5 * (sound2 + va2 - std::sqrt(disc)), 0.0));
      mhd_alfven = amrex::Math::abs(b1_0) / std::sqrt(rho0);
      const amrex::Real bt = std::sqrt(b2_0 * b2_0 + b3_0 * b3_0);
      if (bt > 0.0) {
         alfven_ey = -b3_0 / bt;
         alfven_ez = b2_0 / bt;
      }
      const bool magnetosonic = family_fast_plus || family_fast_minus || family_slow_plus || family_slow_minus;
      if (magnetosonic) {
         const amrex::Real cabs = (family_fast_plus || family_fast_minus) ? mhd_fast : mhd_slow;
         mhd_denominator = cabs * cabs - can2;
         if (amrex::Math::abs(mhd_denominator) < 1.0e-14) {
            amrex::Abort("linear_wave_x magnetosonic family is degenerate for this background field");
         }
      }
      if ((family_alfven_plus || family_alfven_minus) && mhd_alfven <= 0.0) {
         amrex::Abort("linear_wave_x Alfven family requires nonzero b1");
      }
   }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);

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
         const amrex::Real q1 = patch_coordinate_component(sample.mesh, 0);
         const amrex::Real profile = std::sin(two_pi * wave_number * q1);
         amrex::Real drho = 0.0;
         amrex::Real dvx = 0.0;
         amrex::Real dvy = 0.0;
         amrex::Real dvz = 0.0;
         amrex::Real dpressure = 0.0;
         amrex::Real dbx = 0.0;
         amrex::Real dby = 0.0;
         amrex::Real dbz = 0.0;

         if (!mhd) {
            if (family_acoustic_plus || family_acoustic_minus) {
               const amrex::Real c = family_acoustic_plus ? cs : -cs;
               drho = amplitude * profile;
               dvx = c * amplitude * profile / rho0;
               dpressure = sound2 * amplitude * profile;
            } else if (family_entropy) {
               drho = amplitude * profile;
            } else if (family_shear_y) {
               dvy = amplitude * profile;
            } else if (family_shear_z) {
               dvz = amplitude * profile;
            }
         } else {
            if (family_fast_plus || family_fast_minus || family_slow_plus || family_slow_minus) {
               const amrex::Real cabs = (family_fast_plus || family_fast_minus) ? mhd_fast : mhd_slow;
               const amrex::Real c = (family_fast_plus || family_slow_plus) ? cabs : -cabs;
               dvx = amplitude * profile;
               drho = rho0 * amplitude * profile / c;
               dpressure = rho0 * sound2 * amplitude * profile / c;
               dby = b2_0 * c * amplitude * profile / mhd_denominator;
               dbz = b3_0 * c * amplitude * profile / mhd_denominator;
               dvy = -b1_0 * b2_0 * amplitude * profile / (rho0 * mhd_denominator);
               dvz = -b1_0 * b3_0 * amplitude * profile / (rho0 * mhd_denominator);
            } else if (family_alfven_plus || family_alfven_minus) {
               const amrex::Real c = family_alfven_plus ? mhd_alfven : -mhd_alfven;
               dby = amplitude * alfven_ey * profile;
               dbz = amplitude * alfven_ez * profile;
               dvy = -b1_0 * dby / (rho0 * c);
               dvz = -b1_0 * dbz / (rho0 * c);
            } else if (family_entropy) {
               drho = amplitude * profile;
            }
         }

         const amrex::Real rho = rho0 + drho;
         const amrex::Real vx = v1_0 + dvx;
         const amrex::Real vy = v2_0 + dvy;
         const amrex::Real vz = v3_0 + dvz;
         const amrex::Real pressure = pressure0 + dpressure;
         arr(i, j, k, RHO) = rho;
         arr(i, j, k, MOM1) = rho * vx;
         arr(i, j, k, MOM2) = rho * vy;
         arr(i, j, k, MOM3) = rho * vz;
         if (mhd) {
            const amrex::Real bxv = b1_0 + dbx;
            const amrex::Real byv = b2_0 + dby;
            const amrex::Real bzv = b3_0 + dbz;
            arr(i, j, k, ENERGY) = total_mhd_energy_runtime(eos_method, rho, vx, vy, vz, pressure, bxv, byv, bzv, eos);
            arr(i, j, k, B1) = bxv;
            arr(i, j, k, B2) = byv;
            arr(i, j, k, B3) = bzv;
         } else {
            arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho, vx, vy, vz, pressure, eos);
         }
      });
   }
   }
}

inline void initialize_acoustic_wave_x(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const amrex::Real two_pi = constants::two_pi;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho0 = cfg.rho0;
      const amrex::Real pressure_input = cfg.pressure0;
      const amrex::Real amplitude = cfg.amplitude;
      const amrex::Real wave_number = cfg.wave_number;
      const amrex::Real v1_0 = cfg.v1_0;
      const amrex::Real v2_0 = cfg.v2_0;
      const amrex::Real v3_0 = cfg.v3_0;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);
      const amrex::Real pressure0 = base_pressure_runtime(eos_method, rho0, pressure_input, eos);
      const amrex::Real cs0 = sound_speed_from_primitive_runtime(eos_method, rho0, pressure0, eos);

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
         const amrex::Real q1 = patch_coordinate_component(sample.mesh, 0);
         const amrex::Real phase = two_pi * wave_number * q1;
         const amrex::Real drho = amplitude * std::sin(phase);
         const amrex::Real rho = rho0 + drho;
         const amrex::Real pressure = pressure0 + cs0 * cs0 * drho;
         const amrex::Real vx = v1_0 + cs0 * drho / rho0;
         arr(i, j, k, RHO) = rho;
         arr(i, j, k, MOM1) = rho * vx;
         arr(i, j, k, MOM2) = rho * v2_0;
         arr(i, j, k, MOM3) = rho * v3_0;
         arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho, vx, v2_0, v3_0, pressure, eos);
      });
   }
   }
}

inline void initialize_acoustic_packet_x(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const EOSParams eos = eos_params(cfg);
   const int eos_method = eos_method_code(cfg);
   const amrex::Real rho0 = cfg.rho0;
   const amrex::Real pressure0 = base_pressure_runtime(eos_method, rho0, cfg.pressure0, eos);
   const amrex::Real sound_speed = sound_speed_from_primitive_runtime(eos_method, rho0, pressure0, eos);
   const amrex::Real sound_speed2 = sound_speed * sound_speed;
   const amrex::Real amplitude = cfg.amplitude;
   const amrex::Real center_x = cfg.bump_x0;
   const amrex::Real center_y = cfg.bump_y0;
   const amrex::Real center_z = cfg.bump_z0;
   const amrex::Real inv_sigma2 = 1.0 / (cfg.bump_sigma * cfg.bump_sigma);
   const amrex::Real inv_transverse_sigma2 = cfg.acoustic_transverse_sigma > 0.0
      ? 1.0 / (cfg.acoustic_transverse_sigma * cfg.acoustic_transverse_sigma)
      : 0.0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
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
         const amrex::Real delta_x = sample.physical.q1 - center_x;
         amrex::Real transverse_distance2 = 0.0;
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real delta_y = sample.physical.q2 - center_y;
         transverse_distance2 += delta_y * delta_y;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real delta_z = sample.physical.q3 - center_z;
         transverse_distance2 += delta_z * delta_z;
#endif
         const amrex::Real density_perturbation = amplitude * std::exp(
            -0.5 * (delta_x * delta_x * inv_sigma2 + transverse_distance2 * inv_transverse_sigma2)
         );
         const amrex::Real rho = rho0 + density_perturbation;
         const amrex::Real pressure = pressure0 + sound_speed2 * density_perturbation;
         const amrex::Real velocity_x = sound_speed * density_perturbation / rho0;
         const PatchCoordinate velocity = cartesian_velocity_to_mesh_basis(
            mesh_coordinate,
            mesh_axes,
            sample.mesh,
            velocity_x,
            0.0,
            0.0
         );
         const amrex::Real v1 = velocity.q1;
         const amrex::Real v2 = velocity.q2;
         const amrex::Real v3 = velocity.q3;
         arr(i, j, k, RHO) = rho;
         arr(i, j, k, MOM1) = rho * v1;
         arr(i, j, k, MOM2) = rho * v2;
         arr(i, j, k, MOM3) = rho * v3;
         arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho, v1, v2, v3, pressure, eos);
      });
   }
   }
}

inline void initialize_jeans_wave_x(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const amrex::Real two_pi = constants::two_pi;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho0 = cfg.rho0;
      const amrex::Real pressure_input = cfg.pressure0;
      const amrex::Real amplitude = cfg.amplitude;
      const amrex::Real wave_number = cfg.wave_number;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);
      const amrex::Real pressure0 = base_pressure_runtime(eos_method, rho0, pressure_input, eos);
      const amrex::Real cs0 = sound_speed_from_primitive_runtime(eos_method, rho0, pressure0, eos);
      const amrex::Real kx = two_pi * wave_number;
      const amrex::Real omega2 =
         cs0 * cs0 * kx * kx - 4.0 * constants::pi * cfg.gravity_constant * rho0;
      const bool stable = omega2 >= 0.0;
      const amrex::Real rate = stable ? std::sqrt(omega2) : std::sqrt(-omega2);

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
         const amrex::Real q1 = patch_coordinate_component(sample.mesh, 0);
         const amrex::Real phase = kx * q1;
         const amrex::Real drho = amplitude * std::sin(phase);
         const amrex::Real rho = rho0 + drho;
         const amrex::Real pressure = pressure0 + cs0 * cs0 * drho;
         const amrex::Real vx = stable
            ? rate * amplitude * std::sin(phase) / (rho0 * kx)
            : rate * amplitude * std::cos(phase) / (rho0 * kx);
         arr(i, j, k, RHO) = rho;
         arr(i, j, k, MOM1) = rho * vx;
         arr(i, j, k, MOM2) = 0.0;
         arr(i, j, k, MOM3) = 0.0;
         arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho, vx, 0.0, 0.0, pressure, eos);
      });
   }
   }
}

inline void initialize_shear_wave_x(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const amrex::Real two_pi = constants::two_pi;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho0 = cfg.rho0;
      const amrex::Real pressure0 = cfg.pressure0;
      const amrex::Real amplitude = cfg.amplitude;
      const amrex::Real wave_number = cfg.wave_number;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);

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
         const amrex::Real q1 = patch_coordinate_component(sample.mesh, 0);
         const amrex::Real phase = two_pi * wave_number * q1;
         const amrex::Real vy = amplitude * std::sin(phase);
         arr(i, j, k, RHO) = rho0;
         arr(i, j, k, MOM1) = 0.0;
         arr(i, j, k, MOM2) = rho0 * vy;
         arr(i, j, k, MOM3) = 0.0;
         arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho0, 0.0, vy, 0.0, pressure0, eos);
      });
   }
   }
}

inline void initialize_pressure_bump(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   if (config_uses_mixed_patch_system(cfg)) {
      const amrex::IntVect refinement = configured_mixed_patch_refinement_for_domain(cfg, geom.Domain());
      const PatchGridSystem grid_system = configured_mixed_patch_grid_system(cfg, refinement);
      const PatchLayout layout = configured_mixed_patch_layout(cfg, refinement);
      const PatchCoordinate center = make_patch_coordinate(cfg.bump_x0, cfg.bump_y0, cfg.bump_z0);
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
         const auto arr = state.array(mfi);
         const amrex::Real rho0 = cfg.rho0;
         const amrex::Real pressure0 = cfg.pressure0;
         const amrex::Real amplitude = cfg.amplitude;
         const amrex::Real sigma2 = cfg.bump_sigma * cfg.bump_sigma;
         const amrex::Real v1_0 = cfg.v1_0;
         const amrex::Real v2_0 = cfg.v2_0;
         const amrex::Real v3_0 = cfg.v3_0;
         const amrex::Real b1_0 = cfg.b1_0;
         const amrex::Real b2_0 = cfg.b2_0;
         const amrex::Real b3_0 = cfg.b3_0;
         const EOSParams eos = eos_params(cfg);
         const int eos_method = eos_method_code(cfg);
         const bool mhd = state.nComp() >= NVAR_MHD;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const PatchCoordinate local = patch_layout_cell_center_coordinate(grid_patch, layout_patch, i, j, k);
            const PatchCoordinate physical = patch_local_to_global(grid_patch.map, local);
            const amrex::Real dx_physical = physical.q1 - center.q1;
            const amrex::Real dy_physical = physical.q2 - center.q2;
            const amrex::Real dz_physical = physical.q3 - center.q3;
            const amrex::Real r2 = dx_physical * dx_physical + dy_physical * dy_physical + dz_physical * dz_physical;
            const amrex::Real pressure = pressure0 + amplitude * std::exp(-0.5 * r2 / sigma2);
            arr(i, j, k, RHO) = rho0;
            arr(i, j, k, MOM1) = rho0 * v1_0;
            arr(i, j, k, MOM2) = rho0 * v2_0;
            arr(i, j, k, MOM3) = rho0 * v3_0;
            if (mhd) {
               arr(i, j, k, ENERGY) =
                  total_mhd_energy_runtime(eos_method, rho0, v1_0, v2_0, v3_0, pressure, b1_0, b2_0, b3_0, eos);
               arr(i, j, k, B1) = b1_0;
               arr(i, j, k, B2) = b2_0;
               arr(i, j, k, B3) = b3_0;
            } else {
               arr(i, j, k, ENERGY) =
                  total_hydro_energy_runtime(eos_method, rho0, v1_0, v2_0, v3_0, pressure, eos);
            }
         });
      }
      }
      return;
   }

   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const auto periodic = geom.isPeriodicArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho0 = cfg.rho0;
      const amrex::Real pressure0 = cfg.pressure0;
      const amrex::Real amplitude = cfg.amplitude;
      const amrex::Real sigma2 = cfg.bump_sigma * cfg.bump_sigma;
      const amrex::Real x0 = cfg.bump_x0;
      const amrex::Real y0 = cfg.bump_y0;
      const amrex::Real z0 = cfg.bump_z0;
      const PatchCoordinate center_mesh = make_patch_coordinate(x0, y0, z0);
      const PatchCoordinate center = mesh_coordinate_to_physical_point(mesh_coordinate, mesh_axes, center_mesh);
      const amrex::Real v1_0 = cfg.v1_0;
      const amrex::Real v2_0 = cfg.v2_0;
      const amrex::Real v3_0 = cfg.v3_0;
      const amrex::Real b1_0 = cfg.b1_0;
      const amrex::Real b2_0 = cfg.b2_0;
      const amrex::Real b3_0 = cfg.b3_0;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);
      const bool mhd = state.nComp() >= NVAR_MHD;

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
         const amrex::Real pressure = pressure0 + amplitude * std::exp(-0.5 * r2 / sigma2);
         arr(i, j, k, RHO) = rho0;
         arr(i, j, k, MOM1) = rho0 * v1_0;
         arr(i, j, k, MOM2) = rho0 * v2_0;
         arr(i, j, k, MOM3) = rho0 * v3_0;
         if (mhd) {
            arr(i, j, k, ENERGY) = total_mhd_energy_runtime(eos_method, rho0, v1_0, v2_0, v3_0, pressure, b1_0, b2_0, b3_0, eos);
            arr(i, j, k, B1) = b1_0;
            arr(i, j, k, B2) = b2_0;
            arr(i, j, k, B3) = b3_0;
         } else {
            arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho0, v1_0, v2_0, v3_0, pressure, eos);
         }
      });
   }
   }
}

inline void initialize_vortex_blast(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
#if (AMREX_SPACEDIM == 1)
   (void)state;
   (void)geom;
   (void)cfg;
   amrex::Abort("vortex_blast initial data requires a 2D or 3D AMReX executable");
#else
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const auto periodic = geom.isPeriodicArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const bool cartesian_xy = mesh_coordinate == MeshCartesian && mesh_axis(mesh_axes, 0) == MeshAxisX
      && mesh_axis(mesh_axes, 1) == MeshAxisY;
   const bool cylindrical_rphi = mesh_coordinate == MeshCylindrical && mesh_axis(mesh_axes, 0) == MeshAxisR
      && mesh_axis(mesh_axes, 1) == MeshAxisPhi;
   if (!cartesian_xy && !cylindrical_rphi) {
      amrex::Abort("vortex_blast currently supports cartesian x y and cylindrical r phi active axes");
   }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho0 = cfg.rho0;
      const amrex::Real pressure0 = cfg.pressure0;
      const amrex::Real blast_amplitude = cfg.amplitude;
      const amrex::Real sigma2 = cfg.bump_sigma * cfg.bump_sigma;
      const amrex::Real x0 = cfg.bump_x0;
      const amrex::Real y0 = cfg.bump_y0;
      const amrex::Real z0 = cfg.bump_z0;
      const amrex::Real v1_0 = cfg.v1_0;
      const amrex::Real v2_0 = cfg.v2_0;
      const amrex::Real v3_0 = cfg.v3_0;
      const amrex::Real b1_0 = cfg.b1_0;
      const amrex::Real b2_0 = cfg.b2_0;
      const amrex::Real b3_0 = cfg.b3_0;
      const amrex::Real swirl_speed = cfg.kh_secondary_amplitude;
      const amrex::Real asymmetry = cfg.kh_density_amplitude;
      const amrex::Real mode = cfg.kh_secondary_mode;
      const amrex::Real phase = cfg.kh_phase;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);
      const bool mhd = state.nComp() >= NVAR_MHD;

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
         const amrex::Real q1 = patch_coordinate_component(sample.mesh, 0);
         const amrex::Real q2 = patch_coordinate_component(sample.mesh, 1);
#if (AMREX_SPACEDIM == 3)
         const amrex::Real q3 = patch_coordinate_component(sample.mesh, 2);
#else
         const amrex::Real q3 = z0;
#endif
         amrex::Real dx_phys = coordinate_delta(q1, x0, problo[0], probhi[0], periodic[0] != 0);
         amrex::Real dy_phys = coordinate_delta(q2, y0, problo[1], probhi[1], periodic[1] != 0);
         amrex::Real phi = q2;
         if (cylindrical_rphi) {
            const amrex::Real dphi = coordinate_delta(q2, y0, problo[1], probhi[1], periodic[1] != 0);
            const amrex::Real phi_near = y0 + dphi;
            dx_phys = q1 * std::cos(phi_near) - x0 * std::cos(y0);
            dy_phys = q1 * std::sin(phi_near) - x0 * std::sin(y0);
         }
#if (AMREX_SPACEDIM == 3)
         const amrex::Real dz_phys = coordinate_delta(q3, z0, problo[2], probhi[2], periodic[2] != 0);
#else
         const amrex::Real dz_phys = 0.0;
#endif
         const amrex::Real r2 = dx_phys * dx_phys + dy_phys * dy_phys + dz_phys * dz_phys;
         const amrex::Real radius = std::sqrt(dx_phys * dx_phys + dy_phys * dy_phys + 1.0e-300);
         const amrex::Real angle = std::atan2(dy_phys, dx_phys);
         const amrex::Real shape = std::exp(-0.5 * r2 / sigma2);
         const amrex::Real center_regularity = 1.0 - std::exp(-0.5 * r2 / sigma2);
         const amrex::Real angular_shape = amrex::max(
            0.2,
            1.0
               + center_regularity * asymmetry * std::cos(mode * angle + phase)
               + 0.45 * center_regularity * asymmetry * std::sin((mode + 2.0) * angle - 0.7 * phase)
         );
         const amrex::Real pressure = pressure0 + blast_amplitude * shape * angular_shape;

         const amrex::Real swirl_shape = std::exp(-0.5 * r2 / (4.0 * sigma2));
         const amrex::Real regular_swirl = swirl_speed * swirl_shape * center_regularity;
         const amrex::Real vx_swirl = -regular_swirl * dy_phys / radius;
         const amrex::Real vy_swirl = regular_swirl * dx_phys / radius;
         amrex::Real v1 = v1_0 + vx_swirl;
         amrex::Real v2 = v2_0 + vy_swirl;
         if (cylindrical_rphi) {
            const amrex::Real c = std::cos(phi);
            const amrex::Real s = std::sin(phi);
            v1 = v1_0 + vx_swirl * c + vy_swirl * s;
            v2 = v2_0 - vx_swirl * s + vy_swirl * c;
         }

         arr(i, j, k, RHO) = rho0;
         arr(i, j, k, MOM1) = rho0 * v1;
         arr(i, j, k, MOM2) = rho0 * v2;
         arr(i, j, k, MOM3) = rho0 * v3_0;
         if (mhd) {
            arr(i, j, k, ENERGY) = total_mhd_energy_runtime(
               eos_method, rho0, v1, v2, v3_0, pressure, b1_0, b2_0, b3_0, eos
            );
            arr(i, j, k, B1) = b1_0;
            arr(i, j, k, B2) = b2_0;
            arr(i, j, k, B3) = b3_0;
         } else {
            arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho0, v1, v2, v3_0, pressure, eos);
         }
      });
   }
   }
#endif
}

inline void initialize_density_cloud(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const auto periodic = geom.isPeriodicArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho0 = cfg.rho0;
      const amrex::Real pressure0 = cfg.pressure0;
      const amrex::Real amplitude = cfg.amplitude;
      const amrex::Real sigma2 = cfg.bump_sigma * cfg.bump_sigma;
      const amrex::Real x0 = cfg.bump_x0;
      const amrex::Real y0 = cfg.bump_y0;
      const amrex::Real z0 = cfg.bump_z0;
      const PatchCoordinate center_mesh = make_patch_coordinate(x0, y0, z0);
      const PatchCoordinate center = mesh_coordinate_to_physical_point(mesh_coordinate, mesh_axes, center_mesh);
      const amrex::Real v1_0 = cfg.v1_0;
      const amrex::Real v2_0 = cfg.v2_0;
      const amrex::Real v3_0 = cfg.v3_0;
      const amrex::Real b1_0 = cfg.b1_0;
      const amrex::Real b2_0 = cfg.b2_0;
      const amrex::Real b3_0 = cfg.b3_0;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);
      const bool mhd = state.nComp() >= NVAR_MHD;

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
         const amrex::Real rho = rho0 + amplitude * std::exp(-0.5 * r2 / sigma2);
         const PatchCoordinate velocity =
            cartesian_velocity_to_mesh_basis(mesh_coordinate, mesh_axes, sample.mesh, v1_0, v2_0, v3_0);
         const amrex::Real v1 = velocity.q1;
         const amrex::Real v2 = velocity.q2;
         const amrex::Real v3 = velocity.q3;
         arr(i, j, k, RHO) = rho;
         arr(i, j, k, MOM1) = rho * v1;
         arr(i, j, k, MOM2) = rho * v2;
         arr(i, j, k, MOM3) = rho * v3;
         if (mhd) {
            arr(i, j, k, ENERGY) =
               total_mhd_energy_runtime(eos_method, rho, v1, v2, v3, pressure0, b1_0, b2_0, b3_0, eos);
            arr(i, j, k, B1) = b1_0;
            arr(i, j, k, B2) = b2_0;
            arr(i, j, k, B3) = b3_0;
         } else {
            arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho, v1, v2, v3, pressure0, eos);
         }
      });
   }
   }
}

inline void initialize_hse_polytrope_x(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const EOSParams eos = eos_params(cfg);
   const int eos_method = eos_method_code(cfg);
   const amrex::Real hse_gamma = eos_method == EOSIdealGas
      ? eos.gamma
      : eos.segment_gamma[0];
   const amrex::Real hse_k = eos_method == EOSIdealGas
      ? cfg.pressure0 / std::pow(cfg.rho0, hse_gamma)
      : eos.segment_k[0];
   const amrex::Real rho_power0 = std::pow(cfg.rho0, hse_gamma - 1.0);
   const amrex::Real slope = (hse_gamma - 1.0) * cfg.gravity_a1 / (hse_gamma * hse_k);
   const amrex::Real x0 = cfg.bump_x0;
   const amrex::Real x_left = problo[0] + 0.5 * dx[0];
   const amrex::Real x_right = probhi[0] - 0.5 * dx[0];
   const amrex::Real power_left = rho_power0 + slope * (x_left - x0);
   const amrex::Real power_right = rho_power0 + slope * (x_right - x0);
   const amrex::Real min_power = power_left < power_right ? power_left : power_right;
   if (hse_k <= 0.0 || hse_gamma <= 1.0 || min_power <= 0.0) {
      amrex::Abort("hse_polytrope_x parameters produce a non-positive density profile");
   }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real v1_0 = cfg.v1_0;
      const amrex::Real v2_0 = cfg.v2_0;
      const amrex::Real v3_0 = cfg.v3_0;
      const amrex::Real b1_0 = cfg.b1_0;
      const amrex::Real b2_0 = cfg.b2_0;
      const amrex::Real b3_0 = cfg.b3_0;
      const bool mhd = state.nComp() >= NVAR_MHD;

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
         const amrex::Real q1 = patch_coordinate_component(sample.mesh, 0);
         const amrex::Real rho_power = rho_power0 + slope * (q1 - x0);
         const amrex::Real rho = std::pow(rho_power, 1.0 / (hse_gamma - 1.0));
         const amrex::Real pressure = hse_k * std::pow(rho, hse_gamma);
         arr(i, j, k, RHO) = rho;
         arr(i, j, k, MOM1) = rho * v1_0;
         arr(i, j, k, MOM2) = rho * v2_0;
         arr(i, j, k, MOM3) = rho * v3_0;
         if (mhd) {
            arr(i, j, k, ENERGY) = total_mhd_energy_runtime(eos_method, rho, v1_0, v2_0, v3_0, pressure, b1_0, b2_0, b3_0, eos);
            arr(i, j, k, B1) = b1_0;
            arr(i, j, k, B2) = b2_0;
            arr(i, j, k, B3) = b3_0;
         } else {
            arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho, v1_0, v2_0, v3_0, pressure, eos);
         }
      });
   }
   }
}

inline void initialize_hydro_beam(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   if (state.nComp() < NVAR_HYDRO || state.nComp() >= NVAR_MHD) {
      amrex::Abort("hydro_beam requires a hydro state vector");
   }
#if (AMREX_SPACEDIM < 2)
   amrex::Abort("hydro_beam requires the 2D or 3D AMReX executable");
#else
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho_beam = cfg.rho_left;
      const amrex::Real pressure_beam = cfg.pressure_left;
      const amrex::Real vx_beam = cfg.v1_left;
      const amrex::Real vy_beam = cfg.v2_left;
      const amrex::Real vz_beam = cfg.v3_left;
      const amrex::Real rho_ambient = cfg.rho_right;
      const amrex::Real pressure_ambient = cfg.pressure_right;
      const amrex::Real vx_ambient = cfg.v1_right;
      const amrex::Real vy_ambient = cfg.v2_right;
      const amrex::Real vz_ambient = cfg.v3_right;
      const amrex::Real x0 = cfg.bump_x0;
      const amrex::Real y0 = cfg.bump_y0;
      const amrex::Real z0 = cfg.bump_z0;
      const amrex::Real inv_length2 = 1.0 / (cfg.beam_length * cfg.beam_length);
      const amrex::Real inv_radius2 = 1.0 / (cfg.beam_radius * cfg.beam_radius);
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);

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
         const amrex::Real x = sample.physical.q1;
         const amrex::Real y = sample.physical.q2;
#if (AMREX_SPACEDIM == 3)
         const amrex::Real z = sample.physical.q3;
#else
         const amrex::Real z = z0;
#endif
         const amrex::Real rx = x - x0;
         const amrex::Real ry = y - y0;
         const amrex::Real rz = z - z0;
         const amrex::Real envelope = std::exp(-0.5 * (rx * rx * inv_length2 + (ry * ry + rz * rz) * inv_radius2));
         const amrex::Real rho = rho_ambient + envelope * (rho_beam - rho_ambient);
         const amrex::Real pressure = pressure_ambient + envelope * (pressure_beam - pressure_ambient);
         const amrex::Real vx = vx_ambient + envelope * (vx_beam - vx_ambient);
         const amrex::Real vy = vy_ambient + envelope * (vy_beam - vy_ambient);
         const amrex::Real vz = vz_ambient + envelope * (vz_beam - vz_ambient);
         arr(i, j, k, RHO) = rho;
         arr(i, j, k, MOM1) = rho * vx;
         arr(i, j, k, MOM2) = rho * vy;
         arr(i, j, k, MOM3) = rho * vz;
         arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho, vx, vy, vz, pressure, eos);
      });
   }
   }
#endif
}

inline void initialize_sod_plane(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real shock_x0 = cfg.shock_x0;
      const amrex::Real shock_y0 = cfg.shock_y0;
      const amrex::Real shock_z0 = cfg.shock_z0;
      const amrex::Real shock_nx = cfg.shock_nx;
      const amrex::Real shock_ny = cfg.shock_ny;
      const amrex::Real shock_nz = cfg.shock_nz;
      const amrex::Real rho_left = cfg.rho_left;
      const amrex::Real pressure_left = cfg.pressure_left;
      const amrex::Real v1_left = cfg.v1_left;
      const amrex::Real v2_left = cfg.v2_left;
      const amrex::Real v3_left = cfg.v3_left;
      const amrex::Real rho_right = cfg.rho_right;
      const amrex::Real pressure_right = cfg.pressure_right;
      const amrex::Real v1_right = cfg.v1_right;
      const amrex::Real v2_right = cfg.v2_right;
      const amrex::Real v3_right = cfg.v3_right;
      const amrex::Real b1_0 = cfg.b1_0;
      const amrex::Real b2_0 = cfg.b2_0;
      const amrex::Real b3_0 = cfg.b3_0;
      const amrex::Real b1_left = std::isfinite(cfg.b1_left) ? cfg.b1_left : b1_0;
      const amrex::Real b2_left = std::isfinite(cfg.b2_left) ? cfg.b2_left : b2_0;
      const amrex::Real b3_left = std::isfinite(cfg.b3_left) ? cfg.b3_left : b3_0;
      const amrex::Real b1_right = std::isfinite(cfg.b1_right) ? cfg.b1_right : b1_0;
      const amrex::Real b2_right = std::isfinite(cfg.b2_right) ? cfg.b2_right : b2_0;
      const amrex::Real b3_right = std::isfinite(cfg.b3_right) ? cfg.b3_right : b3_0;
      const bool periodic_plane = cfg.ic == "sod_periodic_plane";
      const amrex::Real two_pi = constants::two_pi;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);
      const bool mhd = state.nComp() >= NVAR_MHD;

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
         const amrex::Real x = sample.physical.q1;
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real y = sample.physical.q2;
#else
         const amrex::Real y = shock_y0;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real z = sample.physical.q3;
#else
         const amrex::Real z = shock_z0;
#endif
         const amrex::Real signed_distance = shock_nx * (x - shock_x0)
            + shock_ny * (y - shock_y0)
            + shock_nz * (z - shock_z0);
         const bool left = periodic_plane
            ? std::sin(two_pi * signed_distance) < 0.0
            : signed_distance < 0.0;
         const amrex::Real rho = left ? rho_left : rho_right;
         const amrex::Real pressure = left ? pressure_left : pressure_right;
         const amrex::Real vx = left ? v1_left : v1_right;
         const amrex::Real vy = left ? v2_left : v2_right;
         const amrex::Real vz = left ? v3_left : v3_right;
         const amrex::Real bx_cell = left ? b1_left : b1_right;
         const amrex::Real by_cell = left ? b2_left : b2_right;
         const amrex::Real bz_cell = left ? b3_left : b3_right;
         arr(i, j, k, RHO) = rho;
         arr(i, j, k, MOM1) = rho * vx;
         arr(i, j, k, MOM2) = rho * vy;
         arr(i, j, k, MOM3) = rho * vz;
         if (mhd) {
            arr(i, j, k, ENERGY) = total_mhd_energy_runtime(eos_method, rho, vx, vy, vz, pressure, bx_cell, by_cell, bz_cell, eos);
            arr(i, j, k, B1) = bx_cell;
            arr(i, j, k, B2) = by_cell;
            arr(i, j, k, B3) = bz_cell;
         } else {
            arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho, vx, vy, vz, pressure, eos);
         }
      });
   }
   }
}

inline void initialize_sod_x(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   initialize_sod_plane(state, geom, cfg);
}

inline void initialize_double_mach_reflection(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
   initialize_sod_plane(state, geom, cfg);
}

inline void initialize_sod_plane_face_b_impl(FaceMagneticField& face_b, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);

   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::MultiFab& face = *face_b.b[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(face, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = face.boxArray()[mfi.index()];
         const auto arr = face.array(mfi);
         const amrex::Real shock_x0 = cfg.shock_x0;
         const amrex::Real shock_y0 = cfg.shock_y0;
         const amrex::Real shock_z0 = cfg.shock_z0;
         const amrex::Real shock_nx = cfg.shock_nx;
         const amrex::Real shock_ny = cfg.shock_ny;
         const amrex::Real shock_nz = cfg.shock_nz;
         const amrex::Real b1_0 = cfg.b1_0;
         const amrex::Real b2_0 = cfg.b2_0;
         const amrex::Real b3_0 = cfg.b3_0;
         const amrex::Real b1_left = std::isfinite(cfg.b1_left) ? cfg.b1_left : b1_0;
         const amrex::Real b2_left = std::isfinite(cfg.b2_left) ? cfg.b2_left : b2_0;
         const amrex::Real b3_left = std::isfinite(cfg.b3_left) ? cfg.b3_left : b3_0;
         const amrex::Real b1_right = std::isfinite(cfg.b1_right) ? cfg.b1_right : b1_0;
         const amrex::Real b2_right = std::isfinite(cfg.b2_right) ? cfg.b2_right : b2_0;
         const amrex::Real b3_right = std::isfinite(cfg.b3_right) ? cfg.b3_right : b3_0;
         const bool periodic_plane = cfg.ic == "sod_periodic_plane";
         const amrex::Real two_pi = constants::two_pi;
         const int face_dir = dir;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const InitialDataSamplePoint sample = face_center_initial_data_sample_point(
               mesh_coordinate,
               mesh_axes,
               problo,
               dx,
               face_dir,
               i,
               j,
               k
            );
            const amrex::Real x = sample.physical.q1;
#if (AMREX_SPACEDIM >= 2)
            const amrex::Real y = sample.physical.q2;
#else
            const amrex::Real y = shock_y0;
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real z = sample.physical.q3;
#else
            const amrex::Real z = shock_z0;
#endif
            const amrex::Real signed_distance = shock_nx * (x - shock_x0)
               + shock_ny * (y - shock_y0)
               + shock_nz * (z - shock_z0);
            const bool left = periodic_plane
               ? std::sin(two_pi * signed_distance) < 0.0
               : signed_distance < 0.0;
            if (face_dir == 0) {
               arr(i, j, k, 0) = left ? b1_left : b1_right;
            }
#if (AMREX_SPACEDIM >= 2)
            if (face_dir == 1) {
               arr(i, j, k, 0) = left ? b2_left : b2_right;
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (face_dir == 2) {
               arr(i, j, k, 0) = left ? b3_left : b3_right;
            }
#endif
         });
      }
      }
   }
}

inline void initialize_circular_alfven_x(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const amrex::Real two_pi = constants::two_pi;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho0 = cfg.rho0;
      const amrex::Real pressure0 = cfg.pressure0;
      const amrex::Real b1_0 = cfg.b1_0;
      const amrex::Real amplitude = cfg.alfven_amplitude;
      const amrex::Real wave_number = cfg.wave_number;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);
      const amrex::Real ca = amrex::Math::abs(b1_0) / std::sqrt(rho0);
      const amrex::Real velocity_factor = ca > 0.0 ? -b1_0 / (rho0 * ca) : 0.0;

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
         const amrex::Real q1 = patch_coordinate_component(sample.mesh, 0);
         const amrex::Real phase = two_pi * wave_number * q1;
         const amrex::Real by = amplitude * std::cos(phase);
         const amrex::Real bz = amplitude * std::sin(phase);
         const amrex::Real vy = velocity_factor * by;
         const amrex::Real vz = velocity_factor * bz;
         arr(i, j, k, RHO) = rho0;
         arr(i, j, k, MOM1) = 0.0;
         arr(i, j, k, MOM2) = rho0 * vy;
         arr(i, j, k, MOM3) = rho0 * vz;
         arr(i, j, k, ENERGY) = total_mhd_energy_runtime(eos_method, rho0, 0.0, vy, vz, pressure0, b1_0, by, bz, eos);
         arr(i, j, k, B1) = b1_0;
         arr(i, j, k, B2) = by;
         arr(i, j, k, B3) = bz;
      });
   }
   }
}

inline void initialize_resistive_wave_x(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const amrex::Real two_pi = constants::two_pi;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho0 = cfg.rho0;
      const amrex::Real pressure0 = cfg.pressure0;
      const amrex::Real amplitude = cfg.alfven_amplitude;
      const amrex::Real wave_number = cfg.wave_number;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);
      const amrex::Real total_energy = total_mhd_energy_runtime(eos_method, rho0, 0.0, 0.0, 0.0, pressure0, 0.0, amplitude, 0.0, eos);

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
         const amrex::Real q1 = patch_coordinate_component(sample.mesh, 0);
         const amrex::Real phase = two_pi * wave_number * q1;
         arr(i, j, k, RHO) = rho0;
         arr(i, j, k, MOM1) = 0.0;
         arr(i, j, k, MOM2) = 0.0;
         arr(i, j, k, MOM3) = 0.0;
         arr(i, j, k, ENERGY) = total_energy;
         arr(i, j, k, B1) = 0.0;
         arr(i, j, k, B2) = amplitude * std::sin(phase);
         arr(i, j, k, B3) = amplitude * std::cos(phase);
      });
   }
   }
}

inline void initialize_magnetic_tube_z(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const auto periodic = geom.isPeriodicArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho0 = cfg.rho0;
      const amrex::Real pressure0 = cfg.pressure0;
      const amrex::Real amplitude = cfg.amplitude;
      const amrex::Real sigma2 = cfg.bump_sigma * cfg.bump_sigma;
      const amrex::Real x0 = cfg.bump_x0;
      const amrex::Real y0 = cfg.bump_y0;
      const amrex::Real v1_0 = cfg.v1_0;
      const amrex::Real v2_0 = cfg.v2_0;
      const amrex::Real v3_0 = cfg.v3_0;
      const amrex::Real b1_0 = cfg.b1_0;
      const amrex::Real b2_0 = cfg.b2_0;
      const amrex::Real b3_0 = cfg.b3_0;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);

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
         const amrex::Real x = patch_coordinate_component(sample.mesh, 0);
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real y = patch_coordinate_component(sample.mesh, 1);
#else
         const amrex::Real y = y0;
#endif
         const amrex::Real rx = coordinate_delta(x, x0, problo[0], probhi[0], periodic[0] != 0);
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real ry = coordinate_delta(y, y0, problo[1], probhi[1], periodic[1] != 0);
#else
         const amrex::Real ry = 0.0;
#endif
         const amrex::Real profile = std::exp(-0.5 * (rx * rx + ry * ry) / sigma2);
         const amrex::Real bz = b3_0 + amplitude * profile;
         arr(i, j, k, RHO) = rho0;
         arr(i, j, k, MOM1) = rho0 * v1_0;
         arr(i, j, k, MOM2) = rho0 * v2_0;
         arr(i, j, k, MOM3) = rho0 * v3_0;
         arr(i, j, k, ENERGY) = total_mhd_energy_runtime(eos_method, rho0, v1_0, v2_0, v3_0, pressure0, b1_0, b2_0, bz, eos);
         arr(i, j, k, B1) = b1_0;
         arr(i, j, k, B2) = b2_0;
         arr(i, j, k, B3) = bz;
      });
   }
   }
}

inline void initialize_mhd_rotor(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
#if (AMREX_SPACEDIM == 1)
   (void)state;
   (void)geom;
   (void)cfg;
   amrex::Abort("mhd_rotor requires the 2D or 3D AMReX executable");
#else
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const amrex::Real x0 = cfg.bump_x0;
   const amrex::Real y0 = cfg.bump_y0;
   const amrex::Real rho0 = cfg.rho0;
   const amrex::Real pressure0 = cfg.pressure0;
   const amrex::Real rotor_density = cfg.rotor_density;
   const amrex::Real r0 = cfg.rotor_radius;
   const amrex::Real r1 = cfg.rotor_transition_radius;
   const amrex::Real omega = cfg.rotor_omega;
   const amrex::Real b1_0 = cfg.b1_0;
   const amrex::Real b2_0 = cfg.b2_0;
   const amrex::Real b3_0 = cfg.b3_0;
   const EOSParams eos = eos_params(cfg);
   const int eos_method = eos_method_code(cfg);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
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
         const amrex::Real x = sample.physical.q1;
         const amrex::Real y = sample.physical.q2;
         const amrex::Real rx = x - x0;
         const amrex::Real ry = y - y0;
         const amrex::Real radius = std::sqrt(rx * rx + ry * ry);
         amrex::Real rotor_fraction = 0.0;
         if (radius <= r0) {
            rotor_fraction = 1.0;
         } else if (radius < r1) {
            rotor_fraction = (r1 - radius) / (r1 - r0);
         }
         const amrex::Real rho = rho0 + rotor_fraction * (rotor_density - rho0);
         const amrex::Real vx = -rotor_fraction * omega * ry;
         const amrex::Real vy = rotor_fraction * omega * rx;
         constexpr amrex::Real vz = 0.0;
         arr(i, j, k, RHO) = rho;
         arr(i, j, k, MOM1) = rho * vx;
         arr(i, j, k, MOM2) = rho * vy;
         arr(i, j, k, MOM3) = rho * vz;
         arr(i, j, k, ENERGY) = total_mhd_energy_runtime(eos_method, rho, vx, vy, vz, pressure0, b1_0, b2_0, b3_0, eos);
         arr(i, j, k, B1) = b1_0;
         arr(i, j, k, B2) = b2_0;
         arr(i, j, k, B3) = b3_0;
      });
   }
   }
#endif
}

template <bool Magnetized>
inline void initialize_kelvin_helmholtz_impl(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   char const* ic_name
)
{
#if (AMREX_SPACEDIM == 1)
   (void)state;
   (void)geom;
   (void)cfg;
   (void)ic_name;
   amrex::Abort("Kelvin-Helmholtz initial data requires a 2D or 3D AMReX executable");
#else
   if constexpr (Magnetized) {
      if (state.nComp() < NVAR_MHD) {
         amrex::Abort(std::string(ic_name) + " requires pyfluid.physics = mhd");
      }
   } else {
      if (state.nComp() < NVAR_HYDRO) {
         amrex::Abort(std::string(ic_name) + " requires pyfluid.physics = hydro");
      }
   }

   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const auto periodic = geom.isPeriodicArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const bool x_periodic = periodic[0];
   const bool y_periodic = periodic[1];
#if (AMREX_SPACEDIM == 3)
   const bool z_periodic = periodic[2];
#endif
   const amrex::Real two_pi = constants::two_pi;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho0 = cfg.rho0;
      const amrex::Real pressure0 = cfg.pressure0;
      const amrex::Real shear_speed = cfg.v1_0;
      const amrex::Real perturbation = cfg.amplitude;
      const amrex::Real secondary = cfg.kh_secondary_amplitude;
      const amrex::Real layer_width = amrex::max(cfg.bump_sigma, 1.0e-6);
      const amrex::Real wave_number = cfg.wave_number;
      const amrex::Real secondary_mode = cfg.kh_secondary_mode;
      const amrex::Real z_mode = cfg.kh_z_mode;
      const amrex::Real phase_shift = cfg.kh_phase;
      const amrex::Real envelope_sigma = cfg.kh_envelope_sigma;
      const amrex::Real density_amplitude = cfg.kh_density_amplitude;
      const bool stream_perturbation = cfg.kh_perturbation == "stream";
      const bool vortex_perturbation = cfg.kh_perturbation == "vortex";
      const amrex::Real gaussian_derivative_normalization = std::exp(0.5);
      const amrex::Real envelope_x0 = cfg.bump_x0;
      const amrex::Real sheet_y0 = cfg.bump_y0;
      const amrex::Real envelope_z0 = cfg.bump_z0;
      const amrex::Real b1_0 = cfg.b1_0;
      const amrex::Real b2_0 = cfg.b2_0;
      const amrex::Real b3_0 = cfg.b3_0;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);

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
         const amrex::Real x = patch_coordinate_component(sample.mesh, 0);
         const amrex::Real y = patch_coordinate_component(sample.mesh, 1);
#if (AMREX_SPACEDIM == 3)
         const amrex::Real z = patch_coordinate_component(sample.mesh, 2);
#else
         const amrex::Real z = 0.0;
#endif
         const amrex::Real ymid = y - sheet_y0;
         const amrex::Real ylo = y - 0.25;
         const amrex::Real yhi = y - 0.75;
         const amrex::Real g_mid = std::exp(-0.5 * ymid * ymid / (layer_width * layer_width));
         const amrex::Real g_lo = std::exp(-0.5 * ylo * ylo / (layer_width * layer_width));
         const amrex::Real g_hi = std::exp(-0.5 * yhi * yhi / (layer_width * layer_width));
         const amrex::Real layer = y_periodic ? (g_lo + g_hi) : g_mid;
         const amrex::Real phase_x = two_pi * wave_number * x;
         const amrex::Real phase_x2 = two_pi * secondary_mode * x + phase_shift;
         const amrex::Real sin_x = std::sin(phase_x);
         const amrex::Real sin_x2 = std::sin(phase_x2);
         amrex::Real envelope = 1.0;
         amrex::Real denvelope_dx = 0.0;
         amrex::Real xdist = x - envelope_x0;
         if (x_periodic) {
            xdist -= std::floor(xdist + 0.5);
         }
         if (envelope_sigma > 0.0) {
#if (AMREX_SPACEDIM == 3)
            amrex::Real zdist = z - envelope_z0;
            if (z_periodic) {
               zdist -= std::floor(zdist + 0.5);
            }
            envelope = std::exp(-0.5 * (xdist * xdist + zdist * zdist) / (envelope_sigma * envelope_sigma));
#else
            envelope = std::exp(-0.5 * xdist * xdist / (envelope_sigma * envelope_sigma));
#endif
            denvelope_dx = -xdist * envelope / (envelope_sigma * envelope_sigma);
         }
         const amrex::Real base_vx = y_periodic
            ? shear_speed * (std::tanh(ylo / layer_width) - std::tanh(yhi / layer_width) - 1.0)
            : shear_speed * std::tanh(ymid / layer_width);
         amrex::Real vx = base_vx;
         const amrex::Real shear_profile = shear_speed != 0.0 ? base_vx / shear_speed : 0.0;
         const amrex::Real rho = rho0 * (1.0 + density_amplitude * shear_profile);
         const amrex::Real perturb_sum = perturbation * sin_x + secondary * sin_x2;
         amrex::Real vy = envelope * perturb_sum * layer;
         if (vortex_perturbation) {
            const amrex::Real sx = envelope_sigma > 0.0 ? envelope_sigma : 0.12;
            const amrex::Real sy = amrex::max(2.0 * layer_width, 2.0 * dx[1]);
            const amrex::Real ydist = ymid;
            const amrex::Real vortex = std::exp(-0.5 * (
               (xdist * xdist) / (sx * sx) + (ydist * ydist) / (sy * sy)
            ));
            const amrex::Real stream =
               perturbation * sx * gaussian_derivative_normalization * vortex;
            vx += -stream * ydist / (sy * sy);
            vy = stream * xdist / (sx * sx);
         } else if (stream_perturbation) {
            const amrex::Real k1 = two_pi * wave_number;
            const amrex::Real k2 = two_pi * secondary_mode;
            const amrex::Real primary_mode = perturbation * std::cos(phase_x) / k1;
            const amrex::Real secondary_mode_term = (secondary > 0.0 && secondary_mode > 0.0)
               ? secondary * std::cos(phase_x2) / k2
               : 0.0;
            const amrex::Real stream_mode = primary_mode + secondary_mode_term;
            const amrex::Real stream_mode_dx = -perturb_sum;
            const amrex::Real dlayer_dy = y_periodic
               ? (-ylo * g_lo - yhi * g_hi) / (layer_width * layer_width)
               : -ymid * g_mid / (layer_width * layer_width);
            vx += envelope * stream_mode * dlayer_dy;
            vy = -(denvelope_dx * stream_mode + envelope * stream_mode_dx) * layer;
         }
         amrex::Real vz = 0.0;
#if (AMREX_SPACEDIM == 3)
         const amrex::Real phase_z = two_pi * z_mode * z;
         const amrex::Real z_modulation = 1.0
            + 0.35 * std::sin(phase_z + 0.4)
            + 0.18 * std::sin(2.0 * phase_z + 1.1);
         vy *= z_modulation;
         vz = 0.5 * envelope * (
            perturbation * std::cos(phase_x) * std::sin(phase_z)
            + secondary * std::cos(phase_x2) * std::sin(2.0 * phase_z + 0.7)
         ) * (y_periodic ? (g_lo - g_hi) : g_mid);
#endif
         arr(i, j, k, RHO) = rho;
         arr(i, j, k, MOM1) = rho * vx;
         arr(i, j, k, MOM2) = rho * vy;
         arr(i, j, k, MOM3) = rho * vz;
         if constexpr (Magnetized) {
            arr(i, j, k, ENERGY) = total_mhd_energy_runtime(eos_method, rho, vx, vy, vz, pressure0, b1_0, b2_0, b3_0, eos);
            arr(i, j, k, B1) = b1_0;
            arr(i, j, k, B2) = b2_0;
            arr(i, j, k, B3) = b3_0;
         } else {
            arr(i, j, k, ENERGY) = total_hydro_energy_runtime(eos_method, rho, vx, vy, vz, pressure0, eos);
         }
      });
   }
   }
#endif
}

inline void initialize_kelvin_helmholtz(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   initialize_kelvin_helmholtz_impl<false>(state, geom, cfg, "kelvin_helmholtz");
}

inline void initialize_kelvin_helmholtz_mhd(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   initialize_kelvin_helmholtz_impl<true>(state, geom, cfg, "kelvin_helmholtz_mhd");
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real orszag_tang_vector_potential(
   amrex::Real x,
   amrex::Real y,
   amrex::Real x_lo,
   amrex::Real y_lo,
   amrex::Real magnetic_amplitude,
   amrex::Real kx,
   amrex::Real ky
) noexcept
{
   return magnetic_amplitude * (
      std::cos(ky * (y - y_lo)) / ky
      + std::cos(2.0 * kx * (x - x_lo)) / (2.0 * kx)
   );
}

inline void initialize_orszag_tang(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
#if (AMREX_SPACEDIM == 1)
   (void)state;
   (void)geom;
   (void)cfg;
   amrex::Abort("orszag_tang requires the 2D or 3D AMReX executable");
#else
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const amrex::Real two_pi = constants::two_pi;
   const amrex::Real kx = two_pi * cfg.wave_number / (probhi[0] - problo[0]);
   const amrex::Real ky = two_pi * cfg.wave_number / (probhi[1] - problo[1]);
   const amrex::Real velocity_amplitude = cfg.amplitude;
   const amrex::Real magnetic_amplitude = cfg.b1_0;
   const amrex::Real guide_field = cfg.b3_0;
   const amrex::Real rho0 = cfg.rho0;
   const amrex::Real pressure0 = cfg.pressure0;
   const EOSParams eos = eos_params(cfg);
   const int eos_method = eos_method_code(cfg);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
         const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
         const amrex::Real phase_x = kx * (x - problo[0]);
         const amrex::Real phase_y = ky * (y - problo[1]);
         const amrex::Real v1 = -velocity_amplitude * std::sin(phase_y);
         const amrex::Real v2 = velocity_amplitude * std::sin(phase_x);
         const amrex::Real v3 = 0.0;
         const amrex::Real b1 = -magnetic_amplitude * std::sin(phase_y);
         const amrex::Real b2 = magnetic_amplitude * std::sin(2.0 * phase_x);
         arr(i, j, k, RHO) = rho0;
         arr(i, j, k, MOM1) = rho0 * v1;
         arr(i, j, k, MOM2) = rho0 * v2;
         arr(i, j, k, MOM3) = 0.0;
         arr(i, j, k, ENERGY) =
            total_mhd_energy_runtime(eos_method, rho0, v1, v2, v3, pressure0, b1, b2, guide_field, eos);
         arr(i, j, k, B1) = b1;
         arr(i, j, k, B2) = b2;
         arr(i, j, k, B3) = guide_field;
      });
   }
   }
#endif
}

inline void initialize_orszag_tang_face_b_impl(
   FaceMagneticField& face_b,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
#if (AMREX_SPACEDIM == 1)
   (void)face_b;
   (void)geom;
   (void)cfg;
   amrex::Abort("orszag_tang face initialization requires the 2D or 3D AMReX executable");
#else
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const amrex::Real two_pi = constants::two_pi;
   const amrex::Real kx = two_pi * cfg.wave_number / (probhi[0] - problo[0]);
   const amrex::Real ky = two_pi * cfg.wave_number / (probhi[1] - problo[1]);
   const amrex::Real magnetic_amplitude = cfg.b1_0;

   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::MultiFab& face = *face_b.b[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(face, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto field = face.array(mfi);
         const int direction = dir;
         const amrex::Real guide_field = cfg.b3_0;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (direction == 0) {
               const amrex::Real x = problo[0] + static_cast<amrex::Real>(i) * dx[0];
               const amrex::Real y_lo = problo[1] + static_cast<amrex::Real>(j) * dx[1];
               const amrex::Real az_lo = orszag_tang_vector_potential(
                  x, y_lo, problo[0], problo[1], magnetic_amplitude, kx, ky
               );
               const amrex::Real az_hi = orszag_tang_vector_potential(
                  x, y_lo + dx[1], problo[0], problo[1], magnetic_amplitude, kx, ky
               );
               field(i, j, k) = (az_hi - az_lo) / dx[1];
            } else if (direction == 1) {
               const amrex::Real x_lo = problo[0] + static_cast<amrex::Real>(i) * dx[0];
               const amrex::Real y = problo[1] + static_cast<amrex::Real>(j) * dx[1];
               const amrex::Real az_lo = orszag_tang_vector_potential(
                  x_lo, y, problo[0], problo[1], magnetic_amplitude, kx, ky
               );
               const amrex::Real az_hi = orszag_tang_vector_potential(
                  x_lo + dx[0], y, problo[0], problo[1], magnetic_amplitude, kx, ky
               );
               field(i, j, k) = -(az_hi - az_lo) / dx[0];
            } else {
               field(i, j, k) = guide_field;
            }
         });
      }
      }
   }
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real current_sheet_base_b1(
   amrex::Real y,
   amrex::Real sheet_y1,
   amrex::Real sheet_y2,
   amrex::Real thickness,
   amrex::Real field_strength
) noexcept
{
   return field_strength
      * (std::tanh((y - sheet_y1) / thickness) - std::tanh((y - sheet_y2) / thickness) - 1.0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real current_sheet_vector_potential(
   amrex::Real x,
   amrex::Real y,
   amrex::Real x_lo,
   amrex::Real y_lo,
   amrex::Real sheet_y1,
   amrex::Real sheet_y2,
   amrex::Real thickness,
   amrex::Real field_strength,
   amrex::Real perturbation_potential,
   amrex::Real k1,
   amrex::Real k2
) noexcept
{
   const amrex::Real base = field_strength
      * (thickness * std::log(std::cosh((y - sheet_y1) / thickness))
         - thickness * std::log(std::cosh((y - sheet_y2) / thickness)) - (y - y_lo));
   const amrex::Real perturbation = perturbation_potential * std::cos(k1 * (x - x_lo))
      * std::sin(k2 * (y - y_lo));
   return base + perturbation;
}

inline void initialize_resistive_current_sheet(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
#if (AMREX_SPACEDIM == 1)
   (void)state;
   (void)geom;
   (void)cfg;
   amrex::Abort("resistive_current_sheet requires the 2D or 3D AMReX executable");
#else
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const amrex::Real two_pi = constants::two_pi;
   const amrex::Real length1 = probhi[0] - problo[0];
   const amrex::Real length2 = probhi[1] - problo[1];
   const amrex::Real sheet_y1 = problo[1] + 0.25 * length2;
   const amrex::Real sheet_y2 = problo[1] + 0.75 * length2;
   const amrex::Real thickness = cfg.bump_sigma;
   const amrex::Real field_strength = cfg.b1_0;
   const amrex::Real guide_field = cfg.b3_0;
   const amrex::Real k1 = two_pi * cfg.wave_number / length1;
   const amrex::Real k2 = two_pi / length2;
   const amrex::Real perturbation_potential = cfg.alfven_amplitude / std::max(k1, k2);
   const amrex::Real rho0 = cfg.rho0;
   const amrex::Real pressure0 = cfg.pressure0;
   const amrex::Real v1 = cfg.v1_0;
   const amrex::Real v2 = cfg.v2_0;
   const amrex::Real v3 = cfg.v3_0;
   const EOSParams eos = eos_params(cfg);
   const int eos_method = eos_method_code(cfg);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
         const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
         const amrex::Real base_b1 = current_sheet_base_b1(y, sheet_y1, sheet_y2, thickness, field_strength);
         const amrex::Real b1 = base_b1
            + perturbation_potential * k2 * std::cos(k1 * (x - problo[0])) * std::cos(k2 * (y - problo[1]));
         const amrex::Real b2 = perturbation_potential * k1 * std::sin(k1 * (x - problo[0]))
            * std::sin(k2 * (y - problo[1]));
         const amrex::Real pressure = pressure0 + 0.5 * (field_strength * field_strength - base_b1 * base_b1);
         arr(i, j, k, RHO) = rho0;
         arr(i, j, k, MOM1) = rho0 * v1;
         arr(i, j, k, MOM2) = rho0 * v2;
         arr(i, j, k, MOM3) = rho0 * v3;
         arr(i, j, k, ENERGY) =
            total_mhd_energy_runtime(eos_method, rho0, v1, v2, v3, pressure, b1, b2, guide_field, eos);
         arr(i, j, k, B1) = b1;
         arr(i, j, k, B2) = b2;
         arr(i, j, k, B3) = guide_field;
      });
   }
   }
#endif
}

inline void initialize_resistive_current_sheet_face_b_impl(
   FaceMagneticField& face_b,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
#if (AMREX_SPACEDIM == 1)
   (void)face_b;
   (void)geom;
   (void)cfg;
   amrex::Abort("resistive_current_sheet face initialization requires the 2D or 3D AMReX executable");
#else
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const amrex::Real two_pi = constants::two_pi;
   const amrex::Real length1 = probhi[0] - problo[0];
   const amrex::Real length2 = probhi[1] - problo[1];
   const amrex::Real sheet_y1 = problo[1] + 0.25 * length2;
   const amrex::Real sheet_y2 = problo[1] + 0.75 * length2;
   const amrex::Real thickness = cfg.bump_sigma;
   const amrex::Real field_strength = cfg.b1_0;
   const amrex::Real k1 = two_pi * cfg.wave_number / length1;
   const amrex::Real k2 = two_pi / length2;
   const amrex::Real perturbation_potential = cfg.alfven_amplitude / std::max(k1, k2);

   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::MultiFab& face = *face_b.b[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(face, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto f = face.array(mfi);
         const int d = dir;
         const amrex::Real guide_field = cfg.b3_0;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (d == 0) {
               const amrex::Real x = problo[0] + static_cast<amrex::Real>(i) * dx[0];
               const amrex::Real y_lo = problo[1] + static_cast<amrex::Real>(j) * dx[1];
               const amrex::Real az_lo = current_sheet_vector_potential(
                  x, y_lo, problo[0], problo[1], sheet_y1, sheet_y2, thickness,
                  field_strength, perturbation_potential, k1, k2
               );
               const amrex::Real az_hi = current_sheet_vector_potential(
                  x, y_lo + dx[1], problo[0], problo[1], sheet_y1, sheet_y2, thickness,
                  field_strength, perturbation_potential, k1, k2
               );
               f(i, j, k, 0) = (az_hi - az_lo) / dx[1];
            } else if (d == 1) {
               const amrex::Real x_lo = problo[0] + static_cast<amrex::Real>(i) * dx[0];
               const amrex::Real y = problo[1] + static_cast<amrex::Real>(j) * dx[1];
               const amrex::Real az_lo = current_sheet_vector_potential(
                  x_lo, y, problo[0], problo[1], sheet_y1, sheet_y2, thickness,
                  field_strength, perturbation_potential, k1, k2
               );
               const amrex::Real az_hi = current_sheet_vector_potential(
                  x_lo + dx[0], y, problo[0], problo[1], sheet_y1, sheet_y2, thickness,
                  field_strength, perturbation_potential, k1, k2
               );
               f(i, j, k, 0) = -(az_hi - az_lo) / dx[0];
            }
#if (AMREX_SPACEDIM == 3)
            else {
               f(i, j, k, 0) = guide_field;
            }
#else
            amrex::ignore_unused(guide_field);
#endif
         });
      }
      }
   }
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real reconnecting_shear_vector_potential(
   amrex::Real x,
   amrex::Real y,
   amrex::Real x_lo,
   amrex::Real sheet_y,
   amrex::Real sheet_thickness,
   amrex::Real perturbation_width,
   amrex::Real field_strength,
   amrex::Real magnetic_perturbation,
   amrex::Real primary_wave_number,
   amrex::Real secondary_wave_number,
   amrex::Real secondary_phase
) noexcept
{
   const amrex::Real y_offset = y - sheet_y;
   const amrex::Real envelope = std::exp(
      -0.5 * y_offset * y_offset / (perturbation_width * perturbation_width)
   );
   const amrex::Real primary_phase = primary_wave_number * (x - x_lo);
   const amrex::Real secondary_phase_value = secondary_wave_number * (x - x_lo) + secondary_phase;
   const amrex::Real perturbation = magnetic_perturbation * envelope * (
      std::cos(primary_phase) / primary_wave_number
      + 0.5 * std::cos(secondary_phase_value) / secondary_wave_number
   );
   return field_strength * sheet_thickness * std::log(std::cosh(y_offset / sheet_thickness))
      + perturbation;
}

inline void initialize_reconnecting_shear_layer(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
#if (AMREX_SPACEDIM == 1)
   (void)state;
   (void)geom;
   (void)cfg;
   amrex::Abort("reconnecting_shear_layer requires the 2D or 3D AMReX executable");
#else
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const amrex::Real two_pi = constants::two_pi;
   const amrex::Real length1 = probhi[0] - problo[0];
   const amrex::Real sheet_y = cfg.bump_y0;
   const amrex::Real sheet_thickness = cfg.bump_sigma;
   const amrex::Real perturbation_width = amrex::max(cfg.kh_envelope_sigma, 2.0 * sheet_thickness);
   const amrex::Real field_strength = cfg.b1_0;
   const amrex::Real guide_field = cfg.b3_0;
   const amrex::Real shear_speed = cfg.v1_0;
   const amrex::Real primary_wave_number = two_pi * cfg.wave_number / length1;
   const amrex::Real secondary_wave_number = two_pi * cfg.kh_secondary_mode / length1;
   const amrex::Real secondary_phase = cfg.kh_phase;
   const amrex::Real velocity_perturbation = cfg.amplitude;
   const amrex::Real secondary_velocity_perturbation = cfg.kh_secondary_amplitude;
   const amrex::Real magnetic_perturbation = cfg.alfven_amplitude;
   const amrex::Real rho0 = cfg.rho0;
   const amrex::Real pressure0 = cfg.pressure0;
   const EOSParams eos = eos_params(cfg);
   const int eos_method = eos_method_code(cfg);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
         const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
         const amrex::Real y_offset = y - sheet_y;
         const amrex::Real envelope = std::exp(
            -0.5 * y_offset * y_offset / (perturbation_width * perturbation_width)
         );
         const amrex::Real envelope_derivative = -y_offset * envelope
            / (perturbation_width * perturbation_width);
         const amrex::Real primary_phase = primary_wave_number * (x - problo[0]);
         const amrex::Real secondary_phase_value = secondary_wave_number * (x - problo[0]) + secondary_phase;
         const amrex::Real velocity_stream_derivative_y = envelope_derivative * (
            velocity_perturbation * std::cos(primary_phase) / primary_wave_number
            + secondary_velocity_perturbation * std::cos(secondary_phase_value) / secondary_wave_number
         );
         const amrex::Real v1 = shear_speed * std::tanh(y_offset / sheet_thickness)
            + velocity_stream_derivative_y;
         const amrex::Real v2 = envelope * (
            velocity_perturbation * std::sin(primary_phase)
            + secondary_velocity_perturbation * std::sin(secondary_phase_value)
         );
         const amrex::Real base_b1 = field_strength * std::tanh(y_offset / sheet_thickness);
         const amrex::Real magnetic_stream = magnetic_perturbation * (
            std::cos(primary_phase) / primary_wave_number
            + 0.5 * std::cos(secondary_phase_value) / secondary_wave_number
         );
         const amrex::Real b1 = base_b1 + envelope_derivative * magnetic_stream;
         const amrex::Real b2 = magnetic_perturbation * envelope * (
            std::sin(primary_phase) + 0.5 * std::sin(secondary_phase_value)
         );
         const amrex::Real pressure = pressure0
            + 0.5 * (field_strength * field_strength - base_b1 * base_b1);
         arr(i, j, k, RHO) = rho0;
         arr(i, j, k, MOM1) = rho0 * v1;
         arr(i, j, k, MOM2) = rho0 * v2;
         arr(i, j, k, MOM3) = 0.0;
         arr(i, j, k, ENERGY) = total_mhd_energy_runtime(
            eos_method, rho0, v1, v2, 0.0, pressure, b1, b2, guide_field, eos
         );
         arr(i, j, k, B1) = b1;
         arr(i, j, k, B2) = b2;
         arr(i, j, k, B3) = guide_field;
      });
   }
   }
#endif
}

inline void initialize_reconnecting_shear_layer_face_b_impl(
   FaceMagneticField& face_b,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
#if (AMREX_SPACEDIM == 1)
   (void)face_b;
   (void)geom;
   (void)cfg;
   amrex::Abort("reconnecting_shear_layer face initialization requires the 2D or 3D AMReX executable");
#else
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const amrex::Real two_pi = constants::two_pi;
   const amrex::Real length1 = probhi[0] - problo[0];
   const amrex::Real sheet_y = cfg.bump_y0;
   const amrex::Real sheet_thickness = cfg.bump_sigma;
   const amrex::Real perturbation_width = amrex::max(cfg.kh_envelope_sigma, 2.0 * sheet_thickness);
   const amrex::Real field_strength = cfg.b1_0;
   const amrex::Real guide_field = cfg.b3_0;
   const amrex::Real magnetic_perturbation = cfg.alfven_amplitude;
   const amrex::Real primary_wave_number = two_pi * cfg.wave_number / length1;
   const amrex::Real secondary_wave_number = two_pi * cfg.kh_secondary_mode / length1;
   const amrex::Real secondary_phase = cfg.kh_phase;

   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::MultiFab& face = *face_b.b[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(face, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto field = face.array(mfi);
         const int direction = dir;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (direction == 0) {
               const amrex::Real x = problo[0] + static_cast<amrex::Real>(i) * dx[0];
               const amrex::Real y_lo = problo[1] + static_cast<amrex::Real>(j) * dx[1];
               const amrex::Real potential_lo = reconnecting_shear_vector_potential(
                  x, y_lo, problo[0], sheet_y, sheet_thickness, perturbation_width,
                  field_strength, magnetic_perturbation, primary_wave_number,
                  secondary_wave_number, secondary_phase
               );
               const amrex::Real potential_hi = reconnecting_shear_vector_potential(
                  x, y_lo + dx[1], problo[0], sheet_y, sheet_thickness, perturbation_width,
                  field_strength, magnetic_perturbation, primary_wave_number,
                  secondary_wave_number, secondary_phase
               );
               field(i, j, k, 0) = (potential_hi - potential_lo) / dx[1];
            } else if (direction == 1) {
               const amrex::Real x_lo = problo[0] + static_cast<amrex::Real>(i) * dx[0];
               const amrex::Real y = problo[1] + static_cast<amrex::Real>(j) * dx[1];
               const amrex::Real potential_lo = reconnecting_shear_vector_potential(
                  x_lo, y, problo[0], sheet_y, sheet_thickness, perturbation_width,
                  field_strength, magnetic_perturbation, primary_wave_number,
                  secondary_wave_number, secondary_phase
               );
               const amrex::Real potential_hi = reconnecting_shear_vector_potential(
                  x_lo + dx[0], y, problo[0], sheet_y, sheet_thickness, perturbation_width,
                  field_strength, magnetic_perturbation, primary_wave_number,
                  secondary_wave_number, secondary_phase
               );
               field(i, j, k, 0) = -(potential_hi - potential_lo) / dx[0];
            }
#if (AMREX_SPACEDIM == 3)
            else {
               field(i, j, k, 0) = guide_field;
            }
#else
            amrex::ignore_unused(guide_field);
#endif
         });
      }
      }
   }
#endif
}

inline void initialize_magnetic_blast_noise(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
#if (AMREX_SPACEDIM != 3)
   (void)state;
   (void)geom;
   (void)cfg;
   amrex::Abort("magnetic_blast_noise requires the 3D AMReX executable");
#else
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const auto periodic = geom.isPeriodicArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const amrex::Real two_pi = constants::two_pi;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto arr = state.array(mfi);
      const amrex::Real rho0 = cfg.rho0;
      const amrex::Real pressure0 = cfg.pressure0;
      const amrex::Real blast_amplitude = cfg.amplitude;
      const amrex::Real sigma2 = cfg.bump_sigma * cfg.bump_sigma;
      const amrex::Real x0 = cfg.bump_x0;
      const amrex::Real y0 = cfg.bump_y0;
      const amrex::Real z0 = cfg.bump_z0;
      const amrex::Real b1_0 = cfg.b1_0;
      const amrex::Real b2_0 = cfg.b2_0;
      const amrex::Real b3_0 = cfg.b3_0;
      const amrex::Real k1 = two_pi * cfg.wave_number;
      const amrex::Real a0 = k1 != 0.0 ? cfg.alfven_amplitude / k1 : 0.0;
      const EOSParams eos = eos_params(cfg);
      const int eos_method = eos_method_code(cfg);

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
         const amrex::Real x = patch_coordinate_component(sample.mesh, 0);
         const amrex::Real y = patch_coordinate_component(sample.mesh, 1);
         const amrex::Real z = patch_coordinate_component(sample.mesh, 2);
         const amrex::Real rx = coordinate_delta(x, x0, problo[0], probhi[0], periodic[0] != 0);
         const amrex::Real ry = coordinate_delta(y, y0, problo[1], probhi[1], periodic[1] != 0);
         const amrex::Real rz = coordinate_delta(z, z0, problo[2], probhi[2], periodic[2] != 0);
         const amrex::Real r2 = rx * rx + ry * ry + rz * rz;
         const amrex::Real pressure = pressure0 + blast_amplitude * std::exp(-0.5 * r2 / sigma2);
         const amrex::Real bx_cell = b1_0 + magnetic_blast_noise_bx(x, y, z, a0, k1);
         const amrex::Real by_cell = b2_0 + magnetic_blast_noise_by(x, y, z, a0, k1);
         const amrex::Real bz_cell = b3_0 + magnetic_blast_noise_bz(x, y, z, a0, k1);
         arr(i, j, k, RHO) = rho0;
         arr(i, j, k, MOM1) = 0.0;
         arr(i, j, k, MOM2) = 0.0;
         arr(i, j, k, MOM3) = 0.0;
         arr(i, j, k, ENERGY) = total_mhd_energy_runtime(eos_method, rho0, 0.0, 0.0, 0.0, pressure, bx_cell, by_cell, bz_cell, eos);
         arr(i, j, k, B1) = bx_cell;
         arr(i, j, k, B2) = by_cell;
         arr(i, j, k, B3) = bz_cell;
      });
   }
   }
#endif
}

inline void initialize_magnetic_blast_noise_face_b_impl(FaceMagneticField& face_b, amrex::Geometry const& geom, RunConfig const& cfg)
{
#if (AMREX_SPACEDIM != 3)
   (void)face_b;
   (void)geom;
   (void)cfg;
   amrex::Abort("magnetic_blast_noise face initialization requires the 3D AMReX executable");
#else
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const amrex::Real two_pi = constants::two_pi;
   const amrex::Real b1_0 = cfg.b1_0;
   const amrex::Real b2_0 = cfg.b2_0;
   const amrex::Real b3_0 = cfg.b3_0;
   const amrex::Real k1 = two_pi * cfg.wave_number;
   const amrex::Real a0 = k1 != 0.0 ? cfg.alfven_amplitude / k1 : 0.0;

   for (int dir = 0; dir < 3; ++dir) {
      amrex::MultiFab& face = *face_b.b[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(face, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto f = face.array(mfi);
         const int d = dir;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (d == 0) {
               const amrex::Real x = problo[0] + static_cast<amrex::Real>(i) * dx[0];
               const amrex::Real y0_edge = problo[1] + static_cast<amrex::Real>(j) * dx[1];
               const amrex::Real yc = problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
               const amrex::Real z0_edge = problo[2] + static_cast<amrex::Real>(k) * dx[2];
               const amrex::Real zc = problo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
               const amrex::Real az_hi = magnetic_blast_noise_az(x, y0_edge + dx[1], zc, a0, k1);
               const amrex::Real az_lo = magnetic_blast_noise_az(x, y0_edge, zc, a0, k1);
               const amrex::Real ay_hi = magnetic_blast_noise_ay(x, yc, z0_edge + dx[2], a0, k1);
               const amrex::Real ay_lo = magnetic_blast_noise_ay(x, yc, z0_edge, a0, k1);
               f(i, j, k, 0) = b1_0 + (az_hi - az_lo) / dx[1] - (ay_hi - ay_lo) / dx[2];
            } else if (d == 1) {
               const amrex::Real x0_edge = problo[0] + static_cast<amrex::Real>(i) * dx[0];
               const amrex::Real xc = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
               const amrex::Real y = problo[1] + static_cast<amrex::Real>(j) * dx[1];
               const amrex::Real z0_edge = problo[2] + static_cast<amrex::Real>(k) * dx[2];
               const amrex::Real zc = problo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
               const amrex::Real ax_hi = magnetic_blast_noise_ax(xc, y, z0_edge + dx[2], a0, k1);
               const amrex::Real ax_lo = magnetic_blast_noise_ax(xc, y, z0_edge, a0, k1);
               const amrex::Real az_hi = magnetic_blast_noise_az(x0_edge + dx[0], y, zc, a0, k1);
               const amrex::Real az_lo = magnetic_blast_noise_az(x0_edge, y, zc, a0, k1);
               f(i, j, k, 0) = b2_0 + (ax_hi - ax_lo) / dx[2] - (az_hi - az_lo) / dx[0];
            } else {
               const amrex::Real x0_edge = problo[0] + static_cast<amrex::Real>(i) * dx[0];
               const amrex::Real xc = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
               const amrex::Real y0_edge = problo[1] + static_cast<amrex::Real>(j) * dx[1];
               const amrex::Real yc = problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
               const amrex::Real z = problo[2] + static_cast<amrex::Real>(k) * dx[2];
               const amrex::Real ay_hi = magnetic_blast_noise_ay(x0_edge + dx[0], yc, z, a0, k1);
               const amrex::Real ay_lo = magnetic_blast_noise_ay(x0_edge, yc, z, a0, k1);
               const amrex::Real ax_hi = magnetic_blast_noise_ax(xc, y0_edge + dx[1], z, a0, k1);
               const amrex::Real ax_lo = magnetic_blast_noise_ax(xc, y0_edge, z, a0, k1);
               f(i, j, k, 0) = b3_0 + (ay_hi - ay_lo) / dx[0] - (ax_hi - ax_lo) / dx[1];
            }
         });
      }
      }
   }
#endif
}

inline void apply_newtonian_magnetic_initial_data_policy(
   amrex::MultiFab& state,
   RunConfig const& cfg
)
{
   if (!physical_system_has_magnetic_field(physical_system(cfg))
       || cfg.initial_data_magnetic != "zero") {
      return;
   }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto values = state.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real magnetic_energy = 0.5 * (
            values(i, j, k, B1) * values(i, j, k, B1)
            + values(i, j, k, B2) * values(i, j, k, B2)
            + values(i, j, k, B3) * values(i, j, k, B3)
         );
         values(i, j, k, ENERGY) -= magnetic_energy;
         values(i, j, k, B1) = 0.0;
         values(i, j, k, B2) = 0.0;
         values(i, j, k, B3) = 0.0;
      });
   }
}

inline void initialize_newtonian_state(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
   if (cfg.ic == "uniform") {
      initialize_uniform(state, cfg);
   } else if (cfg.ic == "advection_medley") {
      initialize_advection_medley(state, geom, cfg);
   } else if (cfg.ic == "characteristic_medley") {
      initialize_characteristic_medley(state, geom, cfg);
   } else if (cfg.ic == "mhd_characteristic_medley") {
      initialize_mhd_characteristic_medley(state, geom, cfg);
   } else if (cfg.ic == "cartesian_advection") {
      initialize_cartesian_advection(state, geom, cfg);
   } else if (cfg.ic == "linear_wave_x") {
      initialize_linear_wave_x(state, geom, cfg);
   } else if (cfg.ic == "circular_alfven_x") {
      initialize_circular_alfven_x(state, geom, cfg);
   } else if (cfg.ic == "resistive_wave_x") {
      initialize_resistive_wave_x(state, geom, cfg);
   } else if (cfg.ic == "magnetic_tube_z") {
      initialize_magnetic_tube_z(state, geom, cfg);
   } else if (cfg.ic == "mhd_rotor") {
      initialize_mhd_rotor(state, geom, cfg);
   } else if (cfg.ic == "kelvin_helmholtz") {
      initialize_kelvin_helmholtz(state, geom, cfg);
   } else if (cfg.ic == "kelvin_helmholtz_mhd") {
      initialize_kelvin_helmholtz_mhd(state, geom, cfg);
   } else if (cfg.ic == "orszag_tang") {
      initialize_orszag_tang(state, geom, cfg);
   } else if (cfg.ic == "magnetic_blast_noise") {
      initialize_magnetic_blast_noise(state, geom, cfg);
   } else if (cfg.ic == "resistive_current_sheet") {
      initialize_resistive_current_sheet(state, geom, cfg);
   } else if (cfg.ic == "reconnecting_shear_layer") {
      initialize_reconnecting_shear_layer(state, geom, cfg);
   } else if (cfg.ic == "sod_x" || cfg.ic == "sod_plane" || cfg.ic == "sod_periodic_plane") {
      initialize_sod_plane(state, geom, cfg);
   } else if (cfg.ic == "double_mach_reflection") {
      initialize_double_mach_reflection(state, geom, cfg);
   } else if (cfg.ic == "pressure_bump") {
      initialize_pressure_bump(state, geom, cfg);
   } else if (cfg.ic == "vortex_blast") {
      initialize_vortex_blast(state, geom, cfg);
   } else if (cfg.ic == "density_cloud") {
      initialize_density_cloud(state, geom, cfg);
   } else if (cfg.ic == "hse_polytrope_x") {
      initialize_hse_polytrope_x(state, geom, cfg);
   } else if (cfg.ic == "hydro_beam") {
      initialize_hydro_beam(state, geom, cfg);
   } else if (cfg.ic == "acoustic_wave_x") {
      initialize_acoustic_wave_x(state, geom, cfg);
   } else if (cfg.ic == "acoustic_packet_x") {
      initialize_acoustic_packet_x(state, geom, cfg);
   } else if (cfg.ic == "jeans_wave_x") {
      initialize_jeans_wave_x(state, geom, cfg);
   } else if (cfg.ic == "shear_wave_x") {
      initialize_shear_wave_x(state, geom, cfg);
   } else {
      initialize_sine_wave(state, geom, cfg);
   }
   apply_newtonian_magnetic_initial_data_policy(state, cfg);
   initialize_passive_scalars(state, geom, cfg);
   apply_configured_state_boundary(state, geom, cfg);
}

inline void initialize_state_impl(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   RelativisticInitialDataView const& initial_data = {}
)
{
   if (!physical_system_is_relativistic(physical_system(cfg))) {
      initialize_newtonian_state(state, geom, cfg);
      return;
   }

   if (initial_data.has_fluid() || cfg.ic == "fixed_metric_hse") {
      initialize_relativistic_state(state, geom, cfg, initial_data);
   } else {
      const int newtonian_components = physical_system_has_magnetic_field(
         physical_system(cfg)
      ) ? NVAR_MHD : NVAR_HYDRO;
      amrex::MultiFab newtonian_state(
         state.boxArray(),
         state.DistributionMap(),
         newtonian_components,
         state.nGrow()
      );
      initialize_newtonian_state(newtonian_state, geom, cfg);
      convert_newtonian_initial_state_to_relativistic(
         state, newtonian_state, geom, cfg, initial_data
      );
      apply_configured_state_boundary(state, geom, cfg);
   }
}

} // namespace pyfluid_amrex
