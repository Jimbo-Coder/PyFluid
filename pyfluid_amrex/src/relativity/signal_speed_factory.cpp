#include "pyfluid_amrex/reference_chart.hpp"
#include "pyfluid_amrex/signal_speed_factory.hpp"

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <cmath>
#include <limits>
#include <sstream>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/diffusion_timestep.hpp"
#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/mask_regions.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/patch_grid.hpp"
#include "pyfluid_amrex/patch_layout_builder.hpp"
#include "pyfluid_amrex/patch_metrics.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/reference_metric_spacetime.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/flux.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/z4c_characteristics.hpp"
#include "pyfluid_amrex/z4c_runtime.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rhd_signal_frequency_cell(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& inverse_spacing,
   EOSParams const& eos,
   const SpacetimeMetric& metric
) noexcept
{
   const RHDConserved conserved{
      state(i, j, k, RHO),
      state(i, j, k, MOM1),
      state(i, j, k, MOM2),
      state(i, j, k, MOM3),
      state(i, j, k, ENERGY),
   };
   const RHDRecoveryResult recovered = recover_rhd<EOSMethod>(
      conserved, metric, eos
   );
   if (!recovered.attempt.succeeded()) {
      return std::numeric_limits<amrex::Real>::infinity();
   }
   const RHDPrimitive& primitive = recovered.primitive;
   const amrex::Real sound_speed_squared = relativistic_sound_speed_squared<EOSMethod>(
      primitive.rho, primitive.eps, eos
   );
   const RHDWaveSpeeds speed1 = generated::rhd_wave_speeds<0>(
      primitive, metric, sound_speed_squared
   );
   amrex::Real frequency = amrex::max(
      amrex::Math::abs(speed1.minus), amrex::Math::abs(speed1.plus)
   ) * inverse_spacing[0];
#if (AMREX_SPACEDIM >= 2)
   const RHDWaveSpeeds speed2 = generated::rhd_wave_speeds<1>(
      primitive, metric, sound_speed_squared
   );
   frequency += amrex::max(
      amrex::Math::abs(speed2.minus), amrex::Math::abs(speed2.plus)
   ) * inverse_spacing[1];
#endif
#if (AMREX_SPACEDIM == 3)
   const RHDWaveSpeeds speed3 = generated::rhd_wave_speeds<2>(
      primitive, metric, sound_speed_squared
   );
   frequency += amrex::max(
      amrex::Math::abs(speed3.minus), amrex::Math::abs(speed3.plus)
   ) * inverse_spacing[2];
#endif
   return frequency;
}

template <int EOSMethod>
inline void fill_rhd_signal_frequency(
   amrex::MultiFab& frequency,
   const amrex::MultiFab& state,
   amrex::Geometry const& geom,
   EOSParams const& eos,
   const FixedSpacetimeParameters& spacetime,
   amrex::Real time,
   int coordinate,
   MeshAxes const& axes
)
{
   const auto dx = geom.CellSizeArray();
   const auto lower = geom.ProbLoArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto conserved = state.const_array(mfi);
      const auto cell_frequency = frequency.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> inverse_spacing{};
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            inverse_spacing[direction] = 1.0_rt / mesh_face_gradient_spacing(
               coordinate, axes, lower, dx, direction, i, j, k
            );
         }
         const ReferenceMetricSpacetimeGeometry local_geometry =
            reference_metric_fixed_spacetime_geometry_at_cell(
               spacetime, coordinate, axes, lower, dx, i, j, k, time
            );
         cell_frequency(i, j, k, 0) = rhd_signal_frequency_cell<EOSMethod>(
            conserved, i, j, k, inverse_spacing, eos, local_geometry.metric
         );
      });
   }
}

template <int EOSMethod>
inline void fill_rhd_signal_frequency(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   EOSParams const& eos,
   amrex::MultiFab const& cell_spacetime,
   int coordinate,
   MeshAxes const& axes
)
{
   const auto dx = geom.CellSizeArray();
   const auto lower = geom.ProbLoArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto conserved = state.const_array(mfi);
      const auto spacetime = cell_spacetime.const_array(mfi);
      const auto cell_frequency = frequency.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> inverse_spacing{};
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            inverse_spacing[direction] = 1.0_rt / mesh_face_gradient_spacing(
               coordinate, axes, lower, dx, direction, i, j, k);
         }
         cell_frequency(i, j, k, 0) = rhd_signal_frequency_cell<EOSMethod>(
            conserved,
            i,
            j,
            k,
            inverse_spacing,
            eos,
            generated::z4c_spacetime_metric(load_z4c_state(spacetime, i, j, k))
         );
      });
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rmhd_signal_frequency_cell(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   EOSParams const& eos,
   const SpacetimeMetric& metric
) noexcept
{
   // Preserve the factory's host EOS selection through recovery and speeds.
   // Ideal-gas kernels should not carry the unused tabulated EOS machinery.
   const RMHDConserved conserved = load_rmhd_conserved(state, i, j, k);
   RMHDRecoveryResult recovered;
   if constexpr (EOSMethod == EOSIdealGas) {
      recovered = recover_srmhd(conserved, metric, eos.gamma);
   } else {
      recovered = recover_srmhd(conserved, metric, eos);
   }
   if (!recovered.succeeded()) {
      return std::numeric_limits<amrex::Real>::infinity();
   }
   const RMHDPrimitive& primitive = recovered.primitive;
   const RelativisticCharacteristicState thermodynamics =
      relativistic_characteristic_state<EOSMethod>(
         primitive.rho, primitive.eps, eos
      );
   const RMHDWaveSpeeds speed1 = generated::srmhd_wave_speeds<0>(
      primitive,
      metric,
      thermodynamics.sound_speed_squared,
      thermodynamics.specific_enthalpy
   );
   amrex::Real frequency = amrex::max(
      amrex::Math::abs(speed1.minus), amrex::Math::abs(speed1.plus)
   ) / dx[0];
#if (AMREX_SPACEDIM >= 2)
   const RMHDWaveSpeeds speed2 = generated::srmhd_wave_speeds<1>(
      primitive,
      metric,
      thermodynamics.sound_speed_squared,
      thermodynamics.specific_enthalpy
   );
   frequency += amrex::max(
      amrex::Math::abs(speed2.minus), amrex::Math::abs(speed2.plus)
   ) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
   const RMHDWaveSpeeds speed3 = generated::srmhd_wave_speeds<2>(
      primitive,
      metric,
      thermodynamics.sound_speed_squared,
      thermodynamics.specific_enthalpy
   );
   frequency += amrex::max(
      amrex::Math::abs(speed3.minus), amrex::Math::abs(speed3.plus)
   ) / dx[2];
#endif
   return frequency;
}

template <int EOSMethod>
inline void fill_rmhd_signal_frequency(
   amrex::MultiFab& frequency,
   const amrex::MultiFab& state,
   amrex::Geometry const& geom,
   EOSParams const& eos,
   const FixedSpacetimeParameters& spacetime,
   amrex::Real time
)
{
   const auto dx = geom.CellSizeArray();
   const auto lower = geom.ProbLoArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto conserved = state.const_array(mfi);
      const auto cell_frequency = frequency.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const FixedSpacetimeGeometry local_geometry =
            fixed_spacetime_geometry_at_cell<AMREX_SPACEDIM>(
               spacetime, lower, dx, i, j, k, time
            );
         cell_frequency(i, j, k, 0) = rmhd_signal_frequency_cell<EOSMethod>(
            conserved, i, j, k, dx, eos, local_geometry.metric
         );
      });
   }
}

template <int EOSMethod>
inline void fill_rmhd_signal_frequency(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   EOSParams const& eos,
   amrex::MultiFab const& cell_spacetime
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto conserved = state.const_array(mfi);
      const auto spacetime = cell_spacetime.const_array(mfi);
      const auto cell_frequency = frequency.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         cell_frequency(i, j, k, 0) = rmhd_signal_frequency_cell<EOSMethod>(
            conserved,
            i,
            j,
            k,
            dx,
            eos,
            generated::z4c_spacetime_metric(load_z4c_state(spacetime, i, j, k))
         );
      });
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hydro_signal_frequency_cell(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   EOSParams const& eos
) noexcept
{
   const amrex::Real rho = state(i, j, k, RHO);
   const amrex::Real pressure = pressure_from_state<EOSMethod>(state, i, j, k, eos);
   const amrex::Real cs = sound_speed_from_primitive<EOSMethod>(rho, pressure, eos);
   const StateVector3 velocity = load_velocity_vector3(state, i, j, k);
   amrex::Real value = (amrex::Math::abs(velocity.x) + cs) / dx[0];
#if (AMREX_SPACEDIM >= 2)
   value += (amrex::Math::abs(velocity.y) + cs) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
   value += (amrex::Math::abs(velocity.z) + cs) / dx[2];
#endif
   return value;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hydro_signal_frequency_cell_metric(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int coordinate,
   MeshAxes const& axes,
   EOSParams const& eos
) noexcept
{
   const amrex::Real rho = state(i, j, k, RHO);
   const amrex::Real pressure = pressure_from_state<EOSMethod>(state, i, j, k, eos);
   const amrex::Real cs = sound_speed_from_primitive<EOSMethod>(rho, pressure, eos);
   const StateVector3 velocity = load_velocity_vector3(state, i, j, k);
   amrex::Real value =
      (amrex::Math::abs(velocity.x) + cs)
      / mesh_face_gradient_spacing(coordinate, axes, problo, dx, 0, i, j, k);
#if (AMREX_SPACEDIM >= 2)
   value +=
      (amrex::Math::abs(velocity.y) + cs)
      / mesh_face_gradient_spacing(coordinate, axes, problo, dx, 1, i, j, k);
#endif
#if (AMREX_SPACEDIM == 3)
   value +=
      (amrex::Math::abs(velocity.z) + cs)
      / mesh_face_gradient_spacing(coordinate, axes, problo, dx, 2, i, j, k);
#endif
   return value;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_hydro_signal_frequency_cell(
   amrex::Array4<amrex::Real const> const& state,
   amrex::Array4<amrex::Real> const& frequency,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   EOSParams const& eos
) noexcept
{
   const amrex::Real value = hydro_signal_frequency_cell<EOSMethod>(state, i, j, k, dx, eos);
   frequency(i, j, k, 0) = value;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mhd_signal_frequency_cell(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   EOSParams const& eos
) noexcept
{
   const MHDPrimitive primitive = primitive_from_conserved<EOSMethod>(load_mhd_state(state, i, j, k), eos);
   amrex::Real value = (amrex::Math::abs(primitive.v1) + mhd_fast_speed_from_primitive<EOSMethod>(primitive, 0, eos)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
   value += (amrex::Math::abs(primitive.v2) + mhd_fast_speed_from_primitive<EOSMethod>(primitive, 1, eos)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
   value += (amrex::Math::abs(primitive.v3) + mhd_fast_speed_from_primitive<EOSMethod>(primitive, 2, eos)) / dx[2];
#endif
   return value;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mhd_signal_frequency_cell_metric(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int coordinate,
   MeshAxes const& axes,
   EOSParams const& eos
) noexcept
{
   const MHDPrimitive primitive = primitive_from_conserved<EOSMethod>(load_mhd_state(state, i, j, k), eos);
   amrex::Real value =
      (amrex::Math::abs(primitive.v1) + mhd_fast_speed_from_primitive<EOSMethod>(primitive, 0, eos))
      / mesh_face_gradient_spacing(coordinate, axes, problo, dx, 0, i, j, k);
#if (AMREX_SPACEDIM >= 2)
   value +=
      (amrex::Math::abs(primitive.v2) + mhd_fast_speed_from_primitive<EOSMethod>(primitive, 1, eos))
      / mesh_face_gradient_spacing(coordinate, axes, problo, dx, 1, i, j, k);
#endif
#if (AMREX_SPACEDIM == 3)
   value +=
      (amrex::Math::abs(primitive.v3) + mhd_fast_speed_from_primitive<EOSMethod>(primitive, 2, eos))
      / mesh_face_gradient_spacing(coordinate, axes, problo, dx, 2, i, j, k);
#endif
   return value;
}

template <int EOSMethod>
inline void fill_signal_frequency(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   EOSParams const& eos
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto f = frequency.array(mfi);
      if (curvilinear_mesh_coordinate(coordinate)) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            f(i, j, k, 0) = hydro_signal_frequency_cell_metric<EOSMethod>(s, i, j, k, problo, dx, coordinate, axes, eos);
         });
      } else {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            f(i, j, k, 0) = hydro_signal_frequency_cell<EOSMethod>(s, i, j, k, dx, eos);
         });
      }
   }
   }
}

template <int EOSMethod>
inline void fill_signal_frequency_mapped_hydro(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout,
   EOSParams const& eos
)
{
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
      const auto s = state.const_array(mfi);
      const auto f = frequency.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const StateVector3 velocity = load_velocity_vector3(s, i, j, k);
         const amrex::Real rho = s(i, j, k, RHO);
         const amrex::Real pressure = pressure_from_state<EOSMethod>(s, i, j, k, eos);
         const amrex::Real cs = sound_speed_from_primitive<EOSMethod>(rho, pressure, eos);
         const PhysicalVector physical_velocity = physical_vector(velocity.x, velocity.y, velocity.z);
         const PatchCoordinate center =
            patch_layout_cell_center_coordinate(grid_patch, layout_patch, i, j, k);
         const amrex::Real volume = patch_cell_volume(grid_patch, center);
         if (volume <= 0.0_rt || !amrex::Math::isfinite(volume)) {
            f(i, j, k, 0) = 0.0_rt;
            return;
         }
         amrex::Real frequency_value = 0.0_rt;
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const PatchCoordinate face_lo =
               patch_layout_face_coordinate(grid_patch, layout_patch, dir, i, j, k);
            amrex::IntVect hi_cell(AMREX_D_DECL(i, j, k));
            hi_cell[dir] += 1;
            const int hi_i = hi_cell[0];
#if (AMREX_SPACEDIM >= 2)
            const int hi_j = hi_cell[1];
#else
            const int hi_j = 0;
#endif
#if (AMREX_SPACEDIM == 3)
            const int hi_k = hi_cell[2];
#else
            const int hi_k = 0;
#endif
               const PatchCoordinate face_hi =
                  patch_layout_face_coordinate(grid_patch, layout_patch, dir, hi_i, hi_j, hi_k);
               const PatchFaceMetric metric_lo = patch_face_metric(grid_patch, face_lo, dir);
               const PatchFaceMetric metric_hi = patch_face_metric(grid_patch, face_hi, dir);
            const amrex::Real signal_lo = amrex::Math::abs(dot(physical_velocity, metric_lo.unit_normal)) + cs;
            const amrex::Real signal_hi = amrex::Math::abs(dot(physical_velocity, metric_hi.unit_normal)) + cs;
            frequency_value += 0.5_rt * (
               metric_lo.area * signal_lo
               + metric_hi.area * signal_hi
            ) / volume;
         }
         f(i, j, k, 0) = frequency_value;
      });
   }
   }
}

template <int EOSMethod>
inline void fill_signal_frequency_masked(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   EOSParams const& eos,
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
      const auto s = state.const_array(mfi);
      const auto solid = solid_mask.const_array(mfi);
      const auto f = frequency.array(mfi);
      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
      if (kind == MaskKindNonzero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            f(i, j, k, 0) = 0.0_rt;
         });
         continue;
      }
      if (kind == MaskKindZero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            store_hydro_signal_frequency_cell<EOSMethod>(s, f, i, j, k, dx, eos);
         });
         continue;
      }
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (solid(i, j, k, 0) != 0) {
            f(i, j, k, 0) = 0.0_rt;
            return;
         }
         store_hydro_signal_frequency_cell<EOSMethod>(s, f, i, j, k, dx, eos);
      });
   }
   }
}

template <int EOSMethod>
inline void fill_mhd_signal_frequency(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   EOSParams const& eos
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto f = frequency.array(mfi);
      if (curvilinear_mesh_coordinate(coordinate)) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            f(i, j, k, 0) = mhd_signal_frequency_cell_metric<EOSMethod>(s, i, j, k, problo, dx, coordinate, axes, eos);
         });
      } else {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            f(i, j, k, 0) = mhd_signal_frequency_cell<EOSMethod>(s, i, j, k, dx, eos);
         });
      }
   }
   }
}

template <int EOSMethod>
inline void fill_mhd_signal_frequency_masked(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   EOSParams const& eos,
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
      const auto s = state.const_array(mfi);
      const auto solid = solid_mask.const_array(mfi);
      const auto f = frequency.array(mfi);
      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
      if (kind == MaskKindNonzero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            f(i, j, k, 0) = 0.0_rt;
         });
         continue;
      }
      if (kind == MaskKindZero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            f(i, j, k, 0) = mhd_signal_frequency_cell<EOSMethod>(s, i, j, k, dx, eos);
         });
         continue;
      }
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (solid(i, j, k, 0) != 0) {
            f(i, j, k, 0) = 0.0_rt;
            return;
         }
         f(i, j, k, 0) = mhd_signal_frequency_cell<EOSMethod>(s, i, j, k, dx, eos);
      });
   }
   }
}

template <int EOSMethod>
inline amrex::Real max_hydro_signal_frequency_masked(
   amrex::MultiFab const& state,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   EOSParams const& eos,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
   const bool curvilinear = curvilinear_mesh_coordinate(coordinate);
   amrex::Real max_frequency = 0.0_rt;
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(max:max_frequency) if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
      if (kind == MaskKindNonzero) {
         continue;
      }
      const auto s = state.const_array(mfi);
      const auto solid = solid_mask.const_array(mfi);
      const auto lo = bx.smallEnd();
      const auto hi = bx.bigEnd();
#if (AMREX_SPACEDIM == 1)
      constexpr int jlo = 0;
      constexpr int jhi = 0;
      constexpr int klo = 0;
      constexpr int khi = 0;
#elif (AMREX_SPACEDIM == 2)
      const int jlo = lo[1];
      const int jhi = hi[1];
      constexpr int klo = 0;
      constexpr int khi = 0;
#else
      const int jlo = lo[1];
      const int jhi = hi[1];
      const int klo = lo[2];
      const int khi = hi[2];
#endif
      for (int k = klo; k <= khi; ++k) {
            for (int j = jlo; j <= jhi; ++j) {
               for (int i = lo[0]; i <= hi[0]; ++i) {
                  if (kind == MaskKindMixed && solid(i, j, k, 0) != 0) {
                     continue;
                  }
                  const amrex::Real frequency = curvilinear
                     ? hydro_signal_frequency_cell_metric<EOSMethod>(s, i, j, k, problo, dx, coordinate, axes, eos)
                     : hydro_signal_frequency_cell<EOSMethod>(s, i, j, k, dx, eos);
                  max_frequency = amrex::max(max_frequency, frequency);
            }
         }
      }
   }
   }
   amrex::ParallelDescriptor::ReduceRealMax(max_frequency);
   return max_frequency;
}

template <int EOSMethod>
inline amrex::Real max_mhd_signal_frequency_masked(
   amrex::MultiFab const& state,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   EOSParams const& eos,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
   const bool curvilinear = curvilinear_mesh_coordinate(coordinate);
   amrex::Real max_frequency = 0.0_rt;
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(max:max_frequency) if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const int kind = cached_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
      if (kind == MaskKindNonzero) {
         continue;
      }
      const auto s = state.const_array(mfi);
      const auto solid = solid_mask.const_array(mfi);
      const auto lo = bx.smallEnd();
      const auto hi = bx.bigEnd();
#if (AMREX_SPACEDIM == 1)
      constexpr int jlo = 0;
      constexpr int jhi = 0;
      constexpr int klo = 0;
      constexpr int khi = 0;
#elif (AMREX_SPACEDIM == 2)
      const int jlo = lo[1];
      const int jhi = hi[1];
      constexpr int klo = 0;
      constexpr int khi = 0;
#else
      const int jlo = lo[1];
      const int jhi = hi[1];
      const int klo = lo[2];
      const int khi = hi[2];
#endif
      for (int k = klo; k <= khi; ++k) {
         for (int j = jlo; j <= jhi; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
               if (kind == MaskKindMixed && solid(i, j, k, 0) != 0) {
                  continue;
               }
               const amrex::Real frequency = curvilinear
                  ? mhd_signal_frequency_cell_metric<EOSMethod>(s, i, j, k, problo, dx, coordinate, axes, eos)
                  : mhd_signal_frequency_cell<EOSMethod>(s, i, j, k, dx, eos);
               max_frequency = amrex::max(max_frequency, frequency);
            }
         }
      }
   }
   }
   amrex::ParallelDescriptor::ReduceRealMax(max_frequency);
   return max_frequency;
}

template <int EOSMethod>
inline amrex::Real max_hydro_cutcell_signal_frequency_masked(
   amrex::MultiFab const& state,
   amrex::iMultiFab const& solid_mask,
   amrex::MultiFab const& volume_fraction,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& apertures,
   amrex::Geometry const& geom,
   EOSParams const& eos,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   constexpr amrex::Real alpha_floor = 1.0e-12_rt;
   const auto dx = geom.CellSizeArray();
   const amrex::Box& domain = geom.Domain();
   const auto dlo = domain.smallEnd();
   const auto dhi = domain.bigEnd();
   amrex::Real max_frequency = 0.0_rt;
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(max:max_frequency) if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
      if (kind == MaskKindNonzero) {
         continue;
      }
      const auto s = state.const_array(mfi);
      const auto lo = bx.smallEnd();
      const auto hi = bx.bigEnd();
#if (AMREX_SPACEDIM == 1)
      constexpr int jlo = 0;
      constexpr int jhi = 0;
      constexpr int klo = 0;
      constexpr int khi = 0;
#elif (AMREX_SPACEDIM == 2)
      const int jlo = lo[1];
      const int jhi = hi[1];
      constexpr int klo = 0;
      constexpr int khi = 0;
#else
      const int jlo = lo[1];
      const int jhi = hi[1];
      const int klo = lo[2];
      const int khi = hi[2];
#endif
      if (kind == MaskKindZero) {
         for (int k = klo; k <= khi; ++k) {
            for (int j = jlo; j <= jhi; ++j) {
               for (int i = lo[0]; i <= hi[0]; ++i) {
                  max_frequency = amrex::max(max_frequency, hydro_signal_frequency_cell<EOSMethod>(s, i, j, k, dx, eos));
               }
            }
         }
         continue;
      }
      const auto solid = solid_mask.const_array(mfi);
      const auto alpha = volume_fraction.const_array(mfi);
      const auto ax = apertures[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto ay = apertures[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto az = apertures[2]->const_array(mfi);
#endif
      for (int k = klo; k <= khi; ++k) {
         for (int j = jlo; j <= jhi; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
               const amrex::Real fluid_fraction = alpha(i, j, k, 0);
               if ((kind == MaskKindMixed && solid(i, j, k, 0) != 0) || fluid_fraction <= alpha_floor) {
                  continue;
               }
               const StateVector3 velocity = load_velocity_vector3(s, i, j, k);
               const amrex::Real rho = s(i, j, k, RHO);
               const amrex::Real pressure = pressure_from_state<EOSMethod>(s, i, j, k, eos);
               const amrex::Real cs = sound_speed_from_primitive<EOSMethod>(rho, pressure, eos);

               amrex::Real ax_l = ax(i, j, k, 0);
               if (i > dlo[0] && alpha(i - 1, j, k, 0) <= alpha_floor) {
                  ax_l = 0.0_rt;
               }
               amrex::Real ax_r = ax(i + 1, j, k, 0);
               if (i < dhi[0] && alpha(i + 1, j, k, 0) <= alpha_floor) {
                  ax_r = 0.0_rt;
               }
               amrex::Real frequency =
                  0.5_rt * (ax_l + ax_r) * (amrex::Math::abs(velocity.x) + cs) / dx[0];
#if (AMREX_SPACEDIM >= 2)
               amrex::Real ay_l = ay(i, j, k, 0);
               if (j > dlo[1] && alpha(i, j - 1, k, 0) <= alpha_floor) {
                  ay_l = 0.0_rt;
               }
               amrex::Real ay_r = ay(i, j + 1, k, 0);
               if (j < dhi[1] && alpha(i, j + 1, k, 0) <= alpha_floor) {
                  ay_r = 0.0_rt;
               }
               frequency +=
                  0.5_rt * (ay_l + ay_r) * (amrex::Math::abs(velocity.y) + cs) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
               amrex::Real az_l = az(i, j, k, 0);
               if (k > dlo[2] && alpha(i, j, k - 1, 0) <= alpha_floor) {
                  az_l = 0.0_rt;
               }
               amrex::Real az_r = az(i, j, k + 1, 0);
               if (k < dhi[2] && alpha(i, j, k + 1, 0) <= alpha_floor) {
                  az_r = 0.0_rt;
               }
               frequency +=
                  0.5_rt * (az_l + az_r) * (amrex::Math::abs(velocity.z) + cs) / dx[2];
#endif
               max_frequency = amrex::max(max_frequency, frequency / fluid_fraction);
            }
         }
      }
   }
   }
   amrex::ParallelDescriptor::ReduceRealMax(max_frequency);
   return max_frequency;
}

template <int NComp, int EOSMethod, int DiffusionMethod>
inline amrex::Real adaptive_dt(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Real time,
   amrex::MultiFab const* stored_spacetime = nullptr,
   amrex::iMultiFab const* inactive_mask = nullptr
)
{
   const EOSParams eos = eos_params(cfg);
   if (physical_system(cfg) == PhysicalSystem::RHD) {
      if (stored_spacetime != nullptr) {
         fill_rhd_signal_frequency<EOSMethod>(frequency, state, geom, eos, *stored_spacetime,
            mesh_coordinate_code(cfg.mesh_coordinate), mesh_axes_layout(cfg));
      } else {
         fill_rhd_signal_frequency<EOSMethod>(
            frequency,
            state,
            geom,
            eos,
            fixed_spacetime_parameters(cfg),
            time,
            mesh_coordinate_code(cfg.mesh_coordinate),
            mesh_axes_layout(cfg)
         );
      }
   } else if (physical_system(cfg) == PhysicalSystem::RMHD) {
      if (stored_spacetime != nullptr) {
         fill_rmhd_signal_frequency<EOSMethod>(frequency, state, geom, eos, *stored_spacetime);
      } else {
         fill_rmhd_signal_frequency<EOSMethod>(
            frequency, state, geom, eos, fixed_spacetime_parameters(cfg), time
         );
      }
   } else if constexpr (has_mhd_components<NComp>()) {
      fill_mhd_signal_frequency<EOSMethod>(frequency, state, geom, cfg, eos);
   } else {
      fill_signal_frequency<EOSMethod>(frequency, state, geom, cfg, eos);
   }
   if (inactive_mask != nullptr) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(frequency, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box& box = mfi.tilebox();
         const auto values = frequency.array(mfi);
         const auto inactive = inactive_mask->const_array(mfi);
         amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if (inactive(i, j, k, 0) != 0) {
                  values(i, j, k, 0) = 0.0_rt;
               }
            }
         );
      }
   }
   const amrex::Real max_frequency = frequency.max(0);
   if (!std::isfinite(max_frequency)) {
      const amrex::IntVect location = frequency.maxIndex(0);
      std::ostringstream message;
      const auto spacing = geom.CellSizeArray();
      message
         << "fluid CFL evaluation produced a nonfinite signal frequency at "
         << location << " with first-axis cell spacing " << spacing[0]
         << "; relativistic runs usually indicate a local primitive-recovery "
            "or metric failure";
      amrex::Abort(message.str().c_str());
   }
   const amrex::Real advective_dt = cfg.cfl / max_frequency;
   return amrex::min(advective_dt, diffusion_dt_limit<DiffusionMethod>(geom, cfg));
}

template <int NComp, int EOSMethod, int DiffusionMethod>
inline amrex::Real adaptive_dt_mapped_hydro(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   RunConfig const& cfg,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout
)
{
   static_assert(!has_mhd_components<NComp>(), "mapped patch timestep helper is hydro-only");
   if constexpr (DiffusionMethod != DiffusionNone) {
      amrex::ignore_unused(frequency, state, cfg, grid_system, layout);
      amrex::Abort("mapped patch hydro diffusion timestep support is not implemented");
      return std::numeric_limits<amrex::Real>::max();
   } else {
      const EOSParams eos = eos_params(cfg);
      fill_signal_frequency_mapped_hydro<EOSMethod>(frequency, state, grid_system, layout, eos);
      const amrex::Real max_frequency = frequency.max(0);
      if (max_frequency <= 0.0_rt) {
         return std::numeric_limits<amrex::Real>::max();
      }
      return cfg.cfl / max_frequency;
   }
}

template <int NComp, int EOSMethod, int DiffusionMethod>
inline amrex::Real adaptive_dt_masked(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const EOSParams eos = eos_params(cfg);
   (void)frequency;
   amrex::Real max_frequency = 0.0_rt;
   if constexpr (has_mhd_components<NComp>()) {
      max_frequency = max_mhd_signal_frequency_masked<EOSMethod>(state, solid_mask, geom, cfg, eos, solid_valid_kind);
   } else {
      max_frequency = max_hydro_signal_frequency_masked<EOSMethod>(state, solid_mask, geom, cfg, eos, solid_valid_kind);
   }
   if (max_frequency <= 0.0_rt) {
      return std::numeric_limits<amrex::Real>::max();
   }
   const amrex::Real advective_dt = cfg.cfl / max_frequency;
   return amrex::min(advective_dt, diffusion_dt_limit<DiffusionMethod>(geom, cfg));
}

template <int NComp, int EOSMethod, int DiffusionMethod>
inline amrex::Real adaptive_dt_cutcell_masked(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& state,
   amrex::iMultiFab const& solid_mask,
   amrex::MultiFab const& volume_fraction,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& apertures,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const EOSParams eos = eos_params(cfg);
   (void)frequency;
   amrex::Real max_frequency = 0.0_rt;
   if constexpr (!has_mhd_components<NComp>()) {
      max_frequency = max_hydro_cutcell_signal_frequency_masked<EOSMethod>(
         state,
         solid_mask,
         volume_fraction,
         apertures,
         geom,
         eos,
         solid_valid_kind
      );
   } else {
      max_frequency = max_mhd_signal_frequency_masked<EOSMethod>(state, solid_mask, geom, cfg, eos, solid_valid_kind);
   }
   if (max_frequency <= 0.0_rt) {
      return std::numeric_limits<amrex::Real>::max();
   }
   const amrex::Real advective_dt = cfg.cfl / max_frequency;
   return amrex::min(advective_dt, diffusion_dt_limit<DiffusionMethod>(geom, cfg));
}

} // namespace pyfluid_amrex
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {
namespace {

template <int NumComponents, int EosMethod, int DiffusionMethod>
amrex::Real launch_standard(
   SignalSpeedLaunch const& launch,
   amrex::Geometry const& geometry,
   amrex::Real time,
   amrex::MultiFab const* cell_spacetime
)
{
   return adaptive_dt<NumComponents, EosMethod, DiffusionMethod>(
      launch.frequency,
      launch.state,
      geometry,
      launch.config,
      time,
      cell_spacetime,
      launch.inactive_mask
   );
}

template <int NumComponents, int EosMethod, int DiffusionMethod>
amrex::Real launch_mapped(
   SignalSpeedLaunch const& launch,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout
)
{
   return adaptive_dt_mapped_hydro<NumComponents, EosMethod, DiffusionMethod>(
      launch.frequency,
      launch.state,
      launch.config,
      grid_system,
      layout
   );
}

template <int NumComponents, int EosMethod, int DiffusionMethod>
amrex::Real launch_cutcell(
   SignalSpeedLaunch const& launch,
   amrex::iMultiFab const& solid_mask,
   amrex::MultiFab const& volume_fraction,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& apertures,
   amrex::Geometry const& geometry,
   amrex::Vector<int> const* solid_valid_kind
)
{
   return adaptive_dt_cutcell_masked<NumComponents, EosMethod, DiffusionMethod>(
      launch.frequency,
      launch.state,
      solid_mask,
      volume_fraction,
      apertures,
      geometry,
      launch.config,
      solid_valid_kind
   );
}

template <int NumComponents, int EosMethod, typename Operation>
amrex::Real dispatch_diffusion(int diffusion_method, Operation&& operation)
{
   switch (diffusion_method) {
   case DiffusionNone:
      return operation.template operator()<NumComponents, EosMethod, DiffusionNone>();
   case DiffusionExplicit:
      return operation.template operator()<NumComponents, EosMethod, DiffusionExplicit>();
   case DiffusionImplicit:
      return operation.template operator()<NumComponents, EosMethod, DiffusionImplicit>();
   default:
      amrex::Abort("invalid diffusion method in signal-speed factory");
      return 0.0_rt;
   }
}

template <int NumComponents, typename Operation>
amrex::Real dispatch_runtime_eos(
   SignalSpeedLaunch const& launch,
   Operation&& operation
)
{
   if (eos_method_code(launch.config) == EOSIdealGas) {
      return dispatch_diffusion<NumComponents, EOSIdealGas>(
         launch.diffusion_method, operation
      );
   }
   return dispatch_diffusion<NumComponents, EOSRuntime>(
      launch.diffusion_method, operation
   );
}

template <typename Operation>
amrex::Real dispatch_layout(SignalSpeedLaunch const& launch, Operation&& operation)
{
   switch (launch.num_components) {
   case NVAR_HYDRO:
      return dispatch_runtime_eos<NVAR_HYDRO>(launch, operation);
   case NVAR_HYDRO + 1:
      return dispatch_runtime_eos<NVAR_HYDRO + 1>(launch, operation);
   case NVAR_MHD:
      return dispatch_runtime_eos<NVAR_MHD>(launch, operation);
   case NVAR_MHD + 1:
      return dispatch_runtime_eos<NVAR_MHD + 1>(launch, operation);
   default:
      amrex::Abort("unsupported state layout in signal-speed factory");
      return 0.0_rt;
   }
}

} // namespace

amrex::Real compute_adaptive_dt(
   SignalSpeedLaunch const& launch,
   amrex::Geometry const& geometry,
   amrex::Real time,
   amrex::MultiFab const* cell_spacetime
)
{
   const auto operation = [&]<int NumComponents, int EosMethod, int DiffusionMethod>() {
      return launch_standard<NumComponents, EosMethod, DiffusionMethod>(
         launch, geometry, time, cell_spacetime
      );
   };
   return dispatch_layout(launch, operation);
}

amrex::Real compute_spacetime_dt(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& cell_spacetime,
   amrex::Geometry const& geometry,
   RunConfig const& config
)
{
   const auto spacing = geometry.CellSizeArray();
   const ReferenceChartGeometry reference_chart(geometry, config);
   const bool use_reference_metric = config.spacetime_formulation == "fccz4";
   const auto parameters = z4c_evolution_parameters(config);
   const auto lapse_gauge = z4c_lapse_gauge(config);
   const auto shift_gauge = z4c_shift_gauge(config);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(frequency, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto spacetime = cell_spacetime.const_array(mfi);
      const auto output = frequency.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const Z4cState local = load_z4c_state(spacetime, i, j, k);
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(local);
         const amrex::Real gauge_speed = std::sqrt(z4c_maximum_speed_squared(
            local, parameters, lapse_gauge, shift_gauge));
         const auto chart = reference_chart.at(i, j, k, false);
         amrex::Real value = (
            amrex::Math::abs(metric.shift1)
            + gauge_speed * metric.lapse * std::sqrt(metric.inv_gamma11)
         ) / spacing[0] * (use_reference_metric ? reference_inverse_scale(chart, 0) : 1.0_rt);
#if (AMREX_SPACEDIM >= 2)
         value += (
            amrex::Math::abs(metric.shift2)
            + gauge_speed * metric.lapse * std::sqrt(metric.inv_gamma22)
         ) / spacing[1] * (use_reference_metric ? reference_inverse_scale(chart, 1) : 1.0_rt);
#endif
#if (AMREX_SPACEDIM == 3)
         value += (
            amrex::Math::abs(metric.shift3)
            + gauge_speed * metric.lapse * std::sqrt(metric.inv_gamma33)
         ) / spacing[2] * (use_reference_metric ? reference_inverse_scale(chart, 2) : 1.0_rt);
#endif
         output(i, j, k, 0) = value;
      });
   }
   const amrex::Real max_frequency = frequency.max(0);
   if (!std::isfinite(max_frequency)) {
      const amrex::IntVect location = frequency.maxIndex(0);
      std::ostringstream message;
      const auto spacing = geometry.CellSizeArray();
      message
         << "spacetime CFL evaluation produced a nonfinite characteristic "
            "frequency at " << location << " with cell spacing "
         << spacing[0];
      amrex::Abort(message.str().c_str());
   }
   return config.cfl / max_frequency;
}

amrex::Real compute_mapped_hydro_adaptive_dt(
   SignalSpeedLaunch const& launch,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout
)
{
   const auto operation = [&]<int NumComponents, int EosMethod, int DiffusionMethod>() {
      return launch_mapped<NumComponents, EosMethod, DiffusionMethod>(
         launch, grid_system, layout
      );
   };
   switch (launch.num_components) {
   case NVAR_HYDRO:
      return dispatch_runtime_eos<NVAR_HYDRO>(launch, operation);
   case NVAR_HYDRO + 1:
      return dispatch_runtime_eos<NVAR_HYDRO + 1>(launch, operation);
   default:
      amrex::Abort("mapped-patch signal-speed launch requires a hydro state layout");
      return 0.0_rt;
   }
}

amrex::Real compute_cutcell_adaptive_dt(
   SignalSpeedLaunch const& launch,
   amrex::iMultiFab const& solid_mask,
   amrex::MultiFab const& volume_fraction,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& apertures,
   amrex::Geometry const& geometry,
   amrex::Vector<int> const* solid_valid_kind
)
{
   const auto operation = [&]<int NumComponents, int EosMethod, int DiffusionMethod>() {
      return launch_cutcell<NumComponents, EosMethod, DiffusionMethod>(
         launch,
         solid_mask,
         volume_fraction,
         apertures,
         geometry,
         solid_valid_kind
      );
   };
   return dispatch_layout(launch, operation);
}

} // namespace pyfluid_amrex
