#pragma once

#include <AMReX_GpuLaunch.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <cmath>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/coordinate_locations.hpp"
#include "pyfluid_amrex/gravity.hpp"
#include "pyfluid_amrex/initial_data/sampling.hpp"
#include "pyfluid_amrex/rhs.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

inline bool sources_enabled(RunConfig const& cfg)
{
   return cfg.source != "none" || gravity_enabled(cfg);
}

inline bool uniform_acceleration_enabled(RunConfig const& cfg)
{
   return cfg.source == "uniform_acceleration"
      && (std::abs(cfg.source_a1) > 0.0 || std::abs(cfg.source_a2) > 0.0 || std::abs(cfg.source_a3) > 0.0);
}

inline bool rotating_compact_heating_enabled(RunConfig const& cfg)
{
   return cfg.source == "rotating_compact_heating" && cfg.source_power > 0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real smooth_ramp(amrex::Real time, amrex::Real ramp_time) noexcept
{
   if (!(ramp_time > 0.0)) {
      return 1.0;
   }
   const amrex::Real x = amrex::min(amrex::max(time / ramp_time, 0.0), 1.0);
   return x * x * (3.0 - 2.0 * x);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compact_gaussian(amrex::Real squared_distance_in_sigmas) noexcept
{
   constexpr amrex::Real taper_start = 6.25;
   constexpr amrex::Real support_end = 9.0;
   if (squared_distance_in_sigmas >= support_end) {
      return 0.0;
   }
   const amrex::Real gaussian = std::exp(-0.5 * squared_distance_in_sigmas);
   if (squared_distance_in_sigmas <= taper_start) {
      return gaussian;
   }
   const amrex::Real x = (support_end - squared_distance_in_sigmas) / (support_end - taper_start);
   return gaussian * x * x * (3.0 - 2.0 * x);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rotating_compact_heating_rate(
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k,
   amrex::Real time,
   amrex::Real source_power,
   amrex::Real source_radius,
   amrex::Real inv_sigma2,
   amrex::Real angular_speed,
   amrex::Real phase0,
   int mode,
   amrex::Real envelope
) noexcept
{
   const InitialDataSamplePoint sample = cell_center_initial_data_sample_point(
      mesh_coordinate, mesh_axes, problo, dx, i, j, k
   );
   amrex::Real shape = 0.0;
   for (int n = 0; n < mode; ++n) {
      const amrex::Real phase =
         phase0 + angular_speed * time + PyFluidTwoPi * static_cast<amrex::Real>(n) / static_cast<amrex::Real>(mode);
      const amrex::Real center_x = source_radius * std::cos(phase);
      const amrex::Real center_y = source_radius * std::sin(phase);
      const amrex::Real delta_x = sample.physical.q1 - center_x;
      const amrex::Real delta_y = sample.physical.q2 - center_y;
      shape += compact_gaussian((delta_x * delta_x + delta_y * delta_y) * inv_sigma2);
   }
   return source_power * envelope * shape;
}

struct SourceStageContext {
   RunConfig const* cfg = nullptr;
   amrex::Geometry const* geom = nullptr;
   amrex::Real time = 0.0;

   // Geometry masks are optional source-domain restrictions. They do not
   // change the GF contract; they only tell sources which valid fluid cells
   // receive source terms.
   amrex::iMultiFab const* solid_mask = nullptr;
   amrex::Vector<int> const* solid_valid_kind = nullptr;

   // Gravity is an ordinary source from the RHS point of view. The solver
   // prepares phi/grad_phi at the current RK/subcycling stage time, then passes
   // the cached gradient here. Conservative Athena-style energy coupling also
   // receives the stage mass fluxes from the hyperbolic update.
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const* gravity_gradient = nullptr;
   std::array<amrex::Real, AMREX_SPACEDIM> const* gravity_acceleration_offset = nullptr;
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const* mass_fluxes = nullptr;
   bool suppress_athena_gravity_energy_source = false;
};

template <int NComp>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void add_uniform_acceleration_source_cell(
   amrex::Array4<amrex::Real> const& rhs,
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   amrex::Real ax,
   amrex::Real ay,
   amrex::Real az
) noexcept
{
   const amrex::Real rho = state(i, j, k, RHO);
   const StateVector3 momentum = load_momentum_vector3(state, i, j, k);
   const StateVector3 acceleration = {ax, ay, az};
   rhs(i, j, k, MOM1) += rho * ax;
#if (AMREX_SPACEDIM >= 2)
   rhs(i, j, k, MOM2) += rho * ay;
#else
   (void)ay;
#endif
#if (AMREX_SPACEDIM == 3)
   rhs(i, j, k, MOM3) += rho * az;
#else
   (void)az;
#endif
   rhs(i, j, k, ENERGY) += dot(momentum, acceleration);
}

template <int NComp>
inline void apply_uniform_acceleration_source(
   amrex::MultiFab& rhs,
   amrex::MultiFab const& state,
   RunConfig const& cfg,
   amrex::iMultiFab const* solid_mask = nullptr,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   if (!uniform_acceleration_enabled(cfg)) {
      return;
   }

   const amrex::Real ax = cfg.source_a1;
   const amrex::Real ay = cfg.source_a2;
   const amrex::Real az = cfg.source_a3;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto r = rhs.array(mfi);
      const auto s = state.const_array(mfi);
      if (solid_mask == nullptr) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            add_uniform_acceleration_source_cell<NComp>(r, s, i, j, k, ax, ay, az);
         });
         continue;
      }

      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, *solid_mask, bx);
      if (kind == MaskKindNonzero) {
         continue;
      }
      if (kind == MaskKindZero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            add_uniform_acceleration_source_cell<NComp>(r, s, i, j, k, ax, ay, az);
         });
         continue;
      }

      const auto solid = solid_mask->const_array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (solid(i, j, k, 0) == 0) {
            add_uniform_acceleration_source_cell<NComp>(r, s, i, j, k, ax, ay, az);
         }
      });
   }
   }
}

template <int NComp>
inline void apply_rotating_compact_heating_source(
   amrex::MultiFab& rhs,
   RunConfig const& cfg,
   amrex::Geometry const& geom,
   amrex::Real time,
   amrex::iMultiFab const* solid_mask = nullptr,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   if (!rotating_compact_heating_enabled(cfg)) {
      return;
   }

   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(mesh_coordinate, cfg.mesh_axes);
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const amrex::Real source_power = cfg.source_power;
   const amrex::Real source_radius = cfg.source_radius;
   const amrex::Real inv_sigma2 = 1.0 / (cfg.source_sigma * cfg.source_sigma);
   const amrex::Real angular_speed = cfg.source_omega;
   const amrex::Real phase0 = cfg.source_phase;
   const int mode = cfg.source_azimuthal_mode;
   const amrex::Real envelope = smooth_ramp(time, cfg.source_ramp_time);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto r = rhs.array(mfi);
      if (solid_mask == nullptr) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            r(i, j, k, ENERGY) += rotating_compact_heating_rate(
               mesh_coordinate,
               mesh_axes,
               problo,
               dx,
               i,
               j,
               k,
               time,
               source_power,
               source_radius,
               inv_sigma2,
               angular_speed,
               phase0,
               mode,
               envelope
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
            r(i, j, k, ENERGY) += rotating_compact_heating_rate(
               mesh_coordinate,
               mesh_axes,
               problo,
               dx,
               i,
               j,
               k,
               time,
               source_power,
               source_radius,
               inv_sigma2,
               angular_speed,
               phase0,
               mode,
               envelope
            );
         });
         continue;
      }

      const auto solid = solid_mask->const_array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (solid(i, j, k, 0) != 0) {
            return;
         }
         r(i, j, k, ENERGY) += rotating_compact_heating_rate(
            mesh_coordinate,
            mesh_axes,
            problo,
            dx,
            i,
            j,
            k,
            time,
            source_power,
            source_radius,
            inv_sigma2,
            angular_speed,
            phase0,
            mode,
            envelope
         );
      });
   }
   }
   amrex::ignore_unused(mesh_coordinate);
}

template <int NComp>
inline void apply_sources(
   amrex::MultiFab& rhs,
   amrex::MultiFab const& state,
   SourceStageContext const& context
)
{
   if (context.cfg == nullptr) {
      amrex::Abort("source stage is missing run configuration");
   }
   RunConfig const& cfg = *context.cfg;
   (void)context.time;
   if (!sources_enabled(cfg)) {
      return;
   }
   if (cfg.source == "uniform_acceleration") {
      apply_uniform_acceleration_source<NComp>(rhs, state, cfg, context.solid_mask, context.solid_valid_kind);
   } else if (cfg.source == "rotating_compact_heating") {
      if (context.geom == nullptr) {
         amrex::Abort("rotating compact heating source is missing geometry");
      }
      apply_rotating_compact_heating_source<NComp>(
         rhs,
         cfg,
         *context.geom,
         context.time,
         context.solid_mask,
         context.solid_valid_kind
      );
   } else if (cfg.source != "none") {
      amrex::Abort("unsupported pyfluid.source");
   }
   if (gravity_enabled(cfg)) {
      if (context.gravity_gradient == nullptr) {
         amrex::Abort("gravity source requested before gravity gradient was computed");
      }
      std::array<amrex::Real, AMREX_SPACEDIM> zero_offset{};
      apply_gravity_source<NComp>(
         rhs,
         state,
         *context.gravity_gradient,
         context.gravity_acceleration_offset != nullptr ? *context.gravity_acceleration_offset : zero_offset,
         context.solid_mask,
         context.solid_valid_kind,
         athena_gravity_coupling_enabled(cfg) && !context.suppress_athena_gravity_energy_source,
         context.mass_fluxes,
         context.suppress_athena_gravity_energy_source
      );
   }
}

} // namespace pyfluid_amrex
