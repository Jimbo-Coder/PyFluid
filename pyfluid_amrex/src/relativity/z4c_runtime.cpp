#include "pyfluid_amrex/z4c_runtime.hpp"

#include <AMReX_BC_TYPES.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuAsyncArray.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_Print.H>

#include <cstdint>
#include <cmath>
#include <optional>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/fccz4_runtime.hpp"
#include "pyfluid_amrex/coordinate_locations.hpp"
#include "pyfluid_amrex/initial_data.hpp"
#include "pyfluid_amrex/initial_data/elliptic.hpp"
#include "pyfluid_amrex/initial_data/provider.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/spacetime_fields.hpp"
#include "pyfluid_amrex/z4c_derivatives.hpp"
#include "pyfluid_amrex/spacetime_excision.hpp"
#include "pyfluid_amrex/z4c_excision_derivatives.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

namespace {
struct MetricFailure {
   int found = 0;
   int i = 0, j = 0, k = 0;
   amrex::Real gamma11 = 0.0_rt, minor = 0.0_rt, determinant = 0.0_rt;
   amrex::Real chi = 0.0_rt, lapse = 0.0_rt;
};
} // namespace

// Debug-only stage contract. Algebraic determinant/trace projection cannot
// repair a metric whose signature has already changed, nor may interpolation
// feed such a metric into fluid recovery. Catch that cause before NaN cascades.
void debug_check_spacetime_metric(amrex::MultiFab const& state,
   amrex::IntVect const& ghost_width, bool enabled, bool apply_floors_next)
{
   if (!enabled) return;
   const bool nodal = state.is_nodal();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const auto values = state.const_array(mfi);
      amrex::Gpu::DeviceScalar<MetricFailure> failure(MetricFailure{});
      auto* failure_data = failure.dataPtr();
      amrex::ParallelFor(mfi.growntilebox(ghost_width),
         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const auto point = load_z4c_state(values, i, j, k);
            const amrex::Real determinant = generated::z4c_conformal_metric(point).determinant;
            const amrex::Real minor = point.gamma_tilde11 * point.gamma_tilde22
               - point.gamma_tilde12 * point.gamma_tilde12;
            // The selected nodal projection owns explicit chi/lapse floors.
            // A transferred metric has no such repair step and must already
            // have positive scalars. Neither path can repair metric signature.
            const bool scalars_valid = apply_floors_next || (point.chi > 0.0_rt && point.lapse > 0.0_rt);
            if (!(point.gamma_tilde11 > 0.0_rt && minor > 0.0_rt && determinant > 0.0_rt
                  && amrex::Math::isfinite(minor) && amrex::Math::isfinite(determinant)
                  && scalars_valid)) {
               if (amrex::Gpu::Atomic::CAS(&failure_data->found, 0, 1) == 0) {
                  failure_data->i = i;
                  failure_data->j = j;
                  failure_data->k = k;
                  failure_data->gamma11 = point.gamma_tilde11;
                  failure_data->minor = minor;
                  failure_data->determinant = determinant;
                  failure_data->chi = point.chi;
                  failure_data->lapse = point.lapse;
               }
            }
         });
      // Varargs printing promotes float to double. Transfer one failure record
      // and format it on the host, retaining all diagnostics on FP32-only GPUs.
      const auto invalid = failure.dataValue();
      if (invalid.found) {
         amrex::AllPrint() << "Z4c metric signature failure: nodal=" << nodal
            << " index=(" << invalid.i << ',' << invalid.j << ',' << invalid.k
            << ") gamma11=" << invalid.gamma11 << " minor2=" << invalid.minor
            << " determinant=" << invalid.determinant << " chi=" << invalid.chi
            << " lapse=" << invalid.lapse << '\n';
         amrex::Abort("Z4c stage metric is not positive definite with positive chi/lapse");
      }
   }
}


namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real signed_noise(int i, int j, int k, int component) noexcept
{
   std::uint64_t value = static_cast<std::uint64_t>(i + 104729);
   value ^= static_cast<std::uint64_t>(j + 130363) * 0x9e3779b97f4a7c15ULL;
   value ^= static_cast<std::uint64_t>(k + 155921) * 0xbf58476d1ce4e5b9ULL;
   value ^= static_cast<std::uint64_t>(component + 1) * 0x94d049bb133111ebULL;
   value ^= value >> 30;
   value *= 0xbf58476d1ce4e5b9ULL;
   value ^= value >> 27;
   value *= 0x94d049bb133111ebULL;
   value ^= value >> 31;
   constexpr amrex::Real scale = 1.0_rt / static_cast<amrex::Real>(0x1p53);
   return 2.0_rt * static_cast<amrex::Real>(value >> 11) * scale - 1.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cState gauge_wave_state(amrex::Real phase, amrex::Real amplitude, amrex::Real k) noexcept
{
   Z4cState state;
   const amrex::Real sine = std::sin(phase);
   const amrex::Real cosine = std::cos(phase);
   const amrex::Real h = 1.0_rt - amplitude * sine;
   const amrex::Real chi = 1.0_rt / std::cbrt(h);
   const amrex::Real sqrt_h = std::sqrt(h);
   const amrex::Real k11 = -0.5_rt * amplitude * k * cosine / sqrt_h;
   const amrex::Real trace_k = k11 / h;
   state.chi = chi;
   state.gamma_tilde11 = chi * h;
   state.gamma_tilde22 = chi;
   state.gamma_tilde33 = chi;
   state.k_hat = trace_k;
   state.a_tilde11 = chi * (k11 - h * trace_k / 3.0_rt);
   state.a_tilde22 = -chi * trace_k / 3.0_rt;
   state.a_tilde33 = state.a_tilde22;
   state.lapse = sqrt_h;
   return state;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cState linear_wave_state(amrex::Real phase, amrex::Real amplitude, amrex::Real k) noexcept
{
   Z4cState state;
   const amrex::Real h = amplitude * std::sin(phase);
   const amrex::Real derivative = amplitude * k * std::cos(phase);
   const amrex::Real gamma22 = 1.0_rt + h;
   const amrex::Real gamma33 = 1.0_rt - h;
   const amrex::Real determinant = gamma22 * gamma33;
   const amrex::Real chi = 1.0_rt / std::cbrt(determinant);
   const amrex::Real k22 = 0.5_rt * derivative;
   const amrex::Real k33 = -0.5_rt * derivative;
   const amrex::Real trace_k = k22 / gamma22 + k33 / gamma33;
   state.chi = chi;
   state.gamma_tilde11 = chi;
   state.gamma_tilde22 = chi * gamma22;
   state.gamma_tilde33 = chi * gamma33;
   state.k_hat = trace_k;
   state.a_tilde11 = -chi * trace_k / 3.0_rt;
   state.a_tilde22 = chi * (k22 - gamma22 * trace_k / 3.0_rt);
   state.a_tilde33 = chi * (k33 - gamma33 * trace_k / 3.0_rt);
   return state;
}

template <int spatial_order>
void fill_z4c_connections(
   amrex::Box const& box,
   amrex::Array4<amrex::Real> values,
   ExcisionGridView const& excision,
   amrex::GpuArray<amrex::Real, 3> const& inv_spacing
)
{
   amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      if (excision.region.enabled() && !excision(i, j, k, 0)) return;
      const Z4cState point = load_z4c_state(values, i, j, k);
      Z4cSpatialDerivatives derivatives{};
      if (excision.region.enabled()) {
         // Preserve the full gather's stencil-side selection, including
         // its upwind/KO reach, even though only first derivatives remain.
         const auto plan = excision.derivative_stencil(
            i, j, k, upwind_dissipation_radius(spatial_order));
         derivatives = gather_z4c_metric_derivatives<spatial_order, AMREX_SPACEDIM>(
            values, i, j, k, inv_spacing,
            ExcisionZ4cDerivativeOperator<spatial_order>{plan});
      } else {
         derivatives = gather_z4c_metric_derivatives<spatial_order, AMREX_SPACEDIM>(
            values, i, j, k, inv_spacing);
      }
      const Z4cConformalMetric metric = generated::z4c_conformal_metric(point);
      const Z4cConnections connections = generated::z4c_connections(
         point, derivatives, metric);
      values(i, j, k, Z4C_GAMMA_TILDE1) = connections.contracted_christoffel_tilde1;
      values(i, j, k, Z4C_GAMMA_TILDE2) = connections.contracted_christoffel_tilde2;
      values(i, j, k, Z4C_GAMMA_TILDE3) = connections.contracted_christoffel_tilde3;
   });
}

} // namespace

Z4cInitialDataParameters z4c_initial_data_parameters(RunConfig const& cfg)
{
   Z4cInitialData kind = Z4cInitialData::Minkowski;
   if (cfg.spacetime_initial_data == "gauge_wave") {
      kind = Z4cInitialData::GaugeWave;
   } else if (cfg.spacetime_initial_data == "linear_wave") {
      kind = Z4cInitialData::LinearWave;
   } else if (cfg.spacetime_initial_data == "robust_stability") {
      kind = Z4cInitialData::RobustStability;
   }
   return {kind, cfg.spacetime_amplitude, cfg.spacetime_wave_number};
}

Z4cEvolutionParameters z4c_evolution_parameters(RunConfig const& cfg)
{
   return {
      cfg.spacetime_kappa1,
      cfg.spacetime_kappa2,
      cfg.spacetime_slow_start_strength,
      cfg.spacetime_slow_start_time,
      1.0_rt,
      cfg.spacetime_shift_eta,
   };
}

Z4cLapseGauge z4c_lapse_gauge(RunConfig const& cfg)
{
   if (cfg.spacetime_lapse_gauge == "harmonic") {
      return Z4cLapseGauge::Harmonic;
   }
   if (cfg.spacetime_lapse_gauge == "one_plus_log") {
      return Z4cLapseGauge::OnePlusLog;
   }
   if (cfg.spacetime_lapse_gauge == "trumpet") {
      return Z4cLapseGauge::Trumpet;
   }
   if (cfg.spacetime_lapse_gauge == "slow_start") {
      return Z4cLapseGauge::SlowStart;
   }
   return Z4cLapseGauge::Frozen;
}

Z4cShiftGauge z4c_shift_gauge(RunConfig const& cfg)
{
   return cfg.spacetime_shift_gauge == "gamma_driver"
      ? Z4cShiftGauge::GammaDriver
      : Z4cShiftGauge::Frozen;
}

namespace {

void initialize_z4c_state_at_offset(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   RunConfig const& cfg,
   amrex::Real time,
   RelativisticInitialDataView const& initial_data,
   amrex::Real coordinate_offset,
   bool fill_ghosts,
   InitialDataEllipticHierarchy const* hierarchy = nullptr,
   int level = 0
)
{
   const Z4cInitialDataParameters parameters = z4c_initial_data_parameters(cfg);
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const bool node_centered = coordinate_offset == 0.0_rt;
   const InitialDataFieldSelection fields = initial_data.fields;
   const amrex::Dim3 domain_lo = amrex::lbound(geometry.Domain());
   std::optional<amrex::Gpu::AsyncArray<amrex::Real>> coordinate1_storage;
   amrex::Real const* coordinate1 = nullptr;
   if (hierarchy != nullptr) {
      const InitialDataCoordinateLine line = hierarchy->coordinate_line(
         level, 0
      );
      auto const& values = node_centered ? line.faces : line.centers;
      coordinate1_storage.emplace(values.data(), values.size());
      coordinate1 = coordinate1_storage->data();
   }
#if (AMREX_SPACEDIM >= 2)
   std::optional<amrex::Gpu::AsyncArray<amrex::Real>> coordinate2_storage;
   amrex::Real const* coordinate2 = nullptr;
   if (hierarchy != nullptr) {
      const InitialDataCoordinateLine line = hierarchy->coordinate_line(
         level, 1
      );
      auto const& values = node_centered ? line.faces : line.centers;
      coordinate2_storage.emplace(values.data(), values.size());
      coordinate2 = coordinate2_storage->data();
   }
#endif
#if (AMREX_SPACEDIM == 3)
   std::optional<amrex::Gpu::AsyncArray<amrex::Real>> coordinate3_storage;
   amrex::Real const* coordinate3 = nullptr;
   if (hierarchy != nullptr) {
      const InitialDataCoordinateLine line = hierarchy->coordinate_line(
         level, 2
      );
      auto const& values = node_centered ? line.faces : line.centers;
      coordinate3_storage.emplace(values.data(), values.size());
      coordinate3 = coordinate3_storage->data();
   }
#endif
   const bool mapped_coordinates = hierarchy != nullptr;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion() && !fill_ghosts)
#endif
   for (amrex::MFIter mfi(
           state, fill_ghosts ? false : amrex::TilingIfNotGPU()
        ); mfi.isValid(); ++mfi) {
      const amrex::Box box = fill_ghosts ? mfi.fabbox() : mfi.tilebox();
      const auto output = state.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         amrex::GpuArray<amrex::Real, 3> mesh_point{};
         if (mapped_coordinates) {
            mesh_point[0] = coordinate1[i - domain_lo.x];
#if (AMREX_SPACEDIM >= 2)
            mesh_point[1] = coordinate2[j - domain_lo.y];
#else
            amrex::ignore_unused(j);
#endif
#if (AMREX_SPACEDIM == 3)
            mesh_point[2] = coordinate3[k - domain_lo.z];
#else
            amrex::ignore_unused(k);
#endif
         } else {
            mesh_point = mesh_coordinate_triplet_at_location(
               mesh_coordinate,
               mesh_axes,
               lower,
               spacing,
               i,
               j,
               k,
               node_centered,
               AMREX_SPACEDIM >= 2 && node_centered,
               AMREX_SPACEDIM == 3 && node_centered
            );
         }
         const auto physical = mesh_coordinate_triplet_to_cartesian(
            mesh_coordinate, mesh_point
         );
         const amrex::Real x = physical[0];
         const amrex::Real phase = constants::two_pi
            * parameters.wave_number * (x - time);
         Z4cState point;
         if (initial_data.has_spacetime()) {
            point = mesh_coordinate == MeshSpherical
                  && initial_data.tov.active()
               ? initial_data.sample_tov_radius(mesh_point[0]).spacetime
               : initial_data.sample(
                    {physical[0], physical[1], physical[2]}
                 ).spacetime;
         } else if (parameters.kind == Z4cInitialData::GaugeWave) {
            point = gauge_wave_state(
               phase, parameters.amplitude, constants::two_pi * parameters.wave_number
            );
         } else if (parameters.kind == Z4cInitialData::LinearWave) {
            point = linear_wave_state(
               phase, parameters.amplitude, constants::two_pi * parameters.wave_number
            );
         }
         if (fields.flat_spacetime) {
            const amrex::Real provider_lapse = point.lapse;
            const amrex::Real provider_shift1 = point.shift1;
            const amrex::Real provider_shift2 = point.shift2;
            const amrex::Real provider_shift3 = point.shift3;
            point = Z4cState{};
            point.lapse = provider_lapse;
            point.shift1 = provider_shift1;
            point.shift2 = provider_shift2;
            point.shift3 = provider_shift3;
         }
         if (fields.unit_lapse) {
            point.lapse = 1.0_rt;
         }
         if (fields.zero_shift) {
            point.shift1 = 0.0_rt;
            point.shift2 = 0.0_rt;
            point.shift3 = 0.0_rt;
         }
         store_z4c_state(output, i, j, k, point);
         if (parameters.kind == Z4cInitialData::RobustStability) {
            for (int component = 0; component < Z4C_NUM_COMPONENTS; ++component) {
               const bool initialize_component = component < Z4C_LAPSE
                  ? fields.spacetime_from_id
                  : (component == Z4C_LAPSE
                        ? fields.lapse_from_id
                        : fields.shift_from_id);
               if (!initialize_component) {
                  continue;
               }
               output(i, j, k, component) += parameters.amplitude
                  * signed_noise(i, j, k, component);
            }
         }
      });
   }
}

} // namespace

void initialize_z4c_state(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   RunConfig const& cfg,
   amrex::Real time,
   RelativisticInitialDataView const& initial_data
)
{
   initialize_z4c_state_at_offset(
      state, geometry, cfg, time, initial_data, 0.0_rt, false
   );
}

void initialize_mapped_z4c_state_from_provider(
   amrex::MultiFab& state,
   InitialDataEllipticHierarchy const& hierarchy,
   int level,
   RunConfig const& cfg,
   amrex::Real time,
   RelativisticInitialDataView const& initial_data
)
{
   if (level < 0 || level >= static_cast<int>(hierarchy.geometry.size())) {
      amrex::Abort("mapped Z4c initialization level is invalid");
   }
   initialize_z4c_state_at_offset(
      state,
      hierarchy.geometry[level],
      cfg,
      time,
      initial_data,
      0.0_rt,
      false,
      &hierarchy,
      level
   );
}

void initialize_z4c_cell_state(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   RunConfig const& cfg,
   amrex::Real time,
   RelativisticInitialDataView const& initial_data
)
{
   initialize_z4c_state_at_offset(
      state, geometry, cfg, time, initial_data, 0.5_rt, true
   );
}

void complete_spacetime_connections(
   amrex::MultiFab& state, amrex::Geometry const& geometry, RunConfig const& config)
{
   if (config.spacetime_formulation == "fccz4") {
      complete_fccz4_connections(state, geometry, config);
   } else {
      complete_z4c_contracted_connections(
         state, geometry, spacetime_excision_region(config), config.spacetime_derivative_order);
   }
}

void complete_z4c_contracted_connections(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   SphericalExcisionRegion const& excision_region,
   int derivative_order
)
{
   const auto excision = excision_grid_view(excision_region, geometry, state.ixType());
   const auto spacing = geometry.CellSizeArray();
   const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
      1.0_rt / spacing[0],
#if (AMREX_SPACEDIM >= 2)
      1.0_rt / spacing[1],
#else
      1.0_rt,
#endif
#if (AMREX_SPACEDIM == 3)
      1.0_rt / spacing[2],
#else
      1.0_rt,
#endif
   }};
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto values = state.array(mfi);
      // Select the stencil family at launch scope. Inlining the full runtime
      // derivative gather here embeds every order, including unused second,
      // mixed and advection derivatives, into a single device kernel.
      dispatch_spatial_order(derivative_order, [&](auto order_tag) {
         constexpr int spatial_order = decltype(order_tag)::value;
         fill_z4c_connections<spatial_order>(box, values, excision, inv_spacing);
      });
   }
}

} // namespace pyfluid_amrex
