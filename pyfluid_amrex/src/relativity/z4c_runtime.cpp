#include "pyfluid_amrex/z4c_runtime.hpp"

#include <AMReX_BC_TYPES.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>

#include <cstdint>
#include <cmath>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/coordinate_locations.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/spacetime_fields.hpp"
#include "pyfluid_amrex/z4c_derivatives.hpp"

namespace pyfluid_amrex {

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
   constexpr amrex::Real scale = 1.0 / static_cast<amrex::Real>(0x1p53);
   return 2.0 * static_cast<amrex::Real>(value >> 11) * scale - 1.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cState gauge_wave_state(amrex::Real phase, amrex::Real amplitude, amrex::Real k) noexcept
{
   Z4cState state;
   const amrex::Real sine = std::sin(phase);
   const amrex::Real cosine = std::cos(phase);
   const amrex::Real h = 1.0 - amplitude * sine;
   const amrex::Real chi = 1.0 / std::cbrt(h);
   const amrex::Real sqrt_h = std::sqrt(h);
   const amrex::Real k11 = -0.5 * amplitude * k * cosine / sqrt_h;
   const amrex::Real trace_k = k11 / h;
   state.chi = chi;
   state.gamma_tilde11 = chi * h;
   state.gamma_tilde22 = chi;
   state.gamma_tilde33 = chi;
   state.k_hat = trace_k;
   state.a_tilde11 = chi * (k11 - h * trace_k / 3.0);
   state.a_tilde22 = -chi * trace_k / 3.0;
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
   const amrex::Real gamma22 = 1.0 + h;
   const amrex::Real gamma33 = 1.0 - h;
   const amrex::Real determinant = gamma22 * gamma33;
   const amrex::Real chi = 1.0 / std::cbrt(determinant);
   const amrex::Real k22 = 0.5 * derivative;
   const amrex::Real k33 = -0.5 * derivative;
   const amrex::Real trace_k = k22 / gamma22 + k33 / gamma33;
   state.chi = chi;
   state.gamma_tilde11 = chi;
   state.gamma_tilde22 = chi * gamma22;
   state.gamma_tilde33 = chi * gamma33;
   state.k_hat = trace_k;
   state.a_tilde11 = -chi * trace_k / 3.0;
   state.a_tilde22 = chi * (k22 - gamma22 * trace_k / 3.0);
   state.a_tilde33 = chi * (k33 - gamma33 * trace_k / 3.0);
   return state;
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
      1.0,
      cfg.spacetime_slow_start_time,
      1.0,
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

amrex::Vector<amrex::BCRec> z4c_boundary_records(
   RunConfig const& cfg,
   amrex::Geometry const& geometry
)
{
   const auto periodic = geometry.isPeriodicArray();
   amrex::Vector<amrex::BCRec> records(Z4C_NUM_COMPONENTS);
   int component = 0;
   for (GridFieldInfo const& field : z4c_state_grid_fields().fields) {
      for (GridFieldComponentInfo const& info : field.components) {
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            if (periodic[direction]) {
               records[component].setLo(direction, amrex::BCType::int_dir);
               records[component].setHi(direction, amrex::BCType::int_dir);
               continue;
            }
            const bool reflect_lo = side_boundary_kind(cfg, direction, false)
               == BoundaryKind::Reflecting
               || side_boundary_kind(cfg, direction, false) == BoundaryKind::Symmetry;
            const bool reflect_hi = side_boundary_kind(cfg, direction, true)
               == BoundaryKind::Reflecting
               || side_boundary_kind(cfg, direction, true) == BoundaryKind::Symmetry;
            const int reflection =
               grid_field_reflection_parity(info.symmetry, direction) < 0
               ? amrex::BCType::reflect_odd
               : amrex::BCType::reflect_even;
            records[component].setLo(
               direction, reflect_lo ? reflection : amrex::BCType::foextrap
            );
            records[component].setHi(
               direction, reflect_hi ? reflection : amrex::BCType::foextrap
            );
         }
         ++component;
      }
   }
   return records;
}

namespace {

void initialize_z4c_state_at_offset(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   RunConfig const& cfg,
   amrex::Real time,
   RelativisticInitialDataView const& initial_data,
   amrex::Real coordinate_offset,
   bool fill_ghosts
)
{
   const Z4cInitialDataParameters parameters = z4c_initial_data_parameters(cfg);
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const bool node_centered = coordinate_offset == 0.0;
   const InitialDataFieldSelection fields = initial_data.fields;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion() && !fill_ghosts)
#endif
   for (amrex::MFIter mfi(
           state, fill_ghosts ? false : amrex::TilingIfNotGPU()
        ); mfi.isValid(); ++mfi) {
      const amrex::Box box = fill_ghosts ? mfi.fabbox() : mfi.tilebox();
      const auto output = state.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const auto mesh_point = mesh_coordinate_triplet_at_location(
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
         const auto physical = mesh_coordinate_triplet_to_cartesian(
            mesh_coordinate, mesh_point
         );
         const amrex::Real x = physical[0];
         const amrex::Real phase = constants::two_pi
            * parameters.wave_number * (x - time);
         Z4cState point;
         if (initial_data.has_spacetime()) {
            point = initial_data.sample(
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
            point.lapse = 1.0;
         }
         if (fields.zero_shift) {
            point.shift1 = 0.0;
            point.shift2 = 0.0;
            point.shift3 = 0.0;
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
      state, geometry, cfg, time, initial_data, 0.0, false
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
      state, geometry, cfg, time, initial_data, 0.5, true
   );
}

void complete_z4c_contracted_connections(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry
)
{
   const auto spacing = geometry.CellSizeArray();
   const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
      1.0 / spacing[0],
#if (AMREX_SPACEDIM >= 2)
      1.0 / spacing[1],
#else
      1.0,
#endif
#if (AMREX_SPACEDIM == 3)
      1.0 / spacing[2],
#else
      1.0,
#endif
   }};
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto values = state.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const Z4cState point = load_z4c_state(values, i, j, k);
         const Z4cSpatialDerivatives derivatives =
            gather_z4c_spatial_derivatives<4, AMREX_SPACEDIM>(
               values, i, j, k, inv_spacing, point
            );
         const Z4cConformalMetric metric = generated::z4c_conformal_metric(point);
         const Z4cConnections connections = generated::z4c_connections(
            point, derivatives, metric
         );
         values(i, j, k, Z4C_GAMMA_TILDE1) = connections.contracted_christoffel_tilde1;
         values(i, j, k, Z4C_GAMMA_TILDE2) = connections.contracted_christoffel_tilde2;
         values(i, j, k, Z4C_GAMMA_TILDE3) = connections.contracted_christoffel_tilde3;
      });
   }
}

} // namespace pyfluid_amrex
