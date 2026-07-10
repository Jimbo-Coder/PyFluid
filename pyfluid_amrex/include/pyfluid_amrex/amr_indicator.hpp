#pragma once

#include <AMReX_Array4.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "pyfluid_amrex/coordinate_locations.hpp"
#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mesh_vector_component_for_axis(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int coordinate,
   MeshAxes const& axes,
   int target_axis,
   int component_begin,
   bool divide_by_rho
) noexcept
{
   const amrex::Real inv_rho =
      divide_by_rho ? 1.0 / amrex::max(state(i, j, k, RHO), 1.0e-300) : 1.0;
   for (int component = 0; component < 3; ++component) {
      if (mesh_vector_axis_for_component(coordinate, axes, component) == target_axis) {
         return state(i, j, k, component_begin + component) * inv_rho;
      }
   }
   return 0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> mesh_vector_to_cartesian(
   amrex::Real v1,
   amrex::Real v2,
   amrex::Real v3,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   if (coordinate == MeshCylindrical) {
      const amrex::Real phi =
         runtime_coordinate_for_axis(axes, MeshAxisPhi, problo, dx, i, j, k, false, false, false);
      const amrex::Real cphi = std::cos(phi);
      const amrex::Real sphi = std::sin(phi);
      return {v1 * cphi - v2 * sphi, v1 * sphi + v2 * cphi, v3};
   }

   if (coordinate == MeshSpherical) {
      const amrex::Real theta =
         runtime_coordinate_for_axis(axes, MeshAxisTheta, problo, dx, i, j, k, false, false, false);
      const amrex::Real phi =
         runtime_coordinate_for_axis(axes, MeshAxisPhi, problo, dx, i, j, k, false, false, false);
      const amrex::Real stheta = std::sin(theta);
      const amrex::Real ctheta = std::cos(theta);
      const amrex::Real cphi = std::cos(phi);
      const amrex::Real sphi = std::sin(phi);
      return {
         v1 * stheta * cphi + v2 * ctheta * cphi - v3 * sphi,
         v1 * stheta * sphi + v2 * ctheta * sphi + v3 * cphi,
         v1 * ctheta - v2 * stheta,
      };
   }

   amrex::Real vx = 0.0;
   amrex::Real vy = 0.0;
   amrex::Real vz = 0.0;
   for (int component = 0; component < 3; ++component) {
      const amrex::Real value = component == 0 ? v1 : (component == 1 ? v2 : v3);
      const int axis = mesh_vector_axis_for_component(coordinate, axes, component);
      if (axis == MeshAxisX) {
         vx = value;
      } else if (axis == MeshAxisY) {
         vy = value;
      } else if (axis == MeshAxisZ) {
         vz = value;
      }
   }
   return {vx, vy, vz};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real state_vector_cartesian_component(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int component_begin,
   bool divide_by_rho,
   int cartesian_component
) noexcept
{
   const amrex::GpuArray<int, 3> canonical_axes = mesh_canonical_vector_axes(coordinate);
   const auto axis_component = [&](int axis) noexcept -> amrex::Real {
      if (!mesh_has_axis(axes, axis)) {
         return 0.0;
      }
      return mesh_vector_component_for_axis(
         state,
         i,
         j,
         k,
         coordinate,
         axes,
         axis,
         component_begin,
         divide_by_rho
      );
   };
   const amrex::Real v1 = axis_component(canonical_axes[0]);
   const amrex::Real v2 = axis_component(canonical_axes[1]);
   const amrex::Real v3 = axis_component(canonical_axes[2]);
   const auto cartesian = mesh_vector_to_cartesian(v1, v2, v3, coordinate, axes, problo, dx, i, j, k);
   return cartesian[cartesian_component];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool amr_cell_allows_radial_refinement(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k,
   amrex::Real min_refine_radius
) noexcept
{
   if (min_refine_radius <= 0.0 || coordinate == MeshCartesian) {
      return true;
   }
   const int radial_dir = mesh_direction_for_axis(axes, MeshAxisR);
   if (radial_dir < 0) {
      return true;
   }
   const amrex::Real radius =
      runtime_coordinate_for_axis(axes, MeshAxisR, problo, dx, i, j, k, false, false, false);
   return radius >= min_refine_radius;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real lohner_from_values(
   amrex::Real qm,
   amrex::Real q0,
   amrex::Real qp,
   amrex::Real reference_scale
) noexcept
{
   constexpr amrex::Real smooth_floor = 1.0e-2;
   const amrex::Real numerator = amrex::Math::abs(qp - 2.0 * q0 + qm);
   const amrex::Real denominator =
      amrex::Math::abs(qp - q0)
      + amrex::Math::abs(q0 - qm)
      + smooth_floor * (
         amrex::Math::abs(qp)
         + 2.0 * amrex::Math::abs(q0)
         + amrex::Math::abs(qm)
         + reference_scale
      )
      + 1.0e-50;
   return numerator / denominator;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool amr_direction_is_unresolved(
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int dir,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& min_feature_lengths
) noexcept
{
   const amrex::Real min_feature_length = min_feature_lengths[dir];
   if (min_feature_length <= 0.0) {
      return true;
   }
   return mesh_face_gradient_spacing(coordinate, axes, problo, dx, dir, i, j, k) > min_feature_length;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real directional_indicator_from_values(
   amrex::Real qm,
   amrex::Real q0,
   amrex::Real qp,
   amrex::Real reference_scale
) noexcept
{
   return lohner_from_values(qm, q0, qp, reference_scale);
}

template <typename ValueAt>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void accumulate_metric_aware_directional_indicator(
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& indicators,
   ValueAt value_at,
   int i,
   int j,
   int k,
   amrex::Real reference_scale,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& min_feature_lengths
) noexcept
{
   const amrex::Real q0 = value_at(i, j, k);
   if (amr_direction_is_unresolved(coordinate, axes, problo, dx, 0, i, j, k, min_feature_lengths)) {
      indicators[0] = amrex::max(
         indicators[0],
         directional_indicator_from_values(value_at(i - 1, j, k), q0, value_at(i + 1, j, k), reference_scale)
      );
   }
#if (AMREX_SPACEDIM >= 2)
   if (amr_direction_is_unresolved(coordinate, axes, problo, dx, 1, i, j, k, min_feature_lengths)) {
      indicators[1] = amrex::max(
         indicators[1],
         directional_indicator_from_values(value_at(i, j - 1, k), q0, value_at(i, j + 1, k), reference_scale)
      );
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (amr_direction_is_unresolved(coordinate, axes, problo, dx, 2, i, j, k, min_feature_lengths)) {
      indicators[2] = amrex::max(
         indicators[2],
         directional_indicator_from_values(value_at(i, j, k - 1), q0, value_at(i, j, k + 1), reference_scale)
      );
   }
#endif
}

template <typename ValueAt>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real metric_aware_indicator(
   ValueAt value_at,
   int i,
   int j,
   int k,
   amrex::Real reference_scale,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& min_feature_lengths
) noexcept
{
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> indicators{};
   accumulate_metric_aware_directional_indicator(
      indicators,
      value_at,
      i,
      j,
      k,
      reference_scale,
      coordinate,
      axes,
      problo,
      dx,
      min_feature_lengths
   );
   amrex::Real indicator = indicators[0];
#if (AMREX_SPACEDIM >= 2)
   indicator = amrex::max(indicator, indicators[1]);
#endif
#if (AMREX_SPACEDIM == 3)
   indicator = amrex::max(indicator, indicators[2]);
#endif
   return indicator;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> auto_amr_directional_indicators_from_state(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   int eos_method,
   bool mhd,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& min_feature_lengths,
   amrex::Real min_refine_radius
) noexcept
{
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> indicators{};
   if (!amr_cell_allows_radial_refinement(coordinate, axes, problo, dx, i, j, k, min_refine_radius)) {
      return indicators;
   }
   const auto pressure_at = [=] AMREX_GPU_HOST_DEVICE(int ia, int ja, int ka) noexcept {
      return pressure_from_array_runtime(eos_method, state, ia, ja, ka, eos, mhd);
   };
   const auto rho_at = [=] AMREX_GPU_HOST_DEVICE(int ia, int ja, int ka) noexcept {
      return state(ia, ja, ka, RHO);
   };
   const auto vx_at = [=] AMREX_GPU_HOST_DEVICE(int ia, int ja, int ka) noexcept {
      return state_vector_cartesian_component(
         state, ia, ja, ka, coordinate, axes, problo, dx, MOM1, true, 0
      );
   };
   const auto vy_at = [=] AMREX_GPU_HOST_DEVICE(int ia, int ja, int ka) noexcept {
      return state_vector_cartesian_component(
         state, ia, ja, ka, coordinate, axes, problo, dx, MOM1, true, 1
      );
   };
   const auto vz_at = [=] AMREX_GPU_HOST_DEVICE(int ia, int ja, int ka) noexcept {
      return state_vector_cartesian_component(
         state, ia, ja, ka, coordinate, axes, problo, dx, MOM1, true, 2
      );
   };
   const auto b1_at = [=] AMREX_GPU_HOST_DEVICE(int ia, int ja, int ka) noexcept {
      return state_vector_cartesian_component(
         state, ia, ja, ka, coordinate, axes, problo, dx, B1, false, 0
      );
   };
   const auto b2_at = [=] AMREX_GPU_HOST_DEVICE(int ia, int ja, int ka) noexcept {
      return state_vector_cartesian_component(
         state, ia, ja, ka, coordinate, axes, problo, dx, B1, false, 1
      );
   };
   const auto b3_at = [=] AMREX_GPU_HOST_DEVICE(int ia, int ja, int ka) noexcept {
      return state_vector_cartesian_component(
         state, ia, ja, ka, coordinate, axes, problo, dx, B1, false, 2
      );
   };

   const amrex::Real p0 = amrex::max(amrex::Math::abs(pressure_at(i, j, k)), 1.0e-50);
   const amrex::Real rho0 = amrex::max(amrex::Math::abs(rho_at(i, j, k)), 1.0e-50);
   const amrex::Real v1_0 = vx_at(i, j, k);
   const amrex::Real v2_0 = vy_at(i, j, k);
   const amrex::Real v3_0 = vz_at(i, j, k);
   const amrex::Real vmag = std::sqrt(squared_norm(StateVector3{v1_0, v2_0, v3_0}));
   amrex::Real b2 = 0.0;
   if (mhd) {
      const amrex::Real b1_0 = b1_at(i, j, k);
      const amrex::Real b2_0 = b2_at(i, j, k);
      const amrex::Real b3_0 = b3_at(i, j, k);
      b2 = 2.0 * magnetic_energy_density(StateVector3{b1_0, b2_0, b3_0});
   }
   const amrex::Real cs0 = sound_speed_from_primitive_runtime(eos_method, rho0, p0, eos);
   const amrex::Real signal_scale = amrex::max(vmag + std::sqrt(cs0 * cs0 + b2 / rho0), 1.0e-50);

   accumulate_metric_aware_directional_indicator(
      indicators, pressure_at, i, j, k, p0, coordinate, axes, problo, dx, min_feature_lengths
   );
   accumulate_metric_aware_directional_indicator(
      indicators, rho_at, i, j, k, rho0, coordinate, axes, problo, dx, min_feature_lengths
   );
   accumulate_metric_aware_directional_indicator(
      indicators, vx_at, i, j, k, signal_scale, coordinate, axes, problo, dx, min_feature_lengths
   );
   accumulate_metric_aware_directional_indicator(
      indicators, vy_at, i, j, k, signal_scale, coordinate, axes, problo, dx, min_feature_lengths
   );
   accumulate_metric_aware_directional_indicator(
      indicators, vz_at, i, j, k, signal_scale, coordinate, axes, problo, dx, min_feature_lengths
   );

   if (mhd) {
      const auto magnetic_pressure_at = [=] AMREX_GPU_HOST_DEVICE(int ia, int ja, int ka) noexcept {
         const amrex::Real local_b1 = b1_at(ia, ja, ka);
         const amrex::Real local_b2 = b2_at(ia, ja, ka);
         const amrex::Real local_b3 = b3_at(ia, ja, ka);
         return magnetic_energy_density(StateVector3{local_b1, local_b2, local_b3});
      };
      const amrex::Real b_scale = amrex::max(std::sqrt(b2), std::sqrt(p0));
      const amrex::Real pm_scale = amrex::max(magnetic_pressure_at(i, j, k), p0);
      accumulate_metric_aware_directional_indicator(
         indicators, b1_at, i, j, k, b_scale, coordinate, axes, problo, dx, min_feature_lengths
      );
      accumulate_metric_aware_directional_indicator(
         indicators, b2_at, i, j, k, b_scale, coordinate, axes, problo, dx, min_feature_lengths
      );
      accumulate_metric_aware_directional_indicator(
         indicators, b3_at, i, j, k, b_scale, coordinate, axes, problo, dx, min_feature_lengths
      );
      accumulate_metric_aware_directional_indicator(
         indicators, magnetic_pressure_at, i, j, k, pm_scale, coordinate, axes, problo, dx, min_feature_lengths
      );
   }

   return indicators;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real auto_amr_indicator_from_state(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   int eos_method,
   bool mhd,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& min_feature_lengths,
   amrex::Real min_refine_radius
) noexcept
{
   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> indicators =
      auto_amr_directional_indicators_from_state(
         state,
         i,
         j,
         k,
         eos,
         eos_method,
         mhd,
         coordinate,
         axes,
         problo,
         dx,
         min_feature_lengths,
         min_refine_radius
      );
   amrex::Real indicator = indicators[0];
#if (AMREX_SPACEDIM >= 2)
   indicator = amrex::max(indicator, indicators[1]);
#endif
#if (AMREX_SPACEDIM == 3)
   indicator = amrex::max(indicator, indicators[2]);
#endif
   return indicator;
}

} // namespace pyfluid_amrex
