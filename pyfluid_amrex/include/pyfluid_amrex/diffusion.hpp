#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <limits>

#include "pyfluid_amrex/boundary.hpp"
#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/solid_geometry.hpp"
#include "pyfluid_amrex/kernels_generated/scalar_ops.hpp"
#include "pyfluid_amrex/kernels_generated/vector_ops.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

inline amrex::Real diffusion_inverse_mesh_spacing2_sum(amrex::Geometry const& geom, RunConfig const& cfg)
{
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   amrex::Real value = 0.0;
   const amrex::Real spacing0 = mesh_face_gradient_spacing(coordinate, axes, problo, dx, 0, 0, 0, 0);
   value += 1.0 / (spacing0 * spacing0);
#if (AMREX_SPACEDIM >= 2)
   const amrex::Real spacing1 = mesh_face_gradient_spacing(coordinate, axes, problo, dx, 1, 0, 0, 0);
   value += 1.0 / (spacing1 * spacing1);
#endif
#if (AMREX_SPACEDIM == 3)
   const amrex::Real spacing2 = mesh_face_gradient_spacing(coordinate, axes, problo, dx, 2, 0, 0, 0);
   value += 1.0 / (spacing2 * spacing2);
#endif
   return value;
}

inline amrex::Real max_diffusion_coefficient(RunConfig const& cfg)
{
   amrex::Real coefficient = cfg.viscosity;
   if (resistivity_enabled(cfg)) {
      coefficient = amrex::max(coefficient, cfg.resistivity);
   }
   return coefficient;
}

template <int DiffusionMethod>
inline amrex::Real diffusion_dt_limit(amrex::Geometry const& geom, RunConfig const& cfg)
{
   if constexpr (DiffusionMethod != DiffusionExplicit) {
      return std::numeric_limits<amrex::Real>::max();
   }
   const amrex::Real coefficient = max_diffusion_coefficient(cfg);
   if (coefficient <= 0.0) {
      return std::numeric_limits<amrex::Real>::max();
   }
   return cfg.diffusion_cfl / (2.0 * coefficient * diffusion_inverse_mesh_spacing2_sum(geom, cfg));
}

inline void apply_diffusion_boundary(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   apply_configured_state_boundary(state, geom, cfg);
}

struct ViscousStressTensor {
   amrex::GpuArray<amrex::Real, 9> tau{};
   amrex::GpuArray<amrex::Real, 3> velocity{};
};

struct MomentumOverride {
   bool enabled = false;
   int i = 0;
   int j = 0;
   int k = 0;
   amrex::GpuArray<amrex::Real, 3> momentum{0.0, 0.0, 0.0};

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   bool matches(int qi, int qj, int qk) const noexcept
   {
      return enabled && qi == i && qj == j && qk == k;
   }
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real momentum_component_at_cell(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int component,
   MomentumOverride const& momentum_override
) noexcept
{
   return momentum_override.matches(i, j, k)
      ? momentum_override.momentum[component]
      : state(i, j, k, MOM1 + component);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real velocity_component_at_cell(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int component,
   MomentumOverride const& momentum_override
) noexcept
{
   const amrex::Real inv_rho = 1.0 / amrex::max(state(i, j, k, RHO), 1.0e-300);
   return momentum_component_at_cell(state, i, j, k, component, momentum_override) * inv_rho;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real velocity_component_at_cell(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int component
) noexcept
{
   const MomentumOverride momentum_override{};
   return velocity_component_at_cell(state, i, j, k, component, momentum_override);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<int, 3> offset_cell(amrex::GpuArray<int, 3> cell, int dir, int offset) noexcept
{
   if (dir == 0) {
      cell[0] += offset;
   }
#if (AMREX_SPACEDIM >= 2)
   if (dir == 1) {
      cell[1] += offset;
   }
#else
   amrex::ignore_unused(dir);
#endif
#if (AMREX_SPACEDIM == 3)
   if (dir == 2) {
      cell[2] += offset;
   }
#endif
   return cell;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<int, 3> face_left_cell(int i, int j, int k, int face_dir) noexcept
{
   return offset_cell({i, j, k}, face_dir, -1);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real velocity_component_at_cell(
   amrex::Array4<amrex::Real const> const& state,
   amrex::GpuArray<int, 3> const& cell,
   int component,
   MomentumOverride const& momentum_override
) noexcept
{
   return velocity_component_at_cell(state, cell[0], cell[1], cell[2], component, momentum_override);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real velocity_component_at_cell(
   amrex::Array4<amrex::Real const> const& state,
   amrex::GpuArray<int, 3> const& cell,
   int component
) noexcept
{
   const MomentumOverride momentum_override{};
   return velocity_component_at_cell(state, cell, component, momentum_override);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real velocity_component_at_face(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int face_dir,
   int component,
   MomentumOverride const& momentum_override
) noexcept
{
   const auto left = face_left_cell(i, j, k, face_dir);
   const amrex::GpuArray<int, 3> right{i, j, k};
   return 0.5 * (
      velocity_component_at_cell(state, left, component, momentum_override)
      + velocity_component_at_cell(state, right, component, momentum_override)
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real velocity_component_at_face(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int face_dir,
   int component
) noexcept
{
   const MomentumOverride momentum_override{};
   return velocity_component_at_face(state, i, j, k, face_dir, component, momentum_override);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real velocity_derivative_at_face(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int component,
   int derivative_dir,
   int face_dir,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   MomentumOverride const& momentum_override
) noexcept
{
   const auto left = face_left_cell(i, j, k, face_dir);
   const amrex::GpuArray<int, 3> right{i, j, k};
   if (derivative_dir == face_dir) {
      return (
         velocity_component_at_cell(state, right, component, momentum_override)
         - velocity_component_at_cell(state, left, component, momentum_override)
      ) / dx[derivative_dir];
   }
   const auto left_hi = offset_cell(left, derivative_dir, 1);
   const auto left_lo = offset_cell(left, derivative_dir, -1);
   const auto right_hi = offset_cell(right, derivative_dir, 1);
   const auto right_lo = offset_cell(right, derivative_dir, -1);
   return (
      velocity_component_at_cell(state, left_hi, component, momentum_override)
      - velocity_component_at_cell(state, left_lo, component, momentum_override)
      + velocity_component_at_cell(state, right_hi, component, momentum_override)
      - velocity_component_at_cell(state, right_lo, component, momentum_override)
   ) / (4.0 * dx[derivative_dir]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real velocity_derivative_at_face(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int component,
   int derivative_dir,
   int face_dir,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx
) noexcept
{
   const MomentumOverride momentum_override{};
   return velocity_derivative_at_face(state, i, j, k, component, derivative_dir, face_dir, dx, momentum_override);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real velocity_derivative_at_cell_center(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int component,
   int derivative_dir,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   MomentumOverride const& momentum_override
) noexcept
{
   const amrex::GpuArray<int, 3> center{i, j, k};
   const auto hi = offset_cell(center, derivative_dir, 1);
   const auto lo = offset_cell(center, derivative_dir, -1);
   return (
      velocity_component_at_cell(state, hi, component, momentum_override)
      - velocity_component_at_cell(state, lo, component, momentum_override)
   ) / (2.0 * dx[derivative_dir]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real velocity_derivative_at_cell_center(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int component,
   int derivative_dir,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx
) noexcept
{
   const MomentumOverride momentum_override{};
   return velocity_derivative_at_cell_center(state, i, j, k, component, derivative_dir, dx, momentum_override);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real generated_viscous_stress_component(
   int coordinate,
   int normal,
   int component,
   amrex::Real mu,
   amrex::GpuArray<amrex::Real, 3> const& q,
   amrex::GpuArray<amrex::Real, 3> const& velocity,
   amrex::GpuArray<amrex::Real, 9> const& dv
) noexcept
{
#define PYFLUID_VISCOUS_STRESS_ARGS \
   mu, q[0], q[1], q[2], velocity[0], velocity[1], velocity[2], \
   dv[0], dv[1], dv[2], dv[3], dv[4], dv[5], dv[6], dv[7], dv[8]
   if (coordinate == MeshCylindrical) {
      if (normal == 0 && component == 0) {
         return generated::cylindrical_viscous_stress_11(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      if (normal == 0 && component == 1) {
         return generated::cylindrical_viscous_stress_12(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      if (normal == 0 && component == 2) {
         return generated::cylindrical_viscous_stress_13(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      if (normal == 1 && component == 0) {
         return generated::cylindrical_viscous_stress_21(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      if (normal == 1 && component == 1) {
         return generated::cylindrical_viscous_stress_22(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      if (normal == 1 && component == 2) {
         return generated::cylindrical_viscous_stress_23(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      if (normal == 2 && component == 0) {
         return generated::cylindrical_viscous_stress_31(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      if (normal == 2 && component == 1) {
         return generated::cylindrical_viscous_stress_32(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      return generated::cylindrical_viscous_stress_33(PYFLUID_VISCOUS_STRESS_ARGS);
   }
   if (coordinate == MeshSpherical) {
      if (normal == 0 && component == 0) {
         return generated::spherical_viscous_stress_11(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      if (normal == 0 && component == 1) {
         return generated::spherical_viscous_stress_12(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      if (normal == 0 && component == 2) {
         return generated::spherical_viscous_stress_13(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      if (normal == 1 && component == 0) {
         return generated::spherical_viscous_stress_21(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      if (normal == 1 && component == 1) {
         return generated::spherical_viscous_stress_22(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      if (normal == 1 && component == 2) {
         return generated::spherical_viscous_stress_23(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      if (normal == 2 && component == 0) {
         return generated::spherical_viscous_stress_31(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      if (normal == 2 && component == 1) {
         return generated::spherical_viscous_stress_32(PYFLUID_VISCOUS_STRESS_ARGS);
      }
      return generated::spherical_viscous_stress_33(PYFLUID_VISCOUS_STRESS_ARGS);
   }
   if (normal == 0 && component == 0) {
      return generated::cartesian_viscous_stress_11(PYFLUID_VISCOUS_STRESS_ARGS);
   }
   if (normal == 0 && component == 1) {
      return generated::cartesian_viscous_stress_12(PYFLUID_VISCOUS_STRESS_ARGS);
   }
   if (normal == 0 && component == 2) {
      return generated::cartesian_viscous_stress_13(PYFLUID_VISCOUS_STRESS_ARGS);
   }
   if (normal == 1 && component == 0) {
      return generated::cartesian_viscous_stress_21(PYFLUID_VISCOUS_STRESS_ARGS);
   }
   if (normal == 1 && component == 1) {
      return generated::cartesian_viscous_stress_22(PYFLUID_VISCOUS_STRESS_ARGS);
   }
   if (normal == 1 && component == 2) {
      return generated::cartesian_viscous_stress_23(PYFLUID_VISCOUS_STRESS_ARGS);
   }
   if (normal == 2 && component == 0) {
      return generated::cartesian_viscous_stress_31(PYFLUID_VISCOUS_STRESS_ARGS);
   }
   if (normal == 2 && component == 1) {
      return generated::cartesian_viscous_stress_32(PYFLUID_VISCOUS_STRESS_ARGS);
   }
   return generated::cartesian_viscous_stress_33(PYFLUID_VISCOUS_STRESS_ARGS);
#undef PYFLUID_VISCOUS_STRESS_ARGS
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real generated_tensor_divergence_connection_component(
   int coordinate,
   int component,
   amrex::GpuArray<amrex::Real, 3> const& q,
   amrex::GpuArray<amrex::Real, 9> const& tau
) noexcept
{
#define PYFLUID_TENSOR_CONNECTION_ARGS \
   q[0], q[1], q[2], tau[0], tau[1], tau[2], tau[3], tau[4], tau[5], tau[6], tau[7], tau[8]
   if (coordinate == MeshCylindrical) {
      if (component == 0) {
         return generated::cylindrical_tensor_divergence_connection_1(PYFLUID_TENSOR_CONNECTION_ARGS);
      }
      if (component == 1) {
         return generated::cylindrical_tensor_divergence_connection_2(PYFLUID_TENSOR_CONNECTION_ARGS);
      }
      return generated::cylindrical_tensor_divergence_connection_3(PYFLUID_TENSOR_CONNECTION_ARGS);
   }
   if (coordinate == MeshSpherical) {
      if (component == 0) {
         return generated::spherical_tensor_divergence_connection_1(PYFLUID_TENSOR_CONNECTION_ARGS);
      }
      if (component == 1) {
         return generated::spherical_tensor_divergence_connection_2(PYFLUID_TENSOR_CONNECTION_ARGS);
      }
      return generated::spherical_tensor_divergence_connection_3(PYFLUID_TENSOR_CONNECTION_ARGS);
   }
   if (component == 0) {
      return generated::cartesian_tensor_divergence_connection_1(PYFLUID_TENSOR_CONNECTION_ARGS);
   }
   if (component == 1) {
      return generated::cartesian_tensor_divergence_connection_2(PYFLUID_TENSOR_CONNECTION_ARGS);
   }
   return generated::cartesian_tensor_divergence_connection_3(PYFLUID_TENSOR_CONNECTION_ARGS);
#undef PYFLUID_TENSOR_CONNECTION_ARGS
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ViscousStressTensor viscous_stress_at_location(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real viscosity,
   MomentumOverride const& momentum_override,
   bool nodal0,
   bool nodal1,
   bool nodal2,
   int face_dir
) noexcept
{
   const auto q = mesh_coordinate_triplet_at_location(coordinate, axes, problo, dx, i, j, k, nodal0, nodal1, nodal2);
   amrex::GpuArray<amrex::Real, 3> velocity{0.0, 0.0, 0.0};
   amrex::GpuArray<amrex::Real, 9> dv{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
   for (int component = 0; component < 3; ++component) {
      const int component_axis = mesh_vector_axis_for_component(coordinate, axes, component);
      const int component_frame = mesh_canonical_axis_index(coordinate, component_axis);
      if (component_frame < 0) {
         continue;
      }
      velocity[component_frame] = face_dir >= 0
         ? velocity_component_at_face(state, i, j, k, face_dir, component, momentum_override)
         : velocity_component_at_cell(state, i, j, k, component, momentum_override);
      for (int derivative_dir = 0; derivative_dir < AMREX_SPACEDIM; ++derivative_dir) {
         const int derivative_axis = mesh_axis(axes, derivative_dir);
         const int derivative_frame = mesh_canonical_axis_index(coordinate, derivative_axis);
         if (derivative_frame < 0) {
            continue;
         }
         dv[3 * component_frame + derivative_frame] = face_dir >= 0
            ? velocity_derivative_at_face(
                 state, i, j, k, component, derivative_dir, face_dir, dx, momentum_override
              )
            : velocity_derivative_at_cell_center(state, i, j, k, component, derivative_dir, dx, momentum_override);
      }
   }

   ViscousStressTensor stress;
   stress.velocity = velocity;
   for (int normal = 0; normal < 3; ++normal) {
      for (int component = 0; component < 3; ++component) {
         stress.tau[3 * normal + component] =
            generated_viscous_stress_component(coordinate, normal, component, viscosity, q, velocity, dv);
      }
   }
   return stress;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ViscousStressTensor viscous_stress_at_face(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int face_dir,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real viscosity,
   MomentumOverride const& momentum_override
) noexcept
{
   return viscous_stress_at_location(
      state,
      i,
      j,
      k,
      coordinate,
      axes,
      problo,
      dx,
      viscosity,
      momentum_override,
      face_dir == 0,
      face_dir == 1,
      face_dir == 2,
      face_dir
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ViscousStressTensor viscous_stress_at_face(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int face_dir,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real viscosity
) noexcept
{
   const MomentumOverride momentum_override{};
   return viscous_stress_at_face(state, i, j, k, face_dir, coordinate, axes, problo, dx, viscosity, momentum_override);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ViscousStressTensor viscous_stress_at_cell_center(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real viscosity,
   MomentumOverride const& momentum_override
) noexcept
{
   return viscous_stress_at_location(
      state, i, j, k, coordinate, axes, problo, dx, viscosity, momentum_override, false, false, false, -1
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ViscousStressTensor viscous_stress_at_cell_center(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real viscosity
) noexcept
{
   const MomentumOverride momentum_override{};
   return viscous_stress_at_cell_center(state, i, j, k, coordinate, axes, problo, dx, viscosity, momentum_override);
}

template <int NComp, bool DiffuseMomentum>
inline void add_viscous_stress_connection_sources(
   amrex::MultiFab& rhs,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
   if constexpr (!DiffuseMomentum) {
      amrex::ignore_unused(rhs, state, geom, cfg);
      return;
   }
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   if (coordinate == MeshCartesian) {
      return;
   }
   const MeshAxes axes = mesh_axes_layout(cfg);
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
   const amrex::Real viscosity = cfg.viscosity;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto r = rhs.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const auto q = mesh_coordinate_triplet_at_location(
            coordinate, axes, problo, dx, i, j, k, false, false, false
         );
         const ViscousStressTensor stress =
            viscous_stress_at_cell_center(s, i, j, k, coordinate, axes, problo, dx, viscosity);
         for (int component = 0; component < 3; ++component) {
            const int component_axis = mesh_vector_axis_for_component(coordinate, axes, component);
            const int component_frame = mesh_canonical_axis_index(coordinate, component_axis);
            if (component_frame >= 0) {
               r(i, j, k, MOM1 + component) += generated_tensor_divergence_connection_component(
                  coordinate, component_frame, q, stress.tau
               );
            }
         }
      });
   }
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> viscous_momentum_operator(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real viscosity,
   MomentumOverride const& momentum_override
) noexcept
{
   amrex::GpuArray<amrex::Real, 3> rhs{0.0, 0.0, 0.0};
   const amrex::Real cell_volume = mesh_cell_volume_scale(coordinate, axes, problo, dx, i, j, k);

   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const int normal_axis = mesh_axis(axes, dir);
      const int normal_frame = mesh_canonical_axis_index(coordinate, normal_axis);
      if (normal_frame < 0) {
         continue;
      }
      int ih = i;
      int jh = j;
      int kh = k;
      if (dir == 0) {
         ++ih;
      }
#if (AMREX_SPACEDIM >= 2)
      if (dir == 1) {
         ++jh;
      }
#endif
#if (AMREX_SPACEDIM == 3)
      if (dir == 2) {
         ++kh;
      }
#endif
      const ViscousStressTensor stress_lo =
         viscous_stress_at_face(state, i, j, k, dir, coordinate, axes, problo, dx, viscosity, momentum_override);
      const ViscousStressTensor stress_hi =
         viscous_stress_at_face(state, ih, jh, kh, dir, coordinate, axes, problo, dx, viscosity, momentum_override);
      const amrex::Real area_lo = mesh_face_area_scale(dir, coordinate, axes, problo, dx, i, j, k);
      const amrex::Real area_hi = mesh_face_area_scale(dir, coordinate, axes, problo, dx, ih, jh, kh);
      for (int component = 0; component < 3; ++component) {
         const int component_axis = mesh_vector_axis_for_component(coordinate, axes, component);
         const int component_frame = mesh_canonical_axis_index(coordinate, component_axis);
         if (component_frame < 0) {
            continue;
         }
         const amrex::Real tau_lo = stress_lo.tau[3 * normal_frame + component_frame];
         const amrex::Real tau_hi = stress_hi.tau[3 * normal_frame + component_frame];
         rhs[component] += (area_hi * tau_hi - area_lo * tau_lo) / cell_volume;
      }
   }

   const ViscousStressTensor center_stress =
      viscous_stress_at_cell_center(state, i, j, k, coordinate, axes, problo, dx, viscosity, momentum_override);
   const auto q = mesh_coordinate_triplet_at_location(coordinate, axes, problo, dx, i, j, k, false, false, false);
   for (int component = 0; component < 3; ++component) {
      const int component_axis = mesh_vector_axis_for_component(coordinate, axes, component);
      const int component_frame = mesh_canonical_axis_index(coordinate, component_axis);
      if (component_frame >= 0) {
         rhs[component] += generated_tensor_divergence_connection_component(coordinate, component_frame, q, center_stress.tau);
      }
   }
   return rhs;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real viscous_energy_operator(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real viscosity,
   MomentumOverride const& momentum_override
) noexcept
{
   amrex::Real rhs = 0.0;
   const amrex::Real cell_volume = mesh_cell_volume_scale(coordinate, axes, problo, dx, i, j, k);

   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const int normal_axis = mesh_axis(axes, dir);
      const int normal_frame = mesh_canonical_axis_index(coordinate, normal_axis);
      if (normal_frame < 0) {
         continue;
      }
      int ih = i;
      int jh = j;
      int kh = k;
      if (dir == 0) {
         ++ih;
      }
#if (AMREX_SPACEDIM >= 2)
      if (dir == 1) {
         ++jh;
      }
#endif
#if (AMREX_SPACEDIM == 3)
      if (dir == 2) {
         ++kh;
      }
#endif
      const ViscousStressTensor stress_lo =
         viscous_stress_at_face(state, i, j, k, dir, coordinate, axes, problo, dx, viscosity, momentum_override);
      const ViscousStressTensor stress_hi =
         viscous_stress_at_face(state, ih, jh, kh, dir, coordinate, axes, problo, dx, viscosity, momentum_override);
      amrex::Real flux_lo = 0.0;
      amrex::Real flux_hi = 0.0;
      for (int component = 0; component < 3; ++component) {
         const int component_axis = mesh_vector_axis_for_component(coordinate, axes, component);
         const int component_frame = mesh_canonical_axis_index(coordinate, component_axis);
         if (component_frame < 0) {
            continue;
         }
         flux_lo -= stress_lo.tau[3 * normal_frame + component_frame] * stress_lo.velocity[component_frame];
         flux_hi -= stress_hi.tau[3 * normal_frame + component_frame] * stress_hi.velocity[component_frame];
      }
      const amrex::Real area_lo = mesh_face_area_scale(dir, coordinate, axes, problo, dx, i, j, k);
      const amrex::Real area_hi = mesh_face_area_scale(dir, coordinate, axes, problo, dx, ih, jh, kh);
      rhs -= (area_hi * flux_hi - area_lo * flux_lo) / cell_volume;
   }
   return rhs;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> implicit_viscous_momentum_residual_vector(
   amrex::Array4<amrex::Real const> const& base,
   amrex::Array4<amrex::Real const> const& solution,
   int i,
   int j,
   int k,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real viscosity,
   amrex::Real dt,
   MomentumOverride const& momentum_override
) noexcept
{
   const amrex::GpuArray<amrex::Real, 3> op =
      viscous_momentum_operator(solution, i, j, k, coordinate, axes, problo, dx, viscosity, momentum_override);
   return {
      momentum_component_at_cell(solution, i, j, k, 0, momentum_override) - base(i, j, k, MOM1) - dt * op[0],
      momentum_component_at_cell(solution, i, j, k, 1, momentum_override) - base(i, j, k, MOM2) - dt * op[1],
      momentum_component_at_cell(solution, i, j, k, 2, momentum_override) - base(i, j, k, MOM3) - dt * op[2],
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> solve_3x3(
   amrex::GpuArray<amrex::Real, 9> const& a,
   amrex::GpuArray<amrex::Real, 3> const& b
) noexcept
{
   const amrex::Real a00 = a[0];
   const amrex::Real a01 = a[1];
   const amrex::Real a02 = a[2];
   const amrex::Real a10 = a[3];
   const amrex::Real a11 = a[4];
   const amrex::Real a12 = a[5];
   const amrex::Real a20 = a[6];
   const amrex::Real a21 = a[7];
   const amrex::Real a22 = a[8];
   const amrex::Real det =
      a00 * (a11 * a22 - a12 * a21)
      - a01 * (a10 * a22 - a12 * a20)
      + a02 * (a10 * a21 - a11 * a20);
   if (amrex::Math::abs(det) <= 1.0e-300) {
      return {0.0, 0.0, 0.0};
   }
   const amrex::Real inv_det = 1.0 / det;
   return {
      (b[0] * (a11 * a22 - a12 * a21)
       - a01 * (b[1] * a22 - a12 * b[2])
       + a02 * (b[1] * a21 - a11 * b[2]))
         * inv_det,
      (a00 * (b[1] * a22 - a12 * b[2])
       - b[0] * (a10 * a22 - a12 * a20)
       + a02 * (a10 * b[2] - b[1] * a20))
         * inv_det,
      (a00 * (a11 * b[2] - b[1] * a21)
       - a01 * (a10 * b[2] - b[1] * a20)
       + b[0] * (a10 * a21 - a11 * a20))
         * inv_det,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void relax_implicit_viscous_momentum_cell(
   amrex::Array4<amrex::Real> const& out,
   amrex::Array4<amrex::Real const> const& base,
   amrex::Array4<amrex::Real const> const& solution,
   int i,
   int j,
   int k,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real viscosity,
   amrex::Real dt,
   amrex::Real omega
) noexcept
{
   const amrex::GpuArray<amrex::Real, 3> center_momentum{
      solution(i, j, k, MOM1),
      solution(i, j, k, MOM2),
      solution(i, j, k, MOM3),
   };
   const MomentumOverride no_override{};
   const amrex::GpuArray<amrex::Real, 3> op =
      viscous_momentum_operator(solution, i, j, k, coordinate, axes, problo, dx, viscosity, no_override);
   const amrex::GpuArray<amrex::Real, 3> residual{
      center_momentum[0] - base(i, j, k, MOM1) - dt * op[0],
      center_momentum[1] - base(i, j, k, MOM2) - dt * op[1],
      center_momentum[2] - base(i, j, k, MOM3) - dt * op[2],
   };
   amrex::GpuArray<amrex::Real, 9> jacobian{};
   for (int column = 0; column < 3; ++column) {
      MomentumOverride perturbed{};
      perturbed.enabled = true;
      perturbed.i = i;
      perturbed.j = j;
      perturbed.k = k;
      perturbed.momentum = center_momentum;
      perturbed.momentum[column] += 1.0;
      const amrex::GpuArray<amrex::Real, 3> op_perturbed =
         viscous_momentum_operator(solution, i, j, k, coordinate, axes, problo, dx, viscosity, perturbed);
      for (int row = 0; row < 3; ++row) {
         jacobian[3 * row + column] = (row == column ? 1.0 : 0.0) - dt * (op_perturbed[row] - op[row]);
      }
   }
   const amrex::GpuArray<amrex::Real, 3> rhs{-residual[0], -residual[1], -residual[2]};
   const amrex::GpuArray<amrex::Real, 3> delta = solve_3x3(jacobian, rhs);
   const amrex::GpuArray<amrex::Real, 3> updated_momentum{
      center_momentum[0] + omega * delta[0],
      center_momentum[1] + omega * delta[1],
      center_momentum[2] + omega * delta[2],
   };
   out(i, j, k, MOM1) = updated_momentum[0];
   out(i, j, k, MOM2) = updated_momentum[1];
   out(i, j, k, MOM3) = updated_momentum[2];

   MomentumOverride updated{};
   updated.enabled = true;
   updated.i = i;
   updated.j = j;
   updated.k = k;
   updated.momentum = updated_momentum;
   out(i, j, k, ENERGY) =
      base(i, j, k, ENERGY)
      + dt * viscous_energy_operator(solution, i, j, k, coordinate, axes, problo, dx, viscosity, updated);
}

inline amrex::Real implicit_viscous_momentum_residual(
   amrex::MultiFab& work,
   amrex::MultiFab const& base,
   amrex::MultiFab const& solution,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Real dt
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
   const amrex::Real viscosity = cfg.viscosity;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(solution, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto s = solution.const_array(mfi);
      const auto w = work.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const MomentumOverride no_override{};
         const amrex::GpuArray<amrex::Real, 3> residual =
            implicit_viscous_momentum_residual_vector(
               b, s, i, j, k, coordinate, axes, problo, dx, viscosity, dt, no_override
            );
         amrex::Real residual_norm = 0.0;
         amrex::Real scale = 0.0;
         for (int component = 0; component < 3; ++component) {
            residual_norm += amrex::Math::abs(residual[component]);
            const int q = MOM1 + component;
            scale += amrex::max(amrex::Math::abs(s(i, j, k, q)), amrex::Math::abs(b(i, j, k, q)));
         }
         w(i, j, k, 0) = residual_norm;
         w(i, j, k, 1) = scale;
      });
   }
   }
   return work.sum(0) / amrex::max(work.sum(1), 1.0);
}

template <bool DiffuseMomentum>
inline void sync_implicit_viscous_energy(
   amrex::MultiFab& solution,
   amrex::MultiFab const& base,
   amrex::Geometry const& geom,
   amrex::Real dt,
   RunConfig const& cfg
)
{
   if constexpr (!DiffuseMomentum) {
      amrex::ignore_unused(solution, base, geom, dt, cfg);
      return;
   }
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
   const amrex::Real viscosity = cfg.viscosity;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(solution, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto s = solution.const_array(mfi);
      const auto o = solution.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const MomentumOverride no_override{};
         o(i, j, k, ENERGY) =
            b(i, j, k, ENERGY)
            + dt * viscous_energy_operator(s, i, j, k, coordinate, axes, problo, dx, viscosity, no_override);
      });
   }
   }
}

template <int NComp, bool DiffuseMomentum>
inline void fill_explicit_diffusion_fluxes(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Real time,
   amrex::iMultiFab const* solid_mask = nullptr
)
{
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      fluxes[dir]->setVal(0.0);
   }

   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const amrex::Real viscosity = cfg.viscosity;
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const bool no_slip_wall = cfg.solid_wall_condition == "no_slip";
   const SolidGeometryEvaluator solid_params = solid_geometry_evaluator(cfg, time);

   auto fill_viscous_stress_fluxes = [=] (
      amrex::Array4<amrex::Real> const& flux,
      amrex::Array4<amrex::Real const> const& state_array,
      amrex::Box const& bx,
      int dir,
      amrex::Real coefficient
   ) {
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const int normal_axis = mesh_axis(mesh_axes, dir);
         const int normal_frame = mesh_canonical_axis_index(mesh_coordinate, normal_axis);
         if (normal_frame < 0) {
            return;
         }
         const ViscousStressTensor stress =
            viscous_stress_at_face(state_array, i, j, k, dir, mesh_coordinate, mesh_axes, problo, dx, coefficient);
         amrex::Real energy_flux = 0.0;
         for (int component = 0; component < 3; ++component) {
            const int component_axis = mesh_vector_axis_for_component(mesh_coordinate, mesh_axes, component);
            const int component_frame = mesh_canonical_axis_index(mesh_coordinate, component_axis);
            if (component_frame < 0) {
               continue;
            }
            const amrex::Real tau_nc = stress.tau[3 * normal_frame + component_frame];
            flux(i, j, k, MOM1 + component) = -tau_nc;
            energy_flux -= tau_nc * stress.velocity[component_frame];
         }
         flux(i, j, k, ENERGY) = energy_flux;
      });
   };

   auto fill_component_fluxes_solid = [=] (
      amrex::Array4<amrex::Real> const& flux,
      amrex::Array4<amrex::Real const> const& state_array,
      amrex::Array4<int const> const& solid,
      amrex::Box const& bx,
      int dir,
      int component,
      amrex::Real coefficient,
      bool no_slip,
      SolidGeometryEvaluator const& params
   ) {
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         int li = i;
         int lj = j;
         int lk = k;
         int ri = i;
         int rj = j;
         int rk = k;
         if (dir == 0) {
            li = i - 1;
         }
#if (AMREX_SPACEDIM >= 2)
         else if (dir == 1) {
            lj = j - 1;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         else {
            lk = k - 1;
         }
#endif
         const bool left_solid = solid(li, lj, lk, 0) != 0;
         const bool right_solid = solid(ri, rj, rk, 0) != 0;
         amrex::Real value = 0.0;
         if (!left_solid && !right_solid) {
            value = generated::diffusive_flux(
               state_array(li, lj, lk, component),
               state_array(ri, rj, rk, component),
               coefficient,
               1.0 / dx[dir]
            );
         } else if (left_solid != right_solid && no_slip) {
            const int fi = left_solid ? ri : li;
            const int fj = left_solid ? rj : lj;
            const int fk = left_solid ? rk : lk;
            const amrex::Real fluid_value = left_solid
               ? state_array(ri, rj, rk, component)
               : state_array(li, lj, lk, component);
            const amrex::Real x = problo[0]
               + (static_cast<amrex::Real>(dir == 0 ? i : fi) + (dir == 0 ? 0.0 : 0.5)) * dx[0];
#if (AMREX_SPACEDIM >= 2)
            const amrex::Real y = problo[1]
               + (static_cast<amrex::Real>(dir == 1 ? j : fj) + (dir == 1 ? 0.0 : 0.5)) * dx[1];
#else
            const amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real z = problo[2]
               + (static_cast<amrex::Real>(dir == 2 ? k : fk) + (dir == 2 ? 0.0 : 0.5)) * dx[2];
#else
            const amrex::Real z = 0.0;
#endif
            const int vel_component = component == MOM1 ? 0 : (component == MOM2 ? 1 : 2);
            const amrex::Real wall_value =
               state_array(fi, fj, fk, RHO) * params.velocity_component(params.time, vel_component, x, y, z);
            const amrex::Real left_value = left_solid ? 2.0 * wall_value - fluid_value : fluid_value;
            const amrex::Real right_value = left_solid ? fluid_value : 2.0 * wall_value - fluid_value;
            value = generated::diffusive_flux(left_value, right_value, coefficient, 1.0 / dx[dir]);
         }
         flux(i, j, k, component) = value;
      });
   };

   auto fill_wall_energy_fluxes_solid = [=] (
      amrex::Array4<amrex::Real> const& flux,
      amrex::Array4<amrex::Real const> const& state_array,
      amrex::Array4<int const> const& solid,
      amrex::Box const& bx,
      int dir,
      amrex::Real coefficient,
      bool no_slip,
      SolidGeometryEvaluator const& params
   ) {
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         flux(i, j, k, ENERGY) = 0.0;
         if (!no_slip) {
            return;
         }
         int li = i;
         int lj = j;
         int lk = k;
         int ri = i;
         int rj = j;
         int rk = k;
         if (dir == 0) {
            li = i - 1;
         }
#if (AMREX_SPACEDIM >= 2)
         else if (dir == 1) {
            lj = j - 1;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         else {
            lk = k - 1;
         }
#endif
         const bool left_solid = solid(li, lj, lk, 0) != 0;
         const bool right_solid = solid(ri, rj, rk, 0) != 0;
         if (left_solid == right_solid) {
            return;
         }

         const int fi = left_solid ? ri : li;
         const int fj = left_solid ? rj : lj;
         const int fk = left_solid ? rk : lk;
         const amrex::Real x = problo[0]
            + (static_cast<amrex::Real>(dir == 0 ? i : fi) + (dir == 0 ? 0.0 : 0.5)) * dx[0];
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real y = problo[1]
            + (static_cast<amrex::Real>(dir == 1 ? j : fj) + (dir == 1 ? 0.0 : 0.5)) * dx[1];
#else
         const amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real z = problo[2]
            + (static_cast<amrex::Real>(dir == 2 ? k : fk) + (dir == 2 ? 0.0 : 0.5)) * dx[2];
#else
         const amrex::Real z = 0.0;
#endif
         const amrex::Real sign = left_solid ? -1.0 : 1.0;
         const amrex::Real rho = state_array(fi, fj, fk, RHO);
         amrex::Real energy_flux = 0.0;
         const amrex::Real wall_vx = params.velocity_component(params.time, 0, x, y, z);
         const amrex::Real flux_x =
            sign * 2.0 * coefficient * (state_array(fi, fj, fk, MOM1) - rho * wall_vx) / dx[dir];
         energy_flux += flux_x * wall_vx;
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real wall_vy = params.velocity_component(params.time, 1, x, y, z);
         const amrex::Real flux_y =
            sign * 2.0 * coefficient * (state_array(fi, fj, fk, MOM2) - rho * wall_vy) / dx[dir];
         energy_flux += flux_y * wall_vy;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real wall_vz = params.velocity_component(params.time, 2, x, y, z);
         const amrex::Real flux_z =
            sign * 2.0 * coefficient * (state_array(fi, fj, fk, MOM3) - rho * wall_vz) / dx[dir];
         energy_flux += flux_z * wall_vz;
#endif
         flux(i, j, k, ENERGY) = energy_flux;
      });
   };

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      for (amrex::MFIter mfi(*fluxes[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto s = state.const_array(mfi);
         const auto f = fluxes[dir]->array(mfi);
         const bool use_solid = solid_mask != nullptr;
         const auto solid = use_solid ? solid_mask->const_array(mfi) : amrex::Array4<int const>{};
         if constexpr (DiffuseMomentum) {
            if (use_solid) {
               fill_component_fluxes_solid(f, s, solid, bx, dir, MOM1, viscosity, no_slip_wall, solid_params);
               fill_component_fluxes_solid(f, s, solid, bx, dir, MOM2, viscosity, no_slip_wall, solid_params);
               fill_component_fluxes_solid(f, s, solid, bx, dir, MOM3, viscosity, no_slip_wall, solid_params);
               fill_wall_energy_fluxes_solid(f, s, solid, bx, dir, viscosity, no_slip_wall, solid_params);
            } else {
               fill_viscous_stress_fluxes(f, s, bx, dir, viscosity);
            }
         }
      }
   }
   }
}

template <int NComp, bool DiffuseMomentum>
inline void explicit_diffusion_update_kernel(
   amrex::MultiFab& out,
   amrex::MultiFab const& base,
   amrex::Geometry const& geom,
   amrex::Real dt,
   RunConfig const& cfg
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const amrex::Real viscosity = cfg.viscosity;
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(base, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto o = out.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         for (int q = 0; q < NComp; ++q) {
            o(i, j, k, q) = b(i, j, k, q);
         }
         if constexpr (DiffuseMomentum) {
            const MomentumOverride no_override{};
            const amrex::GpuArray<amrex::Real, 3> dm =
               viscous_momentum_operator(b, i, j, k, mesh_coordinate, mesh_axes, problo, dx, viscosity, no_override);
            o(i, j, k, MOM1) += dt * dm[0];
            o(i, j, k, MOM2) += dt * dm[1];
            o(i, j, k, MOM3) += dt * dm[2];
            o(i, j, k, ENERGY) +=
               dt * viscous_energy_operator(b, i, j, k, mesh_coordinate, mesh_axes, problo, dx, viscosity, no_override);
         }
      });
   }
   }
}

template <bool DiffuseMomentum>
inline void relax_implicit_diffusion_color_kernel(
   amrex::MultiFab& solution,
   amrex::MultiFab const& base,
   amrex::Geometry const& geom,
   amrex::Real dt,
   RunConfig const& cfg,
   int color
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const amrex::Real viscosity = cfg.viscosity;
   const amrex::Real omega = cfg.diffusion_sor_omega;
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(solution, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto s = solution.const_array(mfi);
      const auto o = solution.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (((i + j + k) & 1) != color) {
            return;
         }
         o(i, j, k, RHO) = b(i, j, k, RHO);
         o(i, j, k, ENERGY) = b(i, j, k, ENERGY);
         if constexpr (DiffuseMomentum) {
            relax_implicit_viscous_momentum_cell(
               o, b, s, i, j, k, mesh_coordinate, mesh_axes, problo, dx, viscosity, dt, omega
            );
         }
      });
   }
   }
}

template <bool DiffuseMomentum>
inline amrex::Real implicit_diffusion_residual_impl(
   amrex::MultiFab& work,
   amrex::MultiFab const& base,
   amrex::MultiFab const& solution,
   amrex::Geometry const& geom,
   amrex::Real dt,
   RunConfig const& cfg
)
{
   amrex::ignore_unused(cfg);
   amrex::Real residual = 0.0;
   if constexpr (DiffuseMomentum) {
      residual = amrex::max(residual, implicit_viscous_momentum_residual(work, base, solution, geom, cfg, dt));
   }
   return residual;
}

template <int DiffusionMethod, int NComp, bool DiffuseMomentum>
inline DiffusionStats apply_viscosity_diffusion_impl(
   amrex::MultiFab& state,
   amrex::MultiFab& scratch,
   amrex::MultiFab& work,
   amrex::Geometry const& geom,
   amrex::Real dt,
   RunConfig const& cfg
)
{
   DiffusionStats stats;
   if constexpr (DiffusionMethod == DiffusionNone || !DiffuseMomentum) {
      return stats;
   }

   apply_diffusion_boundary(state, geom, cfg);
   explicit_diffusion_update_kernel<NComp, DiffuseMomentum>(scratch, state, geom, dt, cfg);
   apply_diffusion_boundary(scratch, geom, cfg);

   if constexpr (DiffusionMethod == DiffusionExplicit) {
      amrex::MultiFab::Copy(state, scratch, 0, 0, NComp, 0);
      apply_diffusion_boundary(state, geom, cfg);
      return stats;
   }

   stats.residual = implicit_diffusion_residual_impl<DiffuseMomentum>(work, state, scratch, geom, dt, cfg);
   while (stats.residual > cfg.diffusion_tolerance && stats.iterations < cfg.diffusion_max_iterations) {
      relax_implicit_diffusion_color_kernel<DiffuseMomentum>(scratch, state, geom, dt, cfg, 0);
      apply_diffusion_boundary(scratch, geom, cfg);
      relax_implicit_diffusion_color_kernel<DiffuseMomentum>(scratch, state, geom, dt, cfg, 1);
      apply_diffusion_boundary(scratch, geom, cfg);
      stats.residual = implicit_diffusion_residual_impl<DiffuseMomentum>(work, state, scratch, geom, dt, cfg);
      ++stats.iterations;
   }
   require_implicit_diffusion_converged(stats, cfg, "cell-centered implicit diffusion");
   sync_implicit_viscous_energy<DiffuseMomentum>(scratch, state, geom, dt, cfg);
   apply_diffusion_boundary(scratch, geom, cfg);

   amrex::MultiFab::Copy(state, scratch, 0, 0, NComp, 0);
   apply_diffusion_boundary(state, geom, cfg);
   return stats;
}

template <int DiffusionMethod, int NComp>
inline DiffusionStats apply_viscosity_diffusion(
   amrex::MultiFab& state,
   amrex::MultiFab& scratch,
   amrex::MultiFab& work,
   amrex::Geometry const& geom,
   amrex::Real dt,
   RunConfig const& cfg
)
{
   const bool diffuse_momentum = viscosity_enabled(cfg);
   if (diffuse_momentum) {
      return apply_viscosity_diffusion_impl<DiffusionMethod, NComp, true>(state, scratch, work, geom, dt, cfg);
   }
   return {};
}

} // namespace pyfluid_amrex
