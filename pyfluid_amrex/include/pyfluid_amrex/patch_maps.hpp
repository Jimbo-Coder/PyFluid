#pragma once

#include <AMReX.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/kernels_generated/patch_maps.hpp"
#include "pyfluid_amrex/multipatch.hpp"
#include "pyfluid_amrex/physical_geometry.hpp"

#include <cmath>

namespace pyfluid_amrex {

using namespace amrex::literals;

struct PatchCoordinate {
   amrex::Real q1 = 0.0_rt;
   amrex::Real q2 = 0.0_rt;
   amrex::Real q3 = 0.0_rt;
};

struct PatchMapJacobian {
   amrex::GpuArray<amrex::Real, 9> component{};
};

struct PatchMap {
   int coordinate_map = PatchMapCartesianIdentity;
   PatchCoordinate origin{};
   amrex::Real radial_inner = 0.0_rt;
   amrex::Real radial_outer = 0.0_rt;
};

struct PatchCartesianAxes {
   amrex::GpuArray<int, AMREX_SPACEDIM> component{};
};

enum PatchMapSystemKind {
   PatchMapSystemCartesianSingle = 0,
   PatchMapSystemThornburg04 = 1,
   PatchMapSystemThornburg13 = 2,
   PatchMapSystemCylindrical5 = 3,
   PatchMapSystemCylindrical9 = 4,
};

struct PatchMapSystem {
   int system_kind = PatchMapSystemCartesianSingle;
   PatchCoordinate origin{};
   amrex::Real inner_radius = 0.0_rt;
   amrex::Real medium_radius = 0.0_rt;
   amrex::Real outer_radius = 0.0_rt;
   PatchCartesianAxes active_axes{};
   amrex::Vector<PatchMap> maps;
};

struct PatchMapPoint {
   int patch_id = -1;
   PatchCoordinate local{};
   PatchCoordinate global{};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real patch_coordinate_component(PatchCoordinate const& coordinate, int component) noexcept
{
   if (component == 0) {
      return coordinate.q1;
   }
   if (component == 1) {
      return coordinate.q2;
   }
   return coordinate.q3;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void set_patch_coordinate_component(PatchCoordinate& coordinate, int component, amrex::Real value) noexcept
{
   if (component == 0) {
      coordinate.q1 = value;
   } else if (component == 1) {
      coordinate.q2 = value;
   } else {
      coordinate.q3 = value;
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate make_patch_coordinate(amrex::Real q1, amrex::Real q2, amrex::Real q3 = 0.0_rt) noexcept
{
   return {q1, q2, q3};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCartesianAxes default_patch_cartesian_axes() noexcept
{
   PatchCartesianAxes axes{};
   axes.component[0] = 0;
#if (AMREX_SPACEDIM >= 2)
   axes.component[1] = 1;
#endif
#if (AMREX_SPACEDIM == 3)
   axes.component[2] = 2;
#endif
   return axes;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int patch_cartesian_axis(PatchCartesianAxes const& axes, int dir) noexcept
{
   return axes.component[dir];
}

inline void validate_cylindrical_patch_cartesian_axes(PatchCartesianAxes const& axes)
{
   if (patch_cartesian_axis(axes, 0) != 0) {
      amrex::Abort("mixed cylindrical patches require the first active Cartesian axis to be x");
   }
#if (AMREX_SPACEDIM >= 2)
   if (patch_cartesian_axis(axes, 1) != 1) {
      amrex::Abort("mixed cylindrical 2D patches require the x-y plane, interpreted as cylindrical r-phi");
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (patch_cartesian_axis(axes, 2) != 2) {
      amrex::Abort("mixed cylindrical 3D patches require active Cartesian axes x y z, interpreted as r-phi-z");
   }
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_coordinate_add(PatchCoordinate const& a, PatchCoordinate const& b) noexcept
{
   return {a.q1 + b.q1, a.q2 + b.q2, a.q3 + b.q3};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_coordinate_subtract(PatchCoordinate const& a, PatchCoordinate const& b) noexcept
{
   return {a.q1 - b.q1, a.q2 - b.q2, a.q3 - b.q3};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cartesian_radius(PatchCoordinate const& coordinate) noexcept
{
   return cartesian_radius(coordinate.q1, coordinate.q2, coordinate.q3);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cartesian_cylindrical_radius(PatchCoordinate const& coordinate) noexcept
{
   return cartesian_cylindrical_radius(coordinate.q1, coordinate.q2);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_coordinate_from_generated(generated::PatchPoint const& point) noexcept
{
   return {point.q1, point.q2, point.q3};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchMapJacobian patch_jacobian_from_generated(generated::PatchJacobian const& generated_jacobian) noexcept
{
   PatchMapJacobian jacobian{};
   jacobian.component[0] = generated_jacobian.j11;
   jacobian.component[1] = generated_jacobian.j12;
   jacobian.component[2] = generated_jacobian.j13;
   jacobian.component[3] = generated_jacobian.j21;
   jacobian.component[4] = generated_jacobian.j22;
   jacobian.component[5] = generated_jacobian.j23;
   jacobian.component[6] = generated_jacobian.j31;
   jacobian.component[7] = generated_jacobian.j32;
   jacobian.component[8] = generated_jacobian.j33;
   return jacobian;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real patch_map_jacobian_component(PatchMapJacobian const& jacobian, int global_component, int local_component) noexcept
{
   return jacobian.component[3 * global_component + local_component];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_identity_local_to_global(PatchMap const& map, PatchCoordinate const& local) noexcept
{
   return patch_coordinate_from_generated(generated::patch_identity_local_to_global(
      local.q1,
      local.q2,
      local.q3,
      map.origin.q1,
      map.origin.q2,
      map.origin.q3
   ));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_identity_global_to_local(PatchMap const& map, PatchCoordinate const& global) noexcept
{
   return patch_coordinate_from_generated(generated::patch_identity_global_to_local(
      global.q1,
      global.q2,
      global.q3,
      map.origin.q1,
      map.origin.q2,
      map.origin.q3
   ));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int patch_shell_primary_axis(int coordinate_map) noexcept
{
   if (coordinate_map == PatchMapThornburgShellPlusX
       || coordinate_map == PatchMapThornburgShellMinusX
       || coordinate_map == PatchMapCylindricalShellPlusX
       || coordinate_map == PatchMapCylindricalShellMinusX) {
      return 0;
   }
   if (coordinate_map == PatchMapThornburgShellPlusY
       || coordinate_map == PatchMapThornburgShellMinusY
       || coordinate_map == PatchMapCylindricalShellPlusY
       || coordinate_map == PatchMapCylindricalShellMinusY) {
      return 1;
   }
   return 2;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real patch_shell_primary_sign(int coordinate_map) noexcept
{
   if (coordinate_map == PatchMapThornburgShellMinusX
       || coordinate_map == PatchMapThornburgShellMinusY
       || coordinate_map == PatchMapThornburgShellMinusZ
       || coordinate_map == PatchMapCylindricalShellMinusX
       || coordinate_map == PatchMapCylindricalShellMinusY) {
      return -1.0_rt;
   }
   return 1.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int patch_shell_map_kind_for_face(bool cylindrical, int axis, int sign) noexcept
{
   if (cylindrical) {
      if (axis == 0) {
         return sign >= 0 ? PatchMapCylindricalShellPlusX : PatchMapCylindricalShellMinusX;
      }
      return sign >= 0 ? PatchMapCylindricalShellPlusY : PatchMapCylindricalShellMinusY;
   }
   if (axis == 0) {
      return sign >= 0 ? PatchMapThornburgShellPlusX : PatchMapThornburgShellMinusX;
   }
   if (axis == 1) {
      return sign >= 0 ? PatchMapThornburgShellPlusY : PatchMapThornburgShellMinusY;
   }
   return sign >= 0 ? PatchMapThornburgShellPlusZ : PatchMapThornburgShellMinusZ;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_thornburg_shell_local_to_global(
   PatchMap const& map,
   PatchCoordinate const& local
) noexcept
{
   const int axis = patch_shell_primary_axis(map.coordinate_map);
   const amrex::Real primary_sign = patch_shell_primary_sign(map.coordinate_map);
   if (axis == 0) {
      return patch_coordinate_from_generated(generated::thornburg_shell_local_to_global_axis0(
         local.q1,
         local.q2,
         local.q3,
         map.origin.q1,
         map.origin.q2,
         map.origin.q3,
         primary_sign,
         map.radial_inner,
         map.radial_outer
      ));
   }
   if (axis == 1) {
      return patch_coordinate_from_generated(generated::thornburg_shell_local_to_global_axis1(
         local.q1,
         local.q2,
         local.q3,
         map.origin.q1,
         map.origin.q2,
         map.origin.q3,
         primary_sign,
         map.radial_inner,
         map.radial_outer
      ));
   }
   return patch_coordinate_from_generated(generated::thornburg_shell_local_to_global_axis2(
      local.q1,
      local.q2,
      local.q3,
      map.origin.q1,
      map.origin.q2,
      map.origin.q3,
      primary_sign,
      map.radial_inner,
      map.radial_outer
   ));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_thornburg_shell_global_to_local(
   PatchMap const& map,
   PatchCoordinate const& global
) noexcept
{
   const int axis = patch_shell_primary_axis(map.coordinate_map);
   const amrex::Real primary_sign = patch_shell_primary_sign(map.coordinate_map);
   if (axis == 0) {
      return patch_coordinate_from_generated(generated::thornburg_shell_global_to_local_axis0(
         global.q1,
         global.q2,
         global.q3,
         map.origin.q1,
         map.origin.q2,
         map.origin.q3,
         primary_sign,
         map.radial_inner,
         map.radial_outer
      ));
   }
   if (axis == 1) {
      return patch_coordinate_from_generated(generated::thornburg_shell_global_to_local_axis1(
         global.q1,
         global.q2,
         global.q3,
         map.origin.q1,
         map.origin.q2,
         map.origin.q3,
         primary_sign,
         map.radial_inner,
         map.radial_outer
      ));
   }
   return patch_coordinate_from_generated(generated::thornburg_shell_global_to_local_axis2(
      global.q1,
      global.q2,
      global.q3,
      map.origin.q1,
      map.origin.q2,
      map.origin.q3,
      primary_sign,
      map.radial_inner,
      map.radial_outer
   ));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_cylindrical_shell_local_to_global(
   PatchMap const& map,
   PatchCoordinate const& local
) noexcept
{
   const int axis = patch_shell_primary_axis(map.coordinate_map);
   const amrex::Real primary_sign = patch_shell_primary_sign(map.coordinate_map);
   if (axis == 0) {
      return patch_coordinate_from_generated(generated::cylindrical_shell_local_to_global_axis0(
         local.q1,
         local.q2,
         local.q3,
         map.origin.q1,
         map.origin.q2,
         map.origin.q3,
         primary_sign,
         map.radial_inner,
         map.radial_outer
      ));
   }
   return patch_coordinate_from_generated(generated::cylindrical_shell_local_to_global_axis1(
      local.q1,
      local.q2,
      local.q3,
      map.origin.q1,
      map.origin.q2,
      map.origin.q3,
      primary_sign,
      map.radial_inner,
      map.radial_outer
   ));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_cylindrical_shell_global_to_local(
   PatchMap const& map,
   PatchCoordinate const& global
) noexcept
{
   const int axis = patch_shell_primary_axis(map.coordinate_map);
   const amrex::Real primary_sign = patch_shell_primary_sign(map.coordinate_map);
   if (axis == 0) {
      return patch_coordinate_from_generated(generated::cylindrical_shell_global_to_local_axis0(
         global.q1,
         global.q2,
         global.q3,
         map.origin.q1,
         map.origin.q2,
         map.origin.q3,
         primary_sign,
         map.radial_inner,
         map.radial_outer
      ));
   }
   return patch_coordinate_from_generated(generated::cylindrical_shell_global_to_local_axis1(
      global.q1,
      global.q2,
      global.q3,
      map.origin.q1,
      map.origin.q2,
      map.origin.q3,
      primary_sign,
      map.radial_inner,
      map.radial_outer
   ));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchMapJacobian patch_local_to_global_jacobian(PatchMap const& map, PatchCoordinate const& local) noexcept
{
   if (patch_coordinate_map_is_cartesian(map.coordinate_map)) {
      amrex::ignore_unused(local);
      return patch_jacobian_from_generated(generated::patch_identity_jacobian());
   }
   const int axis = patch_shell_primary_axis(map.coordinate_map);
   const amrex::Real primary_sign = patch_shell_primary_sign(map.coordinate_map);
   if (patch_coordinate_map_is_thornburg_shell(map.coordinate_map)) {
      if (axis == 0) {
         return patch_jacobian_from_generated(generated::thornburg_shell_jacobian_axis0(
            local.q1,
            local.q2,
            local.q3,
            primary_sign,
            map.radial_inner,
            map.radial_outer
         ));
      }
      if (axis == 1) {
         return patch_jacobian_from_generated(generated::thornburg_shell_jacobian_axis1(
            local.q1,
            local.q2,
            local.q3,
            primary_sign,
            map.radial_inner,
            map.radial_outer
         ));
      }
      return patch_jacobian_from_generated(generated::thornburg_shell_jacobian_axis2(
         local.q1,
         local.q2,
         local.q3,
         primary_sign,
         map.radial_inner,
         map.radial_outer
      ));
   }
   if (patch_coordinate_map_is_cylindrical_shell(map.coordinate_map)) {
      if (axis == 0) {
         return patch_jacobian_from_generated(generated::cylindrical_shell_jacobian_axis0(
            local.q1,
            local.q2,
            local.q3,
            primary_sign,
            map.radial_inner,
            map.radial_outer
         ));
      }
      return patch_jacobian_from_generated(generated::cylindrical_shell_jacobian_axis1(
         local.q1,
         local.q2,
         local.q3,
         primary_sign,
         map.radial_inner,
         map.radial_outer
      ));
   }
   amrex::Abort("patch_local_to_global_jacobian is not implemented for this coordinate map");
   return PatchMapJacobian{};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_local_to_global(PatchMap const& map, PatchCoordinate const& local)
{
   if (patch_coordinate_map_is_cartesian(map.coordinate_map)) {
      return patch_identity_local_to_global(map, local);
   }
   if (patch_coordinate_map_is_thornburg_shell(map.coordinate_map)) {
      return patch_thornburg_shell_local_to_global(map, local);
   }
   if (patch_coordinate_map_is_cylindrical_shell(map.coordinate_map)) {
      return patch_cylindrical_shell_local_to_global(map, local);
   }
   amrex::Abort("patch_local_to_global is not implemented for this coordinate map");
   return make_patch_coordinate(0.0_rt, 0.0_rt, 0.0_rt);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PatchCoordinate patch_global_to_local(PatchMap const& map, PatchCoordinate const& global)
{
   if (patch_coordinate_map_is_cartesian(map.coordinate_map)) {
      return patch_identity_global_to_local(map, global);
   }
   if (patch_coordinate_map_is_thornburg_shell(map.coordinate_map)) {
      return patch_thornburg_shell_global_to_local(map, global);
   }
   if (patch_coordinate_map_is_cylindrical_shell(map.coordinate_map)) {
      return patch_cylindrical_shell_global_to_local(map, global);
   }
   amrex::Abort("patch_global_to_local is not implemented for this coordinate map");
   return make_patch_coordinate(0.0_rt, 0.0_rt, 0.0_rt);
}

inline PatchCoordinate patch_local_to_global_checked(PatchMap const& map, PatchCoordinate const& local)
{
   return patch_local_to_global(map, local);
}

inline PatchCoordinate patch_global_to_local_checked(PatchMap const& map, PatchCoordinate const& global)
{
   return patch_global_to_local(map, global);
}

inline amrex::Real patch_coordinate_max_abs_difference(PatchCoordinate const& a, PatchCoordinate const& b)
{
   amrex::Real error = 0.0_rt;
   for (int component = 0; component < 3; ++component) {
      error = amrex::max(
         error,
         amrex::Math::abs(
            patch_coordinate_component(a, component)
            - patch_coordinate_component(b, component)
         )
      );
   }
   return error;
}

inline PatchMap make_patch_map(
   int coordinate_map,
   PatchCoordinate origin,
   amrex::Real radial_inner,
   amrex::Real radial_outer
)
{
   return PatchMap{coordinate_map, origin, radial_inner, radial_outer};
}

inline void validate_patch_map_radii(amrex::Real inner_radius, amrex::Real outer_radius)
{
   if (!(inner_radius > 0.0_rt) || !(outer_radius > inner_radius)) {
      amrex::Abort("patch map radii must satisfy 0 < inner_radius < outer_radius");
   }
}

inline PatchMapSystem make_thornburg04_patch_map_system(
   amrex::Real inner_radius,
   amrex::Real outer_radius,
   PatchCoordinate origin = {},
   PatchCartesianAxes active_axes = default_patch_cartesian_axes()
)
{
   validate_patch_map_radii(inner_radius, outer_radius);
   PatchMapSystem system;
   system.system_kind = PatchMapSystemThornburg04;
   system.origin = origin;
   system.inner_radius = inner_radius;
   system.medium_radius = 0.0_rt;
   system.outer_radius = outer_radius;
   system.active_axes = active_axes;
   system.maps.reserve(1 + 2 * AMREX_SPACEDIM);
   system.maps.push_back(make_patch_map(PatchMapThornburgCentralCartesian, origin, 0.0_rt, inner_radius));
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const int axis = patch_cartesian_axis(active_axes, dir);
      system.maps.push_back(make_patch_map(
         patch_shell_map_kind_for_face(false, axis, 1),
         origin,
         inner_radius,
         outer_radius
      ));
      system.maps.push_back(make_patch_map(
         patch_shell_map_kind_for_face(false, axis, -1),
         origin,
         inner_radius,
         outer_radius
      ));
   }
   return system;
}

inline PatchMapSystem make_thornburg13_patch_map_system(
   amrex::Real inner_radius,
   amrex::Real medium_radius,
   amrex::Real outer_radius,
   PatchCoordinate origin = {},
   PatchCartesianAxes active_axes = default_patch_cartesian_axes()
)
{
   validate_patch_map_radii(inner_radius, outer_radius);
   if (!(medium_radius > inner_radius) || !(outer_radius > medium_radius)) {
      amrex::Abort("Thornburg13 radii must satisfy inner_radius < medium_radius < outer_radius");
   }
   PatchMapSystem system = make_thornburg04_patch_map_system(inner_radius, medium_radius, origin, active_axes);
   system.system_kind = PatchMapSystemThornburg13;
   system.medium_radius = medium_radius;
   system.outer_radius = outer_radius;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const int axis = patch_cartesian_axis(active_axes, dir);
      system.maps.push_back(make_patch_map(
         patch_shell_map_kind_for_face(false, axis, 1),
         origin,
         medium_radius,
         outer_radius
      ));
      system.maps.push_back(make_patch_map(
         patch_shell_map_kind_for_face(false, axis, -1),
         origin,
         medium_radius,
         outer_radius
      ));
   }
   return system;
}

inline PatchMapSystem make_cylindrical5_patch_map_system(
   amrex::Real inner_radius,
   amrex::Real outer_radius,
   PatchCoordinate origin = {},
   PatchCartesianAxes active_axes = default_patch_cartesian_axes()
)
{
   validate_cylindrical_patch_cartesian_axes(active_axes);
   validate_patch_map_radii(inner_radius, outer_radius);
   PatchMapSystem system;
   system.system_kind = PatchMapSystemCylindrical5;
   system.origin = origin;
   system.inner_radius = inner_radius;
   system.medium_radius = 0.0_rt;
   system.outer_radius = outer_radius;
   system.active_axes = active_axes;
   system.maps.reserve(5);
   system.maps.push_back(make_patch_map(PatchMapThornburgCentralCartesian, origin, 0.0_rt, inner_radius));
   for (int axis = 0; axis < 2; ++axis) {
      system.maps.push_back(make_patch_map(
         patch_shell_map_kind_for_face(true, axis, 1),
         origin,
         inner_radius,
         outer_radius
      ));
      system.maps.push_back(make_patch_map(
         patch_shell_map_kind_for_face(true, axis, -1),
         origin,
         inner_radius,
         outer_radius
      ));
   }
   return system;
}

inline PatchMapSystem make_cylindrical9_patch_map_system(
   amrex::Real inner_radius,
   amrex::Real medium_radius,
   amrex::Real outer_radius,
   PatchCoordinate origin = {},
   PatchCartesianAxes active_axes = default_patch_cartesian_axes()
)
{
   validate_cylindrical_patch_cartesian_axes(active_axes);
   validate_patch_map_radii(inner_radius, outer_radius);
   if (!(medium_radius > inner_radius) || !(outer_radius > medium_radius)) {
      amrex::Abort("cylindrical9 radii must satisfy inner_radius < medium_radius < outer_radius");
   }
   PatchMapSystem system = make_cylindrical5_patch_map_system(inner_radius, medium_radius, origin, active_axes);
   system.system_kind = PatchMapSystemCylindrical9;
   system.medium_radius = medium_radius;
   system.outer_radius = outer_radius;
   for (int axis = 0; axis < 2; ++axis) {
      system.maps.push_back(make_patch_map(
         patch_shell_map_kind_for_face(true, axis, 1),
         origin,
         medium_radius,
         outer_radius
      ));
      system.maps.push_back(make_patch_map(
         patch_shell_map_kind_for_face(true, axis, -1),
         origin,
         medium_radius,
         outer_radius
      ));
   }
   return system;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int patch_dominant_axis_xy(PatchCoordinate const& centered) noexcept
{
   return amrex::Math::abs(centered.q2) > amrex::Math::abs(centered.q1) ? 1 : 0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int patch_dominant_active_axis(
   PatchCoordinate const& centered,
   PatchCartesianAxes const& active_axes
) noexcept
{
   int axis = patch_cartesian_axis(active_axes, 0);
   amrex::Real max_abs_component = amrex::Math::abs(patch_coordinate_component(centered, axis));
   for (int dir = 1; dir < AMREX_SPACEDIM; ++dir) {
      const int candidate = patch_cartesian_axis(active_axes, dir);
      const amrex::Real abs_component = amrex::Math::abs(patch_coordinate_component(centered, candidate));
      if (abs_component > max_abs_component) {
         axis = candidate;
         max_abs_component = abs_component;
      }
   }
   return axis;
}

inline int patch_system_shell_patch_id(
   PatchMapSystem const& system,
   int coordinate_map,
   bool outer_shell
)
{
   int begin = 1;
   int end = static_cast<int>(system.maps.size());
   if (system.system_kind == PatchMapSystemThornburg13 || system.system_kind == PatchMapSystemCylindrical9) {
      const int group_size = (static_cast<int>(system.maps.size()) - 1) / 2;
      if (outer_shell) {
         begin = 1 + group_size;
      } else {
         end = 1 + group_size;
      }
   }
   for (int patch_id = begin; patch_id < end; ++patch_id) {
      if (system.maps[patch_id].coordinate_map == coordinate_map) {
         return patch_id;
      }
   }
   return -1;
}

inline int patch_system_owner_patch_id(PatchMapSystem const& system, PatchCoordinate const& global)
{
   const PatchCoordinate centered = patch_coordinate_subtract(global, system.origin);
   if (system.system_kind == PatchMapSystemThornburg04 || system.system_kind == PatchMapSystemThornburg13) {
      const amrex::Real radial_coordinate = cartesian_radius(centered);
      const int axis = patch_dominant_active_axis(centered, system.active_axes);
      if (radial_coordinate < system.inner_radius) {
         return 0;
      }
      if (radial_coordinate > system.outer_radius) {
         return -1;
      }
      const int sign = patch_coordinate_component(centered, axis) >= 0.0_rt ? 1 : -1;
      const bool outer_shell = system.system_kind == PatchMapSystemThornburg13
         && radial_coordinate >= system.medium_radius;
      return patch_system_shell_patch_id(
         system,
         patch_shell_map_kind_for_face(false, axis, sign),
         outer_shell
      );
   }
   if (system.system_kind == PatchMapSystemCylindrical5 || system.system_kind == PatchMapSystemCylindrical9) {
      const amrex::Real radial_coordinate = cartesian_cylindrical_radius(centered);
      if (radial_coordinate < system.inner_radius) {
         return 0;
      }
      if (radial_coordinate > system.outer_radius) {
         return -1;
      }
      const int axis = patch_dominant_axis_xy(centered);
      const int sign = patch_coordinate_component(centered, axis) >= 0.0_rt ? 1 : -1;
      const bool outer_shell = system.system_kind == PatchMapSystemCylindrical9
         && radial_coordinate >= system.medium_radius;
      return patch_system_shell_patch_id(
         system,
         patch_shell_map_kind_for_face(true, axis, sign),
         outer_shell
      );
   }
   return 0;
}

inline PatchMapPoint patch_system_global_to_local(PatchMapSystem const& system, PatchCoordinate const& global)
{
   const int patch_id = patch_system_owner_patch_id(system, global);
   if (patch_id < 0 || patch_id >= static_cast<int>(system.maps.size())) {
      amrex::Abort("global coordinate is outside the patch map system");
   }
   return PatchMapPoint{patch_id, patch_global_to_local(system.maps[patch_id], global), global};
}

inline PatchMapPoint patch_system_local_to_global(
   PatchMapSystem const& system,
   int patch_id,
   PatchCoordinate const& local
)
{
   if (patch_id < 0 || patch_id >= static_cast<int>(system.maps.size())) {
      amrex::Abort("local coordinate patch_id is outside the patch map system");
   }
   const PatchCoordinate global = patch_local_to_global(system.maps[patch_id], local);
   return PatchMapPoint{patch_id, local, global};
}

} // namespace pyfluid_amrex
