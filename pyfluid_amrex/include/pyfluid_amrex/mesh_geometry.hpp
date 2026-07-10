#pragma once

#include <AMReX.H>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

namespace pyfluid_amrex {

enum MeshCoordinateKind {
   MeshCartesian = 0,
   MeshCylindrical = 1,
   MeshSpherical = 2,
};

enum MeshPatchSystem {
   MeshPatchSingle = 0,
   MeshPatchHomogeneousCartesian = 1,
   MeshPatchMixedThornburg04 = 2,
   MeshPatchMixedThornburg13 = 3,
   MeshPatchMixedCylindrical5 = 4,
   MeshPatchMixedCylindrical9 = 5,
};

enum MeshAxis {
   MeshAxisInvalid = 0,
   MeshAxisX = 1,
   MeshAxisY = 2,
   MeshAxisZ = 3,
   MeshAxisR = 4,
   MeshAxisPhi = 5,
   MeshAxisTheta = 6,
};

struct MeshAxes {
   amrex::GpuArray<int, AMREX_SPACEDIM> axis{};
};

struct MeshRuntimeLayout {
   int coordinate = MeshCartesian;
   MeshAxes axes{};
   int patch_system = MeshPatchSingle;
   bool curvilinear = false;
};

inline int mesh_coordinate_code(std::string const& name)
{
   if (name == "cartesian") {
      return MeshCartesian;
   }
   if (name == "cylindrical") {
      return MeshCylindrical;
   }
   if (name == "spherical") {
      return MeshSpherical;
   }
   amrex::Abort("unsupported mesh.coordinate; use cartesian, cylindrical, or spherical");
   return -1;
}

inline char const* mesh_coordinate_name(int code)
{
   if (code == MeshCartesian) {
      return "cartesian";
   }
   if (code == MeshCylindrical) {
      return "cylindrical";
   }
   if (code == MeshSpherical) {
      return "spherical";
   }
   return "unknown";
}

inline std::string mesh_default_axes(int coordinate)
{
   if (coordinate == MeshCartesian) {
#if (AMREX_SPACEDIM == 1)
      return "x";
#elif (AMREX_SPACEDIM == 2)
      return "x y";
#else
      return "x y z";
#endif
   }
   if (coordinate == MeshCylindrical) {
#if (AMREX_SPACEDIM == 1)
      return "r";
#elif (AMREX_SPACEDIM == 2)
      return "r phi";
#else
      return "r phi z";
#endif
   }
   if (coordinate == MeshSpherical) {
#if (AMREX_SPACEDIM == 1)
      return "r";
#elif (AMREX_SPACEDIM == 2)
      return "r theta";
#else
      return "r theta phi";
#endif
   }
   amrex::Abort("invalid mesh coordinate code");
   return "";
}

inline std::string mesh_axes_or_default(int coordinate, std::string axes)
{
   auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
   while (!axes.empty() && is_space(static_cast<unsigned char>(axes.front()))) {
      axes.erase(axes.begin());
   }
   while (!axes.empty() && is_space(static_cast<unsigned char>(axes.back()))) {
      axes.pop_back();
   }
   return axes.empty() ? mesh_default_axes(coordinate) : axes;
}

inline std::vector<std::string> mesh_axis_tokens(int coordinate, std::string const& axes)
{
   std::string normalized = mesh_axes_or_default(coordinate, axes);
   std::replace(normalized.begin(), normalized.end(), ',', ' ');
   std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
   });
   std::stringstream stream(normalized);
   std::vector<std::string> tokens;
   std::string token;
   while (stream >> token) {
      tokens.push_back(token);
   }
   return tokens;
}

inline int mesh_axis_code(std::string const& token)
{
   if (token == "x") {
      return MeshAxisX;
   }
   if (token == "y") {
      return MeshAxisY;
   }
   if (token == "z") {
      return MeshAxisZ;
   }
   if (token == "r") {
      return MeshAxisR;
   }
   if (token == "phi") {
      return MeshAxisPhi;
   }
   if (token == "theta") {
      return MeshAxisTheta;
   }
   amrex::Abort("unsupported mesh.axes token: " + token);
   return MeshAxisInvalid;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int mesh_axis(MeshAxes const& axes, int dir) noexcept
{
   if (dir >= 0 && dir < AMREX_SPACEDIM) {
      return axes.axis[dir];
   }
   return MeshAxisInvalid;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int mesh_direction_for_axis(MeshAxes const& axes, int target_axis) noexcept
{
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (mesh_axis(axes, dir) == target_axis) {
         return dir;
      }
   }
   return -1;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool mesh_has_axis(MeshAxes const& axes, int target_axis) noexcept
{
   return mesh_direction_for_axis(axes, target_axis) >= 0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<int, 3> mesh_canonical_vector_axes(int coordinate) noexcept
{
   if (coordinate == MeshCylindrical) {
      return {MeshAxisR, MeshAxisPhi, MeshAxisZ};
   }
   if (coordinate == MeshSpherical) {
      return {MeshAxisR, MeshAxisTheta, MeshAxisPhi};
   }
   return {MeshAxisX, MeshAxisY, MeshAxisZ};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int mesh_vector_axis_for_component(int coordinate, MeshAxes const& axes, int component) noexcept
{
   if (component < 0 || component >= 3) {
      return MeshAxisInvalid;
   }
   if (component < AMREX_SPACEDIM) {
      return mesh_axis(axes, component);
   }
   const amrex::GpuArray<int, 3> canonical = mesh_canonical_vector_axes(coordinate);
   int missing_index = AMREX_SPACEDIM;
   for (int n = 0; n < 3; ++n) {
      const int axis = canonical[n];
      if (!mesh_has_axis(axes, axis)) {
         if (missing_index == component) {
            return axis;
         }
         ++missing_index;
      }
   }
   return MeshAxisInvalid;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int mesh_canonical_axis_index(int coordinate, int axis) noexcept
{
   if (coordinate == MeshCylindrical) {
      if (axis == MeshAxisR) {
         return 0;
      }
      if (axis == MeshAxisPhi) {
         return 1;
      }
      if (axis == MeshAxisZ) {
         return 2;
      }
      return -1;
   }
   if (coordinate == MeshSpherical) {
      if (axis == MeshAxisR) {
         return 0;
      }
      if (axis == MeshAxisTheta) {
         return 1;
      }
      if (axis == MeshAxisPhi) {
         return 2;
      }
      return -1;
   }
   if (axis == MeshAxisX) {
      return 0;
   }
   if (axis == MeshAxisY) {
      return 1;
   }
   if (axis == MeshAxisZ) {
      return 2;
   }
   return -1;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MeshAxes mesh_default_axes_layout(int coordinate) noexcept
{
   MeshAxes axes{};
   if (coordinate == MeshCartesian) {
      axes.axis[0] = MeshAxisX;
#if (AMREX_SPACEDIM == 1)
      return axes;
#elif (AMREX_SPACEDIM == 2)
      axes.axis[1] = MeshAxisY;
      return axes;
#else
      axes.axis[1] = MeshAxisY;
      axes.axis[2] = MeshAxisZ;
      return axes;
#endif
   }
   if (coordinate == MeshCylindrical) {
      axes.axis[0] = MeshAxisR;
#if (AMREX_SPACEDIM == 1)
      return axes;
#elif (AMREX_SPACEDIM == 2)
      axes.axis[1] = MeshAxisPhi;
      return axes;
#else
      axes.axis[1] = MeshAxisPhi;
      axes.axis[2] = MeshAxisZ;
      return axes;
#endif
   }
   if (coordinate == MeshSpherical) {
      axes.axis[0] = MeshAxisR;
#if (AMREX_SPACEDIM == 1)
      return axes;
#elif (AMREX_SPACEDIM == 2)
      axes.axis[1] = MeshAxisTheta;
      return axes;
#else
      axes.axis[1] = MeshAxisTheta;
      axes.axis[2] = MeshAxisPhi;
      return axes;
#endif
   }
   axes.axis[0] = MeshAxisInvalid;
   return axes;
}

inline bool mesh_axes_equal(MeshAxes const& axes, int a0, int a1 = MeshAxisInvalid, int a2 = MeshAxisInvalid)
{
   if (mesh_axis(axes, 0) != a0) {
      return false;
   }
#if (AMREX_SPACEDIM >= 2)
   if (mesh_axis(axes, 1) != a1) {
      return false;
   }
#else
   amrex::ignore_unused(a1);
#endif
#if (AMREX_SPACEDIM == 3)
   if (mesh_axis(axes, 2) != a2) {
      return false;
   }
#else
   amrex::ignore_unused(a2);
#endif
   return true;
}

inline MeshAxes mesh_axes_layout(int coordinate, std::string const& axes)
{
   // Component indices are active runtime directions. The coordinate family
   // defines one canonical order; lower-dimensional layouts must preserve it.
   const std::vector<std::string> tokens = mesh_axis_tokens(coordinate, axes);
   if (static_cast<int>(tokens.size()) != AMREX_SPACEDIM) {
      amrex::Abort("mesh.axes must list exactly AMREX_SPACEDIM active axes");
   }
   MeshAxes layout{};
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      layout.axis[dir] = mesh_axis_code(tokens[dir]);
   }
   if (coordinate == MeshCartesian) {
#if (AMREX_SPACEDIM == 1)
      if (mesh_axes_equal(layout, MeshAxisX) || mesh_axes_equal(layout, MeshAxisY) || mesh_axes_equal(layout, MeshAxisZ)) {
         return layout;
      }
#elif (AMREX_SPACEDIM == 2)
      if (
         mesh_axes_equal(layout, MeshAxisX, MeshAxisY)
         || mesh_axes_equal(layout, MeshAxisX, MeshAxisZ)
         || mesh_axes_equal(layout, MeshAxisY, MeshAxisZ)
      ) {
         return layout;
      }
#else
      if (mesh_axes_equal(layout, MeshAxisX, MeshAxisY, MeshAxisZ)) {
         return layout;
      }
#endif
      amrex::Abort("cartesian mesh.axes supports ordered subsets of x y z for this executable");
   }
   if (coordinate == MeshCylindrical) {
#if (AMREX_SPACEDIM == 1)
      if (
         mesh_axes_equal(layout, MeshAxisR)
         || mesh_axes_equal(layout, MeshAxisZ)
         || mesh_axes_equal(layout, MeshAxisPhi)
      ) {
         return layout;
      }
#elif (AMREX_SPACEDIM == 2)
      if (
         mesh_axes_equal(layout, MeshAxisR, MeshAxisPhi)
         || mesh_axes_equal(layout, MeshAxisR, MeshAxisZ)
         || mesh_axes_equal(layout, MeshAxisPhi, MeshAxisZ)
      ) {
         return layout;
      }
#else
      if (mesh_axes_equal(layout, MeshAxisR, MeshAxisPhi, MeshAxisZ)) {
         return layout;
      }
#endif
      amrex::Abort("cylindrical mesh.axes supports ordered subsets of r phi z for this executable");
   }
   if (coordinate == MeshSpherical) {
#if (AMREX_SPACEDIM == 1)
      if (
         mesh_axes_equal(layout, MeshAxisR)
         || mesh_axes_equal(layout, MeshAxisTheta)
         || mesh_axes_equal(layout, MeshAxisPhi)
      ) {
         return layout;
      }
#elif (AMREX_SPACEDIM == 2)
      if (
         mesh_axes_equal(layout, MeshAxisR, MeshAxisTheta)
         || mesh_axes_equal(layout, MeshAxisR, MeshAxisPhi)
         || mesh_axes_equal(layout, MeshAxisTheta, MeshAxisPhi)
      ) {
         return layout;
      }
#else
      if (mesh_axes_equal(layout, MeshAxisR, MeshAxisTheta, MeshAxisPhi)) {
         return layout;
      }
#endif
      amrex::Abort("spherical mesh.axes supports ordered subsets of r theta phi for this executable");
   }
   amrex::Abort("invalid mesh coordinate code");
   return layout;
}

inline int mesh_patch_system_code(std::string const& name)
{
   if (name == "single") {
      return MeshPatchSingle;
   }
   if (name == "multi_homogeneous_cartesian" || name == "cartesian_multipatch") {
      return MeshPatchHomogeneousCartesian;
   }
   if (name == "multi_mixed_thornburg04" || name == "mixed_thornburg04") {
      return MeshPatchMixedThornburg04;
   }
   if (name == "multi_mixed_thornburg13" || name == "mixed_thornburg13") {
      return MeshPatchMixedThornburg13;
   }
   if (name == "multi_mixed_cylindrical5" || name == "mixed_cylindrical5") {
      return MeshPatchMixedCylindrical5;
   }
   if (name == "multi_mixed_cylindrical9" || name == "mixed_cylindrical9") {
      return MeshPatchMixedCylindrical9;
   }
   amrex::Abort(
      "unsupported mesh.patch_system; use single, multi_homogeneous_cartesian, "
      "multi_mixed_thornburg04, multi_mixed_thornburg13, "
      "multi_mixed_cylindrical5, or multi_mixed_cylindrical9"
   );
   return -1;
}

inline bool mesh_coordinate_has_metric_terms(int coordinate)
{
   return coordinate == MeshCylindrical || coordinate == MeshSpherical;
}

inline bool mesh_coordinate_uses_amrex_builtin_metric(int coordinate, MeshAxes const& axes)
{
#if (AMREX_SPACEDIM == 2)
   if (coordinate == MeshCylindrical && mesh_axes_equal(axes, MeshAxisR, MeshAxisZ)) {
      return true;
   }
   if (coordinate == MeshSpherical && mesh_axes_equal(axes, MeshAxisR, MeshAxisTheta)) {
      return true;
   }
#elif (AMREX_SPACEDIM == 1)
   if (coordinate == MeshSpherical && mesh_axes_equal(axes, MeshAxisR)) {
      return true;
   }
#else
   amrex::ignore_unused(coordinate, axes);
#endif
   return false;
}

inline bool mesh_coordinate_uses_amrex_builtin_metric(int coordinate)
{
   return mesh_coordinate_uses_amrex_builtin_metric(coordinate, mesh_axes_layout(coordinate, ""));
}

inline bool mesh_patch_system_is_multipatch(int patch_system)
{
   return patch_system == MeshPatchHomogeneousCartesian
      || patch_system == MeshPatchMixedThornburg04
      || patch_system == MeshPatchMixedThornburg13
      || patch_system == MeshPatchMixedCylindrical5
      || patch_system == MeshPatchMixedCylindrical9;
}

inline int amrex_coord_sys_for_mesh_coordinate(int coordinate, MeshAxes const& axes)
{
   if (coordinate == MeshCartesian) {
      return 0;
   }
   if (coordinate == MeshCylindrical) {
#if (AMREX_SPACEDIM == 2)
      return mesh_axes_equal(axes, MeshAxisR, MeshAxisZ) ? 1 : 0;
#else
      amrex::ignore_unused(axes);
      return 0;
#endif
   }
   if (coordinate == MeshSpherical) {
#if (AMREX_SPACEDIM <= 2)
      return mesh_coordinate_uses_amrex_builtin_metric(coordinate, axes) ? 2 : 0;
#else
      amrex::ignore_unused(axes);
      return 0;
#endif
   }
   amrex::Abort("invalid mesh coordinate code");
   return -1;
}

inline int amrex_coord_sys_for_mesh_coordinate(int coordinate)
{
   return amrex_coord_sys_for_mesh_coordinate(coordinate, mesh_axes_layout(coordinate, ""));
}

inline void validate_amrex_geometry_coord_sys(int mesh_coordinate, MeshAxes const& axes, int geometry_coord_sys)
{
   const int expected = amrex_coord_sys_for_mesh_coordinate(mesh_coordinate, axes);
   if (geometry_coord_sys != expected) {
      amrex::Abort(
         std::string("mesh.coordinate=") + mesh_coordinate_name(mesh_coordinate)
         + " conflicts with geometry.coord_sys=" + std::to_string(geometry_coord_sys)
      );
   }
}

inline void validate_amrex_geometry_coord_sys(int mesh_coordinate, int geometry_coord_sys)
{
   validate_amrex_geometry_coord_sys(mesh_coordinate, mesh_axes_layout(mesh_coordinate, ""), geometry_coord_sys);
}

inline void validate_mesh_geometry_for_current_solver(int coordinate, int patch_system)
{
   if (coordinate == MeshCartesian && patch_system == MeshPatchSingle) {
      return;
   }
   if (coordinate == MeshCartesian && patch_system == MeshPatchHomogeneousCartesian) {
      return;
   }
   if (
      coordinate == MeshCartesian
      && (
         patch_system == MeshPatchMixedThornburg04
         || patch_system == MeshPatchMixedThornburg13
         || patch_system == MeshPatchMixedCylindrical5
         || patch_system == MeshPatchMixedCylindrical9
      )
   ) {
      return;
   }
   if (
      (coordinate == MeshCylindrical || coordinate == MeshSpherical)
      && patch_system == MeshPatchSingle
   ) {
      return;
   }
   amrex::Abort(
      "this mesh geometry is not supported by the current finite-volume kernels; "
      "supported production geometries are Cartesian single-patch, Cartesian "
      "homogeneous/mixed multipatch, and cylindrical/spherical single-patch "
      "finite-volume runs"
   );
}

} // namespace pyfluid_amrex
