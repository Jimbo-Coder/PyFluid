#pragma once

#include "pyfluid_amrex/fields.hpp"

#include <AMReX_BC_TYPES.H>
#include <AMReX_BCRec.H>
#include <AMReX_BoxList.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <cmath>
#include <string>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/patch_domain.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int clamp_index(int value, int lo, int hi) noexcept
{
   return value < lo ? lo : (value > hi ? hi : value);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int wrapped_index(int value, int lo, int hi) noexcept
{
   const int n = hi - lo + 1;
   int shifted = (value - lo) % n;
   if (shifted < 0) {
      shifted += n;
   }
   return lo + shifted;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int mirrored_index(int value, int lo, int hi) noexcept
{
   if (value < lo) {
      return clamp_index(lo + (lo - 1 - value), lo, hi);
   }
   if (value > hi) {
      return clamp_index(hi - (value - hi - 1), lo, hi);
   }
   return value;
}

struct BoundaryData {
   amrex::GpuArray<int, AMREX_SPACEDIM> lo_kind{};
   amrex::GpuArray<int, AMREX_SPACEDIM> hi_kind{};
   int mesh_coordinate = MeshCartesian;
   MeshAxes mesh_axes{};
   amrex::GpuArray<int, AMREX_SPACEDIM> origin_lo{};
   amrex::GpuArray<int, AMREX_SPACEDIM * MaxStateComponents> origin_state_parity{};
   EOSParams eos;
   int eos_method = EOSIdealGas;
   amrex::Real inflow_rho = 1.0_rt;
   amrex::Real inflow_pressure = 1.0_rt;
   amrex::Real inflow_v1 = 0.0_rt;
   amrex::Real inflow_v2 = 0.0_rt;
   amrex::Real inflow_v3 = 0.0_rt;
   amrex::Real inflow_b1 = 0.0_rt;
   amrex::Real inflow_b2 = 0.0_rt;
   amrex::Real inflow_b3 = 0.0_rt;
   int inflow_profile = 0;
   amrex::Real inflow_background_rho = 1.0_rt;
   amrex::Real inflow_background_pressure = 1.0_rt;
   amrex::Real inflow_background_v1 = 0.0_rt;
   amrex::Real inflow_background_v2 = 0.0_rt;
   amrex::Real inflow_background_v3 = 0.0_rt;
   amrex::Real inflow_background_b1 = 0.0_rt;
   amrex::Real inflow_background_b2 = 0.0_rt;
   amrex::Real inflow_background_b3 = 0.0_rt;
   int passive_scalars = 0;
   amrex::Real inflow_passive_value = 1.0_rt;
   amrex::Real inflow_background_passive_value = 0.0_rt;
   amrex::Real inflow_center_y = 0.5_rt;
   amrex::Real inflow_center_z = 0.5_rt;
   amrex::Real inflow_radius = 0.08_rt;
   int analytic_profile = 0;
   amrex::Real shock_x0 = 0.5_rt;
   amrex::Real shock_y0 = 0.5_rt;
   amrex::Real shock_z0 = 0.5_rt;
   amrex::Real shock_n1 = 1.0_rt;
   amrex::Real shock_n2 = 0.0_rt;
   amrex::Real shock_n3 = 0.0_rt;
   amrex::Real shock_speed = 0.0_rt;
   amrex::Real rho_left = 1.0_rt;
   amrex::Real pressure_left = 1.0_rt;
   amrex::Real v1_left = 0.0_rt;
   amrex::Real v2_left = 0.0_rt;
   amrex::Real v3_left = 0.0_rt;
   amrex::Real rho_right = 1.0_rt;
   amrex::Real pressure_right = 1.0_rt;
   amrex::Real v1_right = 0.0_rt;
   amrex::Real v2_right = 0.0_rt;
   amrex::Real v3_right = 0.0_rt;
   int mhd = 0;
};

inline BoundaryData make_boundary_data(RunConfig const& cfg)
{
   BoundaryData data;
   data.mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   data.mesh_axes = mesh_axes_layout(data.mesh_coordinate, cfg.mesh_axes);
   data.lo_kind[0] = static_cast<int>(side_boundary_kind(cfg, 0, false));
   data.hi_kind[0] = static_cast<int>(side_boundary_kind(cfg, 0, true));
#if (AMREX_SPACEDIM >= 2)
   data.lo_kind[1] = static_cast<int>(side_boundary_kind(cfg, 1, false));
   data.hi_kind[1] = static_cast<int>(side_boundary_kind(cfg, 1, true));
#endif
#if (AMREX_SPACEDIM == 3)
   data.lo_kind[2] = static_cast<int>(side_boundary_kind(cfg, 2, false));
   data.hi_kind[2] = static_cast<int>(side_boundary_kind(cfg, 2, true));
#endif
   data.eos = eos_params(cfg);
   data.eos_method = eos_method_code(cfg);
   data.inflow_rho = cfg.inflow_rho;
   data.inflow_pressure = cfg.inflow_pressure;
   data.inflow_v1 = cfg.inflow_v1;
   data.inflow_v2 = cfg.inflow_v2;
   data.inflow_v3 = cfg.inflow_v3;
   data.inflow_b1 = cfg.inflow_b1;
   data.inflow_b2 = cfg.inflow_b2;
   data.inflow_b3 = cfg.inflow_b3;
   data.inflow_profile = cfg.inflow_profile == "gaussian_beam" ? 1 : (cfg.inflow_profile == "uniform_beam" ? 2 : 0);
   data.inflow_background_rho = cfg.inflow_background_rho;
   data.inflow_background_pressure = cfg.inflow_background_pressure;
   data.inflow_background_v1 = cfg.inflow_background_v1;
   data.inflow_background_v2 = cfg.inflow_background_v2;
   data.inflow_background_v3 = cfg.inflow_background_v3;
   data.inflow_background_b1 = cfg.inflow_background_b1;
   data.inflow_background_b2 = cfg.inflow_background_b2;
   data.inflow_background_b3 = cfg.inflow_background_b3;
   data.passive_scalars = cfg.passive_scalars;
   const PhysicalSystem system = physical_system(cfg);
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      data.origin_lo[dir] = data.lo_kind[dir] == static_cast<int>(BoundaryKind::Origin) ? 1 : 0;
      for (int component = 0; component < MaxStateComponents; ++component) {
         const int offset = dir * MaxStateComponents + component;
         data.origin_state_parity[offset] =
            data.origin_lo[dir] != 0
            ? state_grid_field_origin_parity(
                 data.mesh_coordinate,
                 data.mesh_axes,
                 system,
                 data.passive_scalars,
                 component
              )
            : 1;
      }
   }
   data.inflow_passive_value = cfg.inflow_passive_value;
   data.inflow_background_passive_value = cfg.inflow_background_passive_value;
   data.inflow_center_y = cfg.inflow_center_y;
   data.inflow_center_z = cfg.inflow_center_z;
   data.inflow_radius = cfg.inflow_radius;
   data.analytic_profile = cfg.boundary_profile == "double_mach_reflection" ? 1 : 0;
   data.shock_x0 = cfg.shock_x0;
   data.shock_y0 = cfg.shock_y0;
   data.shock_z0 = cfg.shock_z0;
   data.shock_n1 = cfg.shock_nx;
   data.shock_n2 = cfg.shock_ny;
   data.shock_n3 = cfg.shock_nz;
   data.shock_speed = cfg.shock_speed;
   data.rho_left = cfg.rho_left;
   data.pressure_left = cfg.pressure_left;
   data.v1_left = cfg.v1_left;
   data.v2_left = cfg.v2_left;
   data.v3_left = cfg.v3_left;
   data.rho_right = cfg.rho_right;
   data.pressure_right = cfg.pressure_right;
   data.v1_right = cfg.v1_right;
   data.v2_right = cfg.v2_right;
   data.v3_right = cfg.v3_right;
   data.mhd = physical_system_has_magnetic_field(system) ? 1 : 0;
   return data;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int origin_state_parity(BoundaryData const& bc, int dir, int component) noexcept
{
   if (dir < 0 || dir >= AMREX_SPACEDIM || component < 0 || component >= MaxStateComponents) {
      return 1;
   }
   return bc.origin_state_parity[dir * MaxStateComponents + component];
}

inline bool needs_radial_antipodal_origin_boundary(BoundaryData const& bc) noexcept
{
   const int radial_dir = mesh_direction_for_axis(bc.mesh_axes, MeshAxisR);
   return radial_dir >= 0 && bc.origin_lo[radial_dir] != 0
      && (mesh_has_axis(bc.mesh_axes, MeshAxisPhi)
          || (bc.mesh_coordinate == MeshSpherical && mesh_has_axis(bc.mesh_axes, MeshAxisTheta)));
}


void validate_origin_boundary_geometry(BoundaryData const& bc, amrex::Geometry const& geom);

inline bool boundary_data_all_periodic(BoundaryData const& bc)
{
   bool all_periodic =
      bc.lo_kind[0] == static_cast<int>(BoundaryKind::Periodic)
      && bc.hi_kind[0] == static_cast<int>(BoundaryKind::Periodic);
#if (AMREX_SPACEDIM >= 2)
   all_periodic =
      all_periodic
      && bc.lo_kind[1] == static_cast<int>(BoundaryKind::Periodic)
      && bc.hi_kind[1] == static_cast<int>(BoundaryKind::Periodic);
#endif
#if (AMREX_SPACEDIM == 3)
   all_periodic =
      all_periodic
      && bc.lo_kind[2] == static_cast<int>(BoundaryKind::Periodic)
      && bc.hi_kind[2] == static_cast<int>(BoundaryKind::Periodic);
#endif
   return all_periodic;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool boundary_kind_is_inflow(int kind) noexcept
{
   return kind == static_cast<int>(BoundaryKind::Inflow);
}

inline int amrex_state_bc_type(BoundaryKind kind, int component, int dir, bool mhd)
{
   if (kind == BoundaryKind::Periodic) {
      return amrex::BCType::int_dir;
   }
   if (kind == BoundaryKind::Inflow) {
      return amrex::BCType::ext_dir;
   }
   if (kind == BoundaryKind::Analytic) {
      return amrex::BCType::ext_dir;
   }
   if (kind == BoundaryKind::Outflow || kind == BoundaryKind::Diode) {
      return amrex::BCType::foextrap;
   }
   if (kind == BoundaryKind::Origin) {
      return amrex::BCType::reflect_even;
   }
   if (kind == BoundaryKind::Reflecting || kind == BoundaryKind::Symmetry) {
      const bool normal_momentum =
         (dir == 0 && component == MOM1)
#if (AMREX_SPACEDIM >= 2)
         || (dir == 1 && component == MOM2)
#endif
#if (AMREX_SPACEDIM == 3)
         || (dir == 2 && component == MOM3)
#endif
         ;
      if (normal_momentum) {
         return amrex::BCType::reflect_odd;
      }
      if (kind == BoundaryKind::Symmetry && mhd) {
         const bool magnetic_component = component == B1 || component == B2 || component == B3;
         const bool normal_magnetic =
            (dir == 0 && component == B1)
#if (AMREX_SPACEDIM >= 2)
            || (dir == 1 && component == B2)
#endif
#if (AMREX_SPACEDIM == 3)
            || (dir == 2 && component == B3)
#endif
            ;
         if (magnetic_component && !normal_magnetic) {
            return amrex::BCType::reflect_odd;
         }
      }
      return amrex::BCType::reflect_even;
   }
   return amrex::BCType::int_dir;
}

inline int amrex_face_b_bc_type(BoundaryKind kind, int face_dir, int boundary_dir)
{
   if (kind == BoundaryKind::Periodic) {
      return amrex::BCType::int_dir;
   }
   if (kind == BoundaryKind::Inflow) {
      return amrex::BCType::ext_dir;
   }
   if (kind == BoundaryKind::Analytic) {
      return amrex::BCType::ext_dir;
   }
   if (kind == BoundaryKind::Outflow || kind == BoundaryKind::Diode) {
      return amrex::BCType::foextrap;
   }
   if (kind == BoundaryKind::Origin) {
      return amrex::BCType::reflect_even;
   }
   if (kind == BoundaryKind::Reflecting) {
      return amrex::BCType::reflect_even;
   }
   if (kind == BoundaryKind::Symmetry) {
      return face_dir == boundary_dir ? amrex::BCType::reflect_even : amrex::BCType::reflect_odd;
   }
   return amrex::BCType::int_dir;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real boundary_profile_state_component(
   BoundaryData const& bc,
   int component,
   amrex::Real rho,
   amrex::Real pressure,
   amrex::Real vx,
   amrex::Real vy,
   amrex::Real vz,
   amrex::Real bx,
   amrex::Real by,
   amrex::Real bz,
   amrex::Real passive_value
) noexcept
{
   if (component == RHO) {
      return rho;
   }
   if (component == MOM1) {
      return rho * vx;
   }
   if (component == MOM2) {
      return rho * vy;
   }
   if (component == MOM3) {
      return rho * vz;
   }
   if (component == ENERGY) {
      if (bc.mhd != 0) {
         return total_mhd_energy_runtime(
            bc.eos_method,
            rho, vx, vy, vz, pressure,
            bx, by, bz, bc.eos
         );
      }
      return total_hydro_energy_runtime(
         bc.eos_method, rho, vx, vy, vz, pressure, bc.eos
      );
   }
   const int passive_begin = bc.mhd != 0 ? NVAR_MHD : NVAR_HYDRO;
   if (component >= passive_begin && component < passive_begin + bc.passive_scalars) {
      return rho * passive_value;
   }
   if (component == B1) {
      return bx;
   }
   if (component == B2) {
      return by;
   }
   return bz;
}

enum AnalyticBoundaryAction {
   AnalyticBoundaryCopy = 0,
   AnalyticBoundaryLeftState = 1,
   AnalyticBoundaryRightState = 2,
   AnalyticBoundaryReflect = 3,
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool planar_shock_left_state(
   BoundaryData const& bc,
   amrex::Real x,
   amrex::Real y,
   amrex::Real z,
   amrex::Real time
) noexcept
{
   const amrex::Real signed_distance = bc.shock_n1 * (x - bc.shock_x0)
      + bc.shock_n2 * (y - bc.shock_y0)
      + bc.shock_n3 * (z - bc.shock_z0)
      - bc.shock_speed * time;
   return signed_distance < 0.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real analytic_shock_state_component(
   BoundaryData const& bc,
   int component,
   bool left_state
) noexcept
{
   return boundary_profile_state_component(
      bc,
      component,
      left_state ? bc.rho_left : bc.rho_right,
      left_state ? bc.pressure_left : bc.pressure_right,
      left_state ? bc.v1_left : bc.v1_right,
      left_state ? bc.v2_left : bc.v2_right,
      left_state ? bc.v3_left : bc.v3_right,
      0.0_rt,
      0.0_rt,
      0.0_rt,
      0.0_rt
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int analytic_boundary_action(
   BoundaryData const& bc,
   int side,
   amrex::Real x,
   amrex::Real y,
   amrex::Real z,
   amrex::Real time
) noexcept
{
   if (bc.analytic_profile != 1) {
      return AnalyticBoundaryCopy;
   }
   if (side == 0) {
      return AnalyticBoundaryLeftState;
   }
#if (AMREX_SPACEDIM >= 2)
   if (side == 2) {
      return x < bc.shock_x0 ? AnalyticBoundaryLeftState : AnalyticBoundaryReflect;
   }
   if (side == 3) {
      return planar_shock_left_state(bc, x, y, z, time)
         ? AnalyticBoundaryLeftState
         : AnalyticBoundaryRightState;
   }
#else
   amrex::ignore_unused(x, y, z, time);
#endif
   return AnalyticBoundaryCopy;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int cell_analytic_boundary_side(
   BoundaryData const& bc,
   amrex::Box const& domain,
   int i,
   int j,
   int k
) noexcept
{
   const int analytic = static_cast<int>(BoundaryKind::Analytic);
   if (i < domain.smallEnd(0) && bc.lo_kind[0] == analytic) {
      return 0;
   }
   if (i > domain.bigEnd(0) && bc.hi_kind[0] == analytic) {
      return 1;
   }
#if (AMREX_SPACEDIM >= 2)
   if (j < domain.smallEnd(1) && bc.lo_kind[1] == analytic) {
      return 2;
   }
   if (j > domain.bigEnd(1) && bc.hi_kind[1] == analytic) {
      return 3;
   }
#else
   amrex::ignore_unused(j);
#endif
#if (AMREX_SPACEDIM == 3)
   if (k < domain.smallEnd(2) && bc.lo_kind[2] == analytic) {
      return 4;
   }
   if (k > domain.bigEnd(2) && bc.hi_kind[2] == analytic) {
      return 5;
   }
#else
   amrex::ignore_unused(k);
#endif
   return -1;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real inflow_profile_weight(BoundaryData const& bc, int boundary_dir, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
   if (bc.inflow_profile == 0) {
      return 1.0_rt;
   }
   amrex::Real r2 = 0.0_rt;
   if (boundary_dir == 0) {
#if (AMREX_SPACEDIM >= 2)
      const amrex::Real dy = y - bc.inflow_center_y;
      r2 += dy * dy;
#endif
#if (AMREX_SPACEDIM == 3)
      const amrex::Real dz = z - bc.inflow_center_z;
      r2 += dz * dz;
#endif
   } else if (boundary_dir == 1) {
      const amrex::Real dy = x - bc.inflow_center_y;
      r2 += dy * dy;
#if (AMREX_SPACEDIM == 3)
      const amrex::Real dz = z - bc.inflow_center_z;
      r2 += dz * dz;
#endif
   } else {
      const amrex::Real dy = x - bc.inflow_center_y;
      const amrex::Real dz = y - bc.inflow_center_z;
      r2 += dy * dy + dz * dz;
   }
   if (bc.inflow_profile == 2) {
      return r2 <= bc.inflow_radius * bc.inflow_radius ? 1.0_rt : 0.0_rt;
   }
   return std::exp(-0.5_rt * r2 / (bc.inflow_radius * bc.inflow_radius));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool inflow_profile_is_fixed(BoundaryData const& bc, int boundary_dir, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
   return bc.inflow_profile != 2 || inflow_profile_weight(bc, boundary_dir, x, y, z) > 0.5_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real inflow_state_component(BoundaryData const& bc, int component) noexcept
{
   return boundary_profile_state_component(
      bc,
      component,
      bc.inflow_rho,
      bc.inflow_pressure,
      bc.inflow_v1,
      bc.inflow_v2,
      bc.inflow_v3,
      bc.inflow_b1,
      bc.inflow_b2,
      bc.inflow_b3,
      bc.inflow_passive_value
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real inflow_state_component(
   BoundaryData const& bc,
   int component,
   int boundary_dir,
   amrex::Real x,
   amrex::Real y,
   amrex::Real z
) noexcept
{
   const amrex::Real w = inflow_profile_weight(bc, boundary_dir, x, y, z);
   const amrex::Real rho = bc.inflow_background_rho + w * (bc.inflow_rho - bc.inflow_background_rho);
   const amrex::Real pressure =
      bc.inflow_background_pressure + w * (bc.inflow_pressure - bc.inflow_background_pressure);
   const amrex::Real vx = bc.inflow_background_v1 + w * (bc.inflow_v1 - bc.inflow_background_v1);
   const amrex::Real vy = bc.inflow_background_v2 + w * (bc.inflow_v2 - bc.inflow_background_v2);
   const amrex::Real vz = bc.inflow_background_v3 + w * (bc.inflow_v3 - bc.inflow_background_v3);
   const amrex::Real bx = bc.inflow_background_b1 + w * (bc.inflow_b1 - bc.inflow_background_b1);
   const amrex::Real by = bc.inflow_background_b2 + w * (bc.inflow_b2 - bc.inflow_background_b2);
   const amrex::Real bz = bc.inflow_background_b3 + w * (bc.inflow_b3 - bc.inflow_background_b3);
   const amrex::Real passive_value =
      bc.inflow_background_passive_value + w * (bc.inflow_passive_value - bc.inflow_background_passive_value);
   return boundary_profile_state_component(bc, component, rho, pressure, vx, vy, vz, bx, by, bz, passive_value);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int cell_inflow_boundary_dir(BoundaryData const& bc, amrex::Box const& domain, int i, int j, int k) noexcept
{
   if ((i < domain.smallEnd(0) && boundary_kind_is_inflow(bc.lo_kind[0]))
      || (i > domain.bigEnd(0) && boundary_kind_is_inflow(bc.hi_kind[0]))) {
      return 0;
   }
#if (AMREX_SPACEDIM >= 2)
   if ((j < domain.smallEnd(1) && boundary_kind_is_inflow(bc.lo_kind[1]))
      || (j > domain.bigEnd(1) && boundary_kind_is_inflow(bc.hi_kind[1]))) {
      return 1;
   }
#else
   amrex::ignore_unused(j);
#endif
#if (AMREX_SPACEDIM == 3)
   if ((k < domain.smallEnd(2) && boundary_kind_is_inflow(bc.lo_kind[2]))
      || (k > domain.bigEnd(2) && boundary_kind_is_inflow(bc.hi_kind[2]))) {
      return 2;
   }
#else
   amrex::ignore_unused(k);
#endif
   return -1;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real inflow_face_b_component(BoundaryData const& bc, int face_dir) noexcept
{
   if (face_dir == 0) {
      return bc.inflow_b1;
   }
#if (AMREX_SPACEDIM >= 2)
   if (face_dir == 1) {
      return bc.inflow_b2;
   }
#endif
   return bc.inflow_b3;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool cell_touches_inflow_boundary(
   BoundaryData const& bc,
   amrex::Box const& domain,
   int i,
   int j,
   int k
) noexcept
{
   if ((i < domain.smallEnd(0) && boundary_kind_is_inflow(bc.lo_kind[0]))
      || (i > domain.bigEnd(0) && boundary_kind_is_inflow(bc.hi_kind[0]))) {
      return true;
   }
#if (AMREX_SPACEDIM >= 2)
   if ((j < domain.smallEnd(1) && boundary_kind_is_inflow(bc.lo_kind[1]))
      || (j > domain.bigEnd(1) && boundary_kind_is_inflow(bc.hi_kind[1]))) {
      return true;
   }
#else
   amrex::ignore_unused(j);
#endif
#if (AMREX_SPACEDIM == 3)
   if ((k < domain.smallEnd(2) && boundary_kind_is_inflow(bc.lo_kind[2]))
      || (k > domain.bigEnd(2) && boundary_kind_is_inflow(bc.hi_kind[2]))) {
      return true;
   }
#else
   amrex::ignore_unused(k);
#endif
   return false;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool face_touches_inflow_boundary(
   BoundaryData const& bc,
   amrex::Box const& cell_domain,
   int face_dir,
   int i,
   int j,
   int k
) noexcept
{
   const int xlo = cell_domain.smallEnd(0);
   const int xhi = cell_domain.bigEnd(0) + (face_dir == 0 ? 1 : 0);
   if ((i < xlo || (face_dir == 0 && i == xlo)) && boundary_kind_is_inflow(bc.lo_kind[0])) {
      return true;
   }
   if ((i > xhi || (face_dir == 0 && i == xhi)) && boundary_kind_is_inflow(bc.hi_kind[0])) {
      return true;
   }
#if (AMREX_SPACEDIM >= 2)
   const int ylo = cell_domain.smallEnd(1);
   const int yhi = cell_domain.bigEnd(1) + (face_dir == 1 ? 1 : 0);
   if ((j < ylo || (face_dir == 1 && j == ylo)) && boundary_kind_is_inflow(bc.lo_kind[1])) {
      return true;
   }
   if ((j > yhi || (face_dir == 1 && j == yhi)) && boundary_kind_is_inflow(bc.hi_kind[1])) {
      return true;
   }
#else
   amrex::ignore_unused(j);
#endif
#if (AMREX_SPACEDIM == 3)
   const int zlo = cell_domain.smallEnd(2);
   const int zhi = cell_domain.bigEnd(2) + (face_dir == 2 ? 1 : 0);
   if ((k < zlo || (face_dir == 2 && k == zlo)) && boundary_kind_is_inflow(bc.lo_kind[2])) {
      return true;
   }
   if ((k > zhi || (face_dir == 2 && k == zhi)) && boundary_kind_is_inflow(bc.hi_kind[2])) {
      return true;
   }
#else
   amrex::ignore_unused(k);
#endif
   return false;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int face_inflow_boundary_dir(
   BoundaryData const& bc,
   amrex::Box const& cell_domain,
   int face_dir,
   int i,
   int j,
   int k
) noexcept
{
   const int xlo = cell_domain.smallEnd(0);
   const int xhi = cell_domain.bigEnd(0) + (face_dir == 0 ? 1 : 0);
   if ((i < xlo || (face_dir == 0 && i == xlo)) && boundary_kind_is_inflow(bc.lo_kind[0])) {
      return 0;
   }
   if ((i > xhi || (face_dir == 0 && i == xhi)) && boundary_kind_is_inflow(bc.hi_kind[0])) {
      return 0;
   }
#if (AMREX_SPACEDIM >= 2)
   const int ylo = cell_domain.smallEnd(1);
   const int yhi = cell_domain.bigEnd(1) + (face_dir == 1 ? 1 : 0);
   if ((j < ylo || (face_dir == 1 && j == ylo)) && boundary_kind_is_inflow(bc.lo_kind[1])) {
      return 1;
   }
   if ((j > yhi || (face_dir == 1 && j == yhi)) && boundary_kind_is_inflow(bc.hi_kind[1])) {
      return 1;
   }
#else
   amrex::ignore_unused(j);
#endif
#if (AMREX_SPACEDIM == 3)
   const int zlo = cell_domain.smallEnd(2);
   const int zhi = cell_domain.bigEnd(2) + (face_dir == 2 ? 1 : 0);
   if ((k < zlo || (face_dir == 2 && k == zlo)) && boundary_kind_is_inflow(bc.lo_kind[2])) {
      return 2;
   }
   if ((k > zhi || (face_dir == 2 && k == zhi)) && boundary_kind_is_inflow(bc.hi_kind[2])) {
      return 2;
   }
#else
   amrex::ignore_unused(k);
#endif
   return -1;
}

struct StateFillPatchBoundary {
   BoundaryData bc;

   AMREX_GPU_DEVICE
   void operator()(
      amrex::IntVect const& iv,
      amrex::Array4<amrex::Real> const& dest,
      int dcomp,
      int numcomp,
      amrex::GeometryData const& geom,
      amrex::Real time,
      amrex::BCRec const*,
      int,
      int orig_comp
   ) const noexcept
   {
      const amrex::Box& domain = geom.Domain();
      const int i = iv[0];
#if (AMREX_SPACEDIM >= 2)
      const int j = iv[1];
#else
      const int j = 0;
#endif
#if (AMREX_SPACEDIM == 3)
      const int k = iv[2];
#else
      const int k = 0;
#endif
      const int boundary_dir = cell_inflow_boundary_dir(bc, domain, i, j, k);
      const int analytic_side = cell_analytic_boundary_side(bc, domain, i, j, k);
      if (boundary_dir < 0 && analytic_side < 0) {
         return;
      }
      const amrex::Real x = geom.ProbLo(0) + (static_cast<amrex::Real>(i) + 0.5_rt) * geom.CellSize(0);
#if (AMREX_SPACEDIM >= 2)
      const amrex::Real y = geom.ProbLo(1) + (static_cast<amrex::Real>(j) + 0.5_rt) * geom.CellSize(1);
#else
      const amrex::Real y = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
      const amrex::Real z = geom.ProbLo(2) + (static_cast<amrex::Real>(k) + 0.5_rt) * geom.CellSize(2);
#else
      const amrex::Real z = 0.0_rt;
#endif
      if (analytic_side >= 0) {
         const int action = analytic_boundary_action(bc, analytic_side, x, y, z, time);
         const int normal_dir = analytic_side / 2;
         int ii = clamp_index(i, domain.smallEnd(0), domain.bigEnd(0));
#if (AMREX_SPACEDIM >= 2)
         int jj = clamp_index(j, domain.smallEnd(1), domain.bigEnd(1));
#else
         const int jj = 0;
#endif
#if (AMREX_SPACEDIM == 3)
         int kk = clamp_index(k, domain.smallEnd(2), domain.bigEnd(2));
#else
         const int kk = 0;
#endif
         if (action == AnalyticBoundaryReflect) {
            if (normal_dir == 0) {
               ii = mirrored_index(i, domain.smallEnd(0), domain.bigEnd(0));
            }
#if (AMREX_SPACEDIM >= 2)
            if (normal_dir == 1) {
               jj = mirrored_index(j, domain.smallEnd(1), domain.bigEnd(1));
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (normal_dir == 2) {
               kk = mirrored_index(k, domain.smallEnd(2), domain.bigEnd(2));
            }
#endif
         }
         for (int n = dcomp; n < dcomp + numcomp; ++n) {
            const int component = orig_comp + n - dcomp;
            if (action == AnalyticBoundaryLeftState || action == AnalyticBoundaryRightState) {
               dest(i, j, k, n) = analytic_shock_state_component(
                  bc, component, action == AnalyticBoundaryLeftState
               );
            } else {
               amrex::Real value = dest(ii, jj, kk, n);
               if (action == AnalyticBoundaryReflect && component == MOM1 + normal_dir) {
                  value = -value;
               }
               dest(i, j, k, n) = value;
            }
         }
         return;
      }
      for (int n = dcomp; n < dcomp + numcomp; ++n) {
         const int component = orig_comp + n - dcomp;
         dest(i, j, k, n) = inflow_state_component(bc, component, boundary_dir, x, y, z);
      }
   }
};

struct FaceBFillPatchBoundary {
   BoundaryData bc;
   int face_dir = 0;

   AMREX_GPU_DEVICE
   void operator()(
      amrex::IntVect const& iv,
      amrex::Array4<amrex::Real> const& dest,
      int dcomp,
      int numcomp,
      amrex::GeometryData const& geom,
      amrex::Real,
      amrex::BCRec const*,
      int,
      int
   ) const noexcept
   {
      const amrex::Box& cell_domain = geom.Domain();
      const int i = iv[0];
#if (AMREX_SPACEDIM >= 2)
      const int j = iv[1];
#else
      const int j = 0;
#endif
#if (AMREX_SPACEDIM == 3)
      const int k = iv[2];
#else
      const int k = 0;
#endif
      if (!face_touches_inflow_boundary(bc, cell_domain, face_dir, i, j, k)) {
         return;
      }
      const amrex::Real value = inflow_face_b_component(bc, face_dir);
      for (int n = dcomp; n < dcomp + numcomp; ++n) {
         dest(i, j, k, n) = value;
      }
   }
};

template <int NComp>
inline void apply_state_boundary(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   BoundaryData const& bc,
   amrex::Real time = 0.0_rt
)
{
   state.FillBoundary(geom.periodicity());
   validate_origin_boundary_geometry(bc, geom);
   if (boundary_data_all_periodic(bc)) {
      return;
   }

   const amrex::Box domain = geom.Domain();
   const auto lo = amrex::lbound(domain);
   const auto hi = amrex::ubound(domain);
   const auto is_periodic = geom.isPeriodicArray();
   const int ng = state.nGrow();
   const int xlo_bc = bc.lo_kind[0];
   const int xhi_bc = bc.hi_kind[0];
#if (AMREX_SPACEDIM >= 2)
   const int ylo_bc = bc.lo_kind[1];
   const int yhi_bc = bc.hi_kind[1];
#endif
#if (AMREX_SPACEDIM == 3)
   const int zlo_bc = bc.lo_kind[2];
   const int zhi_bc = bc.hi_kind[2];
#endif
   const int reflecting_bc = static_cast<int>(BoundaryKind::Reflecting);
   const int symmetry_bc = static_cast<int>(BoundaryKind::Symmetry);
   const int inflow_bc = static_cast<int>(BoundaryKind::Inflow);
   const int analytic_bc = static_cast<int>(BoundaryKind::Analytic);
   const int diode_bc = static_cast<int>(BoundaryKind::Diode);
   const int origin_bc = static_cast<int>(BoundaryKind::Origin);
   const int nonlocal_origin_dir = needs_radial_antipodal_origin_boundary(bc)
      ? mesh_direction_for_axis(bc.mesh_axes, MeshAxisR) : -1;
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	      amrex::Box grown = mfi.growntilebox(ng);
	      grown &= state[mfi].box();
	      amrex::BoxList ghost_boxes = amrex::boxDiff(grown, domain);
	      const auto s = state.array(mfi);
	      for (amrex::Box const& bx : ghost_boxes) {
	      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         bool outside = !is_periodic[0] && (i < lo.x || i > hi.x);
#if (AMREX_SPACEDIM >= 2)
         outside = outside || (!is_periodic[1] && (j < lo.y || j > hi.y));
#endif
#if (AMREX_SPACEDIM == 3)
         outside = outside || (!is_periodic[2] && (k < lo.z || k > hi.z));
#endif
         if (!outside) {
            return;
         }

         // The nonlocal owner completes these ghosts. On a partial AMR level,
         // preserve the hierarchy-filled values between those completions (for
         // example the fixed boundary data of an implicit diffusion iteration).
         if ((nonlocal_origin_dir == 0 && i < lo.x)
#if AMREX_SPACEDIM >= 2
             || (nonlocal_origin_dir == 1 && j < lo.y)
#endif
#if AMREX_SPACEDIM == 3
             || (nonlocal_origin_dir == 2 && k < lo.z)
#endif
         ) {
            return;
         }

         const bool xlo = !is_periodic[0] && i < lo.x;
         const bool xhi = !is_periodic[0] && i > hi.x;
#if (AMREX_SPACEDIM >= 2)
         const bool ylo = !is_periodic[1] && j < lo.y;
         const bool yhi = !is_periodic[1] && j > hi.y;
#else
         const bool ylo = false;
         const bool yhi = false;
#endif
#if (AMREX_SPACEDIM == 3)
         const bool zlo = !is_periodic[2] && k < lo.z;
         const bool zhi = !is_periodic[2] && k > hi.z;
#else
         const bool zlo = false;
         const bool zhi = false;
#endif

         const bool x_origin = xlo && xlo_bc == origin_bc;
         const bool x_reflect = (xlo && (xlo_bc == reflecting_bc || xlo_bc == symmetry_bc || xlo_bc == origin_bc))
            || (xhi && (xhi_bc == reflecting_bc || xhi_bc == symmetry_bc));
         const bool x_symmetry = (xlo && xlo_bc == symmetry_bc) || (xhi && xhi_bc == symmetry_bc);
#if (AMREX_SPACEDIM >= 2)
         const bool y_origin = ylo && ylo_bc == origin_bc;
         const bool y_reflect = (ylo && (ylo_bc == reflecting_bc || ylo_bc == symmetry_bc || ylo_bc == origin_bc))
            || (yhi && (yhi_bc == reflecting_bc || yhi_bc == symmetry_bc));
         const bool y_symmetry = (ylo && ylo_bc == symmetry_bc) || (yhi && yhi_bc == symmetry_bc);
#else
         const bool y_origin = false;
         const bool y_reflect = false;
         const bool y_symmetry = false;
#endif
#if (AMREX_SPACEDIM == 3)
         const bool z_origin = zlo && zlo_bc == origin_bc;
         const bool z_reflect = (zlo && (zlo_bc == reflecting_bc || zlo_bc == symmetry_bc || zlo_bc == origin_bc))
            || (zhi && (zhi_bc == reflecting_bc || zhi_bc == symmetry_bc));
         const bool z_symmetry = (zlo && zlo_bc == symmetry_bc) || (zhi && zhi_bc == symmetry_bc);
#else
         const bool z_origin = false;
         const bool z_reflect = false;
         const bool z_symmetry = false;
#endif

         const bool touches_inflow = (xlo && xlo_bc == inflow_bc) || (xhi && xhi_bc == inflow_bc)
#if (AMREX_SPACEDIM >= 2)
            || (ylo && ylo_bc == inflow_bc) || (yhi && yhi_bc == inflow_bc)
#endif
#if (AMREX_SPACEDIM == 3)
            || (zlo && zlo_bc == inflow_bc) || (zhi && zhi_bc == inflow_bc)
#endif
            ;

         int analytic_side = -1;
         if (xlo && xlo_bc == analytic_bc) {
            analytic_side = 0;
         } else if (xhi && xhi_bc == analytic_bc) {
            analytic_side = 1;
         }
#if (AMREX_SPACEDIM >= 2)
         else if (ylo && ylo_bc == analytic_bc) {
            analytic_side = 2;
         } else if (yhi && yhi_bc == analytic_bc) {
            analytic_side = 3;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         else if (zlo && zlo_bc == analytic_bc) {
            analytic_side = 4;
         } else if (zhi && zhi_bc == analytic_bc) {
            analytic_side = 5;
         }
#endif

         int ii = is_periodic[0] ? i : (x_reflect ? mirrored_index(i, lo.x, hi.x) : clamp_index(i, lo.x, hi.x));
#if (AMREX_SPACEDIM >= 2)
         int jj = is_periodic[1] ? j : (y_reflect ? mirrored_index(j, lo.y, hi.y) : clamp_index(j, lo.y, hi.y));
#else
         const int jj = 0;
#endif
#if (AMREX_SPACEDIM == 3)
         int kk = is_periodic[2] ? k : (z_reflect ? mirrored_index(k, lo.z, hi.z) : clamp_index(k, lo.z, hi.z));
#else
         const int kk = 0;
#endif
         bool fixed_inflow = false;
         if (touches_inflow) {
            int boundary_dir = xlo || xhi ? 0 : -1;
#if (AMREX_SPACEDIM >= 2)
            if (boundary_dir < 0 && (ylo || yhi)) {
               boundary_dir = 1;
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (boundary_dir < 0 && (zlo || zhi)) {
               boundary_dir = 2;
            }
#endif
            const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5_rt) * dx[0];
#if (AMREX_SPACEDIM >= 2)
            const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5_rt) * dx[1];
#else
            const amrex::Real y = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real z = problo[2] + (static_cast<amrex::Real>(k) + 0.5_rt) * dx[2];
#else
            const amrex::Real z = 0.0_rt;
#endif
            fixed_inflow = inflow_profile_is_fixed(bc, boundary_dir, x, y, z);
            if (fixed_inflow) {
               for (int q = 0; q < NComp; ++q) {
                  s(i, j, k, q) = inflow_state_component(bc, q, boundary_dir, x, y, z);
               }
            }
         }
         bool fixed_analytic = false;
         int analytic_action = AnalyticBoundaryCopy;
         int analytic_normal_dir = -1;
         if (analytic_side >= 0) {
            const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5_rt) * dx[0];
#if (AMREX_SPACEDIM >= 2)
            const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5_rt) * dx[1];
#else
            const amrex::Real y = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real z = problo[2] + (static_cast<amrex::Real>(k) + 0.5_rt) * dx[2];
#else
            const amrex::Real z = 0.0_rt;
#endif
            analytic_action = analytic_boundary_action(bc, analytic_side, x, y, z, time);
            analytic_normal_dir = analytic_side / 2;
            if (analytic_action == AnalyticBoundaryLeftState || analytic_action == AnalyticBoundaryRightState) {
               for (int q = 0; q < NComp; ++q) {
                  s(i, j, k, q) = analytic_shock_state_component(
                     bc, q, analytic_action == AnalyticBoundaryLeftState
                  );
               }
               fixed_analytic = true;
            } else if (analytic_action == AnalyticBoundaryReflect) {
               if (analytic_normal_dir == 0) {
                  ii = mirrored_index(i, lo.x, hi.x);
               }
#if (AMREX_SPACEDIM >= 2)
               if (analytic_normal_dir == 1) {
                  jj = mirrored_index(j, lo.y, hi.y);
               }
#endif
#if (AMREX_SPACEDIM == 3)
               if (analytic_normal_dir == 2) {
                  kk = mirrored_index(k, lo.z, hi.z);
               }
#endif
            }
         }
         if (!fixed_inflow && !fixed_analytic) {
            for (int q = 0; q < NComp; ++q) {
               s(i, j, k, q) = s(ii, jj, kk, q);
            }
         }
         if (analytic_action == AnalyticBoundaryReflect && analytic_normal_dir >= 0) {
            s(i, j, k, MOM1 + analytic_normal_dir) = -s(i, j, k, MOM1 + analytic_normal_dir);
         }
         if (x_origin) {
            for (int q = 0; q < NComp; ++q) {
               s(i, j, k, q) *= origin_state_parity(bc, 0, q);
            }
         } else if (x_reflect) {
            s(i, j, k, MOM1) = -s(i, j, k, MOM1);
         }
#if (AMREX_SPACEDIM >= 2)
         if (y_origin) {
            for (int q = 0; q < NComp; ++q) {
               s(i, j, k, q) *= origin_state_parity(bc, 1, q);
            }
         } else if (y_reflect) {
            s(i, j, k, MOM2) = -s(i, j, k, MOM2);
         }
#endif
#if (AMREX_SPACEDIM == 3)
         if (z_origin) {
            for (int q = 0; q < NComp; ++q) {
               s(i, j, k, q) *= origin_state_parity(bc, 2, q);
            }
         } else if (z_reflect) {
            s(i, j, k, MOM3) = -s(i, j, k, MOM3);
         }
#endif
         if constexpr (has_mhd_components<NComp>()) {
            if (x_symmetry) {
               s(i, j, k, B2) = -s(i, j, k, B2);
               s(i, j, k, B3) = -s(i, j, k, B3);
            }
#if (AMREX_SPACEDIM >= 2)
            if (y_symmetry) {
               s(i, j, k, B1) = -s(i, j, k, B1);
               s(i, j, k, B3) = -s(i, j, k, B3);
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (z_symmetry) {
               s(i, j, k, B1) = -s(i, j, k, B1);
               s(i, j, k, B2) = -s(i, j, k, B2);
            }
#endif
         }
         if ((xlo && xlo_bc == diode_bc && s(i, j, k, MOM1) > 0.0_rt)
            || (xhi && xhi_bc == diode_bc && s(i, j, k, MOM1) < 0.0_rt)) {
            s(i, j, k, MOM1) = 0.0_rt;
         }
#if (AMREX_SPACEDIM >= 2)
         if ((ylo && ylo_bc == diode_bc && s(i, j, k, MOM2) > 0.0_rt)
            || (yhi && yhi_bc == diode_bc && s(i, j, k, MOM2) < 0.0_rt)) {
            s(i, j, k, MOM2) = 0.0_rt;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         if ((zlo && zlo_bc == diode_bc && s(i, j, k, MOM3) > 0.0_rt)
            || (zhi && zhi_bc == diode_bc && s(i, j, k, MOM3) < 0.0_rt)) {
            s(i, j, k, MOM3) = 0.0_rt;
         }
#endif
	      });
	      }
	   }
	   }
}


// The hierarchy owner supplies this same bounded donor band through FillPatchNLevels.
amrex::Box radial_origin_source_box(amrex::Geometry const& geom, BoundaryData const& bc,
                                    amrex::IntVect const& num_ghosts);
void fill_radial_origin_from_source(amrex::MultiFab& state, amrex::MultiFab const& source_values,
                                   amrex::Geometry const& geom, BoundaryData const& bc);
void apply_radial_antipodal_origin_boundary(amrex::MultiFab& state, amrex::Geometry const& geom,
                                            BoundaryData const& bc);

template <int NComp>
inline void apply_state_boundary(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
   const BoundaryData bc = make_boundary_data(cfg);
   apply_state_boundary<NComp>(state, geom, bc);
   apply_radial_antipodal_origin_boundary(state, geom, bc);
}

inline void apply_hydro_boundary(amrex::MultiFab& state, amrex::Geometry const& geom, BoundaryData const& bc)
{
   apply_state_boundary<NVAR_HYDRO>(state, geom, bc);
   apply_radial_antipodal_origin_boundary(state, geom, bc);
}

inline void apply_mhd_boundary(amrex::MultiFab& state, amrex::Geometry const& geom, BoundaryData const& bc)
{
   apply_state_boundary<NVAR_MHD>(state, geom, bc);
   apply_radial_antipodal_origin_boundary(state, geom, bc);
}

inline void apply_hydro_boundary(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   apply_state_boundary<NVAR_HYDRO>(state, geom, cfg);
}

inline void apply_mhd_boundary(amrex::MultiFab& state, amrex::Geometry const& geom, RunConfig const& cfg)
{
   apply_state_boundary<NVAR_MHD>(state, geom, cfg);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool patch_union_fab_contains(
   amrex::Dim3 const& lo,
   amrex::Dim3 const& hi,
   int i,
   int j,
   int k
) noexcept
{
   bool contains = i >= lo.x && i <= hi.x;
#if (AMREX_SPACEDIM >= 2)
   contains = contains && j >= lo.y && j <= hi.y;
#endif
#if (AMREX_SPACEDIM == 3)
   contains = contains && k >= lo.z && k <= hi.z;
#endif
   return contains;
}

inline void apply_patch_union_reflecting_boundary(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   PatchDomainBoxes const& patch_domain
)
{
   const int ng = state.nGrow();
   if (ng <= 0) {
      return;
   }

   amrex::iMultiFab valid_mask(state.boxArray(), state.DistributionMap(), 1, state.nGrowVect());
   valid_mask.setVal(0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(valid_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto valid = valid_mask.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         valid(i, j, k, 0) = 1;
      });
   }
   }
   valid_mask.FillBoundary(geom.periodicity());

   const amrex::Box domain = geom.Domain();
   const auto domain_lo = amrex::lbound(domain);
   const auto domain_hi = amrex::ubound(domain);
   const int ncomp = state.nComp();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      amrex::Box grown = mfi.growntilebox(ng);
      grown &= state[mfi].box();
      if (!grown.ok()) {
         continue;
      }
      const auto s = state.array(mfi);
      const auto valid = valid_mask.const_array(mfi);
      const auto fab_lo = amrex::lbound(state[mfi].box());
      const auto fab_hi = amrex::ubound(state[mfi].box());
      amrex::ParallelFor(grown, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (
            !domain.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))
            || patch_domain_contains_cell(patch_domain, i, j, k)
         ) {
            return;
         }

         int reflected_axis = -1;
         int src_i = i;
         int src_j = j;
         int src_k = k;
         for (int d = 1; d <= ng && reflected_axis < 0; ++d) {
            const int candidate_i = i - d;
            if (
               patch_union_fab_contains(fab_lo, fab_hi, candidate_i, j, k)
               && valid(candidate_i, j, k, 0) != 0
               && patch_domain_contains_cell(patch_domain, candidate_i, j, k)
            ) {
               reflected_axis = 0;
               src_i = i - (2 * d - 1);
               if (
                  !patch_union_fab_contains(fab_lo, fab_hi, src_i, j, k)
                  || !patch_domain_contains_cell(patch_domain, src_i, j, k)
                  || valid(src_i, j, k, 0) == 0
               ) {
                  src_i = candidate_i;
               }
            }
            const int candidate_hi_i = i + d;
            if (
               reflected_axis < 0
               && patch_union_fab_contains(fab_lo, fab_hi, candidate_hi_i, j, k)
               && valid(candidate_hi_i, j, k, 0) != 0
               && patch_domain_contains_cell(patch_domain, candidate_hi_i, j, k)
            ) {
               reflected_axis = 0;
               src_i = i + (2 * d - 1);
               if (
                  !patch_union_fab_contains(fab_lo, fab_hi, src_i, j, k)
                  || !patch_domain_contains_cell(patch_domain, src_i, j, k)
                  || valid(src_i, j, k, 0) == 0
               ) {
                  src_i = candidate_hi_i;
               }
            }
         }
#if (AMREX_SPACEDIM >= 2)
         for (int d = 1; d <= ng && reflected_axis < 0; ++d) {
            const int candidate_j = j - d;
            if (
               patch_union_fab_contains(fab_lo, fab_hi, i, candidate_j, k)
               && valid(i, candidate_j, k, 0) != 0
               && patch_domain_contains_cell(patch_domain, i, candidate_j, k)
            ) {
               reflected_axis = 1;
               src_j = j - (2 * d - 1);
               if (
                  !patch_union_fab_contains(fab_lo, fab_hi, i, src_j, k)
                  || !patch_domain_contains_cell(patch_domain, i, src_j, k)
                  || valid(i, src_j, k, 0) == 0
               ) {
                  src_j = candidate_j;
               }
            }
            const int candidate_hi_j = j + d;
            if (
               reflected_axis < 0
               && patch_union_fab_contains(fab_lo, fab_hi, i, candidate_hi_j, k)
               && valid(i, candidate_hi_j, k, 0) != 0
               && patch_domain_contains_cell(patch_domain, i, candidate_hi_j, k)
            ) {
               reflected_axis = 1;
               src_j = j + (2 * d - 1);
               if (
                  !patch_union_fab_contains(fab_lo, fab_hi, i, src_j, k)
                  || !patch_domain_contains_cell(patch_domain, i, src_j, k)
                  || valid(i, src_j, k, 0) == 0
               ) {
                  src_j = candidate_hi_j;
               }
            }
         }
#endif
#if (AMREX_SPACEDIM == 3)
         for (int d = 1; d <= ng && reflected_axis < 0; ++d) {
            const int candidate_k = k - d;
            if (
               patch_union_fab_contains(fab_lo, fab_hi, i, j, candidate_k)
               && valid(i, j, candidate_k, 0) != 0
               && patch_domain_contains_cell(patch_domain, i, j, candidate_k)
            ) {
               reflected_axis = 2;
               src_k = k - (2 * d - 1);
               if (
                  !patch_union_fab_contains(fab_lo, fab_hi, i, j, src_k)
                  || !patch_domain_contains_cell(patch_domain, i, j, src_k)
                  || valid(i, j, src_k, 0) == 0
               ) {
                  src_k = candidate_k;
               }
            }
            const int candidate_hi_k = k + d;
            if (
               reflected_axis < 0
               && patch_union_fab_contains(fab_lo, fab_hi, i, j, candidate_hi_k)
               && valid(i, j, candidate_hi_k, 0) != 0
               && patch_domain_contains_cell(patch_domain, i, j, candidate_hi_k)
            ) {
               reflected_axis = 2;
               src_k = k + (2 * d - 1);
               if (
                  !patch_union_fab_contains(fab_lo, fab_hi, i, j, src_k)
                  || !patch_domain_contains_cell(patch_domain, i, j, src_k)
                  || valid(i, j, src_k, 0) == 0
               ) {
                  src_k = candidate_hi_k;
               }
            }
         }
#endif
         if (reflected_axis < 0) {
            return;
         }
         for (int q = 0; q < ncomp; ++q) {
            s(i, j, k, q) = s(src_i, src_j, src_k, q);
         }
         if (reflected_axis == 0) {
            s(i, j, k, MOM1) = -s(i, j, k, MOM1);
         }
#if (AMREX_SPACEDIM >= 2)
         if (reflected_axis == 1) {
            s(i, j, k, MOM2) = -s(i, j, k, MOM2);
         }
#endif
#if (AMREX_SPACEDIM == 3)
         if (reflected_axis == 2) {
            s(i, j, k, MOM3) = -s(i, j, k, MOM3);
         }
#endif
      });
   }
   }
}

inline void apply_configured_state_boundary(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   BoundaryData const& bc,
   amrex::Real time = 0.0_rt
)
{
   const int ncomp = state.nComp();
   if (bc.mhd != 0) {
      if (ncomp == NVAR_MHD + 1) {
         apply_state_boundary<NVAR_MHD + 1>(state, geom, bc, time);
      } else if (ncomp == NVAR_MHD) {
         apply_state_boundary<NVAR_MHD>(state, geom, bc, time);
      } else {
         amrex::Abort("MHD state component count does not match pyfluid.passive_scalars");
      }
   } else {
      if (ncomp == NVAR_HYDRO + 1) {
         apply_state_boundary<NVAR_HYDRO + 1>(state, geom, bc, time);
      } else if (ncomp == NVAR_HYDRO) {
         apply_state_boundary<NVAR_HYDRO>(state, geom, bc, time);
      } else {
         amrex::Abort("hydro state component count does not match pyfluid.passive_scalars");
      }
   }
   apply_radial_antipodal_origin_boundary(state, geom, bc);
}

inline void apply_configured_state_boundary(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Real time = 0.0_rt
)
{
   apply_configured_state_boundary(state, geom, make_boundary_data(cfg), time);
}

} // namespace pyfluid_amrex
