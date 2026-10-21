#pragma once

#include <AMReX_Array.H>

#include "pyfluid_amrex/boundary_kind.hpp"
#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct RunConfig;

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

BoundaryData make_boundary_data(RunConfig const& cfg);

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

} // namespace pyfluid_amrex
