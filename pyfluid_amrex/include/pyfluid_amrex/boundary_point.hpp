#pragma once

#include <AMReX_BCRec.H>
#include <AMReX_Geometry.H>

#include <cmath>

#include "pyfluid_amrex/boundary_data.hpp"
#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

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

} // namespace pyfluid_amrex
