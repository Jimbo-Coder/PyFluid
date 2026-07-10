#pragma once

#include <AMReX_Array4.H>
#include <AMReX_Algorithm.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/kernels_generated/ideal_mhd_flux_cartesian.hpp"
#include "pyfluid_amrex/kernels_generated/riemann_ops.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/patch_metrics.hpp"
#include "pyfluid_amrex/reconstruction.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

constexpr amrex::Real MAGNETIC_EPS = 1.0e-14;

struct WaveSpeeds {
   amrex::Real sl = 0.0;
   amrex::Real sr = 0.0;
};

struct MHDNormalState {
   amrex::Real rho = 0.0;
   amrex::Real momn = 0.0;
   amrex::Real momt1 = 0.0;
   amrex::Real momt2 = 0.0;
   amrex::Real energy = 0.0;
   amrex::Real bn = 0.0;
   amrex::Real bt1 = 0.0;
   amrex::Real bt2 = 0.0;
};

struct MHDNormalPrimitive {
   amrex::Real rho = 0.0;
   amrex::Real vn = 0.0;
   amrex::Real vt1 = 0.0;
   amrex::Real vt2 = 0.0;
   amrex::Real pressure = 0.0;
   amrex::Real energy = 0.0;
   amrex::Real bn = 0.0;
   amrex::Real bt1 = 0.0;
   amrex::Real bt2 = 0.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int momentum_component(int direction) noexcept
{
   return MOM1 + direction;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real velocity_component(
   HydroState const& state,
   int direction
) noexcept
{
   if (direction == 0) {
      return state.mom1 / state.rho;
   }
   if (direction == 1) {
      return state.mom2 / state.rho;
   }
   return state.mom3 / state.rho;
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real velocity_component(HydroState const& state) noexcept
{
   if constexpr (Direction == 0) {
      return state.mom1 / state.rho;
   } else if constexpr (Direction == 1) {
      return state.mom2 / state.rho;
   } else {
      return state.mom3 / state.rho;
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hydro_flux(
   HydroState const& state,
   amrex::Real pressure,
   int direction
) noexcept
{
   if (direction == 0) {
      return generated::hydro_flux_orthonormal<0>(state, pressure);
   }
   if (direction == 1) {
      return generated::hydro_flux_orthonormal<1>(state, pressure);
   }
   return generated::hydro_flux_orthonormal<2>(state, pressure);
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hydro_flux(
   HydroState const& state,
   amrex::Real pressure
) noexcept
{
   return generated::hydro_flux_orthonormal<Direction>(state, pressure);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hydro_primitive_normal_velocity(HydroPrimitive const& primitive, PhysicalVector normal) noexcept
{
   const StateVector3 velocity{primitive.v1, primitive.v2, primitive.v3};
   return velocity.dot(normal);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hydro_state_normal_velocity(HydroState const& state, PhysicalVector normal) noexcept
{
   const StateVector3 momentum{state.mom1, state.mom2, state.mom3};
   return momentum.dot(normal) / state.rho;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hydro_flux_normal(
   HydroState const& state,
   amrex::Real pressure,
   PhysicalVector normal
) noexcept
{
   const amrex::Real vn = hydro_state_normal_velocity(state, normal);
   const StateVector3 normal_vector = {normal.x, normal.y, normal.z};
   const StateVector3 momentum_flux = StateVector3{state.mom1, state.mom2, state.mom3} * vn
      + normal_vector * pressure;
   return {
      state.rho * vn,
      momentum_flux.x,
      momentum_flux.y,
      momentum_flux.z,
      (state.energy + pressure) * vn,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hllc_star_state_normal(
   HydroState const& state,
   amrex::Real pressure,
   PhysicalVector normal,
   amrex::Real s,
   amrex::Real s_star
) noexcept
{
   const amrex::Real vn = hydro_state_normal_velocity(state, normal);
   const StateVector3 normal_vector = {normal.x, normal.y, normal.z};
   const StateVector3 base_velocity = StateVector3{state.mom1, state.mom2, state.mom3} / state.rho;
   const amrex::Real dvn = s_star - vn;
   const StateVector3 normal_velocity = base_velocity + normal_vector * dvn;
   const amrex::Real rho_star = state.rho * (s - vn) / (s - s_star);
   const amrex::Real p_star = pressure + state.rho * (s - vn) * (s_star - vn);
   return {
      rho_star,
      rho_star * normal_velocity.x,
      rho_star * normal_velocity.y,
      rho_star * normal_velocity.z,
      ((s - vn) * state.energy - pressure * vn + p_star * s_star) / (s - s_star),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hllc_star_state(
   HydroState const& state,
   amrex::Real pressure,
   int direction,
   amrex::Real s,
   amrex::Real s_star
) noexcept
{
   const amrex::Real vn = velocity_component(state, direction);
   const amrex::Real rho_star = state.rho * (s - vn) / (s - s_star);
   const amrex::Real p_star = pressure + state.rho * (s - vn) * (s_star - vn);
   HydroState star{
      rho_star,
      rho_star * state.mom1 / state.rho,
      rho_star * state.mom2 / state.rho,
      rho_star * state.mom3 / state.rho,
      ((s - vn) * state.energy - pressure * vn + p_star * s_star) / (s - s_star),
   };
   if (direction == 0) {
      star.mom1 = rho_star * s_star;
   } else if (direction == 1) {
      star.mom2 = rho_star * s_star;
   } else {
      star.mom3 = rho_star * s_star;
   }
   return star;
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hllc_star_state(
   HydroState const& state,
   amrex::Real pressure,
   amrex::Real s,
   amrex::Real s_star
) noexcept
{
   const amrex::Real vn = velocity_component<Direction>(state);
   const amrex::Real rho_star = state.rho * (s - vn) / (s - s_star);
   const amrex::Real p_star = pressure + state.rho * (s - vn) * (s_star - vn);
   HydroState star{
      rho_star,
      rho_star * state.mom1 / state.rho,
      rho_star * state.mom2 / state.rho,
      rho_star * state.mom3 / state.rho,
      ((s - vn) * state.energy - pressure * vn + p_star * s_star) / (s - s_star),
   };
   if constexpr (Direction == 0) {
      star.mom1 = rho_star * s_star;
   } else if constexpr (Direction == 1) {
      star.mom2 = rho_star * s_star;
   } else {
      star.mom3 = rho_star * s_star;
   }
   return star;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState add_wave_correction(
   HydroState const& flux,
   amrex::Real s,
   HydroState const& star,
   HydroState const& state
) noexcept
{
   return {
      flux.rho + s * (star.rho - state.rho),
      flux.mom1 + s * (star.mom1 - state.mom1),
      flux.mom2 + s * (star.mom2 - state.mom2),
      flux.mom3 + s * (star.mom3 - state.mom3),
      flux.energy + s * (star.energy - state.energy),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool hydro_state_is_finite(HydroState const& state) noexcept
{
   return amrex::Math::isfinite(state.rho)
      && amrex::Math::isfinite(state.mom1)
      && amrex::Math::isfinite(state.mom2)
      && amrex::Math::isfinite(state.mom3)
      && amrex::Math::isfinite(state.energy);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDState mhd_flux(MHDState const& state, amrex::Real pressure, int direction) noexcept
{
   if (direction == 0) {
      return generated::mhd_flux_orthonormal<0>(state, pressure);
   }
   if (direction == 1) {
      return generated::mhd_flux_orthonormal<1>(state, pressure);
   }
   return generated::mhd_flux_orthonormal<2>(state, pressure);
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDState from_normal_flux(MHDNormalState const& flux) noexcept
{
   if constexpr (Direction == 0) {
      return {flux.rho, flux.momn, flux.momt1, flux.momt2, flux.energy, flux.bn, flux.bt1, flux.bt2};
   } else if constexpr (Direction == 1) {
      return {flux.rho, flux.momt1, flux.momn, flux.momt2, flux.energy, flux.bt1, flux.bn, flux.bt2};
   } else {
      return {flux.rho, flux.momt1, flux.momt2, flux.momn, flux.energy, flux.bt1, flux.bt2, flux.bn};
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDNormalState flux_normal_mhd_primitive(MHDNormalPrimitive const& primitive) noexcept
{
   const amrex::Real magnetic = 0.5 * (
      primitive.bn * primitive.bn
      + primitive.bt1 * primitive.bt1
      + primitive.bt2 * primitive.bt2
   );
   const amrex::Real total_pressure = primitive.pressure + magnetic;
   const amrex::Real vdotb = (
      primitive.vn * primitive.bn
      + primitive.vt1 * primitive.bt1
      + primitive.vt2 * primitive.bt2
   );
   return {
      primitive.rho * primitive.vn,
      primitive.rho * primitive.vn * primitive.vn + total_pressure - primitive.bn * primitive.bn,
      primitive.rho * primitive.vt1 * primitive.vn - primitive.bn * primitive.bt1,
      primitive.rho * primitive.vt2 * primitive.vn - primitive.bn * primitive.bt2,
      (primitive.energy + total_pressure) * primitive.vn - primitive.bn * vdotb,
      0.0,
      primitive.bt1 * primitive.vn - primitive.bn * primitive.vt1,
      primitive.bt2 * primitive.vn - primitive.bn * primitive.vt2,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDNormalPrimitive hlld_star_state(
   MHDNormalPrimitive const& primitive,
   amrex::Real wave_speed,
   amrex::Real contact_speed,
   amrex::Real total_pressure_star
) noexcept
{
   const amrex::Real rho_wave = primitive.rho * (wave_speed - primitive.vn);
   const amrex::Real rho_star = rho_wave / (wave_speed - contact_speed);
   const amrex::Real denom = (
      primitive.rho * (wave_speed - primitive.vn) * (wave_speed - contact_speed)
      - primitive.bn * primitive.bn
   );
   amrex::Real vt1_star = primitive.vt1;
   amrex::Real vt2_star = primitive.vt2;
   amrex::Real bt1_star = primitive.bt1;
   amrex::Real bt2_star = primitive.bt2;
   if (amrex::Math::abs(denom) > MAGNETIC_EPS) {
      const amrex::Real factor = primitive.bn * (contact_speed - primitive.vn) / denom;
      vt1_star = primitive.vt1 - primitive.bt1 * factor;
      vt2_star = primitive.vt2 - primitive.bt2 * factor;
      const amrex::Real bfactor = (
         primitive.rho * (wave_speed - primitive.vn) * (wave_speed - primitive.vn)
         - primitive.bn * primitive.bn
      ) / denom;
      bt1_star = primitive.bt1 * bfactor;
      bt2_star = primitive.bt2 * bfactor;
   }
   const amrex::Real vdotb = primitive.vn * primitive.bn + primitive.vt1 * primitive.bt1 + primitive.vt2 * primitive.bt2;
   const amrex::Real vdotb_star = contact_speed * primitive.bn + vt1_star * bt1_star + vt2_star * bt2_star;
   const amrex::Real magnetic = 0.5 * (
      primitive.bn * primitive.bn
      + primitive.bt1 * primitive.bt1
      + primitive.bt2 * primitive.bt2
   );
   const amrex::Real energy_star = (
      (wave_speed - primitive.vn) * primitive.energy
      - (primitive.pressure + magnetic) * primitive.vn
      + total_pressure_star * contact_speed
      + primitive.bn * (vdotb - vdotb_star)
   ) / (wave_speed - contact_speed);
   return {
      rho_star,
      contact_speed,
      vt1_star,
      vt2_star,
      primitive.pressure,
      energy_star,
      primitive.bn,
      bt1_star,
      bt2_star,
   };
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDState mhd_hlld_flux_from_normal_primitives(
   MHDNormalPrimitive const& pl,
   MHDNormalPrimitive const& pr,
   EOSParams const& eos
) noexcept
{
   const amrex::Real bn = 0.5 * (pl.bn + pr.bn);
   const MHDNormalState fl = flux_normal_mhd_primitive(pl);
   const MHDNormalState fr = flux_normal_mhd_primitive(pr);
   const amrex::Real cf_l = mhd_fast_speed_from_components<EOSMethod>(pl.rho, pl.pressure, pl.bn, pl.bt1, pl.bt2, eos);
   const amrex::Real cf_r = mhd_fast_speed_from_components<EOSMethod>(pr.rho, pr.pressure, pr.bn, pr.bt1, pr.bt2, eos);
   const amrex::Real sl = amrex::min(pl.vn - cf_l, pr.vn - cf_r);
   const amrex::Real sr = amrex::max(pl.vn + cf_l, pr.vn + cf_r);
   if (sl >= 0.0) {
      return from_normal_flux<Direction>(fl);
   }
   if (sr <= 0.0) {
      return from_normal_flux<Direction>(fr);
   }

   const amrex::Real total_pressure_l = pl.pressure + 0.5 * (bn * bn + pl.bt1 * pl.bt1 + pl.bt2 * pl.bt2);
   const amrex::Real total_pressure_r = pr.pressure + 0.5 * (bn * bn + pr.bt1 * pr.bt1 + pr.bt2 * pr.bt2);
   const amrex::Real lfac = pl.rho * (sl - pl.vn);
   const amrex::Real rfac = pr.rho * (sr - pr.vn);
   const amrex::Real contact_speed = (rfac * pr.vn - lfac * pl.vn + total_pressure_l - total_pressure_r) / (rfac - lfac);
   const amrex::Real total_pressure_star = total_pressure_l + lfac * (contact_speed - pl.vn);

   const MHDNormalPrimitive pls = hlld_star_state(pl, sl, contact_speed, total_pressure_star);
   const MHDNormalPrimitive prs = hlld_star_state(pr, sr, contact_speed, total_pressure_star);
   const MHDNormalState u_l{pl.rho, pl.rho * pl.vn, pl.rho * pl.vt1, pl.rho * pl.vt2, pl.energy, pl.bn, pl.bt1, pl.bt2};
   const MHDNormalState u_r{pr.rho, pr.rho * pr.vn, pr.rho * pr.vt1, pr.rho * pr.vt2, pr.energy, pr.bn, pr.bt1, pr.bt2};
   const MHDNormalState u_ls{pls.rho, pls.rho * pls.vn, pls.rho * pls.vt1, pls.rho * pls.vt2, pls.energy, pls.bn, pls.bt1, pls.bt2};
   const MHDNormalState u_rs{prs.rho, prs.rho * prs.vn, prs.rho * prs.vt1, prs.rho * prs.vt2, prs.energy, prs.bn, prs.bt1, prs.bt2};
   const MHDNormalState fls{
      fl.rho + sl * (u_ls.rho - u_l.rho),
      fl.momn + sl * (u_ls.momn - u_l.momn),
      fl.momt1 + sl * (u_ls.momt1 - u_l.momt1),
      fl.momt2 + sl * (u_ls.momt2 - u_l.momt2),
      fl.energy + sl * (u_ls.energy - u_l.energy),
      0.0,
      fl.bt1 + sl * (u_ls.bt1 - u_l.bt1),
      fl.bt2 + sl * (u_ls.bt2 - u_l.bt2),
   };
   const MHDNormalState frs{
      fr.rho + sr * (u_rs.rho - u_r.rho),
      fr.momn + sr * (u_rs.momn - u_r.momn),
      fr.momt1 + sr * (u_rs.momt1 - u_r.momt1),
      fr.momt2 + sr * (u_rs.momt2 - u_r.momt2),
      fr.energy + sr * (u_rs.energy - u_r.energy),
      0.0,
      fr.bt1 + sr * (u_rs.bt1 - u_r.bt1),
      fr.bt2 + sr * (u_rs.bt2 - u_r.bt2),
   };

   const amrex::Real abs_bn = amrex::Math::abs(bn);
   if (abs_bn <= MAGNETIC_EPS) {
      return contact_speed >= 0.0 ? from_normal_flux<Direction>(fls) : from_normal_flux<Direction>(frs);
   }

   const amrex::Real sqrt_l = std::sqrt(pls.rho);
   const amrex::Real sqrt_r = std::sqrt(prs.rho);
   const amrex::Real inv_sqrt_sum = 1.0 / (sqrt_l + sqrt_r);
   const amrex::Real sign_bn = bn < 0.0 ? -1.0 : 1.0;
   const amrex::Real sl_star = contact_speed - abs_bn / sqrt_l;
   const amrex::Real sr_star = contact_speed + abs_bn / sqrt_r;
   const amrex::Real vt1_double = (
      sqrt_l * pls.vt1 + sqrt_r * prs.vt1 + (prs.bt1 - pls.bt1) * sign_bn
   ) * inv_sqrt_sum;
   const amrex::Real vt2_double = (
      sqrt_l * pls.vt2 + sqrt_r * prs.vt2 + (prs.bt2 - pls.bt2) * sign_bn
   ) * inv_sqrt_sum;
   const amrex::Real bt1_double = (
      sqrt_l * prs.bt1 + sqrt_r * pls.bt1 + sqrt_l * sqrt_r * (prs.vt1 - pls.vt1) * sign_bn
   ) * inv_sqrt_sum;
   const amrex::Real bt2_double = (
      sqrt_l * prs.bt2 + sqrt_r * pls.bt2 + sqrt_l * sqrt_r * (prs.vt2 - pls.vt2) * sign_bn
   ) * inv_sqrt_sum;
   const amrex::Real vdotb_l_star = contact_speed * bn + pls.vt1 * pls.bt1 + pls.vt2 * pls.bt2;
   const amrex::Real vdotb_r_star = contact_speed * bn + prs.vt1 * prs.bt1 + prs.vt2 * prs.bt2;
   const amrex::Real vdotb_double = contact_speed * bn + vt1_double * bt1_double + vt2_double * bt2_double;
   const amrex::Real energy_l_double = pls.energy - sqrt_l * (vdotb_l_star - vdotb_double) * sign_bn;
   const amrex::Real energy_r_double = prs.energy + sqrt_r * (vdotb_r_star - vdotb_double) * sign_bn;
   const MHDNormalState u_ld{pls.rho, pls.rho * contact_speed, pls.rho * vt1_double, pls.rho * vt2_double, energy_l_double, bn, bt1_double, bt2_double};
   const MHDNormalState u_rd{prs.rho, prs.rho * contact_speed, prs.rho * vt1_double, prs.rho * vt2_double, energy_r_double, bn, bt1_double, bt2_double};
   const MHDNormalState fld{
      fls.rho + sl_star * (u_ld.rho - u_ls.rho),
      fls.momn + sl_star * (u_ld.momn - u_ls.momn),
      fls.momt1 + sl_star * (u_ld.momt1 - u_ls.momt1),
      fls.momt2 + sl_star * (u_ld.momt2 - u_ls.momt2),
      fls.energy + sl_star * (u_ld.energy - u_ls.energy),
      0.0,
      fls.bt1 + sl_star * (u_ld.bt1 - u_ls.bt1),
      fls.bt2 + sl_star * (u_ld.bt2 - u_ls.bt2),
   };
   const MHDNormalState frd{
      frs.rho + sr_star * (u_rd.rho - u_rs.rho),
      frs.momn + sr_star * (u_rd.momn - u_rs.momn),
      frs.momt1 + sr_star * (u_rd.momt1 - u_rs.momt1),
      frs.momt2 + sr_star * (u_rd.momt2 - u_rs.momt2),
      frs.energy + sr_star * (u_rd.energy - u_rs.energy),
      0.0,
      frs.bt1 + sr_star * (u_rd.bt1 - u_rs.bt1),
      frs.bt2 + sr_star * (u_rd.bt2 - u_rs.bt2),
   };

   if (sl_star >= 0.0) {
      return from_normal_flux<Direction>(fls);
   }
   if (contact_speed >= 0.0) {
      return from_normal_flux<Direction>(fld);
   }
   if (sr_star > 0.0) {
      return from_normal_flux<Direction>(frd);
   }
   return from_normal_flux<Direction>(frs);
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDNormalPrimitive to_normal_primitive(
   MHDPrimitive const& primitive,
   amrex::Real bn,
   EOSParams const& eos
) noexcept
{
   const amrex::Real energy = total_mhd_energy<EOSMethod>(
      primitive.rho,
      primitive.v1,
      primitive.v2,
      primitive.v3,
      primitive.pressure,
      primitive.b1,
      primitive.b2,
      primitive.b3,
      eos
   );
   if constexpr (Direction == 1) {
      return {
         primitive.rho,
         primitive.v2,
         primitive.v1,
         primitive.v3,
         primitive.pressure,
         energy,
         bn,
         primitive.b1,
         primitive.b3
      };
   } else if constexpr (Direction == 2) {
      return {
         primitive.rho,
         primitive.v3,
         primitive.v1,
         primitive.v2,
         primitive.pressure,
         energy,
         bn,
         primitive.b1,
         primitive.b2
      };
   } else {
      return {
         primitive.rho,
         primitive.v1,
         primitive.v2,
         primitive.v3,
         primitive.pressure,
         energy,
         bn,
         primitive.b2,
         primitive.b3
      };
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDState mhd_subtract(MHDState const& right, MHDState const& left) noexcept
{
   return {
      right.rho - left.rho,
      right.mom1 - left.mom1,
      right.mom2 - left.mom2,
      right.mom3 - left.mom3,
      right.energy - left.energy,
      right.b1 - left.b1,
      right.b2 - left.b2,
      right.b3 - left.b3,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mhd_primitive_velocity_component(MHDPrimitive const& primitive, int direction) noexcept
{
   if (direction == 0) {
      return primitive.v1;
   }
   if (direction == 1) {
      return primitive.v2;
   }
   return primitive.v3;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mhd_primitive_magnetic_component(MHDPrimitive const& primitive, int direction) noexcept
{
   if (direction == 0) {
      return primitive.b1;
   }
   if (direction == 1) {
      return primitive.b2;
   }
   return primitive.b3;
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mhd_primitive_magnetic_component(MHDPrimitive const& primitive) noexcept
{
   if constexpr (Direction == 0) {
      return primitive.b1;
   } else if constexpr (Direction == 1) {
      return primitive.b2;
   } else {
      return primitive.b3;
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDState mhd_flux_from_primitives(
   MHDPrimitive const& primitive,
   MHDState const& state,
   int direction
) noexcept
{
   return mhd_flux(state, primitive.pressure, direction);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
WaveSpeeds mhd_hll_wave_speeds_from_primitives(
   MHDPrimitive const& primitive_l,
   MHDPrimitive const& primitive_r,
   int direction,
   EOSParams const& eos
) noexcept
{
   const amrex::Real cf_l = mhd_fast_speed_from_primitive<EOSMethod>(primitive_l, direction, eos);
   const amrex::Real cf_r = mhd_fast_speed_from_primitive<EOSMethod>(primitive_r, direction, eos);
   const amrex::Real vn_l = mhd_primitive_velocity_component(primitive_l, direction);
   const amrex::Real vn_r = mhd_primitive_velocity_component(primitive_r, direction);
   return {
      amrex::min(vn_l - cf_l, vn_r - cf_r),
      amrex::max(vn_l + cf_l, vn_r + cf_r),
   };
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
WaveSpeeds mhd_hlle_wave_speeds_from_primitives(
   MHDPrimitive const& primitive_l,
   MHDPrimitive const& primitive_r,
   int direction,
   EOSParams const& eos
) noexcept
{
   if constexpr (EOSMethod == EOSIdealGas) {
      const WaveSpeeds side = mhd_hll_wave_speeds_from_primitives<EOSMethod>(
         primitive_l, primitive_r, direction, eos
      );
      const amrex::Real bn_l = mhd_primitive_magnetic_component(primitive_l, direction);
      const amrex::Real bn_r = mhd_primitive_magnetic_component(primitive_r, direction);
      amrex::Real bt1_l = primitive_l.b2;
      amrex::Real bt2_l = primitive_l.b3;
      amrex::Real bt1_r = primitive_r.b2;
      amrex::Real bt2_r = primitive_r.b3;
      if (direction == 1) {
         bt1_l = primitive_l.b1;
         bt2_l = primitive_l.b3;
         bt1_r = primitive_r.b1;
         bt2_r = primitive_r.b3;
      } else if (direction == 2) {
         bt1_l = primitive_l.b1;
         bt2_l = primitive_l.b2;
         bt1_r = primitive_r.b1;
         bt2_r = primitive_r.b2;
      }
      const amrex::Real sqrt_l = std::sqrt(primitive_l.rho);
      const amrex::Real sqrt_r = std::sqrt(primitive_r.rho);
      const amrex::Real inv_sum = 1.0 / (sqrt_l + sqrt_r);
      const amrex::Real vn_roe = (sqrt_l * mhd_primitive_velocity_component(primitive_l, direction)
         + sqrt_r * mhd_primitive_velocity_component(primitive_r, direction)) * inv_sum;
      const amrex::Real bn_roe = (sqrt_l * bn_l + sqrt_r * bn_r) * inv_sum;
      const amrex::Real bt1_roe = (sqrt_l * bt1_l + sqrt_r * bt1_r) * inv_sum;
      const amrex::Real bt2_roe = (sqrt_l * bt2_l + sqrt_r * bt2_r) * inv_sum;
      const amrex::Real rho_roe = sqrt_l * sqrt_r;
      const amrex::Real p_roe = (sqrt_l * primitive_l.pressure + sqrt_r * primitive_r.pressure) * inv_sum;
      const amrex::Real cf_roe = mhd_fast_speed_from_components<EOSMethod>(
         rho_roe, p_roe, bn_roe, bt1_roe, bt2_roe, eos
      );
      return {amrex::min(side.sl, vn_roe - cf_roe), amrex::max(side.sr, vn_roe + cf_roe)};
   } else {
      return mhd_hll_wave_speeds_from_primitives<EOSMethod>(primitive_l, primitive_r, direction, eos);
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDState mhd_hll_flux_from_primitive_speeds(
   MHDPrimitive const& primitive_l,
   MHDPrimitive const& primitive_r,
   MHDState const& left,
   MHDState const& right,
   int direction,
   WaveSpeeds speeds
) noexcept
{
   const MHDState flux_l = mhd_flux_from_primitives(primitive_l, left, direction);
   const MHDState flux_r = mhd_flux_from_primitives(primitive_r, right, direction);

   if (speeds.sl >= 0.0) {
      return flux_l;
   }
   if (speeds.sr <= 0.0) {
      return flux_r;
   }

   MHDState result = generated::mhd_hll_flux(flux_l, flux_r, left, right, speeds.sl, speeds.sr);
   if (direction == 0) {
      result.b1 = 0.0;
   } else if (direction == 1) {
      result.b2 = 0.0;
   } else {
      result.b3 = 0.0;
   }
   return result;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDState mhd_llf_flux_from_primitives(
   MHDPrimitive const& primitive_l,
   MHDPrimitive const& primitive_r,
   EOSParams const& eos,
   int direction
) noexcept
{
   const MHDState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const MHDState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   const amrex::Real cf_l = mhd_fast_speed_from_primitive<EOSMethod>(primitive_l, direction, eos);
   const amrex::Real cf_r = mhd_fast_speed_from_primitive<EOSMethod>(primitive_r, direction, eos);
   const amrex::Real vn_l = mhd_primitive_velocity_component(primitive_l, direction);
   const amrex::Real vn_r = mhd_primitive_velocity_component(primitive_r, direction);
   const amrex::Real wavespeed = amrex::max(amrex::Math::abs(vn_l) + cf_l, amrex::Math::abs(vn_r) + cf_r);
   const MHDState flux_l = mhd_flux_from_primitives(primitive_l, left, direction);
   const MHDState flux_r = mhd_flux_from_primitives(primitive_r, right, direction);
   MHDState result = generated::mhd_llf_flux(flux_l, flux_r, left, right, wavespeed);
   if (direction == 0) {
      result.b1 = 0.0;
   } else if (direction == 1) {
      result.b2 = 0.0;
   } else {
      result.b3 = 0.0;
   }
   return result;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDState mhd_hll_flux_from_primitives(
   MHDPrimitive const& primitive_l,
   MHDPrimitive const& primitive_r,
   EOSParams const& eos,
   int direction
) noexcept
{
   const MHDState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const MHDState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   return mhd_hll_flux_from_primitive_speeds(
      primitive_l, primitive_r, left, right, direction,
      mhd_hll_wave_speeds_from_primitives<EOSMethod>(primitive_l, primitive_r, direction, eos)
   );
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDState mhd_hlle_flux_from_primitives(
   MHDPrimitive const& primitive_l,
   MHDPrimitive const& primitive_r,
   EOSParams const& eos,
   int direction
) noexcept
{
   const MHDState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const MHDState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   return mhd_hll_flux_from_primitive_speeds(
      primitive_l, primitive_r, left, right, direction,
      mhd_hlle_wave_speeds_from_primitives<EOSMethod>(primitive_l, primitive_r, direction, eos)
   );
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDState mhd_hlld_flux_from_primitives(
   MHDPrimitive const& primitive_l,
   MHDPrimitive const& primitive_r,
   EOSParams const& eos
) noexcept
{
   const amrex::Real bn = 0.5 * (
      mhd_primitive_magnetic_component<Direction>(primitive_l)
      + mhd_primitive_magnetic_component<Direction>(primitive_r)
   );
   const MHDNormalPrimitive pl = to_normal_primitive<Direction, EOSMethod>(primitive_l, bn, eos);
   const MHDNormalPrimitive pr = to_normal_primitive<Direction, EOSMethod>(primitive_r, bn, eos);
   return mhd_hlld_flux_from_normal_primitives<Direction, EOSMethod>(pl, pr, eos);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hydro_primitive_velocity_component(HydroPrimitive const& primitive, int direction) noexcept
{
   if (direction == 0) {
      return primitive.v1;
   }
   if (direction == 1) {
      return primitive.v2;
   }
   return primitive.v3;
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hydro_primitive_velocity_component(HydroPrimitive const& primitive) noexcept
{
   if constexpr (Direction == 0) {
      return primitive.v1;
   } else if constexpr (Direction == 1) {
      return primitive.v2;
   } else {
      return primitive.v3;
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
WaveSpeeds hydro_hll_wave_speeds_from_primitives(
   HydroPrimitive const& left,
   HydroPrimitive const& right,
   int direction,
   EOSParams const& eos
) noexcept
{
   const amrex::Real vn_l = hydro_primitive_velocity_component(left, direction);
   const amrex::Real vn_r = hydro_primitive_velocity_component(right, direction);
   const amrex::Real cs_l = sound_speed_from_primitive<EOSMethod>(left.rho, left.pressure, eos);
   const amrex::Real cs_r = sound_speed_from_primitive<EOSMethod>(right.rho, right.pressure, eos);
   return {amrex::min(vn_l - cs_l, vn_r - cs_r), amrex::max(vn_l + cs_l, vn_r + cs_r)};
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
WaveSpeeds hydro_hll_wave_speeds_from_primitives(
   HydroPrimitive const& left,
   HydroPrimitive const& right,
   EOSParams const& eos
) noexcept
{
   const amrex::Real vn_l = hydro_primitive_velocity_component<Direction>(left);
   const amrex::Real vn_r = hydro_primitive_velocity_component<Direction>(right);
   const amrex::Real cs_l = sound_speed_from_primitive<EOSMethod>(left.rho, left.pressure, eos);
   const amrex::Real cs_r = sound_speed_from_primitive<EOSMethod>(right.rho, right.pressure, eos);
   return {amrex::min(vn_l - cs_l, vn_r - cs_r), amrex::max(vn_l + cs_l, vn_r + cs_r)};
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
WaveSpeeds hydro_hlle_wave_speeds_from_primitives(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   int direction,
   EOSParams const& eos
) noexcept
{
   if constexpr (EOSMethod == EOSIdealGas) {
      const WaveSpeeds side = hydro_hll_wave_speeds_from_primitives<EOSMethod>(
         primitive_l, primitive_r, direction, eos
      );
      const amrex::Real sqrt_l = std::sqrt(primitive_l.rho);
      const amrex::Real sqrt_r = std::sqrt(primitive_r.rho);
      const amrex::Real inv_sum = 1.0 / (sqrt_l + sqrt_r);
      const amrex::Real vx_roe = (sqrt_l * primitive_l.v1 + sqrt_r * primitive_r.v1) * inv_sum;
      const amrex::Real vy_roe = (sqrt_l * primitive_l.v2 + sqrt_r * primitive_r.v2) * inv_sum;
      const amrex::Real vz_roe = (sqrt_l * primitive_l.v3 + sqrt_r * primitive_r.v3) * inv_sum;
      const amrex::Real energy_l = total_hydro_energy<EOSMethod>(
         primitive_l.rho, primitive_l.v1, primitive_l.v2, primitive_l.v3, primitive_l.pressure, eos
      );
      const amrex::Real energy_r = total_hydro_energy<EOSMethod>(
         primitive_r.rho, primitive_r.v1, primitive_r.v2, primitive_r.v3, primitive_r.pressure, eos
      );
      const amrex::Real h_l = (energy_l + primitive_l.pressure) / primitive_l.rho;
      const amrex::Real h_r = (energy_r + primitive_r.pressure) / primitive_r.rho;
      const amrex::Real h_roe = (sqrt_l * h_l + sqrt_r * h_r) * inv_sum;
      const amrex::Real kinetic_roe = 0.5 * (vx_roe * vx_roe + vy_roe * vy_roe + vz_roe * vz_roe);
      const amrex::Real sound2_roe = amrex::max((eos.gamma - 1.0) * (h_roe - kinetic_roe), 0.0);
      const amrex::Real sound_roe = std::sqrt(sound2_roe);
      const amrex::Real vn_roe = direction == 0 ? vx_roe : (direction == 1 ? vy_roe : vz_roe);
      const amrex::Real shock_factor = (eos.gamma + 1.0) / (2.0 * eos.gamma);
      const amrex::Real q_l = std::sqrt(1.0 + shock_factor * amrex::max(primitive_r.pressure / primitive_l.pressure - 1.0, 0.0));
      const amrex::Real q_r = std::sqrt(1.0 + shock_factor * amrex::max(primitive_l.pressure / primitive_r.pressure - 1.0, 0.0));
      return {amrex::min(side.sl, vn_roe - sound_roe * q_l), amrex::max(side.sr, vn_roe + sound_roe * q_r)};
   } else {
      return hydro_hll_wave_speeds_from_primitives<EOSMethod>(primitive_l, primitive_r, direction, eos);
   }
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
WaveSpeeds hydro_hlle_wave_speeds_from_primitives(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   EOSParams const& eos
) noexcept
{
   if constexpr (EOSMethod == EOSIdealGas) {
      const WaveSpeeds side = hydro_hll_wave_speeds_from_primitives<Direction, EOSMethod>(
         primitive_l, primitive_r, eos
      );
      const amrex::Real sqrt_l = std::sqrt(primitive_l.rho);
      const amrex::Real sqrt_r = std::sqrt(primitive_r.rho);
      const amrex::Real inv_sum = 1.0 / (sqrt_l + sqrt_r);
      const amrex::Real vx_roe = (sqrt_l * primitive_l.v1 + sqrt_r * primitive_r.v1) * inv_sum;
      const amrex::Real vy_roe = (sqrt_l * primitive_l.v2 + sqrt_r * primitive_r.v2) * inv_sum;
      const amrex::Real vz_roe = (sqrt_l * primitive_l.v3 + sqrt_r * primitive_r.v3) * inv_sum;
      const amrex::Real energy_l = total_hydro_energy<EOSMethod>(
         primitive_l.rho, primitive_l.v1, primitive_l.v2, primitive_l.v3, primitive_l.pressure, eos
      );
      const amrex::Real energy_r = total_hydro_energy<EOSMethod>(
         primitive_r.rho, primitive_r.v1, primitive_r.v2, primitive_r.v3, primitive_r.pressure, eos
      );
      const amrex::Real h_l = (energy_l + primitive_l.pressure) / primitive_l.rho;
      const amrex::Real h_r = (energy_r + primitive_r.pressure) / primitive_r.rho;
      const amrex::Real h_roe = (sqrt_l * h_l + sqrt_r * h_r) * inv_sum;
      const amrex::Real kinetic_roe = 0.5 * (vx_roe * vx_roe + vy_roe * vy_roe + vz_roe * vz_roe);
      const amrex::Real sound2_roe = amrex::max((eos.gamma - 1.0) * (h_roe - kinetic_roe), 0.0);
      const amrex::Real sound_roe = std::sqrt(sound2_roe);
      amrex::Real vn_roe = vx_roe;
      if constexpr (Direction == 1) {
         vn_roe = vy_roe;
      } else if constexpr (Direction == 2) {
         vn_roe = vz_roe;
      }
      const amrex::Real shock_factor = (eos.gamma + 1.0) / (2.0 * eos.gamma);
      const amrex::Real q_l = std::sqrt(1.0 + shock_factor * amrex::max(primitive_r.pressure / primitive_l.pressure - 1.0, 0.0));
      const amrex::Real q_r = std::sqrt(1.0 + shock_factor * amrex::max(primitive_l.pressure / primitive_r.pressure - 1.0, 0.0));
      return {amrex::min(side.sl, vn_roe - sound_roe * q_l), amrex::max(side.sr, vn_roe + sound_roe * q_r)};
   } else {
      return hydro_hll_wave_speeds_from_primitives<Direction, EOSMethod>(primitive_l, primitive_r, eos);
   }
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
WaveSpeeds hydro_hll_wave_speeds_from_primitives_normal(
   HydroPrimitive const& left,
   HydroPrimitive const& right,
   PhysicalVector normal,
   EOSParams const& eos
) noexcept
{
   const amrex::Real vn_l = hydro_primitive_normal_velocity(left, normal);
   const amrex::Real vn_r = hydro_primitive_normal_velocity(right, normal);
   const amrex::Real cs_l = sound_speed_from_primitive<EOSMethod>(left.rho, left.pressure, eos);
   const amrex::Real cs_r = sound_speed_from_primitive<EOSMethod>(right.rho, right.pressure, eos);
   return {amrex::min(vn_l - cs_l, vn_r - cs_r), amrex::max(vn_l + cs_l, vn_r + cs_r)};
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
WaveSpeeds hydro_hlle_wave_speeds_from_primitives_normal(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   PhysicalVector normal,
   EOSParams const& eos
) noexcept
{
   if constexpr (EOSMethod == EOSIdealGas) {
      const WaveSpeeds side = hydro_hll_wave_speeds_from_primitives_normal<EOSMethod>(
         primitive_l, primitive_r, normal, eos
      );
      const amrex::Real sqrt_l = std::sqrt(primitive_l.rho);
      const amrex::Real sqrt_r = std::sqrt(primitive_r.rho);
      const amrex::Real inv_sum = 1.0 / (sqrt_l + sqrt_r);
      const amrex::Real vx_roe = (sqrt_l * primitive_l.v1 + sqrt_r * primitive_r.v1) * inv_sum;
      const amrex::Real vy_roe = (sqrt_l * primitive_l.v2 + sqrt_r * primitive_r.v2) * inv_sum;
      const amrex::Real vz_roe = (sqrt_l * primitive_l.v3 + sqrt_r * primitive_r.v3) * inv_sum;
      const amrex::Real energy_l = total_hydro_energy<EOSMethod>(
         primitive_l.rho, primitive_l.v1, primitive_l.v2, primitive_l.v3, primitive_l.pressure, eos
      );
      const amrex::Real energy_r = total_hydro_energy<EOSMethod>(
         primitive_r.rho, primitive_r.v1, primitive_r.v2, primitive_r.v3, primitive_r.pressure, eos
      );
      const amrex::Real h_l = (energy_l + primitive_l.pressure) / primitive_l.rho;
      const amrex::Real h_r = (energy_r + primitive_r.pressure) / primitive_r.rho;
      const amrex::Real h_roe = (sqrt_l * h_l + sqrt_r * h_r) * inv_sum;
      const amrex::Real kinetic_roe = 0.5 * (vx_roe * vx_roe + vy_roe * vy_roe + vz_roe * vz_roe);
      const amrex::Real sound2_roe = amrex::max((eos.gamma - 1.0) * (h_roe - kinetic_roe), 0.0);
      const amrex::Real sound_roe = std::sqrt(sound2_roe);
      const amrex::Real vn_roe = vx_roe * normal.x + vy_roe * normal.y + vz_roe * normal.z;
      const amrex::Real shock_factor = (eos.gamma + 1.0) / (2.0 * eos.gamma);
      const amrex::Real q_l = std::sqrt(
         1.0 + shock_factor * amrex::max(primitive_r.pressure / primitive_l.pressure - 1.0, 0.0)
      );
      const amrex::Real q_r = std::sqrt(
         1.0 + shock_factor * amrex::max(primitive_l.pressure / primitive_r.pressure - 1.0, 0.0)
      );
      return {amrex::min(side.sl, vn_roe - sound_roe * q_l), amrex::max(side.sr, vn_roe + sound_roe * q_r)};
   } else {
      return hydro_hll_wave_speeds_from_primitives_normal<EOSMethod>(primitive_l, primitive_r, normal, eos);
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hll_flux_from_primitive_speeds_normal(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   HydroState const& left,
   HydroState const& right,
   PhysicalVector normal,
   WaveSpeeds speeds
) noexcept
{
   const HydroState flux_l = hydro_flux_normal(left, primitive_l.pressure, normal);
   const HydroState flux_r = hydro_flux_normal(right, primitive_r.pressure, normal);
   if (speeds.sl >= 0.0) {
      return flux_l;
   }
   if (speeds.sr <= 0.0) {
      return flux_r;
   }
   return generated::hydro_hll_flux(flux_l, flux_r, left, right, speeds.sl, speeds.sr);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState llf_flux_from_primitives_normal(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   EOSParams const& eos,
   PhysicalVector normal
) noexcept
{
   const HydroState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const HydroState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   const amrex::Real vn_l = hydro_primitive_normal_velocity(primitive_l, normal);
   const amrex::Real vn_r = hydro_primitive_normal_velocity(primitive_r, normal);
   const amrex::Real cs_l = sound_speed_from_primitive<EOSMethod>(primitive_l.rho, primitive_l.pressure, eos);
   const amrex::Real cs_r = sound_speed_from_primitive<EOSMethod>(primitive_r.rho, primitive_r.pressure, eos);
   const amrex::Real wavespeed = amrex::max(amrex::Math::abs(vn_l) + cs_l, amrex::Math::abs(vn_r) + cs_r);
   const HydroState flux_l = hydro_flux_normal(left, primitive_l.pressure, normal);
   const HydroState flux_r = hydro_flux_normal(right, primitive_r.pressure, normal);
   return generated::hydro_llf_flux(flux_l, flux_r, left, right, wavespeed);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hll_flux_from_primitives_normal(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   EOSParams const& eos,
   PhysicalVector normal
) noexcept
{
   const HydroState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const HydroState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   return hll_flux_from_primitive_speeds_normal(
      primitive_l, primitive_r, left, right, normal,
      hydro_hll_wave_speeds_from_primitives_normal<EOSMethod>(primitive_l, primitive_r, normal, eos)
   );
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hlle_flux_from_primitives_normal(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   EOSParams const& eos,
   PhysicalVector normal
) noexcept
{
   const HydroState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const HydroState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   return hll_flux_from_primitive_speeds_normal(
      primitive_l, primitive_r, left, right, normal,
      hydro_hlle_wave_speeds_from_primitives_normal<EOSMethod>(primitive_l, primitive_r, normal, eos)
   );
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hllc_flux_from_primitives_normal(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   EOSParams const& eos,
   PhysicalVector normal
) noexcept
{
   const HydroState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const HydroState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   const amrex::Real vn_l = hydro_primitive_normal_velocity(primitive_l, normal);
   const amrex::Real vn_r = hydro_primitive_normal_velocity(primitive_r, normal);
   const WaveSpeeds speeds = hydro_hlle_wave_speeds_from_primitives_normal<EOSMethod>(
      primitive_l, primitive_r, normal, eos
   );
   const amrex::Real sl = speeds.sl;
   const amrex::Real sr = speeds.sr;
   const HydroState flux_l = hydro_flux_normal(left, primitive_l.pressure, normal);
   const HydroState flux_r = hydro_flux_normal(right, primitive_r.pressure, normal);
   if (sl >= 0.0) {
      return flux_l;
   }
   if (sr <= 0.0) {
      return flux_r;
   }
   const amrex::Real denom = left.rho * (sl - vn_l) - right.rho * (sr - vn_r);
   if (!amrex::Math::isfinite(denom) || amrex::Math::abs(denom) <= 1.0e-300) {
      return hlle_flux_from_primitives_normal<EOSMethod>(primitive_l, primitive_r, eos, normal);
   }
   const amrex::Real s_star = (
      primitive_r.pressure - primitive_l.pressure
      + left.rho * vn_l * (sl - vn_l)
      - right.rho * vn_r * (sr - vn_r)
   ) / denom;
   if (!amrex::Math::isfinite(s_star) || s_star <= sl || s_star >= sr) {
      return hlle_flux_from_primitives_normal<EOSMethod>(primitive_l, primitive_r, eos, normal);
   }
   if (s_star >= 0.0) {
      const HydroState star_l = hllc_star_state_normal(left, primitive_l.pressure, normal, sl, s_star);
      if (!hydro_state_is_finite(star_l) || star_l.rho <= 0.0 || star_l.energy <= 0.0) {
         return hlle_flux_from_primitives_normal<EOSMethod>(primitive_l, primitive_r, eos, normal);
      }
      return add_wave_correction(flux_l, sl, star_l, left);
   }
   const HydroState star_r = hllc_star_state_normal(right, primitive_r.pressure, normal, sr, s_star);
   if (!hydro_state_is_finite(star_r) || star_r.rho <= 0.0 || star_r.energy <= 0.0) {
      return hlle_flux_from_primitives_normal<EOSMethod>(primitive_l, primitive_r, eos, normal);
   }
   return add_wave_correction(flux_r, sr, star_r, right);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hll_flux_from_primitive_speeds(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   HydroState const& left,
   HydroState const& right,
   int direction,
   WaveSpeeds speeds
) noexcept
{
   const HydroState flux_l = hydro_flux(left, primitive_l.pressure, direction);
   const HydroState flux_r = hydro_flux(right, primitive_r.pressure, direction);
   if (speeds.sl >= 0.0) {
      return flux_l;
   }
   if (speeds.sr <= 0.0) {
      return flux_r;
   }
   return generated::hydro_hll_flux(flux_l, flux_r, left, right, speeds.sl, speeds.sr);
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hll_flux_from_primitive_speeds(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   HydroState const& left,
   HydroState const& right,
   WaveSpeeds speeds
) noexcept
{
   const HydroState flux_l = hydro_flux<Direction>(left, primitive_l.pressure);
   const HydroState flux_r = hydro_flux<Direction>(right, primitive_r.pressure);
   if (speeds.sl >= 0.0) {
      return flux_l;
   }
   if (speeds.sr <= 0.0) {
      return flux_r;
   }
   return generated::hydro_hll_flux(flux_l, flux_r, left, right, speeds.sl, speeds.sr);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState llf_flux_from_primitives(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   EOSParams const& eos,
   int direction
) noexcept
{
   const HydroState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const HydroState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   const amrex::Real vn_l = hydro_primitive_velocity_component(primitive_l, direction);
   const amrex::Real vn_r = hydro_primitive_velocity_component(primitive_r, direction);
   const amrex::Real cs_l = sound_speed_from_primitive<EOSMethod>(primitive_l.rho, primitive_l.pressure, eos);
   const amrex::Real cs_r = sound_speed_from_primitive<EOSMethod>(primitive_r.rho, primitive_r.pressure, eos);
   const amrex::Real wavespeed = amrex::max(amrex::Math::abs(vn_l) + cs_l, amrex::Math::abs(vn_r) + cs_r);
   const HydroState flux_l = hydro_flux(left, primitive_l.pressure, direction);
   const HydroState flux_r = hydro_flux(right, primitive_r.pressure, direction);
   return generated::hydro_llf_flux(flux_l, flux_r, left, right, wavespeed);
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState llf_flux_from_primitives(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   EOSParams const& eos
) noexcept
{
   const HydroState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const HydroState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   const amrex::Real vn_l = hydro_primitive_velocity_component<Direction>(primitive_l);
   const amrex::Real vn_r = hydro_primitive_velocity_component<Direction>(primitive_r);
   const amrex::Real cs_l = sound_speed_from_primitive<EOSMethod>(primitive_l.rho, primitive_l.pressure, eos);
   const amrex::Real cs_r = sound_speed_from_primitive<EOSMethod>(primitive_r.rho, primitive_r.pressure, eos);
   const amrex::Real wavespeed = amrex::max(amrex::Math::abs(vn_l) + cs_l, amrex::Math::abs(vn_r) + cs_r);
   const HydroState flux_l = hydro_flux<Direction>(left, primitive_l.pressure);
   const HydroState flux_r = hydro_flux<Direction>(right, primitive_r.pressure);
   return generated::hydro_llf_flux(flux_l, flux_r, left, right, wavespeed);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hll_flux_from_primitives(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   EOSParams const& eos,
   int direction
) noexcept
{
   const HydroState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const HydroState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   return hll_flux_from_primitive_speeds(
      primitive_l, primitive_r, left, right, direction,
      hydro_hll_wave_speeds_from_primitives<EOSMethod>(primitive_l, primitive_r, direction, eos)
   );
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hll_flux_from_primitives(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   EOSParams const& eos
) noexcept
{
   const HydroState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const HydroState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   return hll_flux_from_primitive_speeds<Direction>(
      primitive_l, primitive_r, left, right,
      hydro_hll_wave_speeds_from_primitives<Direction, EOSMethod>(primitive_l, primitive_r, eos)
   );
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hlle_flux_from_primitives(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   EOSParams const& eos,
   int direction
) noexcept
{
   const HydroState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const HydroState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   return hll_flux_from_primitive_speeds(
      primitive_l, primitive_r, left, right, direction,
      hydro_hlle_wave_speeds_from_primitives<EOSMethod>(primitive_l, primitive_r, direction, eos)
   );
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hlle_flux_from_primitives(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   EOSParams const& eos
) noexcept
{
   const HydroState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const HydroState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   return hll_flux_from_primitive_speeds<Direction>(
      primitive_l, primitive_r, left, right,
      hydro_hlle_wave_speeds_from_primitives<Direction, EOSMethod>(primitive_l, primitive_r, eos)
   );
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hllc_flux_from_primitives(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   EOSParams const& eos,
   int direction
) noexcept
{
   const HydroState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const HydroState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   const amrex::Real vn_l = hydro_primitive_velocity_component(primitive_l, direction);
   const amrex::Real vn_r = hydro_primitive_velocity_component(primitive_r, direction);
   const WaveSpeeds speeds = hydro_hlle_wave_speeds_from_primitives<EOSMethod>(
      primitive_l, primitive_r, direction, eos
   );
   const amrex::Real sl = speeds.sl;
   const amrex::Real sr = speeds.sr;
   const HydroState flux_l = hydro_flux(left, primitive_l.pressure, direction);
   const HydroState flux_r = hydro_flux(right, primitive_r.pressure, direction);
   if (sl >= 0.0) {
      return flux_l;
   }
   if (sr <= 0.0) {
      return flux_r;
   }
   const amrex::Real denom = left.rho * (sl - vn_l) - right.rho * (sr - vn_r);
   if (!amrex::Math::isfinite(denom) || amrex::Math::abs(denom) <= 1.0e-300) {
      return hlle_flux_from_primitives<EOSMethod>(primitive_l, primitive_r, eos, direction);
   }
   const amrex::Real s_star = (
      primitive_r.pressure - primitive_l.pressure
      + left.rho * vn_l * (sl - vn_l)
      - right.rho * vn_r * (sr - vn_r)
   ) / denom;
   if (!amrex::Math::isfinite(s_star) || s_star <= sl || s_star >= sr) {
      return hlle_flux_from_primitives<EOSMethod>(primitive_l, primitive_r, eos, direction);
   }
   if (s_star >= 0.0) {
      const HydroState star_l = hllc_star_state(left, primitive_l.pressure, direction, sl, s_star);
      if (!hydro_state_is_finite(star_l) || star_l.rho <= 0.0 || star_l.energy <= 0.0) {
         return hlle_flux_from_primitives<EOSMethod>(primitive_l, primitive_r, eos, direction);
      }
      return add_wave_correction(flux_l, sl, star_l, left);
   }
   const HydroState star_r = hllc_star_state(right, primitive_r.pressure, direction, sr, s_star);
   if (!hydro_state_is_finite(star_r) || star_r.rho <= 0.0 || star_r.energy <= 0.0) {
      return hlle_flux_from_primitives<EOSMethod>(primitive_l, primitive_r, eos, direction);
   }
   return add_wave_correction(flux_r, sr, star_r, right);
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hllc_flux_from_primitives(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   EOSParams const& eos
) noexcept
{
   const HydroState left = conserved_from_primitive<EOSMethod>(primitive_l, eos);
   const HydroState right = conserved_from_primitive<EOSMethod>(primitive_r, eos);
   const amrex::Real vn_l = hydro_primitive_velocity_component<Direction>(primitive_l);
   const amrex::Real vn_r = hydro_primitive_velocity_component<Direction>(primitive_r);
   const WaveSpeeds speeds = hydro_hlle_wave_speeds_from_primitives<Direction, EOSMethod>(
      primitive_l, primitive_r, eos
   );
   const amrex::Real sl = speeds.sl;
   const amrex::Real sr = speeds.sr;
   const HydroState flux_l = hydro_flux<Direction>(left, primitive_l.pressure);
   const HydroState flux_r = hydro_flux<Direction>(right, primitive_r.pressure);
   if (sl >= 0.0) {
      return flux_l;
   }
   if (sr <= 0.0) {
      return flux_r;
   }
   const amrex::Real denom = left.rho * (sl - vn_l) - right.rho * (sr - vn_r);
   if (!amrex::Math::isfinite(denom) || amrex::Math::abs(denom) <= 1.0e-300) {
      return hlle_flux_from_primitives<Direction, EOSMethod>(primitive_l, primitive_r, eos);
   }
   const amrex::Real s_star = (
      primitive_r.pressure - primitive_l.pressure
      + left.rho * vn_l * (sl - vn_l)
      - right.rho * vn_r * (sr - vn_r)
   ) / denom;
   if (!amrex::Math::isfinite(s_star) || s_star <= sl || s_star >= sr) {
      return hlle_flux_from_primitives<Direction, EOSMethod>(primitive_l, primitive_r, eos);
   }
   if (s_star >= 0.0) {
      const HydroState star_l = hllc_star_state<Direction>(left, primitive_l.pressure, sl, s_star);
      if (!hydro_state_is_finite(star_l) || star_l.rho <= 0.0 || star_l.energy <= 0.0) {
         return hlle_flux_from_primitives<Direction, EOSMethod>(primitive_l, primitive_r, eos);
      }
      return add_wave_correction(flux_l, sl, star_l, left);
   }
   const HydroState star_r = hllc_star_state<Direction>(right, primitive_r.pressure, sr, s_star);
   if (!hydro_state_is_finite(star_r) || star_r.rho <= 0.0 || star_r.energy <= 0.0) {
      return hlle_flux_from_primitives<Direction, EOSMethod>(primitive_l, primitive_r, eos);
   }
   return add_wave_correction(flux_r, sr, star_r, right);
}

template <int ReconMethod, int RiemannMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hydro_riemann_flux_normal(
   amrex::Array4<amrex::Real const> const& state,
   int il,
   int jl,
   int kl,
   int ir,
   int jr,
   int kr,
   EOSParams const& eos,
   PhysicalVector normal
) noexcept
{
   const HydroPrimitiveFaceStates face = hydro_primitive_face_states<ReconMethod, Direction, EOSMethod>(
      state, il, jl, kl, ir, jr, kr, eos
   );
   if constexpr (RiemannMethod == RiemannHLL) {
      return hll_flux_from_primitives_normal<EOSMethod>(face.left, face.right, eos, normal);
   } else if constexpr (RiemannMethod == RiemannHLLE) {
      return hlle_flux_from_primitives_normal<EOSMethod>(face.left, face.right, eos, normal);
   } else if constexpr (RiemannMethod == RiemannHLLC) {
      return hllc_flux_from_primitives_normal<EOSMethod>(face.left, face.right, eos, normal);
   } else {
      return llf_flux_from_primitives_normal<EOSMethod>(face.left, face.right, eos, normal);
   }
}

template <int ReconMethod, int RiemannMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hydro_riemann_flux(
   amrex::Array4<amrex::Real const> const& state,
   int il,
   int jl,
   int kl,
   int ir,
   int jr,
   int kr,
   EOSParams const& eos
) noexcept
{
   const HydroPrimitiveFaceStates face = hydro_primitive_face_states<ReconMethod, Direction, EOSMethod>(
      state, il, jl, kl, ir, jr, kr, eos
   );
   if constexpr (RiemannMethod == RiemannHLL) {
      return hll_flux_from_primitives<Direction, EOSMethod>(face.left, face.right, eos);
   } else if constexpr (RiemannMethod == RiemannHLLE) {
      return hlle_flux_from_primitives<Direction, EOSMethod>(face.left, face.right, eos);
   } else if constexpr (RiemannMethod == RiemannHLLC) {
      return hllc_flux_from_primitives<Direction, EOSMethod>(face.left, face.right, eos);
   } else {
      return llf_flux_from_primitives<Direction, EOSMethod>(face.left, face.right, eos);
   }
}

template <int ReconMethod, int RiemannMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDState mhd_riemann_flux(
   amrex::Array4<amrex::Real const> const& state,
   int il,
   int jl,
   int kl,
   int ir,
   int jr,
   int kr,
   EOSParams const& eos
) noexcept
{
   const MHDPrimitiveFaceStates face = mhd_primitive_face_states<ReconMethod, Direction, EOSMethod>(
      state, il, jl, kl, ir, jr, kr, eos
   );
   if constexpr (RiemannMethod == RiemannHLL) {
      return mhd_hll_flux_from_primitives<EOSMethod>(face.left, face.right, eos, Direction);
   } else if constexpr (RiemannMethod == RiemannHLLE) {
      return mhd_hlle_flux_from_primitives<EOSMethod>(face.left, face.right, eos, Direction);
   } else if constexpr (RiemannMethod == RiemannHLLD) {
      return mhd_hlld_flux_from_primitives<Direction, EOSMethod>(face.left, face.right, eos);
   } else {
      return mhd_llf_flux_from_primitives<EOSMethod>(face.left, face.right, eos, Direction);
   }
}

} // namespace pyfluid_amrex
