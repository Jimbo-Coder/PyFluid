#pragma once

#include <AMReX_Array4.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "pyfluid_amrex/characteristic_basis.hpp"
#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/kernels_generated/characteristic_transforms.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

constexpr amrex::Real WENO_Z_EPS = 1.0e-8;
constexpr amrex::Real TENO5_EPS = 1.0e-40;
constexpr amrex::Real TENO5_CUTOFF = 1.0e-5;
constexpr amrex::Real MP5_ALPHA = 4.0;
constexpr amrex::Real MP5_EPS = 1.0e-12;

struct HydroFaceStates {
   HydroState left;
   HydroState right;
};

struct MHDFaceStates {
   MHDState left;
   MHDState right;
};

struct HydroPrimitiveFaceStates {
   HydroPrimitive left;
   HydroPrimitive right;
};

struct MHDPrimitiveFaceStates {
   MHDPrimitive left;
   MHDPrimitive right;
};

struct ScalarFaceValues {
   amrex::Real left = 0.0;
   amrex::Real right = 0.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real minmod(amrex::Real a, amrex::Real b) noexcept
{
   if (a * b <= 0.0) {
      return 0.0;
   }
   return a > 0.0 ? amrex::min(a, b) : amrex::max(a, b);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real minmod4(amrex::Real a, amrex::Real b, amrex::Real c, amrex::Real d) noexcept
{
   if (a > 0.0 && b > 0.0 && c > 0.0 && d > 0.0) {
      return amrex::min(amrex::min(a, b), amrex::min(c, d));
   }
   if (a < 0.0 && b < 0.0 && c < 0.0 && d < 0.0) {
      return amrex::max(amrex::max(a, b), amrex::max(c, d));
   }
   return 0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mc_slope(amrex::Real qm, amrex::Real q0, amrex::Real qp) noexcept
{
   const amrex::Real centered = 0.5 * (qp - qm);
   const amrex::Real limited = minmod(2.0 * (q0 - qm), 2.0 * (qp - q0));
   return minmod(centered, limited);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitive primitive_at(
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& /*eos*/
) noexcept
{
   return {
      primitive(i, j, k, RHO),
      primitive(i, j, k, MOM1),
      primitive(i, j, k, MOM2),
      primitive(i, j, k, MOM3),
      primitive(i, j, k, ENERGY),
   };
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitive primitive_neighbor(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int offset,
   EOSParams const& eos
) noexcept
{
   if constexpr (Direction == 0) {
      return primitive_at(state, i + offset, j, k, eos);
   }
#if (AMREX_SPACEDIM >= 2)
   else if constexpr (Direction == 1) {
      return primitive_at(state, i, j + offset, k, eos);
   }
#endif
   return primitive_at(state, i, j, k + offset, eos);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDPrimitive mhd_primitive_at(
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& /*eos*/
) noexcept
{
   return {
      primitive(i, j, k, RHO),
      primitive(i, j, k, MOM1),
      primitive(i, j, k, MOM2),
      primitive(i, j, k, MOM3),
      primitive(i, j, k, ENERGY),
      primitive(i, j, k, B1),
      primitive(i, j, k, B2),
      primitive(i, j, k, B3),
   };
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDPrimitive mhd_primitive_neighbor(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int offset,
   EOSParams const& eos
) noexcept
{
   if constexpr (Direction == 0) {
      return mhd_primitive_at(state, i + offset, j, k, eos);
   }
#if (AMREX_SPACEDIM >= 2)
   else if constexpr (Direction == 1) {
      return mhd_primitive_at(state, i, j + offset, k, eos);
   }
#endif
   return mhd_primitive_at(state, i, j, k + offset, eos);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real positivity_limiter_theta(amrex::Real face, amrex::Real center, amrex::Real floor) noexcept
{
   if (face >= floor) {
      return 1.0;
   }
   if (!amrex::Math::isfinite(face) || !amrex::Math::isfinite(center) || center <= floor) {
      return 0.0;
   }
   const amrex::Real denom = center - face;
   if (denom <= 0.0) {
      return 0.0;
   }
   return amrex::max(0.0, amrex::min(1.0, (center - floor) / denom));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitive positivity_limit_hydro_primitive(
   HydroPrimitive face,
   HydroPrimitive const& center,
   EOSParams const& eos
) noexcept
{
   amrex::Real theta = positivity_limiter_theta(face.rho, center.rho, eos.rho_floor);
   theta = amrex::min(theta, positivity_limiter_theta(face.pressure, center.pressure, eos.pressure_floor));
   if (!amrex::Math::isfinite(face.v1) || !amrex::Math::isfinite(face.v2) || !amrex::Math::isfinite(face.v3)) {
      theta = 0.0;
   }
   face.rho = center.rho + theta * (face.rho - center.rho);
   face.v1 = center.v1 + theta * (face.v1 - center.v1);
   face.v2 = center.v2 + theta * (face.v2 - center.v2);
   face.v3 = center.v3 + theta * (face.v3 - center.v3);
   face.pressure = center.pressure + theta * (face.pressure - center.pressure);
   return face;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDPrimitive positivity_limit_mhd_primitive(
   MHDPrimitive face,
   MHDPrimitive const& center,
   EOSParams const& eos
) noexcept
{
   amrex::Real theta = positivity_limiter_theta(face.rho, center.rho, eos.rho_floor);
   theta = amrex::min(theta, positivity_limiter_theta(face.pressure, center.pressure, eos.pressure_floor));
   if (
      !amrex::Math::isfinite(face.v1)
      || !amrex::Math::isfinite(face.v2)
      || !amrex::Math::isfinite(face.v3)
      || !amrex::Math::isfinite(face.b1)
      || !amrex::Math::isfinite(face.b2)
      || !amrex::Math::isfinite(face.b3)
   ) {
      theta = 0.0;
   }
   face.rho = center.rho + theta * (face.rho - center.rho);
   face.v1 = center.v1 + theta * (face.v1 - center.v1);
   face.v2 = center.v2 + theta * (face.v2 - center.v2);
   face.v3 = center.v3 + theta * (face.v3 - center.v3);
   face.pressure = center.pressure + theta * (face.pressure - center.pressure);
   face.b1 = center.b1 + theta * (face.b1 - center.b1);
   face.b2 = center.b2 + theta * (face.b2 - center.b2);
   face.b3 = center.b3 + theta * (face.b3 - center.b3);
   return face;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitiveFaceStates positivity_limit_hydro_face_states(
   HydroPrimitiveFaceStates face,
   HydroPrimitive const& left_center,
   HydroPrimitive const& right_center,
   EOSParams const& eos
) noexcept
{
   return {
      positivity_limit_hydro_primitive(face.left, left_center, eos),
      positivity_limit_hydro_primitive(face.right, right_center, eos),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDPrimitiveFaceStates positivity_limit_mhd_face_states(
   MHDPrimitiveFaceStates face,
   MHDPrimitive const& left_center,
   MHDPrimitive const& right_center,
   EOSParams const& eos
) noexcept
{
   return {
      positivity_limit_mhd_primitive(face.left, left_center, eos),
      positivity_limit_mhd_primitive(face.right, right_center, eos),
   };
}

template <int Direction, int Component>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real primitive_component_neighbor(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int offset
) noexcept
{
   if constexpr (Direction == 0) {
      return state(i + offset, j, k, Component);
   }
#if (AMREX_SPACEDIM >= 2)
   else if constexpr (Direction == 1) {
      return state(i, j + offset, k, Component);
   }
#endif
   return state(i, j, k + offset, Component);
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hydro_normal_velocity(HydroPrimitive const& q) noexcept
{
   if constexpr (Direction == 0) {
      return q.v1;
   } else if constexpr (Direction == 1) {
      return q.v2;
   } else {
      return q.v3;
   }
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hydro_tangent1_velocity(HydroPrimitive const& q) noexcept
{
   if constexpr (Direction == 0) {
      return q.v2;
   } else {
      return q.v1;
   }
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hydro_tangent2_velocity(HydroPrimitive const& q) noexcept
{
   if constexpr (Direction == 2) {
      return q.v2;
   } else {
      return q.v3;
   }
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitive hydro_primitive_from_normal(
   amrex::Real rho,
   amrex::Real vn,
   amrex::Real vt1,
   amrex::Real vt2,
   amrex::Real pressure
) noexcept
{
   if constexpr (Direction == 0) {
      return {rho, vn, vt1, vt2, pressure};
   } else if constexpr (Direction == 1) {
      return {rho, vt1, vn, vt2, pressure};
   } else {
      return {rho, vt1, vt2, vn, pressure};
   }
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mhd_normal_b(MHDPrimitive const& q) noexcept
{
   if constexpr (Direction == 0) {
      return q.b1;
   } else if constexpr (Direction == 1) {
      return q.b2;
   } else {
      return q.b3;
   }
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDReducedPrimitive mhd_reduced_primitive(MHDPrimitive const& q) noexcept
{
   if constexpr (Direction == 0) {
      return {q.rho, q.v1, q.v2, q.v3, q.pressure, q.b2, q.b3};
   } else if constexpr (Direction == 1) {
      return {q.rho, q.v2, q.v1, q.v3, q.pressure, q.b1, q.b3};
   } else {
      return {q.rho, q.v3, q.v1, q.v2, q.pressure, q.b1, q.b2};
   }
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDPrimitive mhd_primitive_from_reduced(MHDReducedPrimitive const& q, amrex::Real bn) noexcept
{
   if constexpr (Direction == 0) {
      return {q.rho, q.vn, q.vt1, q.vt2, q.pressure, bn, q.bt1, q.bt2};
   } else if constexpr (Direction == 1) {
      return {q.rho, q.vt1, q.vn, q.vt2, q.pressure, q.bt1, bn, q.bt2};
   } else {
      return {q.rho, q.vt1, q.vt2, q.vn, q.pressure, q.bt1, q.bt2, bn};
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real weno_z_left(amrex::Real vmm, amrex::Real vm, amrex::Real v0, amrex::Real vp, amrex::Real vpp) noexcept
{
   const amrex::Real d0 = vmm - 2.0 * vm + v0;
   const amrex::Real e0 = vmm - 4.0 * vm + 3.0 * v0;
   const amrex::Real d1 = vm - 2.0 * v0 + vp;
   const amrex::Real e1 = vm - vp;
   const amrex::Real d2 = v0 - 2.0 * vp + vpp;
   const amrex::Real e2 = 3.0 * v0 - 4.0 * vp + vpp;
   const amrex::Real beta0 = (13.0 / 12.0) * d0 * d0 + 0.25 * e0 * e0;
   const amrex::Real beta1 = (13.0 / 12.0) * d1 * d1 + 0.25 * e1 * e1;
   const amrex::Real beta2 = (13.0 / 12.0) * d2 * d2 + 0.25 * e2 * e2;
   const amrex::Real tau5 = amrex::Math::abs(beta0 - beta2);
   const amrex::Real r0 = tau5 / (WENO_Z_EPS + beta0);
   const amrex::Real r1 = tau5 / (WENO_Z_EPS + beta1);
   const amrex::Real r2 = tau5 / (WENO_Z_EPS + beta2);
   const amrex::Real a0 = 0.1 * (1.0 + r0 * r0);
   const amrex::Real a1 = 0.6 * (1.0 + r1 * r1);
   const amrex::Real a2 = 0.3 * (1.0 + r2 * r2);
   const amrex::Real inv = 1.0 / (a0 + a1 + a2);
   const amrex::Real p0 = (2.0 * vmm - 7.0 * vm + 11.0 * v0) / 6.0;
   const amrex::Real p1 = (-vm + 5.0 * v0 + 2.0 * vp) / 6.0;
   const amrex::Real p2 = (2.0 * v0 + 5.0 * vp - vpp) / 6.0;
   return (a0 * p0 + a1 * p1 + a2 * p2) * inv;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real weno_z_right(amrex::Real vm, amrex::Real v0, amrex::Real vp, amrex::Real vpp, amrex::Real vppp) noexcept
{
   return weno_z_left(vppp, vpp, vp, v0, vm);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real teno5_left(amrex::Real vmm, amrex::Real vm, amrex::Real v0, amrex::Real vp, amrex::Real vpp) noexcept
{
   const amrex::Real d0 = vmm - 2.0 * vm + v0;
   const amrex::Real e0 = vmm - 4.0 * vm + 3.0 * v0;
   const amrex::Real d1 = vm - 2.0 * v0 + vp;
   const amrex::Real e1 = vm - vp;
   const amrex::Real d2 = v0 - 2.0 * vp + vpp;
   const amrex::Real e2 = 3.0 * v0 - 4.0 * vp + vpp;
   const amrex::Real beta0 = (13.0 / 12.0) * d0 * d0 + 0.25 * e0 * e0;
   const amrex::Real beta1 = (13.0 / 12.0) * d1 * d1 + 0.25 * e1 * e1;
   const amrex::Real beta2 = (13.0 / 12.0) * d2 * d2 + 0.25 * e2 * e2;
   const amrex::Real tau5 = amrex::Math::abs(beta0 - beta2);
   const amrex::Real r0 = 1.0 + tau5 / (beta0 + TENO5_EPS);
   const amrex::Real r1 = 1.0 + tau5 / (beta1 + TENO5_EPS);
   const amrex::Real r2 = 1.0 + tau5 / (beta2 + TENO5_EPS);
   const amrex::Real g0 = r0 * r0 * r0 * r0 * r0 * r0;
   const amrex::Real g1 = r1 * r1 * r1 * r1 * r1 * r1;
   const amrex::Real g2 = r2 * r2 * r2 * r2 * r2 * r2;
   const amrex::Real inv_gamma = 1.0 / (g0 + g1 + g2);
   const amrex::Real a0 = (g0 * inv_gamma >= TENO5_CUTOFF) ? 0.1 : 0.0;
   const amrex::Real a1 = (g1 * inv_gamma >= TENO5_CUTOFF) ? 0.6 : 0.0;
   const amrex::Real a2 = (g2 * inv_gamma >= TENO5_CUTOFF) ? 0.3 : 0.0;
   const amrex::Real inv_alpha = 1.0 / (a0 + a1 + a2);
   const amrex::Real p0 = (2.0 * vmm - 7.0 * vm + 11.0 * v0) / 6.0;
   const amrex::Real p1 = (-vm + 5.0 * v0 + 2.0 * vp) / 6.0;
   const amrex::Real p2 = (2.0 * v0 + 5.0 * vp - vpp) / 6.0;
   return (a0 * p0 + a1 * p1 + a2 * p2) * inv_alpha;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real mp5_left(amrex::Real vmm, amrex::Real vm, amrex::Real v0, amrex::Real vp, amrex::Real vpp) noexcept
{
   const amrex::Real unlimited = (2.0 * vmm - 13.0 * vm + 47.0 * v0 + 27.0 * vp - 3.0 * vpp) / 60.0;
   const amrex::Real monotone = v0 + minmod(vp - v0, MP5_ALPHA * (v0 - vm));
   const amrex::Real scale = amrex::max(
      amrex::max(amrex::Math::abs(vmm), amrex::Math::abs(vm)),
      amrex::max(amrex::Math::abs(v0), amrex::max(amrex::Math::abs(vp), amrex::Math::abs(vpp)))
   );
   if ((unlimited - v0) * (unlimited - monotone) <= MP5_EPS * amrex::max(1.0, scale * scale)) {
      return unlimited;
   }

   const amrex::Real dm = vmm - 2.0 * vm + v0;
   const amrex::Real d0 = vm - 2.0 * v0 + vp;
   const amrex::Real dp = v0 - 2.0 * vp + vpp;
   const amrex::Real d_m4_left = minmod4(4.0 * dm - d0, 4.0 * d0 - dm, dm, d0);
   const amrex::Real d_m4_right = minmod4(4.0 * d0 - dp, 4.0 * dp - d0, d0, dp);
   const amrex::Real upper_limit = v0 + MP5_ALPHA * (v0 - vm);
   const amrex::Real average = 0.5 * (v0 + vp);
   const amrex::Real median = average - 0.5 * d_m4_right;
   const amrex::Real large_curvature = v0 + 0.5 * (v0 - vm) + (4.0 / 3.0) * d_m4_left;
   const amrex::Real lower_bound = amrex::max(
      amrex::min(amrex::min(v0, vp), median),
      amrex::min(amrex::min(v0, upper_limit), large_curvature)
   );
   const amrex::Real upper_bound = amrex::min(
      amrex::max(amrex::max(v0, vp), median),
      amrex::max(amrex::max(v0, upper_limit), large_curvature)
   );
   return unlimited + minmod(lower_bound - unlimited, upper_bound - unlimited);
}

template <int ReconMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real reconstruct_left(
   amrex::Real vmm,
   amrex::Real vm,
   amrex::Real v0,
   amrex::Real vp,
   amrex::Real vpp
) noexcept
{
   static_assert(is_five_point_reconstruction(ReconMethod));
   if constexpr (ReconMethod == ReconMP5 || ReconMethod == ReconMP5Char) {
      return mp5_left(vmm, vm, v0, vp, vpp);
   } else if constexpr (ReconMethod == ReconTENO5 || ReconMethod == ReconTENO5Char) {
      return teno5_left(vmm, vm, v0, vp, vpp);
   } else {
      return weno_z_left(vmm, vm, v0, vp, vpp);
   }
}

template <int ReconMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real reconstruct_right(
   amrex::Real vm,
   amrex::Real v0,
   amrex::Real vp,
   amrex::Real vpp,
   amrex::Real vppp
) noexcept
{
   return reconstruct_left<ReconMethod>(vppp, vpp, vp, v0, vm);
}

template <int ReconMethod, int Direction, int Component>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ScalarFaceValues five_point_primitive_component_face_values(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k
) noexcept
{
   const amrex::Real vmm = primitive_component_neighbor<Direction, Component>(state, i, j, k, -2);
   const amrex::Real vm = primitive_component_neighbor<Direction, Component>(state, i, j, k, -1);
   const amrex::Real v0 = primitive_component_neighbor<Direction, Component>(state, i, j, k, 0);
   const amrex::Real vp = primitive_component_neighbor<Direction, Component>(state, i, j, k, 1);
   const amrex::Real vpp = primitive_component_neighbor<Direction, Component>(state, i, j, k, 2);
   const amrex::Real vppp = primitive_component_neighbor<Direction, Component>(state, i, j, k, 3);
   return {
      reconstruct_left<ReconMethod>(vmm, vm, v0, vp, vpp),
      reconstruct_right<ReconMethod>(vm, v0, vp, vpp, vppp),
   };
}

template <int Direction, int Component>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ScalarFaceValues pwl_primitive_component_face_values(
   amrex::Array4<amrex::Real const> const& state,
   int il,
   int jl,
   int kl,
   int ir,
   int jr,
   int kr
) noexcept
{
   const amrex::Real qlm = primitive_component_neighbor<Direction, Component>(state, il, jl, kl, -1);
   const amrex::Real ql = state(il, jl, kl, Component);
   const amrex::Real qr = state(ir, jr, kr, Component);
   const amrex::Real qrp = primitive_component_neighbor<Direction, Component>(state, ir, jr, kr, 1);
   return {
      ql + 0.5 * mc_slope(qlm, ql, qr),
      qr - 0.5 * mc_slope(ql, qr, qrp),
   };
}

template <int ReconMethod, int Direction, int Component>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ScalarFaceValues primitive_component_face_values(
   amrex::Array4<amrex::Real const> const& state,
   int il,
   int jl,
   int kl,
   int ir,
   int jr,
   int kr
) noexcept
{
   if constexpr (ReconMethod == ReconPWL) {
      return pwl_primitive_component_face_values<Direction, Component>(state, il, jl, kl, ir, jr, kr);
   } else if constexpr (is_five_point_reconstruction(ReconMethod)) {
      static_cast<void>(ir);
      static_cast<void>(jr);
      static_cast<void>(kr);
      return five_point_primitive_component_face_values<ReconMethod, Direction, Component>(state, il, jl, kl);
   } else {
      return {
         state(il, jl, kl, Component),
         state(ir, jr, kr, Component),
      };
   }
}

template <int ReconMethod, int Component>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real reconstruct_left_component(
   HydroCharacteristics const& qmm,
   HydroCharacteristics const& qm,
   HydroCharacteristics const& q0,
   HydroCharacteristics const& qp,
   HydroCharacteristics const& qpp
) noexcept
{
   static_assert(Component >= 0 && Component < 5);
   if constexpr (Component == 0) {
      return reconstruct_left<ReconMethod>(qmm.entropy, qm.entropy, q0.entropy, qp.entropy, qpp.entropy);
   } else if constexpr (Component == 1) {
      return reconstruct_left<ReconMethod>(qmm.shear1, qm.shear1, q0.shear1, qp.shear1, qpp.shear1);
   } else if constexpr (Component == 2) {
      return reconstruct_left<ReconMethod>(qmm.shear2, qm.shear2, q0.shear2, qp.shear2, qpp.shear2);
   } else if constexpr (Component == 3) {
      return reconstruct_left<ReconMethod>(qmm.sound_minus, qm.sound_minus, q0.sound_minus, qp.sound_minus, qpp.sound_minus);
   } else {
      return reconstruct_left<ReconMethod>(qmm.sound_plus, qm.sound_plus, q0.sound_plus, qp.sound_plus, qpp.sound_plus);
   }
}

template <int ReconMethod, int Component>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real reconstruct_right_component(
   HydroCharacteristics const& qm,
   HydroCharacteristics const& q0,
   HydroCharacteristics const& qp,
   HydroCharacteristics const& qpp,
   HydroCharacteristics const& qppp
) noexcept
{
   static_assert(Component >= 0 && Component < 5);
   if constexpr (Component == 0) {
      return reconstruct_right<ReconMethod>(qm.entropy, q0.entropy, qp.entropy, qpp.entropy, qppp.entropy);
   } else if constexpr (Component == 1) {
      return reconstruct_right<ReconMethod>(qm.shear1, q0.shear1, qp.shear1, qpp.shear1, qppp.shear1);
   } else if constexpr (Component == 2) {
      return reconstruct_right<ReconMethod>(qm.shear2, q0.shear2, qp.shear2, qpp.shear2, qppp.shear2);
   } else if constexpr (Component == 3) {
      return reconstruct_right<ReconMethod>(qm.sound_minus, q0.sound_minus, qp.sound_minus, qpp.sound_minus, qppp.sound_minus);
   } else {
      return reconstruct_right<ReconMethod>(qm.sound_plus, q0.sound_plus, qp.sound_plus, qpp.sound_plus, qppp.sound_plus);
   }
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroCharacteristics hydro_characteristics(
   HydroPrimitive const& q,
   HydroCharacteristicBasis const& basis
) noexcept
{
   return generated::hydro_characteristics(
      {
         q.rho,
         hydro_normal_velocity<Direction>(q),
         hydro_tangent1_velocity<Direction>(q),
         hydro_tangent2_velocity<Direction>(q),
         q.pressure,
      },
      basis
   );
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitive hydro_primitive_from_characteristics(
   HydroCharacteristics const& w,
   HydroCharacteristicBasis const& basis
) noexcept
{
   const HydroNormalPrimitive primitive =
      generated::hydro_normal_primitive_from_characteristics(w, basis);
   return hydro_primitive_from_normal<Direction>(
      primitive.rho,
      primitive.vn,
      primitive.vt1,
      primitive.vt2,
      primitive.pressure
   );
}

template <int ReconMethod, int Component>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real reconstruct_left_component(
   MHDCharacteristics const& qmm,
   MHDCharacteristics const& qm,
   MHDCharacteristics const& q0,
   MHDCharacteristics const& qp,
   MHDCharacteristics const& qpp
) noexcept
{
   static_assert(Component >= 0 && Component < 7);
   if constexpr (Component == 0) {
      return reconstruct_left<ReconMethod>(qmm.fast_minus, qm.fast_minus, q0.fast_minus, qp.fast_minus, qpp.fast_minus);
   } else if constexpr (Component == 1) {
      return reconstruct_left<ReconMethod>(qmm.alfven_minus, qm.alfven_minus, q0.alfven_minus, qp.alfven_minus, qpp.alfven_minus);
   } else if constexpr (Component == 2) {
      return reconstruct_left<ReconMethod>(qmm.slow_minus, qm.slow_minus, q0.slow_minus, qp.slow_minus, qpp.slow_minus);
   } else if constexpr (Component == 3) {
      return reconstruct_left<ReconMethod>(qmm.entropy, qm.entropy, q0.entropy, qp.entropy, qpp.entropy);
   } else if constexpr (Component == 4) {
      return reconstruct_left<ReconMethod>(qmm.slow_plus, qm.slow_plus, q0.slow_plus, qp.slow_plus, qpp.slow_plus);
   } else if constexpr (Component == 5) {
      return reconstruct_left<ReconMethod>(qmm.alfven_plus, qm.alfven_plus, q0.alfven_plus, qp.alfven_plus, qpp.alfven_plus);
   } else {
      return reconstruct_left<ReconMethod>(qmm.fast_plus, qm.fast_plus, q0.fast_plus, qp.fast_plus, qpp.fast_plus);
   }
}

template <int ReconMethod, int Component>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real reconstruct_right_component(
   MHDCharacteristics const& qm,
   MHDCharacteristics const& q0,
   MHDCharacteristics const& qp,
   MHDCharacteristics const& qpp,
   MHDCharacteristics const& qppp
) noexcept
{
   static_assert(Component >= 0 && Component < 7);
   if constexpr (Component == 0) {
      return reconstruct_right<ReconMethod>(qm.fast_minus, q0.fast_minus, qp.fast_minus, qpp.fast_minus, qppp.fast_minus);
   } else if constexpr (Component == 1) {
      return reconstruct_right<ReconMethod>(qm.alfven_minus, q0.alfven_minus, qp.alfven_minus, qpp.alfven_minus, qppp.alfven_minus);
   } else if constexpr (Component == 2) {
      return reconstruct_right<ReconMethod>(qm.slow_minus, q0.slow_minus, qp.slow_minus, qpp.slow_minus, qppp.slow_minus);
   } else if constexpr (Component == 3) {
      return reconstruct_right<ReconMethod>(qm.entropy, q0.entropy, qp.entropy, qpp.entropy, qppp.entropy);
   } else if constexpr (Component == 4) {
      return reconstruct_right<ReconMethod>(qm.slow_plus, q0.slow_plus, qp.slow_plus, qpp.slow_plus, qppp.slow_plus);
   } else if constexpr (Component == 5) {
      return reconstruct_right<ReconMethod>(qm.alfven_plus, q0.alfven_plus, qp.alfven_plus, qpp.alfven_plus, qppp.alfven_plus);
   } else {
      return reconstruct_right<ReconMethod>(qm.fast_plus, q0.fast_plus, qp.fast_plus, qpp.fast_plus, qppp.fast_plus);
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDCharacteristics mhd_characteristics(MHDReducedPrimitive const& q, MHDCharacteristicBasis const& basis) noexcept
{
   return generated::mhd_characteristics(q, basis);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDReducedPrimitive mhd_reduced_from_characteristics(
   MHDCharacteristics const& w,
   MHDCharacteristicBasis const& basis
) noexcept
{
   return generated::mhd_reduced_from_characteristics(w, basis);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitive pwl_hydro_primitive(
   HydroPrimitive const& qm,
   HydroPrimitive const& q0,
   HydroPrimitive const& qp,
   amrex::Real side
) noexcept
{
   return {
      q0.rho + 0.5 * side * mc_slope(qm.rho, q0.rho, qp.rho),
      q0.v1 + 0.5 * side * mc_slope(qm.v1, q0.v1, qp.v1),
      q0.v2 + 0.5 * side * mc_slope(qm.v2, q0.v2, qp.v2),
      q0.v3 + 0.5 * side * mc_slope(qm.v3, q0.v3, qp.v3),
      q0.pressure + 0.5 * side * mc_slope(qm.pressure, q0.pressure, qp.pressure),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDPrimitive pwl_mhd_primitive(
   MHDPrimitive const& qm,
   MHDPrimitive const& q0,
   MHDPrimitive const& qp,
   amrex::Real side
) noexcept
{
   return {
      q0.rho + 0.5 * side * mc_slope(qm.rho, q0.rho, qp.rho),
      q0.v1 + 0.5 * side * mc_slope(qm.v1, q0.v1, qp.v1),
      q0.v2 + 0.5 * side * mc_slope(qm.v2, q0.v2, qp.v2),
      q0.v3 + 0.5 * side * mc_slope(qm.v3, q0.v3, qp.v3),
      q0.pressure + 0.5 * side * mc_slope(qm.pressure, q0.pressure, qp.pressure),
      q0.b1 + 0.5 * side * mc_slope(qm.b1, q0.b1, qp.b1),
      q0.b2 + 0.5 * side * mc_slope(qm.b2, q0.b2, qp.b2),
      q0.b3 + 0.5 * side * mc_slope(qm.b3, q0.b3, qp.b3),
   };
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitiveFaceStates hydro_pwl_primitive_face_states(
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
   const HydroPrimitive qlm = primitive_neighbor<Direction>(state, il, jl, kl, -1, eos);
   const HydroPrimitive ql = primitive_at(state, il, jl, kl, eos);
   const HydroPrimitive qr = primitive_at(state, ir, jr, kr, eos);
   const HydroPrimitive qrp = primitive_neighbor<Direction>(state, ir, jr, kr, 1, eos);
   return {
      pwl_hydro_primitive(qlm, ql, qr, 1.0),
      pwl_hydro_primitive(ql, qr, qrp, -1.0),
   };
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroFaceStates hydro_pwl_face_states(
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
   const HydroPrimitiveFaceStates face = hydro_pwl_primitive_face_states<Direction>(state, il, jl, kl, ir, jr, kr, eos);
   return {conserved_from_primitive<EOSMethod>(face.left, eos), conserved_from_primitive<EOSMethod>(face.right, eos)};
}

template <int ReconMethod, int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitiveFaceStates hydro_five_point_primitive_face_states(
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
   static_cast<void>(ir);
   static_cast<void>(jr);
   static_cast<void>(kr);
   static_cast<void>(eos);
   const ScalarFaceValues rho = five_point_primitive_component_face_values<ReconMethod, Direction, RHO>(state, il, jl, kl);
   const ScalarFaceValues vx = five_point_primitive_component_face_values<ReconMethod, Direction, MOM1>(state, il, jl, kl);
   const ScalarFaceValues vy = five_point_primitive_component_face_values<ReconMethod, Direction, MOM2>(state, il, jl, kl);
   const ScalarFaceValues vz = five_point_primitive_component_face_values<ReconMethod, Direction, MOM3>(state, il, jl, kl);
   const ScalarFaceValues pressure = five_point_primitive_component_face_values<ReconMethod, Direction, ENERGY>(state, il, jl, kl);
   return {
      {rho.left, vx.left, vy.left, vz.left, pressure.left},
      {rho.right, vx.right, vy.right, vz.right, pressure.right},
   };
}

template <int ReconMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroFaceStates hydro_five_point_face_states(
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
   const HydroPrimitiveFaceStates face = hydro_five_point_primitive_face_states<ReconMethod, Direction>(state, il, jl, kl, ir, jr, kr, eos);
   return {conserved_from_primitive<EOSMethod>(face.left, eos), conserved_from_primitive<EOSMethod>(face.right, eos)};
}

template <int ReconMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitiveFaceStates hydro_five_point_char_primitive_face_states(
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
   const HydroPrimitive q0 = primitive_neighbor<Direction>(state, il, jl, kl, -2, eos);
   const HydroPrimitive q1 = primitive_neighbor<Direction>(state, il, jl, kl, -1, eos);
   const HydroPrimitive q2 = primitive_at(state, il, jl, kl, eos);
   const HydroPrimitive q3 = primitive_at(state, ir, jr, kr, eos);
   const HydroPrimitive q4 = primitive_neighbor<Direction>(state, ir, jr, kr, 1, eos);
   const HydroPrimitive q5 = primitive_neighbor<Direction>(state, ir, jr, kr, 2, eos);

   const amrex::Real rho_ref = 0.5 * (q2.rho + q3.rho);
   const amrex::Real pressure_ref = 0.5 * (q2.pressure + q3.pressure);
   const amrex::Real sound_speed = sound_speed_from_primitive<EOSMethod>(rho_ref, pressure_ref, eos);
   const amrex::Real inv_rho = 1.0 / rho_ref;
   const amrex::Real inv_sound_speed = 1.0 / sound_speed;
   const HydroCharacteristicBasis basis{
      rho_ref,
      inv_rho,
      sound_speed,
      sound_speed * sound_speed,
      inv_sound_speed,
      inv_sound_speed * inv_sound_speed,
   };
   const HydroCharacteristics c0 = hydro_characteristics<Direction>(q0, basis);
   const HydroCharacteristics c1 = hydro_characteristics<Direction>(q1, basis);
   const HydroCharacteristics c2 = hydro_characteristics<Direction>(q2, basis);
   const HydroCharacteristics c3 = hydro_characteristics<Direction>(q3, basis);
   const HydroCharacteristics c4 = hydro_characteristics<Direction>(q4, basis);
   const HydroCharacteristics c5 = hydro_characteristics<Direction>(q5, basis);
   const HydroCharacteristics wl{
      reconstruct_left_component<ReconMethod, 0>(c0, c1, c2, c3, c4),
      reconstruct_left_component<ReconMethod, 1>(c0, c1, c2, c3, c4),
      reconstruct_left_component<ReconMethod, 2>(c0, c1, c2, c3, c4),
      reconstruct_left_component<ReconMethod, 3>(c0, c1, c2, c3, c4),
      reconstruct_left_component<ReconMethod, 4>(c0, c1, c2, c3, c4),
   };
   const HydroCharacteristics wr{
      reconstruct_right_component<ReconMethod, 0>(c1, c2, c3, c4, c5),
      reconstruct_right_component<ReconMethod, 1>(c1, c2, c3, c4, c5),
      reconstruct_right_component<ReconMethod, 2>(c1, c2, c3, c4, c5),
      reconstruct_right_component<ReconMethod, 3>(c1, c2, c3, c4, c5),
      reconstruct_right_component<ReconMethod, 4>(c1, c2, c3, c4, c5),
   };

   return {
      hydro_primitive_from_characteristics<Direction>(wl, basis),
      hydro_primitive_from_characteristics<Direction>(wr, basis),
   };
}

template <int ReconMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroFaceStates hydro_five_point_char_face_states(
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
   const HydroPrimitiveFaceStates face = hydro_five_point_char_primitive_face_states<ReconMethod, Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   return {conserved_from_primitive<EOSMethod>(face.left, eos), conserved_from_primitive<EOSMethod>(face.right, eos)};
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDPrimitiveFaceStates mhd_pwl_primitive_face_states(
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
   const MHDPrimitive qlm = mhd_primitive_neighbor<Direction>(state, il, jl, kl, -1, eos);
   const MHDPrimitive ql = mhd_primitive_at(state, il, jl, kl, eos);
   const MHDPrimitive qr = mhd_primitive_at(state, ir, jr, kr, eos);
   const MHDPrimitive qrp = mhd_primitive_neighbor<Direction>(state, ir, jr, kr, 1, eos);
   return {
      pwl_mhd_primitive(qlm, ql, qr, 1.0),
      pwl_mhd_primitive(ql, qr, qrp, -1.0),
   };
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDFaceStates mhd_pwl_face_states(
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
   const MHDPrimitiveFaceStates face = mhd_pwl_primitive_face_states<Direction>(state, il, jl, kl, ir, jr, kr, eos);
   return {conserved_from_primitive<EOSMethod>(face.left, eos), conserved_from_primitive<EOSMethod>(face.right, eos)};
}

template <int ReconMethod, int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDPrimitiveFaceStates mhd_five_point_primitive_face_states(
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
   static_cast<void>(ir);
   static_cast<void>(jr);
   static_cast<void>(kr);
   static_cast<void>(eos);
   const ScalarFaceValues rho = five_point_primitive_component_face_values<ReconMethod, Direction, RHO>(state, il, jl, kl);
   const ScalarFaceValues vx = five_point_primitive_component_face_values<ReconMethod, Direction, MOM1>(state, il, jl, kl);
   const ScalarFaceValues vy = five_point_primitive_component_face_values<ReconMethod, Direction, MOM2>(state, il, jl, kl);
   const ScalarFaceValues vz = five_point_primitive_component_face_values<ReconMethod, Direction, MOM3>(state, il, jl, kl);
   const ScalarFaceValues pressure = five_point_primitive_component_face_values<ReconMethod, Direction, ENERGY>(state, il, jl, kl);
   const ScalarFaceValues bx = five_point_primitive_component_face_values<ReconMethod, Direction, B1>(state, il, jl, kl);
   const ScalarFaceValues by = five_point_primitive_component_face_values<ReconMethod, Direction, B2>(state, il, jl, kl);
   const ScalarFaceValues bz = five_point_primitive_component_face_values<ReconMethod, Direction, B3>(state, il, jl, kl);
   return {
      {rho.left, vx.left, vy.left, vz.left, pressure.left, bx.left, by.left, bz.left},
      {rho.right, vx.right, vy.right, vz.right, pressure.right, bx.right, by.right, bz.right},
   };
}

template <int ReconMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDFaceStates mhd_five_point_face_states(
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
   const MHDPrimitiveFaceStates face = mhd_five_point_primitive_face_states<ReconMethod, Direction>(state, il, jl, kl, ir, jr, kr, eos);
   return {conserved_from_primitive<EOSMethod>(face.left, eos), conserved_from_primitive<EOSMethod>(face.right, eos)};
}

template <int ReconMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDPrimitiveFaceStates mhd_five_point_char_primitive_face_states(
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
   const MHDPrimitive q0 = mhd_primitive_neighbor<Direction>(state, il, jl, kl, -2, eos);
   const MHDPrimitive q1 = mhd_primitive_neighbor<Direction>(state, il, jl, kl, -1, eos);
   const MHDPrimitive q2 = mhd_primitive_at(state, il, jl, kl, eos);
   const MHDPrimitive q3 = mhd_primitive_at(state, ir, jr, kr, eos);
   const MHDPrimitive q4 = mhd_primitive_neighbor<Direction>(state, ir, jr, kr, 1, eos);
   const MHDPrimitive q5 = mhd_primitive_neighbor<Direction>(state, ir, jr, kr, 2, eos);

   const MHDReducedPrimitive r0 = mhd_reduced_primitive<Direction>(q0);
   const MHDReducedPrimitive r1 = mhd_reduced_primitive<Direction>(q1);
   const MHDReducedPrimitive r2 = mhd_reduced_primitive<Direction>(q2);
   const MHDReducedPrimitive r3 = mhd_reduced_primitive<Direction>(q3);
   const MHDReducedPrimitive r4 = mhd_reduced_primitive<Direction>(q4);
   const MHDReducedPrimitive r5 = mhd_reduced_primitive<Direction>(q5);

   const amrex::Real b_normal_left = reconstruct_left<ReconMethod>(
      mhd_normal_b<Direction>(q0),
      mhd_normal_b<Direction>(q1),
      mhd_normal_b<Direction>(q2),
      mhd_normal_b<Direction>(q3),
      mhd_normal_b<Direction>(q4)
   );
   const amrex::Real b_normal_right = reconstruct_right<ReconMethod>(
      mhd_normal_b<Direction>(q1),
      mhd_normal_b<Direction>(q2),
      mhd_normal_b<Direction>(q3),
      mhd_normal_b<Direction>(q4),
      mhd_normal_b<Direction>(q5)
   );
   const amrex::Real b_normal = 0.5 * (b_normal_left + b_normal_right);
   const MHDCharacteristicBasis basis = mhd_characteristic_basis<EOSMethod>(
      0.5 * (q2.rho + q3.rho),
      0.5 * (q2.pressure + q3.pressure),
      b_normal,
      0.5 * (r2.bt1 + r3.bt1),
      0.5 * (r2.bt2 + r3.bt2),
      eos
   );

   const MHDCharacteristics c0 = mhd_characteristics(r0, basis);
   const MHDCharacteristics c1 = mhd_characteristics(r1, basis);
   const MHDCharacteristics c2 = mhd_characteristics(r2, basis);
   const MHDCharacteristics c3 = mhd_characteristics(r3, basis);
   const MHDCharacteristics c4 = mhd_characteristics(r4, basis);
   const MHDCharacteristics c5 = mhd_characteristics(r5, basis);
   const MHDCharacteristics wl{
      reconstruct_left_component<ReconMethod, 0>(c0, c1, c2, c3, c4),
      reconstruct_left_component<ReconMethod, 1>(c0, c1, c2, c3, c4),
      reconstruct_left_component<ReconMethod, 2>(c0, c1, c2, c3, c4),
      reconstruct_left_component<ReconMethod, 3>(c0, c1, c2, c3, c4),
      reconstruct_left_component<ReconMethod, 4>(c0, c1, c2, c3, c4),
      reconstruct_left_component<ReconMethod, 5>(c0, c1, c2, c3, c4),
      reconstruct_left_component<ReconMethod, 6>(c0, c1, c2, c3, c4),
   };
   const MHDCharacteristics wr{
      reconstruct_right_component<ReconMethod, 0>(c1, c2, c3, c4, c5),
      reconstruct_right_component<ReconMethod, 1>(c1, c2, c3, c4, c5),
      reconstruct_right_component<ReconMethod, 2>(c1, c2, c3, c4, c5),
      reconstruct_right_component<ReconMethod, 3>(c1, c2, c3, c4, c5),
      reconstruct_right_component<ReconMethod, 4>(c1, c2, c3, c4, c5),
      reconstruct_right_component<ReconMethod, 5>(c1, c2, c3, c4, c5),
      reconstruct_right_component<ReconMethod, 6>(c1, c2, c3, c4, c5),
   };

   return {
      mhd_primitive_from_reduced<Direction>(mhd_reduced_from_characteristics(wl, basis), b_normal),
      mhd_primitive_from_reduced<Direction>(mhd_reduced_from_characteristics(wr, basis), b_normal),
   };
}

template <int ReconMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDFaceStates mhd_five_point_char_face_states(
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
   const MHDPrimitiveFaceStates face = mhd_five_point_char_primitive_face_states<ReconMethod, Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   return {conserved_from_primitive<EOSMethod>(face.left, eos), conserved_from_primitive<EOSMethod>(face.right, eos)};
}

template <int ReconMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitiveFaceStates hydro_primitive_face_states(
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
   HydroPrimitiveFaceStates face;
   if constexpr (ReconMethod == ReconPWL) {
      face = hydro_pwl_primitive_face_states<Direction>(state, il, jl, kl, ir, jr, kr, eos);
   } else if constexpr (is_characteristic_reconstruction(ReconMethod)) {
      face = hydro_five_point_char_primitive_face_states<ReconMethod, Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   } else if constexpr (is_five_point_reconstruction(ReconMethod)) {
      face = hydro_five_point_primitive_face_states<ReconMethod, Direction>(state, il, jl, kl, ir, jr, kr, eos);
   } else {
      face = {primitive_at(state, il, jl, kl, eos), primitive_at(state, ir, jr, kr, eos)};
   }
   return positivity_limit_hydro_face_states(
      face,
      primitive_at(state, il, jl, kl, eos),
      primitive_at(state, ir, jr, kr, eos),
      eos
   );
}

template <int ReconMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroFaceStates hydro_face_states(
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
   if constexpr (ReconMethod == ReconPWL) {
      return hydro_pwl_face_states<Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   } else if constexpr (is_characteristic_reconstruction(ReconMethod)) {
      return hydro_five_point_char_face_states<ReconMethod, Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   } else if constexpr (is_five_point_reconstruction(ReconMethod)) {
      return hydro_five_point_face_states<ReconMethod, Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   } else {
      return {
         conserved_from_primitive<EOSMethod>(primitive_at(state, il, jl, kl, eos), eos),
         conserved_from_primitive<EOSMethod>(primitive_at(state, ir, jr, kr, eos), eos)
      };
   }
}

template <int ReconMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDPrimitiveFaceStates mhd_primitive_face_states(
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
   MHDPrimitiveFaceStates face;
   if constexpr (ReconMethod == ReconPWL) {
      face = mhd_pwl_primitive_face_states<Direction>(state, il, jl, kl, ir, jr, kr, eos);
   } else if constexpr (is_characteristic_reconstruction(ReconMethod)) {
      face = mhd_five_point_char_primitive_face_states<ReconMethod, Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   } else if constexpr (is_five_point_reconstruction(ReconMethod)) {
      face = mhd_five_point_primitive_face_states<ReconMethod, Direction>(state, il, jl, kl, ir, jr, kr, eos);
   } else {
      face = {mhd_primitive_at(state, il, jl, kl, eos), mhd_primitive_at(state, ir, jr, kr, eos)};
   }
   return positivity_limit_mhd_face_states(
      face,
      mhd_primitive_at(state, il, jl, kl, eos),
      mhd_primitive_at(state, ir, jr, kr, eos),
      eos
   );
}

template <int ReconMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDFaceStates mhd_face_states(
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
   if constexpr (ReconMethod == ReconPWL) {
      return mhd_pwl_face_states<Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   } else if constexpr (is_characteristic_reconstruction(ReconMethod)) {
      return mhd_five_point_char_face_states<ReconMethod, Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   } else if constexpr (is_five_point_reconstruction(ReconMethod)) {
      return mhd_five_point_face_states<ReconMethod, Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   } else {
      return {
         conserved_from_primitive<EOSMethod>(mhd_primitive_at(state, il, jl, kl, eos), eos),
         conserved_from_primitive<EOSMethod>(mhd_primitive_at(state, ir, jr, kr, eos), eos)
      };
   }
}

} // namespace pyfluid_amrex
