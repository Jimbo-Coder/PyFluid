#pragma once

#include <AMReX_Array4.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

constexpr amrex::Real WENO_Z_EPS = 1.0e-8;

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

struct HydroCharacteristics {
   amrex::Real entropy = 0.0;
   amrex::Real shear1 = 0.0;
   amrex::Real shear2 = 0.0;
   amrex::Real sound_minus = 0.0;
   amrex::Real sound_plus = 0.0;
};

struct MHDReducedPrimitive {
   amrex::Real rho = 0.0;
   amrex::Real vn = 0.0;
   amrex::Real vt1 = 0.0;
   amrex::Real vt2 = 0.0;
   amrex::Real pressure = 0.0;
   amrex::Real bt1 = 0.0;
   amrex::Real bt2 = 0.0;
};

struct MHDCharacteristics {
   amrex::Real fast_minus = 0.0;
   amrex::Real alfven_minus = 0.0;
   amrex::Real slow_minus = 0.0;
   amrex::Real entropy = 0.0;
   amrex::Real slow_plus = 0.0;
   amrex::Real alfven_plus = 0.0;
   amrex::Real fast_plus = 0.0;
};

struct MHDCharacteristicBasis {
   amrex::Real rho = 0.0;
   amrex::Real sqrt_rho = 0.0;
   amrex::Real sound_speed = 0.0;
   amrex::Real fast_speed = 0.0;
   amrex::Real slow_speed = 0.0;
   amrex::Real alpha_f = 0.0;
   amrex::Real alpha_s = 0.0;
   amrex::Real beta_t1 = 0.0;
   amrex::Real beta_t2 = 0.0;
   amrex::Real normal_sign = 1.0;
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

template <int Direction, int Component>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ScalarFaceValues weno_z_primitive_component_face_values(
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
      weno_z_left(vmm, vm, v0, vp, vpp),
      weno_z_right(vm, v0, vp, vpp, vppp),
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
   } else if constexpr (ReconMethod == ReconWENOZ || ReconMethod == ReconWENOZChar) {
      static_cast<void>(ir);
      static_cast<void>(jr);
      static_cast<void>(kr);
      return weno_z_primitive_component_face_values<Direction, Component>(state, il, jl, kl);
   } else {
      return {
         state(il, jl, kl, Component),
         state(ir, jr, kr, Component),
      };
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real weno_z_left_component(
   HydroCharacteristics const& qmm,
   HydroCharacteristics const& qm,
   HydroCharacteristics const& q0,
   HydroCharacteristics const& qp,
   HydroCharacteristics const& qpp,
   int component
) noexcept
{
   if (component == 0) {
      return weno_z_left(qmm.entropy, qm.entropy, q0.entropy, qp.entropy, qpp.entropy);
   }
   if (component == 1) {
      return weno_z_left(qmm.shear1, qm.shear1, q0.shear1, qp.shear1, qpp.shear1);
   }
   if (component == 2) {
      return weno_z_left(qmm.shear2, qm.shear2, q0.shear2, qp.shear2, qpp.shear2);
   }
   if (component == 3) {
      return weno_z_left(qmm.sound_minus, qm.sound_minus, q0.sound_minus, qp.sound_minus, qpp.sound_minus);
   }
   return weno_z_left(qmm.sound_plus, qm.sound_plus, q0.sound_plus, qp.sound_plus, qpp.sound_plus);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real weno_z_right_component(
   HydroCharacteristics const& qm,
   HydroCharacteristics const& q0,
   HydroCharacteristics const& qp,
   HydroCharacteristics const& qpp,
   HydroCharacteristics const& qppp,
   int component
) noexcept
{
   if (component == 0) {
      return weno_z_right(qm.entropy, q0.entropy, qp.entropy, qpp.entropy, qppp.entropy);
   }
   if (component == 1) {
      return weno_z_right(qm.shear1, q0.shear1, qp.shear1, qpp.shear1, qppp.shear1);
   }
   if (component == 2) {
      return weno_z_right(qm.shear2, q0.shear2, qp.shear2, qpp.shear2, qppp.shear2);
   }
   if (component == 3) {
      return weno_z_right(qm.sound_minus, q0.sound_minus, qp.sound_minus, qpp.sound_minus, qppp.sound_minus);
   }
   return weno_z_right(qm.sound_plus, q0.sound_plus, qp.sound_plus, qpp.sound_plus, qppp.sound_plus);
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroCharacteristics hydro_characteristics(
   HydroPrimitive const& q,
   amrex::Real rho_ref,
   amrex::Real sound_speed
) noexcept
{
   const amrex::Real inv_sound2 = 1.0 / (sound_speed * sound_speed);
   const amrex::Real vn = hydro_normal_velocity<Direction>(q);
   return {
      q.rho - q.pressure * inv_sound2,
      hydro_tangent1_velocity<Direction>(q),
      hydro_tangent2_velocity<Direction>(q),
      0.5 * (q.pressure * inv_sound2 - rho_ref * vn / sound_speed),
      0.5 * (q.pressure * inv_sound2 + rho_ref * vn / sound_speed),
   };
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitive hydro_primitive_from_characteristics(
   HydroCharacteristics const& w,
   amrex::Real rho_ref,
   amrex::Real sound_speed
) noexcept
{
   const amrex::Real rho = w.entropy + w.sound_minus + w.sound_plus;
   const amrex::Real vn = sound_speed * (w.sound_plus - w.sound_minus) / rho_ref;
   const amrex::Real pressure = sound_speed * sound_speed * (w.sound_minus + w.sound_plus);
   return hydro_primitive_from_normal<Direction>(rho, vn, w.shear1, w.shear2, pressure);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real weno_z_left_component(
   MHDCharacteristics const& qmm,
   MHDCharacteristics const& qm,
   MHDCharacteristics const& q0,
   MHDCharacteristics const& qp,
   MHDCharacteristics const& qpp,
   int component
) noexcept
{
   if (component == 0) {
      return weno_z_left(qmm.fast_minus, qm.fast_minus, q0.fast_minus, qp.fast_minus, qpp.fast_minus);
   }
   if (component == 1) {
      return weno_z_left(qmm.alfven_minus, qm.alfven_minus, q0.alfven_minus, qp.alfven_minus, qpp.alfven_minus);
   }
   if (component == 2) {
      return weno_z_left(qmm.slow_minus, qm.slow_minus, q0.slow_minus, qp.slow_minus, qpp.slow_minus);
   }
   if (component == 3) {
      return weno_z_left(qmm.entropy, qm.entropy, q0.entropy, qp.entropy, qpp.entropy);
   }
   if (component == 4) {
      return weno_z_left(qmm.slow_plus, qm.slow_plus, q0.slow_plus, qp.slow_plus, qpp.slow_plus);
   }
   if (component == 5) {
      return weno_z_left(qmm.alfven_plus, qm.alfven_plus, q0.alfven_plus, qp.alfven_plus, qpp.alfven_plus);
   }
   return weno_z_left(qmm.fast_plus, qm.fast_plus, q0.fast_plus, qp.fast_plus, qpp.fast_plus);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real weno_z_right_component(
   MHDCharacteristics const& qm,
   MHDCharacteristics const& q0,
   MHDCharacteristics const& qp,
   MHDCharacteristics const& qpp,
   MHDCharacteristics const& qppp,
   int component
) noexcept
{
   if (component == 0) {
      return weno_z_right(qm.fast_minus, q0.fast_minus, qp.fast_minus, qpp.fast_minus, qppp.fast_minus);
   }
   if (component == 1) {
      return weno_z_right(qm.alfven_minus, q0.alfven_minus, qp.alfven_minus, qpp.alfven_minus, qppp.alfven_minus);
   }
   if (component == 2) {
      return weno_z_right(qm.slow_minus, q0.slow_minus, qp.slow_minus, qpp.slow_minus, qppp.slow_minus);
   }
   if (component == 3) {
      return weno_z_right(qm.entropy, q0.entropy, qp.entropy, qpp.entropy, qppp.entropy);
   }
   if (component == 4) {
      return weno_z_right(qm.slow_plus, q0.slow_plus, qp.slow_plus, qpp.slow_plus, qppp.slow_plus);
   }
   if (component == 5) {
      return weno_z_right(qm.alfven_plus, q0.alfven_plus, qp.alfven_plus, qpp.alfven_plus, qppp.alfven_plus);
   }
   return weno_z_right(qm.fast_plus, q0.fast_plus, qp.fast_plus, qpp.fast_plus, qppp.fast_plus);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDCharacteristicBasis mhd_characteristic_basis(
   amrex::Real rho_ref,
   amrex::Real pressure_ref,
   amrex::Real b_normal,
   amrex::Real b_tangent1,
   amrex::Real b_tangent2,
   EOSParams const& eos
) noexcept
{
   const amrex::Real sound_speed = sound_speed_from_primitive<EOSMethod>(rho_ref, pressure_ref, eos);
   const amrex::Real sound2 = sound_speed * sound_speed;
   const amrex::Real b2 = (b_normal * b_normal + b_tangent1 * b_tangent1 + b_tangent2 * b_tangent2) / rho_ref;
   const amrex::Real ca2 = b_normal * b_normal / rho_ref;
   amrex::Real discr = (sound2 + b2) * (sound2 + b2) - 4.0 * sound2 * ca2;
   if (discr < 0.0 && discr > -1.0e-14) {
      discr = 0.0;
   }
   discr = amrex::max(discr, 0.0);
   const amrex::Real root = std::sqrt(discr);
   const amrex::Real fast2 = amrex::max(0.5 * (sound2 + b2 + root), 0.0);
   const amrex::Real slow2 = amrex::max(0.5 * (sound2 + b2 - root), 0.0);
   const amrex::Real b_perp = std::sqrt(b_tangent1 * b_tangent1 + b_tangent2 * b_tangent2);

   amrex::Real beta_t1 = 1.0 / std::sqrt(2.0);
   amrex::Real beta_t2 = beta_t1;
   if (b_perp > 1.0e-14) {
      beta_t1 = b_tangent1 / b_perp;
      beta_t2 = b_tangent2 / b_perp;
   }

   amrex::Real alpha_f = 1.0;
   amrex::Real alpha_s = 0.0;
   const amrex::Real denom = fast2 - slow2;
   if (denom > 1.0e-14) {
      alpha_f = std::sqrt(amrex::max((sound2 - slow2) / denom, 0.0));
      alpha_s = std::sqrt(amrex::max((fast2 - sound2) / denom, 0.0));
   }

   return {
      rho_ref,
      std::sqrt(rho_ref),
      sound_speed,
      std::sqrt(fast2),
      std::sqrt(slow2),
      alpha_f,
      alpha_s,
      beta_t1,
      beta_t2,
      b_normal < 0.0 ? -1.0 : 1.0,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDCharacteristics mhd_characteristics(MHDReducedPrimitive const& q, MHDCharacteristicBasis const& basis) noexcept
{
   const amrex::Real inv_sound2 = 1.0 / (basis.sound_speed * basis.sound_speed);
   const amrex::Real compressive = q.pressure * inv_sound2 / basis.rho;
   const amrex::Real b_parallel = (
      basis.beta_t1 * q.bt1 + basis.beta_t2 * q.bt2
   ) / (basis.sound_speed * basis.sqrt_rho);
   const amrex::Real fast_sum = basis.alpha_f * compressive + basis.alpha_s * b_parallel;
   const amrex::Real slow_sum = basis.alpha_s * compressive - basis.alpha_f * b_parallel;

   amrex::Real denom = (
      basis.alpha_f * basis.alpha_f * basis.fast_speed * basis.fast_speed
      + basis.alpha_s * basis.alpha_s * basis.slow_speed * basis.slow_speed
   );
   if (denom <= 1.0e-14) {
      denom = basis.sound_speed * basis.sound_speed;
   }

   const amrex::Real v_parallel = basis.normal_sign * (basis.beta_t1 * q.vt1 + basis.beta_t2 * q.vt2);
   const amrex::Real fast_diff = (
      basis.alpha_f * basis.fast_speed * q.vn
      - basis.alpha_s * basis.slow_speed * v_parallel
   ) / denom;
   const amrex::Real slow_diff = (
      basis.alpha_s * basis.slow_speed * q.vn
      + basis.alpha_f * basis.fast_speed * v_parallel
   ) / denom;
   const amrex::Real alfven_sum = (basis.beta_t2 * q.bt1 - basis.beta_t1 * q.bt2) / basis.sqrt_rho;
   const amrex::Real alfven_diff = basis.normal_sign * (basis.beta_t2 * q.vt1 - basis.beta_t1 * q.vt2);

   return {
      0.5 * (fast_sum - fast_diff),
      0.5 * (alfven_sum + alfven_diff),
      0.5 * (slow_sum - slow_diff),
      q.rho - q.pressure * inv_sound2,
      0.5 * (slow_sum + slow_diff),
      0.5 * (alfven_sum - alfven_diff),
      0.5 * (fast_sum + fast_diff),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDReducedPrimitive mhd_reduced_from_characteristics(
   MHDCharacteristics const& w,
   MHDCharacteristicBasis const& basis
) noexcept
{
   const amrex::Real fast_sum = w.fast_minus + w.fast_plus;
   const amrex::Real fast_diff = w.fast_plus - w.fast_minus;
   const amrex::Real slow_sum = w.slow_minus + w.slow_plus;
   const amrex::Real slow_diff = w.slow_plus - w.slow_minus;
   const amrex::Real alfven_sum = w.alfven_minus + w.alfven_plus;
   const amrex::Real alfven_diff = w.alfven_minus - w.alfven_plus;
   const amrex::Real sound2 = basis.sound_speed * basis.sound_speed;

   return {
      basis.alpha_f * basis.rho * fast_sum + basis.alpha_s * basis.rho * slow_sum + w.entropy,
      basis.alpha_f * basis.fast_speed * fast_diff + basis.alpha_s * basis.slow_speed * slow_diff,
      basis.normal_sign * (
         -basis.alpha_s * basis.slow_speed * basis.beta_t1 * fast_diff
         + basis.alpha_f * basis.fast_speed * basis.beta_t1 * slow_diff
         + basis.beta_t2 * alfven_diff
      ),
      basis.normal_sign * (
         -basis.alpha_s * basis.slow_speed * basis.beta_t2 * fast_diff
         + basis.alpha_f * basis.fast_speed * basis.beta_t2 * slow_diff
         - basis.beta_t1 * alfven_diff
      ),
      basis.alpha_f * basis.rho * sound2 * fast_sum + basis.alpha_s * basis.rho * sound2 * slow_sum,
      (
         basis.alpha_s * basis.sound_speed * basis.sqrt_rho * basis.beta_t1 * fast_sum
         - basis.alpha_f * basis.sound_speed * basis.sqrt_rho * basis.beta_t1 * slow_sum
         + basis.beta_t2 * basis.sqrt_rho * alfven_sum
      ),
      (
         basis.alpha_s * basis.sound_speed * basis.sqrt_rho * basis.beta_t2 * fast_sum
         - basis.alpha_f * basis.sound_speed * basis.sqrt_rho * basis.beta_t2 * slow_sum
         - basis.beta_t1 * basis.sqrt_rho * alfven_sum
      ),
   };
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

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitiveFaceStates hydro_wenoz_primitive_face_states(
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
   const ScalarFaceValues rho = weno_z_primitive_component_face_values<Direction, RHO>(state, il, jl, kl);
   const ScalarFaceValues vx = weno_z_primitive_component_face_values<Direction, MOM1>(state, il, jl, kl);
   const ScalarFaceValues vy = weno_z_primitive_component_face_values<Direction, MOM2>(state, il, jl, kl);
   const ScalarFaceValues vz = weno_z_primitive_component_face_values<Direction, MOM3>(state, il, jl, kl);
   const ScalarFaceValues pressure = weno_z_primitive_component_face_values<Direction, ENERGY>(state, il, jl, kl);
   return {
      {rho.left, vx.left, vy.left, vz.left, pressure.left},
      {rho.right, vx.right, vy.right, vz.right, pressure.right},
   };
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroFaceStates hydro_wenoz_face_states(
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
   const HydroPrimitiveFaceStates face = hydro_wenoz_primitive_face_states<Direction>(state, il, jl, kl, ir, jr, kr, eos);
   return {conserved_from_primitive<EOSMethod>(face.left, eos), conserved_from_primitive<EOSMethod>(face.right, eos)};
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroPrimitiveFaceStates hydro_wenoz_char_primitive_face_states(
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
   const HydroCharacteristics c0 = hydro_characteristics<Direction>(q0, rho_ref, sound_speed);
   const HydroCharacteristics c1 = hydro_characteristics<Direction>(q1, rho_ref, sound_speed);
   const HydroCharacteristics c2 = hydro_characteristics<Direction>(q2, rho_ref, sound_speed);
   const HydroCharacteristics c3 = hydro_characteristics<Direction>(q3, rho_ref, sound_speed);
   const HydroCharacteristics c4 = hydro_characteristics<Direction>(q4, rho_ref, sound_speed);
   const HydroCharacteristics c5 = hydro_characteristics<Direction>(q5, rho_ref, sound_speed);
   const HydroCharacteristics wl{
      weno_z_left_component(c0, c1, c2, c3, c4, 0),
      weno_z_left_component(c0, c1, c2, c3, c4, 1),
      weno_z_left_component(c0, c1, c2, c3, c4, 2),
      weno_z_left_component(c0, c1, c2, c3, c4, 3),
      weno_z_left_component(c0, c1, c2, c3, c4, 4),
   };
   const HydroCharacteristics wr{
      weno_z_right_component(c1, c2, c3, c4, c5, 0),
      weno_z_right_component(c1, c2, c3, c4, c5, 1),
      weno_z_right_component(c1, c2, c3, c4, c5, 2),
      weno_z_right_component(c1, c2, c3, c4, c5, 3),
      weno_z_right_component(c1, c2, c3, c4, c5, 4),
   };

   return {
      hydro_primitive_from_characteristics<Direction>(wl, rho_ref, sound_speed),
      hydro_primitive_from_characteristics<Direction>(wr, rho_ref, sound_speed),
   };
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroFaceStates hydro_wenoz_char_face_states(
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
   const HydroPrimitiveFaceStates face = hydro_wenoz_char_primitive_face_states<Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
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

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDPrimitiveFaceStates mhd_wenoz_primitive_face_states(
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
   const ScalarFaceValues rho = weno_z_primitive_component_face_values<Direction, RHO>(state, il, jl, kl);
   const ScalarFaceValues vx = weno_z_primitive_component_face_values<Direction, MOM1>(state, il, jl, kl);
   const ScalarFaceValues vy = weno_z_primitive_component_face_values<Direction, MOM2>(state, il, jl, kl);
   const ScalarFaceValues vz = weno_z_primitive_component_face_values<Direction, MOM3>(state, il, jl, kl);
   const ScalarFaceValues pressure = weno_z_primitive_component_face_values<Direction, ENERGY>(state, il, jl, kl);
   const ScalarFaceValues bx = weno_z_primitive_component_face_values<Direction, B1>(state, il, jl, kl);
   const ScalarFaceValues by = weno_z_primitive_component_face_values<Direction, B2>(state, il, jl, kl);
   const ScalarFaceValues bz = weno_z_primitive_component_face_values<Direction, B3>(state, il, jl, kl);
   return {
      {rho.left, vx.left, vy.left, vz.left, pressure.left, bx.left, by.left, bz.left},
      {rho.right, vx.right, vy.right, vz.right, pressure.right, bx.right, by.right, bz.right},
   };
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDFaceStates mhd_wenoz_face_states(
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
   const MHDPrimitiveFaceStates face = mhd_wenoz_primitive_face_states<Direction>(state, il, jl, kl, ir, jr, kr, eos);
   return {conserved_from_primitive<EOSMethod>(face.left, eos), conserved_from_primitive<EOSMethod>(face.right, eos)};
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDPrimitiveFaceStates mhd_wenoz_char_primitive_face_states(
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

   const amrex::Real b_normal_left = weno_z_left(
      mhd_normal_b<Direction>(q0),
      mhd_normal_b<Direction>(q1),
      mhd_normal_b<Direction>(q2),
      mhd_normal_b<Direction>(q3),
      mhd_normal_b<Direction>(q4)
   );
   const amrex::Real b_normal_right = weno_z_right(
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
      weno_z_left_component(c0, c1, c2, c3, c4, 0),
      weno_z_left_component(c0, c1, c2, c3, c4, 1),
      weno_z_left_component(c0, c1, c2, c3, c4, 2),
      weno_z_left_component(c0, c1, c2, c3, c4, 3),
      weno_z_left_component(c0, c1, c2, c3, c4, 4),
      weno_z_left_component(c0, c1, c2, c3, c4, 5),
      weno_z_left_component(c0, c1, c2, c3, c4, 6),
   };
   const MHDCharacteristics wr{
      weno_z_right_component(c1, c2, c3, c4, c5, 0),
      weno_z_right_component(c1, c2, c3, c4, c5, 1),
      weno_z_right_component(c1, c2, c3, c4, c5, 2),
      weno_z_right_component(c1, c2, c3, c4, c5, 3),
      weno_z_right_component(c1, c2, c3, c4, c5, 4),
      weno_z_right_component(c1, c2, c3, c4, c5, 5),
      weno_z_right_component(c1, c2, c3, c4, c5, 6),
   };

   return {
      mhd_primitive_from_reduced<Direction>(mhd_reduced_from_characteristics(wl, basis), b_normal),
      mhd_primitive_from_reduced<Direction>(mhd_reduced_from_characteristics(wr, basis), b_normal),
   };
}

template <int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDFaceStates mhd_wenoz_char_face_states(
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
   const MHDPrimitiveFaceStates face = mhd_wenoz_char_primitive_face_states<Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
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
   } else if constexpr (ReconMethod == ReconWENOZChar) {
      face = hydro_wenoz_char_primitive_face_states<Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   } else if constexpr (ReconMethod == ReconWENOZ) {
      face = hydro_wenoz_primitive_face_states<Direction>(state, il, jl, kl, ir, jr, kr, eos);
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
   } else if constexpr (ReconMethod == ReconWENOZChar) {
      return hydro_wenoz_char_face_states<Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   } else if constexpr (ReconMethod == ReconWENOZ) {
      return hydro_wenoz_face_states<Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
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
   } else if constexpr (ReconMethod == ReconWENOZChar) {
      face = mhd_wenoz_char_primitive_face_states<Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   } else if constexpr (ReconMethod == ReconWENOZ) {
      face = mhd_wenoz_primitive_face_states<Direction>(state, il, jl, kl, ir, jr, kr, eos);
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
   } else if constexpr (ReconMethod == ReconWENOZChar) {
      return mhd_wenoz_char_face_states<Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   } else if constexpr (ReconMethod == ReconWENOZ) {
      return mhd_wenoz_face_states<Direction, EOSMethod>(state, il, jl, kl, ir, jr, kr, eos);
   } else {
      return {
         conserved_from_primitive<EOSMethod>(mhd_primitive_at(state, il, jl, kl, eos), eos),
         conserved_from_primitive<EOSMethod>(mhd_primitive_at(state, ir, jr, kr, eos), eos)
      };
   }
}

} // namespace pyfluid_amrex
