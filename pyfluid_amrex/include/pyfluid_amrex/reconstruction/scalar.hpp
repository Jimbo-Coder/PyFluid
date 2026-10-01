#pragma once

#include <AMReX_Array4.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct ScalarFaceValues {
   amrex::Real left = 0.0_rt;
   amrex::Real right = 0.0_rt;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real minmod(amrex::Real a, amrex::Real b) noexcept
{
   if (a * b <= 0.0_rt) {
      return 0.0_rt;
   }
   return a > 0.0_rt ? amrex::min(a, b) : amrex::max(a, b);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real mc_slope(amrex::Real qm, amrex::Real q0, amrex::Real qp) noexcept
{
   const amrex::Real centered = 0.5_rt * (qp - qm);
   const amrex::Real limited = minmod(2.0_rt * (q0 - qm), 2.0_rt * (qp - q0));
   return minmod(centered, limited);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real positivity_limiter_theta(amrex::Real face, amrex::Real center,
                                                                              amrex::Real floor) noexcept
{
   if (face >= floor) {
      return 1.0_rt;
   }
   if (!amrex::Math::isfinite(face) || !amrex::Math::isfinite(center) || center <= floor) {
      return 0.0_rt;
   }
   const amrex::Real denominator = center - face;
   if (denominator <= 0.0_rt) {
      return 0.0_rt;
   }
   return amrex::max(0.0_rt, amrex::min(1.0_rt, (center - floor) / denominator));
}

template <int Direction, int Component>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
primitive_component_neighbor(amrex::Array4<amrex::Real const> const& state, int i, int j, int k, int offset) noexcept
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

template <int ReconstructionMethod> struct FivePointReconstruction;

template <int ReconstructionMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real reconstruct_left(amrex::Real vmm, amrex::Real vm, amrex::Real v0,
                                                                      amrex::Real vp, amrex::Real vpp) noexcept
{
   static_assert(is_five_point_reconstruction(ReconstructionMethod));
   return FivePointReconstruction<ReconstructionMethod>::left(vmm, vm, v0, vp, vpp);
}

template <int ReconstructionMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real reconstruct_right(amrex::Real vm, amrex::Real v0, amrex::Real vp,
                                                                       amrex::Real vpp, amrex::Real vppp) noexcept
{
   return reconstruct_left<ReconstructionMethod>(vppp, vpp, vp, v0, vm);
}

template <int ReconstructionMethod, int Direction, int Component>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ScalarFaceValues
five_point_primitive_component_face_values(amrex::Array4<amrex::Real const> const& state, int i, int j, int k) noexcept
{
   const amrex::Real vmm = primitive_component_neighbor<Direction, Component>(state, i, j, k, -2);
   const amrex::Real vm = primitive_component_neighbor<Direction, Component>(state, i, j, k, -1);
   const amrex::Real v0 = primitive_component_neighbor<Direction, Component>(state, i, j, k, 0);
   const amrex::Real vp = primitive_component_neighbor<Direction, Component>(state, i, j, k, 1);
   const amrex::Real vpp = primitive_component_neighbor<Direction, Component>(state, i, j, k, 2);
   const amrex::Real vppp = primitive_component_neighbor<Direction, Component>(state, i, j, k, 3);
   return {
      reconstruct_left<ReconstructionMethod>(vmm, vm, v0, vp, vpp),
      reconstruct_right<ReconstructionMethod>(vm, v0, vp, vpp, vppp),
   };
}

template <int Direction, int Component>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ScalarFaceValues pwl_primitive_component_face_values(
   amrex::Array4<amrex::Real const> const& state, int il, int jl, int kl, int ir, int jr, int kr) noexcept
{
   const amrex::Real qlm = primitive_component_neighbor<Direction, Component>(state, il, jl, kl, -1);
   const amrex::Real ql = state(il, jl, kl, Component);
   const amrex::Real qr = state(ir, jr, kr, Component);
   const amrex::Real qrp = primitive_component_neighbor<Direction, Component>(state, ir, jr, kr, 1);
   return {
      ql + 0.5_rt * mc_slope(qlm, ql, qr),
      qr - 0.5_rt * mc_slope(ql, qr, qrp),
   };
}

template <int ReconstructionMethod, int Direction, int Component>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ScalarFaceValues primitive_component_face_values(
   amrex::Array4<amrex::Real const> const& state, int il, int jl, int kl, int ir, int jr, int kr) noexcept
{
   if constexpr (ReconstructionMethod == ReconPWL) {
      return pwl_primitive_component_face_values<Direction, Component>(state, il, jl, kl, ir, jr, kr);
   } else if constexpr (is_five_point_reconstruction(ReconstructionMethod)) {
      static_cast<void>(ir);
      static_cast<void>(jr);
      static_cast<void>(kr);
      return five_point_primitive_component_face_values<ReconstructionMethod, Direction, Component>(state, il, jl, kl);
   } else {
      return {
         state(il, jl, kl, Component),
         state(ir, jr, kr, Component),
      };
   }
}

} // namespace pyfluid_amrex
