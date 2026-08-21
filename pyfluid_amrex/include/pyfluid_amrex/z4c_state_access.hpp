#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/kernels_generated/differential_ops.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"
#include "pyfluid_amrex/spacetime_fields.hpp"
#include "pyfluid_amrex/z4c_state.hpp"

namespace pyfluid_amrex {

template <typename ComponentReader>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cState load_z4c_state_components(ComponentReader const& component) noexcept
{
   return {
      component(Z4C_CHI),
      component(Z4C_GAMMA_TILDE11),
      component(Z4C_GAMMA_TILDE12),
      component(Z4C_GAMMA_TILDE13),
      component(Z4C_GAMMA_TILDE22),
      component(Z4C_GAMMA_TILDE23),
      component(Z4C_GAMMA_TILDE33),
      component(Z4C_K_HAT),
      component(Z4C_A_TILDE11),
      component(Z4C_A_TILDE12),
      component(Z4C_A_TILDE13),
      component(Z4C_A_TILDE22),
      component(Z4C_A_TILDE23),
      component(Z4C_A_TILDE33),
      component(Z4C_GAMMA_TILDE1),
      component(Z4C_GAMMA_TILDE2),
      component(Z4C_GAMMA_TILDE3),
      component(Z4C_THETA),
      component(Z4C_LAPSE),
      component(Z4C_SHIFT1),
      component(Z4C_SHIFT2),
      component(Z4C_SHIFT3),
   };
}

template <typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cState load_z4c_state(Array const& state, int i, int j, int k) noexcept
{
   struct Reader {
      Array const& state;
      int i;
      int j;
      int k;

      AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
      amrex::Real operator()(int component) const noexcept
      {
         return state(i, j, k, component);
      }
   };
   return load_z4c_state_components(Reader{state, i, j, k});
}

template <int Order, int Direction, typename Array>
struct Z4cCellToFaceReader {
   Array const& state;
   int i;
   int j;
   int k;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real operator()(int component) const noexcept
   {
      return generated::cell_to_face_ordered<Order, Direction>(
         state, i, j, k, component
      );
   }
};

template <int Order, int Direction, typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SpacetimeMetric stored_spacetime_metric_at_face(
   Array const& cell_spacetime,
   int i,
   int j,
   int k
) noexcept
{
   const Z4cState state = load_z4c_state_components(
      Z4cCellToFaceReader<Order, Direction, Array>{cell_spacetime, i, j, k}
   );
   return generated::z4c_spacetime_metric(state);
}

template <typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_z4c_state(
   Array const& state,
   int i,
   int j,
   int k,
   Z4cState const& point
) noexcept
{
   state(i, j, k, Z4C_CHI) = point.chi;
   state(i, j, k, Z4C_GAMMA_TILDE11) = point.gamma_tilde11;
   state(i, j, k, Z4C_GAMMA_TILDE12) = point.gamma_tilde12;
   state(i, j, k, Z4C_GAMMA_TILDE13) = point.gamma_tilde13;
   state(i, j, k, Z4C_GAMMA_TILDE22) = point.gamma_tilde22;
   state(i, j, k, Z4C_GAMMA_TILDE23) = point.gamma_tilde23;
   state(i, j, k, Z4C_GAMMA_TILDE33) = point.gamma_tilde33;
   state(i, j, k, Z4C_K_HAT) = point.k_hat;
   state(i, j, k, Z4C_A_TILDE11) = point.a_tilde11;
   state(i, j, k, Z4C_A_TILDE12) = point.a_tilde12;
   state(i, j, k, Z4C_A_TILDE13) = point.a_tilde13;
   state(i, j, k, Z4C_A_TILDE22) = point.a_tilde22;
   state(i, j, k, Z4C_A_TILDE23) = point.a_tilde23;
   state(i, j, k, Z4C_A_TILDE33) = point.a_tilde33;
   state(i, j, k, Z4C_GAMMA_TILDE1) = point.gamma_tilde1;
   state(i, j, k, Z4C_GAMMA_TILDE2) = point.gamma_tilde2;
   state(i, j, k, Z4C_GAMMA_TILDE3) = point.gamma_tilde3;
   state(i, j, k, Z4C_THETA) = point.theta;
   state(i, j, k, Z4C_LAPSE) = point.lapse;
   state(i, j, k, Z4C_SHIFT1) = point.shift1;
   state(i, j, k, Z4C_SHIFT2) = point.shift2;
   state(i, j, k, Z4C_SHIFT3) = point.shift3;
}

template <typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void project_z4c_state_at(
   Array const& state,
   int i,
   int j,
   int k,
   amrex::Real chi_floor,
   amrex::Real lapse_floor
) noexcept
{
   Z4cState point = load_z4c_state(state, i, j, k);
   point.chi = amrex::max(point.chi, chi_floor);
   point.lapse = amrex::max(point.lapse, lapse_floor);
   store_z4c_state(
      state,
      i,
      j,
      k,
      generated::z4c_project_algebraic_constraints(point)
   );
}

} // namespace pyfluid_amrex
