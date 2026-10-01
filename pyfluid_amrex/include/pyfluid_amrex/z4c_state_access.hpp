#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/kernels_generated/differential_ops.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"
#include "pyfluid_amrex/spacetime_fields.hpp"
#include "pyfluid_amrex/z4c_state.hpp"
#include "pyfluid_amrex/z4c_centering_representation.hpp"
#include "pyfluid_amrex/excision_geometry.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

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
      return represented_cell_to_face_ordered<
         Z4cPositiveScalarCenteringRepresentation, Order, Direction
      >(
         state, i, j, k, component
      );
   }
};

template <int Order, int Direction, typename Array>
struct Z4cExteriorCellToFaceReader {
   Array const& state;
   int i, j, k;
   amrex::GpuArray<int, 3> first, last;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real operator()(int component) const noexcept
   {
      const int index[3]{i, j, k};
      const int first_offset = first[Direction] - index[Direction];
      amrex::Real value = 0.0_rt;
      for (int ordinal = 0; ordinal < Order; ++ordinal) {
         value += generated::bounded_centering_transfer_axis_weight<Order>(
            0, 1 << Direction, Direction, first_offset, ordinal)
            * Z4cPositiveScalarCenteringRepresentation::encode(
               component,
               generated::sample_direction<Direction>(
                  state, i, j, k, component, first_offset + ordinal
               )
            );
      }
      return Z4cPositiveScalarCenteringRepresentation::decode(component, value);
   }
};

// Keep the boundary closure out of every reconstruction/Riemann inlining
// product. The same small per-order/direction routine serves all methods;
// this header-owned device implementation does not require CUDA RDC.
template <int Order, int Direction, typename Array>
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE inline
SpacetimeMetric stored_exterior_spacetime_metric_at_face(
   Array const& cell_spacetime,
   int i,
   int j,
   int k,
   ExcisionGridView const& excision
) noexcept
{
   if (excision.region.enabled()) {
      amrex::GpuArray<int, 3> left{{i, j, k}};
      --left[Direction];
      const bool left_active = excision(left[0], left[1], left[2], 0);
      const bool right_active = excision(i, j, k, 0);
      if (!left_active && !right_active) {
         // A wholly inactive face has no physical flux. Its metric is only
         // finite scratch for the common face loop, which is overwritten.
         return generated::z4c_spacetime_metric(load_z4c_state(cell_spacetime, i, j, k));
      }
      amrex::GpuArray<int, 3> first{{i, j, k}}, last{{i, j, k}};
      first[Direction] -= Order / 2;
      last[Direction] = first[Direction] + Order - 1;
      if (excision.region.intersects(excision.position(first[0], first[1], first[2]),
                                    excision.position(last[0], last[1], last[2]))) {
         auto target = excision.position(i, j, k);
         target[Direction] -= 0.5_rt * excision.spacing[Direction];
         const bool positive = !left_active || (right_active
            && target[Direction] >= excision.region.center[Direction]);
         const int index[3]{i, j, k};
         first[Direction] = index[Direction] - (positive ? 0 : Order);
         last[Direction] = first[Direction] + Order - 1;
         return generated::z4c_spacetime_metric(load_z4c_state_components(
            Z4cExteriorCellToFaceReader<Order, Direction, Array>{
               cell_spacetime, i, j, k, first, last}));
      }
   }
   const Z4cState state = load_z4c_state_components(
      Z4cCellToFaceReader<Order, Direction, Array>{cell_spacetime, i, j, k}
   );
   return generated::z4c_spacetime_metric(state);
}

template <int Order, int Direction, typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SpacetimeMetric stored_spacetime_metric_at_face(
   Array const& cell_spacetime, int i, int j, int k,
   ExcisionGridView const& excision = {}
) noexcept
{
   if (excision.region.enabled()) {
      return stored_exterior_spacetime_metric_at_face<Order, Direction>(
         cell_spacetime, i, j, k, excision);
   }
   return generated::z4c_spacetime_metric(load_z4c_state_components(
      Z4cCellToFaceReader<Order, Direction, Array>{cell_spacetime, i, j, k}));
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
