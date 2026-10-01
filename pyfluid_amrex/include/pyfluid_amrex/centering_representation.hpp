#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/kernels_generated/differential_ops.hpp"

namespace pyfluid_amrex {

// Centering-transfer representations are opt-in. The default is an inline
// identity transformation for every existing scalar field.
struct IdentityCenteringTransferRepresentation {
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   static amrex::Real encode(int, amrex::Real value) noexcept
   {
      return value;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   static amrex::Real decode(int, amrex::Real value) noexcept
   {
      return value;
   }
};

template <typename Representation, typename Array>
struct CenteringTransferSource {
   Array values;
   bool use_representation = true;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real operator()(int i, int j, int k, int component) const noexcept
   {
      const amrex::Real value = values(i, j, k, component);
      return use_representation ? Representation::encode(component, value) : value;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real decode(int component, amrex::Real value) const noexcept
   {
      return use_representation ? Representation::decode(component, value) : value;
   }
};

template <typename Representation, int Order, int Direction, typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real represented_cell_to_face_ordered(
   Array const& field,
   int i,
   int j,
   int k,
   int component
) noexcept
{
   const CenteringTransferSource<Representation, Array> source{field};
   return Representation::decode(
      component,
      generated::cell_to_face_ordered<Order, Direction>(
         source, i, j, k, component
      )
   );
}

} // namespace pyfluid_amrex
