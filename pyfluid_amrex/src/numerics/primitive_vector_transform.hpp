#pragma once

#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

template <int NumComponents>
inline void transform_primitive_vectors(
   amrex::MultiFab& primitive,
   amrex::Geometry const& geometry,
   int coordinate,
   MeshAxes const& axes,
   bool to_cartesian
)
{
   if (coordinate == MeshCartesian) {
      return;
   }
   const auto problo = geometry.ProbLoArray();
   const auto dx = geometry.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(primitive); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.fabbox();
      const auto values = primitive.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const auto q = mesh_coordinate_triplet_at_location(
            coordinate, axes, problo, dx, i, j, k, false, false, false
         );
         StateVector3 velocity{
            values(i, j, k, MOM1), values(i, j, k, MOM2), values(i, j, k, MOM3)
         };
         velocity = to_cartesian
            ? mesh_basis_vector_to_cartesian(coordinate, axes, q, velocity)
            : cartesian_vector_to_mesh_basis(coordinate, axes, q, velocity);
         values(i, j, k, MOM1) = velocity.x;
         values(i, j, k, MOM2) = velocity.y;
         values(i, j, k, MOM3) = velocity.z;
         if constexpr (has_mhd_components<NumComponents>()) {
            StateVector3 magnetic{
               values(i, j, k, B1), values(i, j, k, B2), values(i, j, k, B3)
            };
            magnetic = to_cartesian
               ? mesh_basis_vector_to_cartesian(coordinate, axes, q, magnetic)
               : cartesian_vector_to_mesh_basis(coordinate, axes, q, magnetic);
            values(i, j, k, B1) = magnetic.x;
            values(i, j, k, B2) = magnetic.y;
            values(i, j, k, B3) = magnetic.z;
         }
      });
   }
   }
}

} // namespace pyfluid_amrex
