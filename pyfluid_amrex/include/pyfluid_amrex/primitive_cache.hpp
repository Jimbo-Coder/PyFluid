#pragma once

#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/mask_regions.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

template <int NComp, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_primitive_cache_cell(
   amrex::Array4<amrex::Real const> const& state,
   amrex::Array4<amrex::Real> const& primitive,
   int i,
   int j,
   int k,
   EOSParams const& eos
) noexcept
{
   if constexpr (has_mhd_components<NComp>()) {
      const MHDPrimitive q = primitive_from_conserved<EOSMethod>(load_mhd_state(state, i, j, k), eos);
      primitive(i, j, k, RHO) = q.rho;
      primitive(i, j, k, MOM1) = q.v1;
      primitive(i, j, k, MOM2) = q.v2;
      primitive(i, j, k, MOM3) = q.v3;
      primitive(i, j, k, ENERGY) = q.pressure;
      primitive(i, j, k, B1) = q.b1;
      primitive(i, j, k, B2) = q.b2;
      primitive(i, j, k, B3) = q.b3;
   } else {
      const HydroPrimitive q = primitive_from_conserved<EOSMethod>(load_hydro_state(state, i, j, k), eos);
      primitive(i, j, k, RHO) = q.rho;
      primitive(i, j, k, MOM1) = q.v1;
      primitive(i, j, k, MOM2) = q.v2;
      primitive(i, j, k, MOM3) = q.v3;
      primitive(i, j, k, ENERGY) = q.pressure;
   }
   for (int q = base_component_count<NComp>(); q < NComp; ++q) {
      primitive(i, j, k, q) =
         state(i, j, k, q) / amrex::max(state(i, j, k, RHO), std::numeric_limits<amrex::Real>::min());
   }
}

template <int NComp, int EOSMethod>
inline void fill_primitive_cache(amrex::MultiFab& primitive, amrex::MultiFab const& state, EOSParams const& eos)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(primitive); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.fabbox();
      const auto s = state.const_array(mfi);
      const auto p = primitive.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         store_primitive_cache_cell<NComp, EOSMethod>(s, p, i, j, k, eos);
      });
   }
   }
}

template <int NComp>
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
         StateVector3 velocity{values(i, j, k, MOM1), values(i, j, k, MOM2), values(i, j, k, MOM3)};
         velocity = to_cartesian
            ? mesh_basis_vector_to_cartesian(coordinate, axes, q, velocity)
            : cartesian_vector_to_mesh_basis(coordinate, axes, q, velocity);
         values(i, j, k, MOM1) = velocity.x;
         values(i, j, k, MOM2) = velocity.y;
         values(i, j, k, MOM3) = velocity.z;
         if constexpr (has_mhd_components<NComp>()) {
            StateVector3 magnetic{values(i, j, k, B1), values(i, j, k, B2), values(i, j, k, B3)};
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

template <int NComp, int EOSMethod>
inline void fill_primitive_cache_masked(
   amrex::MultiFab& primitive,
   amrex::MultiFab const& state,
   amrex::iMultiFab const& primitive_needed,
   EOSParams const& eos,
   amrex::Vector<int> const* primitive_needed_kind = nullptr
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(primitive); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.fabbox();
      const auto s = state.const_array(mfi);
      const auto needed = primitive_needed.const_array(mfi);
      const auto p = primitive.array(mfi);
      const int kind = cached_mask_kind(primitive_needed_kind, mfi, primitive_needed, bx);
      if (kind == MaskKindZero) {
         continue;
      }
      if (kind == MaskKindNonzero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            store_primitive_cache_cell<NComp, EOSMethod>(s, p, i, j, k, eos);
         });
         continue;
      }
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (needed(i, j, k, 0) == 0) {
            return;
         }
         store_primitive_cache_cell<NComp, EOSMethod>(s, p, i, j, k, eos);
      });
   }
   }
}

} // namespace pyfluid_amrex
