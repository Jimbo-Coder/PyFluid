#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <string>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/flux_divergence.hpp"
#include "pyfluid_amrex/flux_update.hpp"
#include "pyfluid_amrex/mask_regions.hpp"
#include "pyfluid_amrex/patch_domain.hpp"
#include "pyfluid_amrex/patch_metrics.hpp"
#include "pyfluid_amrex/riemann.hpp"
#include "relativity/rhd_system.hpp"
#include "relativity/rmhd_system.hpp"
#include "numerics/mapped_patch_interfaces.hpp"
#include "pyfluid_amrex/solid_geometry.hpp"
#include "pyfluid_amrex/solid_wall_power.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/state_update.hpp"

namespace pyfluid_amrex {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_hydro_flux(amrex::Array4<amrex::Real> const& flux, int i, int j, int k, HydroState const& value) noexcept
{
   flux(i, j, k, RHO) = value.rho;
   flux(i, j, k, MOM1) = value.mom1;
   flux(i, j, k, MOM2) = value.mom2;
   flux(i, j, k, MOM3) = value.mom3;
   flux(i, j, k, ENERGY) = value.energy;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_mhd_flux(amrex::Array4<amrex::Real> const& flux, int i, int j, int k, MHDState const& value) noexcept
{
   flux(i, j, k, RHO) = value.rho;
   flux(i, j, k, MOM1) = value.mom1;
   flux(i, j, k, MOM2) = value.mom2;
   flux(i, j, k, MOM3) = value.mom3;
   flux(i, j, k, ENERGY) = value.energy;
   flux(i, j, k, B1) = value.b1;
   flux(i, j, k, B2) = value.b2;
   flux(i, j, k, B3) = value.b3;
}

template <int ReconMethod, int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool solid_reconstruction_stencil_is_fluid(
   amrex::Array4<int const> const& solid,
   int i,
   int j,
   int k
) noexcept;

template <int NComp>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_zero_flux(amrex::Array4<amrex::Real> const& flux, int i, int j, int k) noexcept
{
   for (int n = 0; n < NComp; ++n) {
      flux(i, j, k, n) = 0.0;
   }
}

template <int NComp, int ReconMethod, int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_passive_fluxes(
   amrex::Array4<amrex::Real> const& flux,
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   int li,
   int lj,
   int lk,
   int ri,
   int rj,
   int rk,
   amrex::Real mass_flux
) noexcept
{
   if constexpr (passive_component_count<NComp>() > 0) {
      constexpr int Component = base_component_count<NComp>();
      const ScalarFaceValues dye = primitive_component_face_values<ReconMethod, Direction, Component>(
         primitive, li, lj, lk, ri, rj, rk
      );
      flux(i, j, k, Component) = mass_flux * (mass_flux >= 0.0 ? dye.left : dye.right);
   }
}

template <int NComp>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_wall_passive_fluxes(
   amrex::Array4<amrex::Real> const& flux,
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   int fi,
   int fj,
   int fk,
   amrex::Real mass_flux
) noexcept
{
   if constexpr (passive_component_count<NComp>() > 0) {
      constexpr int Component = base_component_count<NComp>();
      flux(i, j, k, Component) = mass_flux * primitive(fi, fj, fk, Component);
   }
}

template <int NComp, int ReconMethod, int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_passive_fluxes_limited_by_solid_stencil(
   amrex::Array4<amrex::Real> const& flux,
   amrex::Array4<amrex::Real const> const& primitive,
   amrex::Array4<int const> const& solid,
   int i,
   int j,
   int k,
   int li,
   int lj,
   int lk,
   int ri,
   int rj,
   int rk,
   amrex::Real mass_flux
) noexcept
{
   if constexpr (passive_component_count<NComp>() > 0) {
      if constexpr (ReconMethod == ReconPWC) {
         store_passive_fluxes<NComp, ReconPWC, Direction>(flux, primitive, i, j, k, li, lj, lk, ri, rj, rk, mass_flux);
      } else {
         if (solid_reconstruction_stencil_is_fluid<ReconMethod, Direction>(solid, li, lj, lk)) {
            store_passive_fluxes<NComp, ReconMethod, Direction>(
               flux, primitive, i, j, k, li, lj, lk, ri, rj, rk, mass_flux
            );
         } else {
            store_passive_fluxes<NComp, ReconPWC, Direction>(
               flux, primitive, i, j, k, li, lj, lk, ri, rj, rk, mass_flux
            );
         }
      }
   }
}

template <int NComp, int ReconMethod, int RiemannMethod, int Direction, int EOSMethod>
inline void fill_hydro_flux_direction(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   int mesh_coordinate = MeshCartesian,
   MeshAxes mesh_axes = mesh_default_axes_layout(MeshCartesian),
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo = {},
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = {},
   bool skip_zero_area_metric_faces = false
)
{
   amrex::MultiFab& flux = *fluxes[Direction];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const auto p = primitive.const_array(mfi);
      const auto f = flux.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (skip_zero_area_metric_faces) {
            const amrex::Real area = mesh_face_area_scale(Direction, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            if (area == 0.0) {
               store_zero_flux<NComp>(f, i, j, k);
               return;
            }
         }
         int li = i;
         int lj = j;
         int lk = k;
         int ri = i;
         int rj = j;
         int rk = k;
         HydroState value;
         if constexpr (Direction == 0) {
            li = i - 1;
         }
#if (AMREX_SPACEDIM >= 2)
         else if constexpr (Direction == 1) {
            lj = j - 1;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         else {
            lk = k - 1;
         }
#endif
         if (mesh_layout_spans_physical_space(mesh_coordinate, mesh_axes)) {
            const MeshFaceMetric metric = mesh_physical_face_metric(
               Direction, mesh_coordinate, mesh_axes, problo, dx, i, j, k
            );
            value = hydro_riemann_flux_normal<ReconMethod, RiemannMethod, Direction, EOSMethod>(
               p, li, lj, lk, ri, rj, rk, eos, metric.unit_normal
            );
         } else {
            value = hydro_riemann_flux<ReconMethod, RiemannMethod, Direction, EOSMethod>(
               p, li, lj, lk, ri, rj, rk, eos
            );
         }
         store_hydro_flux(f, i, j, k, value);
         store_passive_fluxes<NComp, ReconMethod, Direction>(f, p, i, j, k, li, lj, lk, ri, rj, rk, value.rho);
      });
   }
   }
}

template <int NComp, int ReconMethod, int RiemannMethod, int Direction, int EOSMethod>
inline void fill_hydro_flux_direction_mapped(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout
)
{
   amrex::MultiFab& flux = *fluxes[Direction];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const amrex::Box& cell_box = primitive.boxArray()[mfi.index()];
      PatchDescriptor const& layout_patch_ref = patch_layout_descriptor_containing_box(layout, cell_box);
      PatchGridDescriptor const& grid_patch_ref = patch_grid_descriptor_for_layout_patch(grid_system, layout_patch_ref);
      const PatchDescriptor layout_patch = layout_patch_ref;
      const PatchGridDescriptor grid_patch = grid_patch_ref;
      const auto p = primitive.const_array(mfi);
      const auto f = flux.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         int li = i;
         int lj = j;
         int lk = k;
         int ri = i;
         int rj = j;
         int rk = k;
         if constexpr (Direction == 0) {
            li = i - 1;
         }
#if (AMREX_SPACEDIM >= 2)
         else if constexpr (Direction == 1) {
            lj = j - 1;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         else {
            lk = k - 1;
         }
#endif
         const PatchCoordinate face_local = patch_layout_face_coordinate(grid_patch, layout_patch, Direction, i, j, k);
         const PatchFaceMetric metric = patch_face_metric(grid_patch, face_local, Direction);
         HydroState value = hydro_riemann_flux_normal<ReconMethod, RiemannMethod, Direction, EOSMethod>(
            p, li, lj, lk, ri, rj, rk, eos, metric.unit_normal
         );
         value.rho *= metric.area;
         value.mom1 *= metric.area;
         value.mom2 *= metric.area;
         value.mom3 *= metric.area;
         value.energy *= metric.area;
         store_hydro_flux(f, i, j, k, value);
         store_passive_fluxes<NComp, ReconMethod, Direction>(f, p, i, j, k, li, lj, lk, ri, rj, rk, value.rho);
      });
   }
   }
}

template <int ReconMethod, int RiemannMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hydro_flux_limited_by_solid_stencil(
   amrex::Array4<amrex::Real const> const& p,
   amrex::Array4<int const> const& solid,
   int li,
   int lj,
   int lk,
   int ri,
   int rj,
   int rk,
   EOSParams const& eos
) noexcept;

template <int RiemannMethod, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hydro_flux_from_primitive_riemann(
   HydroPrimitive const& left,
   HydroPrimitive const& right,
   EOSParams const& eos,
   int direction
) noexcept
{
   if constexpr (RiemannMethod == RiemannHLL) {
      return hll_flux_from_primitives<EOSMethod>(left, right, eos, direction);
   } else if constexpr (RiemannMethod == RiemannHLLE) {
      return hlle_flux_from_primitives<EOSMethod>(left, right, eos, direction);
   } else if constexpr (RiemannMethod == RiemannHLLC) {
      return hllc_flux_from_primitives<EOSMethod>(left, right, eos, direction);
   } else {
      return llf_flux_from_primitives<EOSMethod>(left, right, eos, direction);
   }
}

template <int RiemannMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState moving_slip_wall_flux(
   amrex::Array4<amrex::Real const> const& p,
   int fi,
   int fj,
   int fk,
   bool fluid_on_left,
   amrex::Real wall_vn,
   EOSParams const& eos
) noexcept
{
   const HydroPrimitive fluid = primitive_at(p, fi, fj, fk, eos);
   HydroPrimitive ghost = fluid;
   if constexpr (Direction == 0) {
      ghost.v1 = 2.0 * wall_vn - fluid.v1;
   }
#if (AMREX_SPACEDIM >= 2)
   else if constexpr (Direction == 1) {
      ghost.v2 = 2.0 * wall_vn - fluid.v2;
   }
#endif
#if (AMREX_SPACEDIM == 3)
   else {
      ghost.v3 = 2.0 * wall_vn - fluid.v3;
   }
#endif
   return fluid_on_left
      ? hydro_flux_from_primitive_riemann<RiemannMethod, EOSMethod>(fluid, ghost, eos, Direction)
      : hydro_flux_from_primitive_riemann<RiemannMethod, EOSMethod>(ghost, fluid, eos, Direction);
}

template <int NComp, int ReconMethod, int RiemannMethod, int Direction, int EOSMethod>
inline void fill_hydro_flux_direction_with_solid_walls(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   EOSParams const& eos,
   SolidGeometryEvaluator const& solid_params,
   amrex::Vector<int> const* solid_flux_kind = nullptr
)
{
   amrex::MultiFab& flux = *fluxes[Direction];
   const amrex::Box domain = geom.Domain();
   const auto lo = domain.smallEnd();
   const auto hi = domain.bigEnd();
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
	   for (amrex::MFIter mfi(flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.validbox();
	      const auto p = primitive.const_array(mfi);
	      const auto f = flux.array(mfi);
	      const auto solid = solid_mask.const_array(mfi);
	      const amrex::Box cell_region = solid_flux_stencil_region(bx, Direction, ReconMethod, domain);
		      const int kind = cached_or_region_mask_kind(solid_flux_kind, mfi, solid_mask, cell_region);
		      if (kind == MaskKindNonzero) {
		         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		            for (int q = 0; q < NComp; ++q) {
		               f(i, j, k, q) = 0.0;
		            }
		         });
		         continue;
		      }
	      if (kind == MaskKindZero) {
	         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	            int li = i;
	            int lj = j;
	            int lk = k;
	            int ri = i;
	            int rj = j;
	            int rk = k;
	            if constexpr (Direction == 0) {
	               li = i - 1;
	            }
#if (AMREX_SPACEDIM >= 2)
	            else if constexpr (Direction == 1) {
	               lj = j - 1;
	            }
#endif
#if (AMREX_SPACEDIM == 3)
	            else {
	               lk = k - 1;
	            }
#endif
	            const HydroState value =
	               hydro_riemann_flux<ReconMethod, RiemannMethod, Direction, EOSMethod>(p, li, lj, lk, ri, rj, rk, eos);
	            store_hydro_flux(f, i, j, k, value);
	            store_passive_fluxes<NComp, ReconMethod, Direction>(f, p, i, j, k, li, lj, lk, ri, rj, rk, value.rho);
	         });
	         continue;
	      }
	      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	         int li = i;
         int lj = j;
         int lk = k;
         int ri = i;
         int rj = j;
         int rk = k;
         if constexpr (Direction == 0) {
            li = i - 1;
         }
#if (AMREX_SPACEDIM >= 2)
         else if constexpr (Direction == 1) {
            lj = j - 1;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         else {
            lk = k - 1;
         }
#endif

         bool left_in_domain = li >= lo[0] && li <= hi[0];
         bool right_in_domain = ri >= lo[0] && ri <= hi[0];
#if (AMREX_SPACEDIM >= 2)
         left_in_domain = left_in_domain && lj >= lo[1] && lj <= hi[1];
         right_in_domain = right_in_domain && rj >= lo[1] && rj <= hi[1];
#endif
#if (AMREX_SPACEDIM == 3)
         left_in_domain = left_in_domain && lk >= lo[2] && lk <= hi[2];
         right_in_domain = right_in_domain && rk >= lo[2] && rk <= hi[2];
#endif
         const bool left_solid = left_in_domain && solid(li, lj, lk, 0) != 0;
         const bool right_solid = right_in_domain && solid(ri, rj, rk, 0) != 0;

         if (!left_solid && !right_solid) {
            const HydroState value = hydro_flux_limited_by_solid_stencil<
               ReconMethod, RiemannMethod, Direction, EOSMethod
            >(p, solid, li, lj, lk, ri, rj, rk, eos);
            store_hydro_flux(f, i, j, k, value);
            store_passive_fluxes_limited_by_solid_stencil<NComp, ReconMethod, Direction>(
               f, p, solid, i, j, k, li, lj, lk, ri, rj, rk, value.rho
            );
            return;
         }

         for (int q = 0; q < NComp; ++q) {
            f(i, j, k, q) = 0.0;
         }
         if (left_solid == right_solid) {
            return;
         }

         const int fi = left_solid ? ri : li;
         const int fj = left_solid ? rj : lj;
         const int fk = left_solid ? rk : lk;
         amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
         amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
#else
         amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
         amrex::Real z = problo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
#else
         amrex::Real z = 0.0;
#endif
         if constexpr (Direction == 0) {
            x = problo[0] + static_cast<amrex::Real>(i) * dx[0];
         }
#if (AMREX_SPACEDIM >= 2)
         else if constexpr (Direction == 1) {
            y = problo[1] + static_cast<amrex::Real>(j) * dx[1];
         }
#endif
#if (AMREX_SPACEDIM == 3)
         else {
            z = problo[2] + static_cast<amrex::Real>(k) * dx[2];
         }
#endif
         const amrex::Real wall_vn = solid_params.motion == SolidMotionRotate
            ? 0.0
            : solid_params.velocity_component(solid_params.time, Direction, x, y, z);
         const HydroState value = moving_slip_wall_flux<RiemannMethod, Direction, EOSMethod>(
            p,
            fi,
            fj,
            fk,
            right_solid,
            wall_vn,
            eos
         );
         store_hydro_flux(f, i, j, k, value);
         store_wall_passive_fluxes<NComp>(f, p, i, j, k, fi, fj, fk, value.rho);
	      });
	   }
	   }
}

template <int ReconMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int solid_reconstruction_offset_lo() noexcept
{
   if constexpr (is_five_point_reconstruction(ReconMethod)) {
      return -2;
   } else if constexpr (ReconMethod == ReconPWL) {
      return -1;
   } else {
      return 0;
   }
}

template <int ReconMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int solid_reconstruction_offset_hi() noexcept
{
   if constexpr (is_five_point_reconstruction(ReconMethod)) {
      return 3;
   } else if constexpr (ReconMethod == ReconPWL) {
      return 2;
   } else {
      return 1;
   }
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool solid_mask_at_offset(
   amrex::Array4<int const> const& solid,
   int i,
   int j,
   int k,
   int offset
) noexcept
{
   if constexpr (Direction == 0) {
      return solid(i + offset, j, k, 0) != 0;
   }
#if (AMREX_SPACEDIM >= 2)
   else if constexpr (Direction == 1) {
      return solid(i, j + offset, k, 0) != 0;
   }
#endif
   return solid(i, j, k + offset, 0) != 0;
}

template <int ReconMethod, int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool solid_reconstruction_stencil_is_fluid(
   amrex::Array4<int const> const& solid,
   int i,
   int j,
   int k
) noexcept
{
   if constexpr (ReconMethod == ReconPWC) {
      return true;
   }
   for (int offset = solid_reconstruction_offset_lo<ReconMethod>();
        offset <= solid_reconstruction_offset_hi<ReconMethod>();
        ++offset) {
      if (solid_mask_at_offset<Direction>(solid, i, j, k, offset)) {
         return false;
      }
   }
   return true;
}

template <int ReconMethod, int RiemannMethod, int Direction, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState hydro_flux_limited_by_solid_stencil(
   amrex::Array4<amrex::Real const> const& p,
   amrex::Array4<int const> const& solid,
   int li,
   int lj,
   int lk,
   int ri,
   int rj,
   int rk,
   EOSParams const& eos
) noexcept
{
   if constexpr (ReconMethod == ReconPWC) {
      return hydro_riemann_flux<ReconPWC, RiemannMethod, Direction, EOSMethod>(p, li, lj, lk, ri, rj, rk, eos);
   }
   if (solid_reconstruction_stencil_is_fluid<ReconMethod, Direction>(solid, li, lj, lk)) {
      return hydro_riemann_flux<ReconMethod, RiemannMethod, Direction, EOSMethod>(p, li, lj, lk, ri, rj, rk, eos);
   }
   return hydro_riemann_flux<ReconPWC, RiemannMethod, Direction, EOSMethod>(p, li, lj, lk, ri, rj, rk, eos);
}

template <int NComp, int ReconMethod, int RiemannMethod, int Direction, int EOSMethod>
inline void fill_mhd_flux_direction(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   int mesh_coordinate = MeshCartesian,
   MeshAxes mesh_axes = mesh_default_axes_layout(MeshCartesian),
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo = {},
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = {},
   bool skip_zero_area_metric_faces = false
)
{
   amrex::MultiFab& flux = *fluxes[Direction];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const auto p = primitive.const_array(mfi);
      const auto f = flux.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (skip_zero_area_metric_faces) {
            const amrex::Real area = mesh_face_area_scale(Direction, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            if (area == 0.0) {
               store_zero_flux<NComp>(f, i, j, k);
               return;
            }
         }
         int li = i;
         int lj = j;
         int lk = k;
         int ri = i;
         int rj = j;
         int rk = k;
         MHDState value;
         if constexpr (Direction == 0) {
            li = i - 1;
         }
#if (AMREX_SPACEDIM >= 2)
         else if constexpr (Direction == 1) {
            lj = j - 1;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         else {
            lk = k - 1;
         }
#endif
         value = mhd_riemann_flux<ReconMethod, RiemannMethod, Direction, EOSMethod>(p, li, lj, lk, ri, rj, rk, eos);
         store_mhd_flux(f, i, j, k, value);
         store_passive_fluxes<NComp, ReconMethod, Direction>(f, p, i, j, k, li, lj, lk, ri, rj, rk, value.rho);
      });
   }
   }
}

template <
   int NComp,
   int ReconMethod,
   int RiemannMethod,
   int Direction,
   int EOSMethod,
   bool SolidWalls = false,
   bool MappedPatches = false
>
inline void fill_flux_direction_for_components(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   int mesh_coordinate = MeshCartesian,
   MeshAxes mesh_axes = mesh_default_axes_layout(MeshCartesian),
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo = {},
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = {},
   bool skip_zero_area_metric_faces = false,
   amrex::iMultiFab const* solid_mask = nullptr,
   amrex::Geometry const* geom = nullptr,
   SolidGeometryEvaluator const& solid_params = SolidGeometryEvaluator{},
   amrex::Vector<int> const* solid_flux_kind = nullptr,
   PatchGridSystem const* grid_system = nullptr,
   PatchLayout const* layout = nullptr
)
{
   if constexpr (has_mhd_components<NComp>()) {
      if constexpr (MappedPatches) {
         amrex::Abort("mapped patch fluxes are currently implemented for hydro only");
      }
      if constexpr (SolidWalls) {
         amrex::Abort("solid wall flux skipping is only implemented for hydro");
      }
      fill_mhd_flux_direction<NComp, ReconMethod, RiemannMethod, Direction, EOSMethod>(
         fluxes,
         primitive,
         eos,
         mesh_coordinate,
         mesh_axes,
         problo,
         dx,
         skip_zero_area_metric_faces
      );
   } else {
      if constexpr (MappedPatches) {
         if constexpr (SolidWalls) {
            amrex::Abort("mapped patch fluxes do not use solid wall flux skipping");
         }
         if (grid_system == nullptr || layout == nullptr) {
            amrex::Abort("mapped patch fluxes require a patch grid system and patch layout");
         }
         fill_hydro_flux_direction_mapped<NComp, ReconMethod, RiemannMethod, Direction, EOSMethod>(
            fluxes,
            primitive,
            eos,
            *grid_system,
            *layout
         );
      } else if constexpr (SolidWalls) {
         if (solid_mask == nullptr || geom == nullptr) {
            amrex::Abort("solid wall flux skipping requires solid mask and geometry");
         }
	         fill_hydro_flux_direction_with_solid_walls<NComp, ReconMethod, RiemannMethod, Direction, EOSMethod>(
	            fluxes,
	            primitive,
	            *solid_mask,
	            *geom,
	            eos,
	            solid_params,
	            solid_flux_kind
	         );
      } else {
         fill_hydro_flux_direction<NComp, ReconMethod, RiemannMethod, Direction, EOSMethod>(
            fluxes,
            primitive,
            eos,
            mesh_coordinate,
            mesh_axes,
            problo,
            dx,
            skip_zero_area_metric_faces
         );
      }
   }
}

template <
   int NComp,
   int ReconMethod,
   int RiemannMethod,
   int EOSMethod,
   bool SolidWalls = false,
   bool MappedPatches = false
>
inline void fill_fluxes_for_components(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   RunConfig const& cfg,
   amrex::iMultiFab const* solid_mask = nullptr,
   amrex::Geometry const* geom = nullptr,
   amrex::Array<amrex::Vector<int>, AMREX_SPACEDIM> const* solid_flux_kind = nullptr,
   amrex::Real time = 0.0,
   PatchGridSystem const* grid_system = nullptr,
   PatchLayout const* layout = nullptr,
   EOSThermodynamicState const* atmosphere = nullptr
)
{
   if (physical_system(cfg) == PhysicalSystem::RMHD) {
#if PYFLUID_AMREX_ENABLE_MHD
      if constexpr (NComp != NVAR_MHD
                    || is_characteristic_reconstruction(ReconMethod)
                    || SolidWalls || MappedPatches
                    || EOSMethod != EOSIdealGas) {
         amrex::Abort(
            "unsupported RMHD state-layout or flux-topology product reached "
            "the generic flux path"
         );
      } else if constexpr (RiemannMethod == RiemannHLLC
                    || RiemannMethod == RiemannHLLD) {
         amrex::Abort(
            "advanced RMHD Riemann solver must be routed by the flux factory"
         );
      } else {
         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(geom != nullptr, "RMHD fluxes require level geometry");
         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            atmosphere != nullptr, "RMHD fluxes require the recovery atmosphere state"
         );
         const FixedSpacetimeParameters spacetime = fixed_spacetime_parameters(cfg);
         fill_rmhd_flux_direction<ReconMethod, RiemannMethod, 0>(
            fluxes, primitive, eos_params(cfg), *geom, spacetime,
            *atmosphere, time
         );
#if (AMREX_SPACEDIM >= 2)
         fill_rmhd_flux_direction<ReconMethod, RiemannMethod, 1>(
            fluxes, primitive, eos_params(cfg), *geom, spacetime,
            *atmosphere, time
         );
#endif
#if (AMREX_SPACEDIM == 3)
         fill_rmhd_flux_direction<ReconMethod, RiemannMethod, 2>(
            fluxes, primitive, eos_params(cfg), *geom, spacetime,
            *atmosphere, time
         );
#endif
      }
#else
      amrex::Abort("RMHD fluxes are disabled in this build");
#endif
      return;
   }
   if (physical_system(cfg) == PhysicalSystem::RHD) {
      if constexpr (NComp != NVAR_HYDRO
                    || is_characteristic_reconstruction(ReconMethod)
                    || SolidWalls || MappedPatches) {
         amrex::Abort(
            "unsupported RHD state-layout or flux-topology product reached "
            "the generic flux path"
         );
      } else if constexpr (RiemannMethod == RiemannHLLC
                    || RiemannMethod == RiemannHLLD) {
         amrex::Abort(
            "unsupported advanced RHD Riemann solver reached the generic flux path"
         );
      } else {
         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(geom != nullptr, "RHD fluxes require level geometry");
         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            atmosphere != nullptr, "RHD fluxes require the recovery atmosphere state"
         );
         const FixedSpacetimeParameters spacetime = fixed_spacetime_parameters(cfg);
         fill_rhd_flux_direction<ReconMethod, RiemannMethod, 0, EOSMethod>(
            fluxes, primitive, eos_params(cfg), *geom, spacetime,
            *atmosphere, time
         );
#if (AMREX_SPACEDIM >= 2)
         fill_rhd_flux_direction<ReconMethod, RiemannMethod, 1, EOSMethod>(
            fluxes, primitive, eos_params(cfg), *geom, spacetime,
            *atmosphere, time
         );
#endif
#if (AMREX_SPACEDIM == 3)
         fill_rhd_flux_direction<ReconMethod, RiemannMethod, 2, EOSMethod>(
            fluxes, primitive, eos_params(cfg), *geom, spacetime,
            *atmosphere, time
         );
#endif
      }
      return;
   }
   if constexpr (EOSMethod == EOSCompose) {
      amrex::Abort("native cold CompOSE fluxes require RHD");
   } else {
   const EOSParams eos = eos_params(cfg);
   const SolidGeometryEvaluator solid_params = solid_geometry_evaluator(cfg, time);
   const int mesh_coordinate = geom == nullptr ? MeshCartesian : mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(mesh_coordinate, cfg.mesh_axes);
   const auto problo = geom == nullptr
      ? amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>{}
      : geom->ProbLoArray();
   const auto dx = geom == nullptr
      ? amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>{}
      : geom->CellSizeArray();
   const bool skip_zero_area_metric_faces =
      geom != nullptr && !MappedPatches && mesh_coordinate != MeshCartesian;
   fill_flux_direction_for_components<NComp, ReconMethod, RiemannMethod, 0, EOSMethod, SolidWalls, MappedPatches>(
      fluxes,
      primitive,
      eos,
      mesh_coordinate,
      mesh_axes,
      problo,
      dx,
      skip_zero_area_metric_faces,
      solid_mask,
      geom,
      solid_params,
      solid_flux_kind == nullptr ? nullptr : &(*solid_flux_kind)[0],
      grid_system,
      layout
   );
#if (AMREX_SPACEDIM >= 2)
   fill_flux_direction_for_components<NComp, ReconMethod, RiemannMethod, 1, EOSMethod, SolidWalls, MappedPatches>(
      fluxes,
      primitive,
      eos,
      mesh_coordinate,
      mesh_axes,
      problo,
      dx,
      skip_zero_area_metric_faces,
      solid_mask,
      geom,
      solid_params,
      solid_flux_kind == nullptr ? nullptr : &(*solid_flux_kind)[1],
      grid_system,
      layout
   );
#endif
#if (AMREX_SPACEDIM == 3)
   fill_flux_direction_for_components<NComp, ReconMethod, RiemannMethod, 2, EOSMethod, SolidWalls, MappedPatches>(
      fluxes,
      primitive,
      eos,
      mesh_coordinate,
      mesh_axes,
      problo,
      dx,
      skip_zero_area_metric_faces,
      solid_mask,
      geom,
      solid_params,
      solid_flux_kind == nullptr ? nullptr : &(*solid_flux_kind)[2],
      grid_system,
      layout
   );
#endif
   if constexpr (MappedPatches && !has_mhd_components<NComp>()) {
      if (grid_system == nullptr || layout == nullptr) {
         amrex::Abort("mapped patch flux correction requires a patch grid system and patch layout");
      }
      correct_mapped_patch_hydro_interface_fluxes<NComp, ReconMethod, RiemannMethod, EOSMethod>(
         fluxes,
         primitive,
         eos,
         *grid_system,
         *layout
      );
   }
   }
}

inline void abort_disabled_flux_feature(char const* kind, char const* name)
{
   amrex::Abort(
      std::string("pyfluid.") + kind + "=" + name
      + " was selected, but that branch is disabled in this AMReX build"
   );
}

template <int NComp, int ReconMethod, int EOSMethod, bool SolidWalls = false, bool MappedPatches = false>
inline void fill_fluxes_for_riemann(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   RunConfig const& cfg,
   int riemann_method,
   amrex::iMultiFab const* solid_mask = nullptr,
   amrex::Geometry const* geom = nullptr,
   amrex::Array<amrex::Vector<int>, AMREX_SPACEDIM> const* solid_flux_kind = nullptr,
   amrex::Real time = 0.0,
   PatchGridSystem const* grid_system = nullptr,
   PatchLayout const* layout = nullptr,
   EOSThermodynamicState const* atmosphere = nullptr
)
{
   switch (riemann_method) {
   case RiemannHLLD:
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLLD
	      fill_fluxes_for_components<NComp, ReconMethod, RiemannHLLD, EOSMethod, SolidWalls, MappedPatches>(
	         fluxes, primitive, cfg, solid_mask, geom, solid_flux_kind, time, grid_system, layout, atmosphere
	      );
      return;
#else
      abort_disabled_flux_feature("Riemann solver", "hlld");
      return;
#endif
   case RiemannHLLC:
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLLC
	      fill_fluxes_for_components<NComp, ReconMethod, RiemannHLLC, EOSMethod, SolidWalls, MappedPatches>(
	         fluxes, primitive, cfg, solid_mask, geom, solid_flux_kind, time, grid_system, layout, atmosphere
	      );
      return;
#else
      abort_disabled_flux_feature("Riemann solver", "hllc");
      return;
#endif
   case RiemannHLLE:
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLLE
	      fill_fluxes_for_components<NComp, ReconMethod, RiemannHLLE, EOSMethod, SolidWalls, MappedPatches>(
	         fluxes, primitive, cfg, solid_mask, geom, solid_flux_kind, time, grid_system, layout, atmosphere
	      );
      return;
#else
      abort_disabled_flux_feature("Riemann solver", "hlle");
      return;
#endif
   case RiemannHLL:
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLL
	      fill_fluxes_for_components<NComp, ReconMethod, RiemannHLL, EOSMethod, SolidWalls, MappedPatches>(
	         fluxes, primitive, cfg, solid_mask, geom, solid_flux_kind, time, grid_system, layout, atmosphere
	      );
      return;
#else
      abort_disabled_flux_feature("Riemann solver", "hll");
      return;
#endif
   case RiemannLLF:
#if PYFLUID_AMREX_ENABLE_RIEMANN_LLF
	      fill_fluxes_for_components<NComp, ReconMethod, RiemannLLF, EOSMethod, SolidWalls, MappedPatches>(
	         fluxes, primitive, cfg, solid_mask, geom, solid_flux_kind, time, grid_system, layout, atmosphere
	      );
      return;
#else
      abort_disabled_flux_feature("Riemann solver", "llf");
      return;
#endif
   default:
      amrex::Abort("Invalid parsed Riemann solver code");
   }
}

template <int NComp, int ReconMethod, int Direction, int EOSMethod, bool SolidWalls = false, bool MappedPatches = false>
inline void fill_flux_direction_for_riemann(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   RunConfig const& cfg,
   int riemann_method,
   amrex::iMultiFab const* solid_mask = nullptr,
   amrex::Geometry const* geom = nullptr,
   amrex::Vector<int> const* solid_flux_kind = nullptr,
   amrex::Real time = 0.0,
   PatchGridSystem const* grid_system = nullptr,
   PatchLayout const* layout = nullptr,
   amrex::MultiFab const* stored_spacetime = nullptr,
   EOSThermodynamicState const* atmosphere = nullptr
)
{
   const EOSParams eos = eos_params(cfg);
   if (physical_system(cfg) == PhysicalSystem::RMHD) {
#if PYFLUID_AMREX_ENABLE_MHD
      if constexpr (NComp != NVAR_MHD
                    || is_characteristic_reconstruction(ReconMethod)
                    || SolidWalls || MappedPatches
                    || EOSMethod != EOSIdealGas) {
         amrex::Abort(
            "unsupported RMHD state-layout or flux-topology product reached "
            "the directional flux path"
         );
      } else {
         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(geom != nullptr, "RMHD fluxes require level geometry");
         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            atmosphere != nullptr, "RMHD fluxes require the recovery atmosphere state"
         );
         if (stored_spacetime != nullptr) {
            fill_rmhd_flux_direction_runtime<ReconMethod, Direction>(
               fluxes, primitive, eos, riemann_method, *stored_spacetime,
               *atmosphere
            );
         } else {
            fill_rmhd_flux_direction_runtime<ReconMethod, Direction>(
               fluxes, primitive, eos, riemann_method, *geom,
               fixed_spacetime_parameters(cfg), *atmosphere,
               time
            );
         }
      }
#else
      amrex::Abort("RMHD fluxes are disabled in this build");
#endif
      return;
   }
   if (physical_system(cfg) == PhysicalSystem::RHD) {
      if constexpr (NComp != NVAR_HYDRO
                    || is_characteristic_reconstruction(ReconMethod)
                    || SolidWalls || MappedPatches) {
         amrex::Abort(
            "unsupported RHD state-layout or flux-topology product reached "
            "the directional flux path"
         );
      } else {
         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(geom != nullptr, "RHD fluxes require level geometry");
         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            atmosphere != nullptr, "RHD fluxes require the recovery atmosphere state"
         );
         if (stored_spacetime != nullptr) {
            fill_rhd_flux_direction_runtime<ReconMethod, Direction, EOSMethod>(
               fluxes, primitive, eos, riemann_method, *stored_spacetime,
               *atmosphere
            );
         } else {
            fill_rhd_flux_direction_runtime<ReconMethod, Direction, EOSMethod>(
               fluxes, primitive, eos, riemann_method, *geom,
               fixed_spacetime_parameters(cfg), *atmosphere,
               time
            );
         }
      }
      return;
   }
   if constexpr (EOSMethod == EOSCompose) {
      amrex::Abort("native cold CompOSE fluxes require RHD");
   } else {
   const SolidGeometryEvaluator solid_params = solid_geometry_evaluator(cfg, time);
   const int mesh_coordinate = geom == nullptr ? MeshCartesian : mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(mesh_coordinate, cfg.mesh_axes);
   const auto problo = geom == nullptr
      ? amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>{}
      : geom->ProbLoArray();
   const auto dx = geom == nullptr
      ? amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>{}
      : geom->CellSizeArray();
   const bool skip_zero_area_metric_faces =
      geom != nullptr && !MappedPatches && mesh_coordinate != MeshCartesian;
   switch (riemann_method) {
   case RiemannHLLD:
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLLD
	      fill_flux_direction_for_components<NComp, ReconMethod, RiemannHLLD, Direction, EOSMethod, SolidWalls, MappedPatches>(
	         fluxes, primitive, eos, mesh_coordinate, mesh_axes, problo, dx, skip_zero_area_metric_faces,
	         solid_mask, geom, solid_params, solid_flux_kind, grid_system, layout
	      );
      return;
#else
      abort_disabled_flux_feature("Riemann solver", "hlld");
      return;
#endif
   case RiemannHLLC:
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLLC
	      fill_flux_direction_for_components<NComp, ReconMethod, RiemannHLLC, Direction, EOSMethod, SolidWalls, MappedPatches>(
	         fluxes, primitive, eos, mesh_coordinate, mesh_axes, problo, dx, skip_zero_area_metric_faces,
	         solid_mask, geom, solid_params, solid_flux_kind, grid_system, layout
	      );
      return;
#else
      abort_disabled_flux_feature("Riemann solver", "hllc");
      return;
#endif
   case RiemannHLLE:
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLLE
	      fill_flux_direction_for_components<NComp, ReconMethod, RiemannHLLE, Direction, EOSMethod, SolidWalls, MappedPatches>(
	         fluxes, primitive, eos, mesh_coordinate, mesh_axes, problo, dx, skip_zero_area_metric_faces,
	         solid_mask, geom, solid_params, solid_flux_kind, grid_system, layout
	      );
      return;
#else
      abort_disabled_flux_feature("Riemann solver", "hlle");
      return;
#endif
   case RiemannHLL:
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLL
	      fill_flux_direction_for_components<NComp, ReconMethod, RiemannHLL, Direction, EOSMethod, SolidWalls, MappedPatches>(
	         fluxes, primitive, eos, mesh_coordinate, mesh_axes, problo, dx, skip_zero_area_metric_faces,
	         solid_mask, geom, solid_params, solid_flux_kind, grid_system, layout
	      );
      return;
#else
      abort_disabled_flux_feature("Riemann solver", "hll");
      return;
#endif
   case RiemannLLF:
#if PYFLUID_AMREX_ENABLE_RIEMANN_LLF
	      fill_flux_direction_for_components<NComp, ReconMethod, RiemannLLF, Direction, EOSMethod, SolidWalls, MappedPatches>(
	         fluxes, primitive, eos, mesh_coordinate, mesh_axes, problo, dx, skip_zero_area_metric_faces,
	         solid_mask, geom, solid_params, solid_flux_kind, grid_system, layout
	      );
      return;
#else
      abort_disabled_flux_feature("Riemann solver", "llf");
      return;
#endif
   default:
      amrex::Abort("Invalid parsed Riemann solver code");
   }
   }
}

} // namespace pyfluid_amrex
