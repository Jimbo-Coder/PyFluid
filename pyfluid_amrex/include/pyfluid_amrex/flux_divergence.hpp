#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

#include <array>

#include "pyfluid_amrex/kernels_generated/vector/finite_volume.hpp"
#include "pyfluid_amrex/mask_regions.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/patch_metrics.hpp"
#include "pyfluid_amrex/reference_metric_valencia.hpp"
#include "pyfluid_amrex/solid_geometry_point.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

template <int NComp>
inline void compute_rhs_from_fluxes(
   amrex::MultiFab& rhs,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::Geometry const& geom
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.tilebox();
      const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto fz = fluxes[2]->const_array(mfi);
#endif
      const auto r = rhs.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         for (int q = 0; q < NComp; ++q) {
            amrex::Real value = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
            value -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
            value -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
            r(i, j, k, q) = value;
         }
      });
   }
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real finite_volume_divergence_from_face_integrals(
   amrex::Real flux_x_hi,
   amrex::Real flux_x_lo,
   amrex::Real flux_y_hi,
   amrex::Real flux_y_lo,
   amrex::Real flux_z_hi,
   amrex::Real flux_z_lo,
   amrex::Real volume
) noexcept
{
#if (AMREX_SPACEDIM == 1)
   amrex::ignore_unused(flux_y_hi, flux_y_lo, flux_z_hi, flux_z_lo);
   return generated::finite_volume_divergence_1d(flux_x_hi, flux_x_lo, volume);
#elif (AMREX_SPACEDIM == 2)
   amrex::ignore_unused(flux_z_hi, flux_z_lo);
   return generated::finite_volume_divergence_2d(flux_x_hi, flux_x_lo, flux_y_hi, flux_y_lo, volume);
#else
   return generated::finite_volume_divergence_3d(
      flux_x_hi, flux_x_lo, flux_y_hi, flux_y_lo, flux_z_hi, flux_z_lo, volume
   );
#endif
}

template <int NComp>
inline void compute_rhs_from_fluxes_mapped_patches(
   amrex::MultiFab& rhs,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.tilebox();
	      PatchDescriptor const& layout_patch_ref = patch_layout_descriptor_containing_box(layout, mfi.validbox());
	      PatchGridDescriptor const& grid_patch_ref = patch_grid_descriptor_for_layout_patch(grid_system, layout_patch_ref);
	      const PatchDescriptor layout_patch = layout_patch_ref;
	      const PatchGridDescriptor grid_patch = grid_patch_ref;
	      const PatchGridDescriptor central_patch = grid_system.patches.front();
	      const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
	      const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
	      const auto fz = fluxes[2]->const_array(mfi);
#endif
	      const auto r = rhs.array(mfi);
	      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	         const PatchCoordinate cell_center = patch_layout_cell_center_coordinate(grid_patch, layout_patch, i, j, k);
	         const PatchCoordinate global = patch_local_to_global(grid_patch.map, cell_center);
	         const bool physical_owner =
	            layout_patch.nominal_patch_id == central_patch.patch_id
	            || !patch_grid_contains_global_point(central_patch, global, 1.0e-12_rt);
	         if (!physical_owner) {
	            for (int q = 0; q < NComp; ++q) {
	               r(i, j, k, q) = 0.0_rt;
	            }
	            return;
	         }
	         const amrex::Real volume = patch_cell_volume(grid_patch, cell_center);
	         for (int q = 0; q < NComp; ++q) {
	            const amrex::Real flux_xlo = fx(i, j, k, q);
	            const amrex::Real flux_xhi = fx(i + 1, j, k, q);
	            amrex::Real flux_ylo = 0.0_rt;
	            amrex::Real flux_yhi = 0.0_rt;
	            amrex::Real flux_zlo = 0.0_rt;
	            amrex::Real flux_zhi = 0.0_rt;
#if (AMREX_SPACEDIM >= 2)
	            flux_ylo = fy(i, j, k, q);
	            flux_yhi = fy(i, j + 1, k, q);
#endif
#if (AMREX_SPACEDIM == 3)
	            flux_zlo = fz(i, j, k, q);
	            flux_zhi = fz(i, j, k + 1, q);
#endif
	            r(i, j, k, q) = -finite_volume_divergence_from_face_integrals(
	               flux_xhi, flux_xlo, flux_yhi, flux_ylo, flux_zhi, flux_zlo, volume
	            );
	         }
	      });
	   }
	   }
}

template <int NComp, bool AddHydroSources>
inline void compute_rhs_from_fluxes_metric_impl(
   amrex::MultiFab& rhs,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const* primitive,
   amrex::Geometry const& geom,
   int coordinate,
   MeshAxes const& axes
)
{
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.tilebox();
	      const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
	      const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
	      const auto fz = fluxes[2]->const_array(mfi);
#endif
	      const auto prim = AddHydroSources ? primitive->const_array(mfi) : amrex::Array4<amrex::Real const>{};
	      const auto r = rhs.array(mfi);
	      amrex::ParallelFor(bx, [r, prim, coordinate, axes, problo, dx, AMREX_D_DECL(fx, fy, fz)] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	         const amrex::Real cell_volume = mesh_cell_volume_scale(coordinate, axes, problo, dx, i, j, k);
	         const amrex::Real area_xlo = mesh_face_area_scale(0, coordinate, axes, problo, dx, i, j, k);
	         const amrex::Real area_xhi = mesh_face_area_scale(0, coordinate, axes, problo, dx, i + 1, j, k);
#if (AMREX_SPACEDIM >= 2)
	         const amrex::Real area_ylo = mesh_face_area_scale(1, coordinate, axes, problo, dx, i, j, k);
	         const amrex::Real area_yhi = mesh_face_area_scale(1, coordinate, axes, problo, dx, i, j + 1, k);
#endif
#if (AMREX_SPACEDIM == 3)
	         const amrex::Real area_zlo = mesh_face_area_scale(2, coordinate, axes, problo, dx, i, j, k);
	         const amrex::Real area_zhi = mesh_face_area_scale(2, coordinate, axes, problo, dx, i, j, k + 1);
#endif
	         amrex::Real rho = 0.0_rt;
	         amrex::Real pressure = 0.0_rt;
	         StateVector3 momentum{};
	         StateVector3 magnetic{};
	         amrex::Real source_pressure = 0.0_rt;
	         amrex::Real radius = 0.0_rt;
	         amrex::Real theta = 0.0_rt;
	         if constexpr (AddHydroSources) {
	            radius = runtime_coordinate_for_axis(axes, MeshAxisR, -1, problo, dx, i, j, k);
	            theta = runtime_coordinate_for_axis(axes, MeshAxisTheta, -1, problo, dx, i, j, k);
	            rho = prim(i, j, k, RHO);
	            momentum = load_momentum_vector3(prim, i, j, k);
	            pressure = prim(i, j, k, ENERGY);
	            source_pressure = pressure;
	            if constexpr (has_mhd_components<NComp>()) {
	               magnetic = load_magnetic_vector3(prim, i, j, k);
	               source_pressure += 0.5_rt * dot(magnetic, magnetic);
	            }
	         }
	         for (int q = 0; q < NComp; ++q) {
	            amrex::Real flux_xlo = area_xlo * fx(i, j, k, q);
	            amrex::Real flux_xhi = area_xhi * fx(i + 1, j, k, q);
	            amrex::Real flux_ylo = 0.0_rt;
	            amrex::Real flux_yhi = 0.0_rt;
	            amrex::Real flux_zlo = 0.0_rt;
	            amrex::Real flux_zhi = 0.0_rt;
#if (AMREX_SPACEDIM >= 2)
	            flux_ylo = area_ylo * fy(i, j, k, q);
	            flux_yhi = area_yhi * fy(i, j + 1, k, q);
#endif
#if (AMREX_SPACEDIM == 3)
	            flux_zlo = area_zlo * fz(i, j, k, q);
	            flux_zhi = area_zhi * fz(i, j, k + 1, q);
#endif
	            int component_dir = -1;
	            if constexpr (AddHydroSources) {
	               if (q == MOM1 || q == MOM2 || q == MOM3) {
	                  const int component_axis = mesh_vector_axis_for_component(coordinate, axes, q - MOM1);
	                  component_dir = mesh_direction_for_axis(axes, component_axis);
	                  if (component_dir == 0) {
	                     flux_xlo -= source_pressure * area_xlo;
	                     flux_xhi -= source_pressure * area_xhi;
	                  }
#if (AMREX_SPACEDIM >= 2)
	                  if (component_dir == 1) {
	                     flux_ylo -= source_pressure * area_ylo;
	                     flux_yhi -= source_pressure * area_yhi;
	                  }
#endif
#if (AMREX_SPACEDIM == 3)
	                  if (component_dir == 2) {
	                     flux_zlo -= source_pressure * area_zlo;
	                     flux_zhi -= source_pressure * area_zhi;
	                  }
#endif
	               }
	            }
	            amrex::Real value = -finite_volume_divergence_from_face_integrals(
	               flux_xhi, flux_xlo, flux_yhi, flux_ylo, flux_zhi, flux_zlo, cell_volume
	            );
	            if constexpr (AddHydroSources) {
	               if (q == MOM1 || q == MOM2 || q == MOM3) {
	                  if constexpr (has_mhd_components<NComp>()) {
	                     value += curvilinear_mhd_stress_curvature_source(
	                        coordinate,
	                        axes,
	                        q,
	                        rho,
	                        momentum.x,
	                        momentum.y,
	                        momentum.z,
	                        magnetic.x,
	                        magnetic.y,
	                        magnetic.z,
	                        radius,
	                        theta
	                     );
	                  } else {
	                     value += curvilinear_momentum_curvature_source(
	                        coordinate,
	                        axes,
	                        q,
	                        rho,
	                        momentum.x,
	                        momentum.y,
	                        momentum.z,
	                        radius,
	                        theta
	                     );
	                  }
	               }
	            }
	            r(i, j, k, q) = value;
	         }
	      });
	   }
	   }
}

template <int NComp>
inline void compute_rhs_from_fluxes_mesh_divergence(
   amrex::MultiFab& rhs,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::Geometry const& geom,
   int coordinate,
   MeshAxes const& axes
)
{
   compute_rhs_from_fluxes_metric_impl<NComp, false>(rhs, fluxes, nullptr, geom, coordinate, axes);
}

template <int NComp>
inline void compute_rhs_from_physical_fluxes_metric(
   amrex::MultiFab& rhs,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::Geometry const& geom,
   int coordinate,
   MeshAxes const& axes
)
{
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto flux0 = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto flux1 = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto flux2 = fluxes[2]->const_array(mfi);
#endif
      const auto output = rhs.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real volume = mesh_cell_volume_scale(coordinate, axes, problo, dx, i, j, k);
         const MeshFaceMetric face0_lo = mesh_physical_face_metric(0, coordinate, axes, problo, dx, i, j, k);
         const MeshFaceMetric face0_hi = mesh_physical_face_metric(0, coordinate, axes, problo, dx, i + 1, j, k);
#if (AMREX_SPACEDIM >= 2)
         const MeshFaceMetric face1_lo = mesh_physical_face_metric(1, coordinate, axes, problo, dx, i, j, k);
         const MeshFaceMetric face1_hi = mesh_physical_face_metric(1, coordinate, axes, problo, dx, i, j + 1, k);
#endif
#if (AMREX_SPACEDIM == 3)
         const MeshFaceMetric face2_lo = mesh_physical_face_metric(2, coordinate, axes, problo, dx, i, j, k);
         const MeshFaceMetric face2_hi = mesh_physical_face_metric(2, coordinate, axes, problo, dx, i, j, k + 1);
#endif
         amrex::Real divergence[MaxStateComponents]{};
         for (int component = 0; component < NComp; ++component) {
            amrex::Real value = -(
               face0_hi.area * flux0(i + 1, j, k, component)
               - face0_lo.area * flux0(i, j, k, component)
            ) / volume;
#if (AMREX_SPACEDIM >= 2)
            value -= (
               face1_hi.area * flux1(i, j + 1, k, component)
               - face1_lo.area * flux1(i, j, k, component)
            ) / volume;
#endif
#if (AMREX_SPACEDIM == 3)
            value -= (
               face2_hi.area * flux2(i, j, k + 1, component)
               - face2_lo.area * flux2(i, j, k, component)
            ) / volume;
#endif
            divergence[component] = value;
         }
         const auto q = mesh_coordinate_triplet_at_location(
            coordinate, axes, problo, dx, i, j, k, false, false, false
         );
         const StateVector3 local_momentum = cartesian_vector_to_mesh_basis(
            coordinate,
            axes,
            q,
            {divergence[MOM1], divergence[MOM2], divergence[MOM3]}
         );
         divergence[MOM1] = local_momentum.x;
         divergence[MOM2] = local_momentum.y;
         divergence[MOM3] = local_momentum.z;
         for (int component = 0; component < NComp; ++component) {
            output(i, j, k, component) = divergence[component];
         }
      });
   }
   }
}

template <int NComp>
inline void compute_rhs_from_fluxes_metric(
   amrex::MultiFab& rhs,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   amrex::Geometry const& geom,
   int coordinate,
   MeshAxes const& axes
)
{
   compute_rhs_from_fluxes_metric_impl<NComp, true>(rhs, fluxes, &primitive, geom, coordinate, axes);
}

template <int NComp>
inline void compute_rhs_from_fluxes_masked(
   amrex::MultiFab& rhs,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::iMultiFab const& solid_mask,
   amrex::Geometry const& geom,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.tilebox();
      const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto fz = fluxes[2]->const_array(mfi);
#endif
	      const auto solid = solid_mask.const_array(mfi);
		      const auto r = rhs.array(mfi);
		      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
		      if (kind == MaskKindNonzero) {
		         continue;
		      }
	      if (kind == MaskKindZero) {
	         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	            for (int q = 0; q < NComp; ++q) {
	               amrex::Real value = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
	               value -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
	               value -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
	               r(i, j, k, q) = value;
	            }
	         });
	         continue;
	      }
	      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	         if (solid(i, j, k, 0) != 0) {
	            for (int q = 0; q < NComp; ++q) {
               r(i, j, k, q) = 0.0_rt;
            }
            return;
         }
         for (int q = 0; q < NComp; ++q) {
            amrex::Real value = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
            value -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
            value -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
            r(i, j, k, q) = value;
         }
      });
   }
   }
}

template <int NComp>
inline void compute_rhs_from_fluxes_cutcell(
   amrex::MultiFab& rhs,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::iMultiFab const& solid_mask,
   amrex::MultiFab const& volume_fraction,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& apertures,
   amrex::MultiFab const& primitive,
   amrex::Geometry const& geom,
   SolidGeometryEvaluator const& solid_params,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
   constexpr amrex::Real alpha_floor = 1.0e-12_rt;
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const amrex::Box& domain = geom.Domain();
   const auto lo = domain.smallEnd();
   const auto hi = domain.bigEnd();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto fx = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto fy = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto fz = fluxes[2]->const_array(mfi);
#endif
      const auto ax = apertures[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto ay = apertures[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto az = apertures[2]->const_array(mfi);
#endif
      const auto solid = solid_mask.const_array(mfi);
      const auto alpha = volume_fraction.const_array(mfi);
      const auto p = primitive.const_array(mfi);
	      const auto r = rhs.array(mfi);
	      const int kind = cached_or_region_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
	      if (kind == MaskKindNonzero) {
	         continue;
	      }
      if (kind == MaskKindZero) {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            for (int q = 0; q < NComp; ++q) {
               amrex::Real value = -(fx(i + 1, j, k, q) - fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
               value -= (fy(i, j + 1, k, q) - fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
               value -= (fz(i, j, k + 1, q) - fz(i, j, k, q)) / dx[2];
#endif
               r(i, j, k, q) = value;
            }
         });
         continue;
      }
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real fluid_fraction = alpha(i, j, k, 0);
         if (solid(i, j, k, 0) != 0 || fluid_fraction <= alpha_floor) {
            for (int q = 0; q < NComp; ++q) {
               r(i, j, k, q) = 0.0_rt;
            }
            return;
         }

         amrex::Real ax_l = ax(i, j, k, 0);
         if (i > lo[0] && alpha(i - 1, j, k, 0) <= alpha_floor) {
            ax_l = 0.0_rt;
         }
         amrex::Real ax_r = ax(i + 1, j, k, 0);
         if (i < hi[0] && alpha(i + 1, j, k, 0) <= alpha_floor) {
            ax_r = 0.0_rt;
         }
#if (AMREX_SPACEDIM >= 2)
         amrex::Real ay_l = ay(i, j, k, 0);
         if (j > lo[1] && alpha(i, j - 1, k, 0) <= alpha_floor) {
            ay_l = 0.0_rt;
         }
         amrex::Real ay_r = ay(i, j + 1, k, 0);
         if (j < hi[1] && alpha(i, j + 1, k, 0) <= alpha_floor) {
            ay_r = 0.0_rt;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         amrex::Real az_l = az(i, j, k, 0);
         if (k > lo[2] && alpha(i, j, k - 1, 0) <= alpha_floor) {
            az_l = 0.0_rt;
         }
         amrex::Real az_r = az(i, j, k + 1, 0);
         if (k < hi[2] && alpha(i, j, k + 1, 0) <= alpha_floor) {
            az_r = 0.0_rt;
         }
#endif
         const amrex::Real inv_alpha = 1.0_rt / fluid_fraction;
         for (int q = 0; q < NComp; ++q) {
            amrex::Real value = -(ax_r * fx(i + 1, j, k, q) - ax_l * fx(i, j, k, q)) / dx[0];
#if (AMREX_SPACEDIM >= 2)
            value -= (ay_r * fy(i, j + 1, k, q) - ay_l * fy(i, j, k, q)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
            value -= (az_r * fz(i, j, k + 1, q) - az_l * fz(i, j, k, q)) / dx[2];
#endif
            r(i, j, k, q) = value * inv_alpha;
         }

         const amrex::Real pressure = p(i, j, k, ENERGY);
         const amrex::Real sx = (ax_r - ax_l) / dx[0];
         r(i, j, k, MOM1) += pressure * sx * inv_alpha;
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real sy = (ay_r - ay_l) / dx[1];
         r(i, j, k, MOM2) += pressure * sy * inv_alpha;
#else
         const amrex::Real sy = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real sz = (az_r - az_l) / dx[2];
         r(i, j, k, MOM3) += pressure * sz * inv_alpha;
#else
         const amrex::Real sz = 0.0_rt;
#endif
         if (solid_params.motion == SolidMotionTranslate || solid_params.motion == SolidMotionPulse) {
            const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5_rt) * dx[0];
#if (AMREX_SPACEDIM >= 2)
            const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5_rt) * dx[1];
#else
            const amrex::Real y = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real z = problo[2] + (static_cast<amrex::Real>(k) + 0.5_rt) * dx[2];
#else
            const amrex::Real z = 0.0_rt;
#endif
            const amrex::Real wall_vx = solid_params.velocity_component(solid_params.time, 0, x, y, z);
#if (AMREX_SPACEDIM >= 2)
            const amrex::Real wall_vy = solid_params.velocity_component(solid_params.time, 1, x, y, z);
#else
            const amrex::Real wall_vy = 0.0_rt;
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real wall_vz = solid_params.velocity_component(solid_params.time, 2, x, y, z);
#else
            const amrex::Real wall_vz = 0.0_rt;
#endif
            r(i, j, k, ENERGY) += pressure * (wall_vx * sx + wall_vy * sy + wall_vz * sz) * inv_alpha;
         }
      });
   }
   }
}



} // namespace pyfluid_amrex
