#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/ct_fields.hpp"
#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/kernels_generated/vector_ops.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

inline void fill_cell_divergence(
   amrex::MultiFab& divb,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   bool absolute_value
)
{
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto d = divb.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real dbx_dx = (s(i + 1, j, k, B1) - s(i - 1, j, k, B1)) / (2.0 * dx[0]);
         amrex::Real dby_dy = 0.0;
#if (AMREX_SPACEDIM >= 2)
         dby_dy = (s(i, j + 1, k, B2) - s(i, j - 1, k, B2)) / (2.0 * dx[1]);
#endif
         amrex::Real dbz_dz = 0.0;
#if (AMREX_SPACEDIM == 3)
         dbz_dz = (s(i, j, k + 1, B3) - s(i, j, k - 1, B3)) / (2.0 * dx[2]);
#endif
         const amrex::Real divergence = generated::divergence_cartesian(dbx_dx, dby_dy, dbz_dz);
         d(i, j, k, 0) = absolute_value ? amrex::Math::abs(divergence) : divergence;
      });
   }
   }
}

inline void fill_magnetic_energy_density(
   amrex::MultiFab& magnetic_energy,
   amrex::MultiFab const& state
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto e = magnetic_energy.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         e(i, j, k, 0) = magnetic_energy_density(s(i, j, k, B1), s(i, j, k, B2), s(i, j, k, B3));
      });
   }
   }
}

inline void fill_face_divergence(
   amrex::MultiFab& divb,
   FaceMagneticField const& face_b,
   amrex::Geometry const& geom,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   bool absolute_value = true
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(divb, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto d = divb.array(mfi);
      const auto bx_face = face_b.b[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto by_face = face_b.b[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto bz_face = face_b.b[2]->const_array(mfi);
#endif
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real xlo_area =
            mesh_face_area_scale(0, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
         const amrex::Real xhi_area =
            mesh_face_area_scale(0, mesh_coordinate, mesh_axes, problo, dx, i + 1, j, k);
         const amrex::Real xlo_flux = generated::magnetic_face_flux(bx_face(i, j, k, 0), xlo_area);
         const amrex::Real xhi_flux = generated::magnetic_face_flux(bx_face(i + 1, j, k, 0), xhi_area);
         const amrex::Real volume = mesh_cell_volume_scale(mesh_coordinate, mesh_axes, problo, dx, i, j, k);
#if (AMREX_SPACEDIM == 1)
         amrex::Real divergence = generated::finite_volume_divergence_1d(xhi_flux, xlo_flux, volume);
#else
         const amrex::Real ylo_area =
            mesh_face_area_scale(1, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
         const amrex::Real yhi_area =
            mesh_face_area_scale(1, mesh_coordinate, mesh_axes, problo, dx, i, j + 1, k);
         const amrex::Real ylo_flux = generated::magnetic_face_flux(by_face(i, j, k, 0), ylo_area);
         const amrex::Real yhi_flux = generated::magnetic_face_flux(by_face(i, j + 1, k, 0), yhi_area);
#if (AMREX_SPACEDIM == 2)
         amrex::Real divergence =
            generated::finite_volume_divergence_2d(xhi_flux, xlo_flux, yhi_flux, ylo_flux, volume);
#else
         const amrex::Real zlo_area =
            mesh_face_area_scale(2, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
         const amrex::Real zhi_area =
            mesh_face_area_scale(2, mesh_coordinate, mesh_axes, problo, dx, i, j, k + 1);
         const amrex::Real zlo_flux = generated::magnetic_face_flux(bz_face(i, j, k, 0), zlo_area);
         const amrex::Real zhi_flux = generated::magnetic_face_flux(bz_face(i, j, k + 1, 0), zhi_area);
         amrex::Real divergence = generated::finite_volume_divergence_3d(
            xhi_flux,
            xlo_flux,
            yhi_flux,
            ylo_flux,
            zhi_flux,
            zlo_flux,
            volume
         );
#endif
#endif
         d(i, j, k, 0) = absolute_value ? amrex::Math::abs(divergence) : divergence;
      });
   }
   }
}

inline void fill_face_divergence(
   amrex::MultiFab& divb,
   FaceMagneticField const& face_b,
   amrex::Geometry const& geom,
   bool absolute_value = true
)
{
   fill_face_divergence(
      divb,
      face_b,
      geom,
      MeshCartesian,
      mesh_default_axes_layout(MeshCartesian),
      absolute_value
   );
}

} // namespace pyfluid_amrex
