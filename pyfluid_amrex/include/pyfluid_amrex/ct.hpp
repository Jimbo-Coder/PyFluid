#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <memory>

#include "pyfluid_amrex/boundary.hpp"
#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/ct_diagnostics.hpp"
#include "pyfluid_amrex/ct_fields.hpp"
#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/kernels_generated/scalar_ops.hpp"
#include "pyfluid_amrex/kernels_generated/vector_ops.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

inline void enforce_edge_electric_periodicity(EdgeElectricField& edge_e, amrex::Geometry const& geom);
inline void enforce_edge_electric_boundary(
   EdgeElectricField& edge_e,
   amrex::Geometry const& geom,
   RunConfig const& cfg
);

inline bool ct_enabled(RunConfig const& cfg)
{
   return is_mhd(cfg) && cfg.ct == "on";
}

inline void fill_face_b_from_cell_b(FaceMagneticField& face_b, amrex::MultiFab const& state)
{
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::MultiFab& face = *face_b.b[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(face, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box bx = amrex::convert(
            state.boxArray()[mfi.index()],
            amrex::IntVect::TheDimensionVector(dir)
         );
         const auto s = state.const_array(mfi);
         const auto f = face.array(mfi);
         const int d = dir;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (d == 0) {
               f(i, j, k, 0) = 0.5 * (s(i - 1, j, k, B1) + s(i, j, k, B1));
            }
#if (AMREX_SPACEDIM >= 2)
            if (d == 1) {
               f(i, j, k, 0) = 0.5 * (s(i, j - 1, k, B2) + s(i, j, k, B2));
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (d == 2) {
               f(i, j, k, 0) = 0.5 * (s(i, j, k - 1, B3) + s(i, j, k, B3));
            }
#endif
         });
      }
      }
   }
}

inline void fill_uniform_face_b(
   FaceMagneticField& face_b,
   amrex::Real bx,
   amrex::Real by,
   amrex::Real bz
)
{
   face_b.b[0]->setVal(bx, 0, 1, face_b.b[0]->nGrowVect());
#if (AMREX_SPACEDIM >= 2)
   face_b.b[1]->setVal(by, 0, 1, face_b.b[1]->nGrowVect());
#else
   (void)by;
#endif
#if (AMREX_SPACEDIM == 3)
   face_b.b[2]->setVal(bz, 0, 1, face_b.b[2]->nGrowVect());
#else
   (void)bz;
#endif
}

inline void enforce_physical_face_divergence_boundary(
   FaceMagneticField& face_b,
   amrex::Geometry const& geom
)
{
   const amrex::Box domain = geom.Domain();
   const auto cell_lo = amrex::lbound(domain);
   const auto cell_hi = amrex::ubound(domain);
   const auto is_periodic = geom.isPeriodicArray();
   const auto dx = geom.CellSizeArray();

   if (!is_periodic[0]) {
      amrex::MultiFab& bx_face_mf = *face_b.b[0];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(bx_face_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box bx = mfi.tilebox();
         const auto bx_face = bx_face_mf.array(mfi);
#if (AMREX_SPACEDIM >= 2)
         const auto by_face = face_b.b[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
         const auto bz_face = face_b.b[2]->const_array(mfi);
#endif
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (i != cell_lo.x && i != cell_hi.x + 1) {
               return;
            }
#if (AMREX_SPACEDIM >= 2)
            if (j < cell_lo.y || j > cell_hi.y) {
               return;
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (k < cell_lo.z || k > cell_hi.z) {
               return;
            }
#endif
            const int cell_i = (i == cell_lo.x) ? cell_lo.x : cell_hi.x;
            amrex::Real tangential_div = 0.0;
#if (AMREX_SPACEDIM >= 2)
            tangential_div += (by_face(cell_i, j + 1, k, 0) - by_face(cell_i, j, k, 0)) / dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
            tangential_div += (bz_face(cell_i, j, k + 1, 0) - bz_face(cell_i, j, k, 0)) / dx[2];
#endif
            if (i == cell_lo.x) {
               bx_face(i, j, k, 0) = bx_face(i + 1, j, k, 0) + dx[0] * tangential_div;
            } else {
               bx_face(i, j, k, 0) = bx_face(i - 1, j, k, 0) - dx[0] * tangential_div;
            }
         });
      }
      }
   }

#if (AMREX_SPACEDIM >= 2)
   if (!is_periodic[1]) {
      amrex::MultiFab& by_face_mf = *face_b.b[1];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(by_face_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box bx = mfi.tilebox();
         const auto bx_face = face_b.b[0]->const_array(mfi);
         const auto by_face = by_face_mf.array(mfi);
#if (AMREX_SPACEDIM == 3)
         const auto bz_face = face_b.b[2]->const_array(mfi);
#endif
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (j != cell_lo.y && j != cell_hi.y + 1) {
               return;
            }
            if (!is_periodic[0] && (i == cell_lo.x || i == cell_hi.x)) {
               return;
            }
#if (AMREX_SPACEDIM == 3)
            if (k < cell_lo.z || k > cell_hi.z) {
               return;
            }
#endif
            const int cell_j = (j == cell_lo.y) ? cell_lo.y : cell_hi.y;
            amrex::Real tangential_div = (bx_face(i + 1, cell_j, k, 0) - bx_face(i, cell_j, k, 0)) / dx[0];
#if (AMREX_SPACEDIM == 3)
            tangential_div += (bz_face(i, cell_j, k + 1, 0) - bz_face(i, cell_j, k, 0)) / dx[2];
#endif
            if (j == cell_lo.y) {
               by_face(i, j, k, 0) = by_face(i, j + 1, k, 0) + dx[1] * tangential_div;
            } else {
               by_face(i, j, k, 0) = by_face(i, j - 1, k, 0) - dx[1] * tangential_div;
            }
         });
      }
      }
   }
#endif

#if (AMREX_SPACEDIM == 3)
   if (!is_periodic[2]) {
      amrex::MultiFab& bz_face_mf = *face_b.b[2];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(bz_face_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box bx = mfi.tilebox();
         const auto bx_face = face_b.b[0]->const_array(mfi);
         const auto by_face = face_b.b[1]->const_array(mfi);
         const auto bz_face = bz_face_mf.array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (k != cell_lo.z && k != cell_hi.z + 1) {
               return;
            }
            if (!is_periodic[0] && (i == cell_lo.x || i == cell_hi.x)) {
               return;
            }
            if (!is_periodic[1] && (j == cell_lo.y || j == cell_hi.y)) {
               return;
            }
            const int cell_k = (k == cell_lo.z) ? cell_lo.z : cell_hi.z;
            const amrex::Real tangential_div =
               (bx_face(i + 1, j, cell_k, 0) - bx_face(i, j, cell_k, 0)) / dx[0]
               + (by_face(i, j + 1, cell_k, 0) - by_face(i, j, cell_k, 0)) / dx[1];
            if (k == cell_lo.z) {
               bz_face(i, j, k, 0) = bz_face(i, j, k + 1, 0) + dx[2] * tangential_div;
            } else {
               bz_face(i, j, k, 0) = bz_face(i, j, k - 1, 0) - dx[2] * tangential_div;
            }
         });
      }
      }
   }
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int mirrored_origin_face_index(int value, int lo, int hi, bool nodal_on_origin) noexcept
{
   if (value < lo) {
      const int mirrored = nodal_on_origin ? lo + (lo - value) : lo + (lo - 1 - value);
      return clamp_index(mirrored, lo, hi);
   }
   return clamp_index(value, lo, hi);
}

inline void fill_nonfinite_physical_face_boundary(
   FaceMagneticField& face_b,
   amrex::Geometry const& geom
)
{
   const auto is_periodic = geom.isPeriodicArray();
   bool all_periodic = true;
   for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
      all_periodic = all_periodic && is_periodic[dim];
   }
   if (all_periodic) {
      return;
   }

   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::MultiFab& face = *face_b.b[dir];
      const amrex::Box face_domain = amrex::convert(geom.Domain(), amrex::IntVect::TheDimensionVector(dir));
      const auto lo = amrex::lbound(face_domain);
      const auto hi = amrex::ubound(face_domain);
      const int ng = face.nGrow();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(face, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto f = face.array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (amrex::Math::isfinite(f(i, j, k, 0))) {
               return;
            }

            int ii = i;
            int jj = j;
            int kk = k;
            if (!is_periodic[0]) {
               if (i <= lo.x + ng - 1) {
                  ii = (lo.x + ng <= hi.x) ? lo.x + ng : hi.x;
               } else if (i >= hi.x - ng + 1) {
                  ii = (hi.x - ng >= lo.x) ? hi.x - ng : lo.x;
               }
            }
#if (AMREX_SPACEDIM >= 2)
            if (!is_periodic[1]) {
               if (j <= lo.y + ng - 1) {
                  jj = (lo.y + ng <= hi.y) ? lo.y + ng : hi.y;
               } else if (j >= hi.y - ng + 1) {
                  jj = (hi.y - ng >= lo.y) ? hi.y - ng : lo.y;
               }
            }
#else
            (void)jj;
#endif
#if (AMREX_SPACEDIM == 3)
            if (!is_periodic[2]) {
               if (k <= lo.z + ng - 1) {
                  kk = (lo.z + ng <= hi.z) ? lo.z + ng : hi.z;
               } else if (k >= hi.z - ng + 1) {
                  kk = (hi.z - ng >= lo.z) ? hi.z - ng : lo.z;
               }
            }
#else
            (void)kk;
#endif
            f(i, j, k, 0) = f(ii, jj, kk, 0);
         });
      }
      }
   }
}

inline void apply_face_b_boundary(
   FaceMagneticField& face_b,
   amrex::Geometry const& geom,
   BoundaryData const& bc
)
{
   validate_origin_boundary_geometry(bc, geom);
   const auto geom_periodic = geom.isPeriodicArray();
   bool all_periodic = true;
   for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
      all_periodic = all_periodic && geom_periodic[dim];
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      face_b.b[dir]->FillBoundary(geom.periodicity());
   }
   if (!all_periodic) {
      fill_nonfinite_physical_face_boundary(face_b, geom);
      if (bc.mesh_coordinate == MeshCartesian) {
         enforce_physical_face_divergence_boundary(face_b, geom);
      }
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::MultiFab& face = *face_b.b[dir];
      face.FillBoundary(geom.periodicity());
      if (all_periodic) {
         continue;
      }

      const amrex::Box face_domain = amrex::convert(geom.Domain(), amrex::IntVect::TheDimensionVector(dir));
      const amrex::Box cell_domain = geom.Domain();
      const auto lo = amrex::lbound(face_domain);
      const auto hi = amrex::ubound(face_domain);
      const auto is_periodic = geom.isPeriodicArray();
      const int ng = face.nGrow();
      const int xlo_bc = bc.lo_kind[0];
      const int xhi_bc = bc.hi_kind[0];
#if (AMREX_SPACEDIM >= 2)
      const int ylo_bc = bc.lo_kind[1];
      const int yhi_bc = bc.hi_kind[1];
#endif
#if (AMREX_SPACEDIM == 3)
      const int zlo_bc = bc.lo_kind[2];
      const int zhi_bc = bc.hi_kind[2];
#endif
      const int reflecting_bc = static_cast<int>(BoundaryKind::Reflecting);
      const int symmetry_bc = static_cast<int>(BoundaryKind::Symmetry);
      const int inflow_bc = static_cast<int>(BoundaryKind::Inflow);
      const int origin_bc = static_cast<int>(BoundaryKind::Origin);
      const auto problo = geom.ProbLoArray();
      const auto dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(face, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box bx = amrex::grow(mfi.tilebox(), ng);
         const auto f = face.array(mfi);
         const int d = dir;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const int face_boundary_dir = face_inflow_boundary_dir(bc, cell_domain, dir, i, j, k);
            const bool touches_inflow = face_boundary_dir >= 0;
            bool outside = touches_inflow || (!is_periodic[0] && (i < lo.x || i > hi.x));
#if (AMREX_SPACEDIM >= 2)
            outside = outside || (!is_periodic[1] && (j < lo.y || j > hi.y));
#endif
#if (AMREX_SPACEDIM == 3)
            outside = outside || (!is_periodic[2] && (k < lo.z || k > hi.z));
#endif
            if (!outside) {
               return;
            }

            const bool xlo = !is_periodic[0] && i < lo.x;
            const bool xhi = !is_periodic[0] && i > hi.x;
#if (AMREX_SPACEDIM >= 2)
            const bool ylo = !is_periodic[1] && j < lo.y;
            const bool yhi = !is_periodic[1] && j > hi.y;
#else
            const bool ylo = false;
            const bool yhi = false;
#endif
#if (AMREX_SPACEDIM == 3)
            const bool zlo = !is_periodic[2] && k < lo.z;
            const bool zhi = !is_periodic[2] && k > hi.z;
#else
            const bool zlo = false;
            const bool zhi = false;
#endif

            const bool x_origin = xlo && xlo_bc == origin_bc;
            const bool x_reflect = (xlo && (xlo_bc == reflecting_bc || xlo_bc == symmetry_bc || xlo_bc == origin_bc))
               || (xhi && (xhi_bc == reflecting_bc || xhi_bc == symmetry_bc));
            const bool x_symmetry = (xlo && xlo_bc == symmetry_bc) || (xhi && xhi_bc == symmetry_bc);
#if (AMREX_SPACEDIM >= 2)
            const bool y_origin = ylo && ylo_bc == origin_bc;
            const bool y_reflect = (ylo && (ylo_bc == reflecting_bc || ylo_bc == symmetry_bc || ylo_bc == origin_bc))
               || (yhi && (yhi_bc == reflecting_bc || yhi_bc == symmetry_bc));
            const bool y_symmetry = (ylo && ylo_bc == symmetry_bc) || (yhi && yhi_bc == symmetry_bc);
#else
            const bool y_origin = false;
            const bool y_reflect = false;
            const bool y_symmetry = false;
#endif
#if (AMREX_SPACEDIM == 3)
            const bool z_origin = zlo && zlo_bc == origin_bc;
            const bool z_reflect = (zlo && (zlo_bc == reflecting_bc || zlo_bc == symmetry_bc || zlo_bc == origin_bc))
               || (zhi && (zhi_bc == reflecting_bc || zhi_bc == symmetry_bc));
            const bool z_symmetry = (zlo && zlo_bc == symmetry_bc) || (zhi && zhi_bc == symmetry_bc);
#else
            const bool z_origin = false;
            const bool z_reflect = false;
            const bool z_symmetry = false;
#endif
            const bool ghost_fixed_inflow = (xlo && xlo_bc == inflow_bc) || (xhi && xhi_bc == inflow_bc)
#if (AMREX_SPACEDIM >= 2)
               || (ylo && ylo_bc == inflow_bc) || (yhi && yhi_bc == inflow_bc)
#endif
#if (AMREX_SPACEDIM == 3)
               || (zlo && zlo_bc == inflow_bc) || (zhi && zhi_bc == inflow_bc)
#endif
               ;

            const int ii = is_periodic[0] ? i : (
               x_origin ? mirrored_origin_face_index(i, lo.x, hi.x, d == 0) :
               (x_reflect ? mirrored_index(i, lo.x, hi.x) : clamp_index(i, lo.x, hi.x))
            );
#if (AMREX_SPACEDIM >= 2)
            const int jj = is_periodic[1] ? j : (
               y_origin ? mirrored_origin_face_index(j, lo.y, hi.y, d == 1) :
               (y_reflect ? mirrored_index(j, lo.y, hi.y) : clamp_index(j, lo.y, hi.y))
            );
#else
            const int jj = 0;
#endif
#if (AMREX_SPACEDIM == 3)
            const int kk = is_periodic[2] ? k : (
               z_origin ? mirrored_origin_face_index(k, lo.z, hi.z, d == 2) :
               (z_reflect ? mirrored_index(k, lo.z, hi.z) : clamp_index(k, lo.z, hi.z))
            );
#else
            const int kk = 0;
#endif
            int boundary_dir = face_boundary_dir;
            if (boundary_dir < 0 && ghost_fixed_inflow) {
               boundary_dir = xlo || xhi ? 0 : -1;
#if (AMREX_SPACEDIM >= 2)
               if (boundary_dir < 0 && (ylo || yhi)) {
                  boundary_dir = 1;
               }
#endif
#if (AMREX_SPACEDIM == 3)
               if (boundary_dir < 0 && (zlo || zhi)) {
                  boundary_dir = 2;
               }
#endif
            }
            bool fixed_inflow = false;
            if (boundary_dir >= 0) {
               const amrex::Real x =
                  problo[0] + (static_cast<amrex::Real>(i) + (dir == 0 ? 0.0 : 0.5)) * dx[0];
#if (AMREX_SPACEDIM >= 2)
               const amrex::Real y =
                  problo[1] + (static_cast<amrex::Real>(j) + (dir == 1 ? 0.0 : 0.5)) * dx[1];
#else
               const amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
               const amrex::Real z =
                  problo[2] + (static_cast<amrex::Real>(k) + (dir == 2 ? 0.0 : 0.5)) * dx[2];
#else
               const amrex::Real z = 0.0;
#endif
               fixed_inflow = inflow_profile_is_fixed(bc, boundary_dir, x, y, z);
            }
            if (fixed_inflow) {
               f(i, j, k, 0) = inflow_face_b_component(bc, dir);
            } else {
               amrex::Real sign = 1.0;
               if (x_origin) {
                  sign *= origin_state_parity(bc, 0, B1 + d);
               }
#if (AMREX_SPACEDIM >= 2)
               if (y_origin) {
                  sign *= origin_state_parity(bc, 1, B1 + d);
               }
#endif
#if (AMREX_SPACEDIM == 3)
               if (z_origin) {
                  sign *= origin_state_parity(bc, 2, B1 + d);
               }
#endif
               if (x_symmetry && dir != 0) {
                  sign = -sign;
               }
#if (AMREX_SPACEDIM >= 2)
               if (y_symmetry && dir != 1) {
                  sign = -sign;
               }
#endif
#if (AMREX_SPACEDIM == 3)
               if (z_symmetry && dir != 2) {
                  sign = -sign;
               }
#endif
               f(i, j, k, 0) = sign * f(ii, jj, kk, 0);
            }
         });
      }
      }
   }

   const int radial_dir = mesh_direction_for_axis(bc.mesh_axes, MeshAxisR);
   if (radial_dir >= 0 && bc.origin_lo[radial_dir] != 0 && face_b.b[radial_dir]) {
      amrex::MultiFab& radial_face = *face_b.b[radial_dir];
      const amrex::Box face_domain = amrex::convert(geom.Domain(), amrex::IntVect::TheDimensionVector(radial_dir));
      const int origin_index = face_domain.smallEnd(radial_dir);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(radial_face, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box bx = mfi.tilebox();
         const auto f = radial_face.array(mfi);
         const int d = radial_dir;
         const int o = origin_index;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            bool on_origin = (d == 0 && i == o);
#if (AMREX_SPACEDIM >= 2)
            on_origin = on_origin || (d == 1 && j == o);
#endif
#if (AMREX_SPACEDIM == 3)
            on_origin = on_origin || (d == 2 && k == o);
#endif
            if (on_origin) {
               f(i, j, k, 0) = 0.0;
            }
         });
      }
      }
   }
}

inline void apply_face_b_boundary(
   FaceMagneticField& face_b,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
   apply_face_b_boundary(face_b, geom, make_boundary_data(cfg));
}

inline void sync_cell_b_from_faces_with_axis_mask(
   amrex::MultiFab& state,
   FaceMagneticField const& face_b,
   bool preserve_pressure,
   bool sync_bx,
   bool sync_by,
   bool sync_bz
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.array(mfi);
      const auto bx_face = face_b.b[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto by_face = face_b.b[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto bz_face = face_b.b[2]->const_array(mfi);
#endif
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real old_bx = s(i, j, k, B1);
         const amrex::Real old_by = s(i, j, k, B2);
         const amrex::Real old_bz = s(i, j, k, B3);
         const amrex::Real face_bx = 0.5 * (bx_face(i, j, k, 0) + bx_face(i + 1, j, k, 0));
         const amrex::Real new_bx = sync_bx ? face_bx : old_bx;
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real face_by = 0.5 * (by_face(i, j, k, 0) + by_face(i, j + 1, k, 0));
         const amrex::Real new_by = sync_by ? face_by : old_by;
#else
         const amrex::Real new_by = old_by;
         (void)sync_by;
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real face_bz = 0.5 * (bz_face(i, j, k, 0) + bz_face(i, j, k + 1, 0));
         const amrex::Real new_bz = sync_bz ? face_bz : old_bz;
#else
         const amrex::Real new_bz = old_bz;
         (void)sync_bz;
#endif
         if (preserve_pressure) {
            const amrex::Real old_mag = magnetic_energy_density(old_bx, old_by, old_bz);
            const amrex::Real new_mag = magnetic_energy_density(new_bx, new_by, new_bz);
            s(i, j, k, ENERGY) += new_mag - old_mag;
            if (s(i, j, k, ENERGY) < 1.0e-300) {
               s(i, j, k, ENERGY) = 1.0e-300;
            }
         }
         s(i, j, k, B1) = new_bx;
         s(i, j, k, B2) = new_by;
         s(i, j, k, B3) = new_bz;
      });
   }
   }
}

inline void sync_cell_b_from_faces(
   amrex::MultiFab& state,
   FaceMagneticField const& face_b,
   bool preserve_pressure
)
{
   sync_cell_b_from_faces_with_axis_mask(state, face_b, preserve_pressure, true, true, true);
}

inline void sync_cell_b_from_faces(
   amrex::MultiFab& state,
   FaceMagneticField const& face_b,
   amrex::Geometry const& geom,
   bool preserve_pressure
)
{
   const amrex::Box domain = geom.Domain();
   const bool sync_bx = domain.length(0) > 1;
#if (AMREX_SPACEDIM >= 2)
   const bool sync_by = domain.length(1) > 1;
#else
   const bool sync_by = true;
#endif
#if (AMREX_SPACEDIM == 3)
   const bool sync_bz = domain.length(2) > 1;
#else
   const bool sync_bz = true;
#endif
   sync_cell_b_from_faces_with_axis_mask(state, face_b, preserve_pressure, sync_bx, sync_by, sync_bz);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cell_current_x(
   amrex::Array4<amrex::Real const> const& s,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx
) noexcept
{
   amrex::Real dbz_dy = 0.0;
#if (AMREX_SPACEDIM >= 2)
   dbz_dy = (s(i, j + 1, k, B3) - s(i, j - 1, k, B3)) / (2.0 * dx[1]);
#endif
   amrex::Real dby_dz = 0.0;
#if (AMREX_SPACEDIM == 3)
   dby_dz = (s(i, j, k + 1, B2) - s(i, j, k - 1, B2)) / (2.0 * dx[2]);
#endif
   return generated::curl_x_cartesian(dbz_dy, dby_dz);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cell_current_y(
   amrex::Array4<amrex::Real const> const& s,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx
) noexcept
{
   amrex::Real dbx_dz = 0.0;
#if (AMREX_SPACEDIM == 3)
   dbx_dz = (s(i, j, k + 1, B1) - s(i, j, k - 1, B1)) / (2.0 * dx[2]);
#endif
   const amrex::Real dbz_dx = (s(i + 1, j, k, B3) - s(i - 1, j, k, B3)) / (2.0 * dx[0]);
   return generated::curl_y_cartesian(dbx_dz, dbz_dx);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cell_current_z(
   amrex::Array4<amrex::Real const> const& s,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx
) noexcept
{
   const amrex::Real dby_dx = (s(i + 1, j, k, B2) - s(i - 1, j, k, B2)) / (2.0 * dx[0]);
   amrex::Real dbx_dy = 0.0;
#if (AMREX_SPACEDIM >= 2)
   dbx_dy = (s(i, j + 1, k, B1) - s(i, j - 1, k, B1)) / (2.0 * dx[1]);
#endif
   return generated::curl_z_cartesian(dby_dx, dbx_dy);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cell_emf_component(
   amrex::Array4<amrex::Real const> const& s,
   int i,
   int j,
   int k,
   int component,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real eta
) noexcept
{
   const amrex::Real rho = s(i, j, k, RHO);
   const StateVector3 momentum = load_momentum_vector3(s, i, j, k);
   const StateVector3 velocity = velocity_from_momentum(momentum, rho);
   const StateVector3 magnetic = load_magnetic_vector3(s, i, j, k);
   if (component == 0) {
      const amrex::Real ideal = generated::ideal_emf_x(
         velocity.y, velocity.z, magnetic.y, magnetic.z
      );
      return eta == 0.0 ? ideal : generated::total_emf(ideal, eta, cell_current_x(s, i, j, k, dx));
   }
   if (component == 1) {
      const amrex::Real ideal = generated::ideal_emf_y(
         velocity.z, velocity.x, magnetic.z, magnetic.x
      );
      return eta == 0.0 ? ideal : generated::total_emf(ideal, eta, cell_current_y(s, i, j, k, dx));
   }
   const amrex::Real ideal = generated::ideal_emf_z(
      velocity.x, velocity.y, magnetic.x, magnetic.y
   );
   return eta == 0.0 ? ideal : generated::total_emf(ideal, eta, cell_current_z(s, i, j, k, dx));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cell_resistive_emf_component(
   amrex::Array4<amrex::Real const> const& s,
   int i,
   int j,
   int k,
   int component,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real eta
) noexcept
{
   if (component == 0) {
      return generated::resistive_emf(eta, cell_current_x(s, i, j, k, dx));
   }
   if (component == 1) {
      return generated::resistive_emf(eta, cell_current_y(s, i, j, k, dx));
   }
   return generated::resistive_emf(eta, cell_current_z(s, i, j, k, dx));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ct_face_flux(
   amrex::Real face_b,
   int face_dir,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   const amrex::Real area = mesh_face_area_scale(face_dir, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
   return generated::magnetic_face_flux(face_b, area);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ct_face_b_from_flux(
   amrex::Real flux,
   int face_dir,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   const amrex::Real area = mesh_face_area_scale(face_dir, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
   if (area == 0.0) {
      // The radial-origin cap has zero area, so its finite-volume magnetic flux is identically zero.
      return 0.0;
   }
   return generated::magnetic_face_component(flux, area);
}

inline void fill_face_fillpatch_from_face_b(
   FaceMagneticField& face_fillpatch,
   FaceMagneticField const& face_b,
   amrex::Geometry const& geom,
   int mesh_coordinate,
   MeshAxes mesh_axes
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::MultiFab& dst = *face_fillpatch.b[dir];
      amrex::MultiFab const& src = *face_b.b[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.growntilebox(dst.nGrowVect());
         const auto out = dst.array(mfi);
         const auto in = src.const_array(mfi);
         const int d = dir;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real physical_flux =
               ct_face_flux(in(i, j, k, 0), d, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            const amrex::Real logical_area = cartesian_face_area_scale(d, dx);
            out(i, j, k, 0) = physical_flux / logical_area;
         });
      }
      }
   }
}

inline void fill_face_b_from_face_fillpatch(
   FaceMagneticField& face_b,
   FaceMagneticField const& face_fillpatch,
   amrex::Geometry const& geom,
   int mesh_coordinate,
   MeshAxes mesh_axes
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::MultiFab& dst = *face_b.b[dir];
      amrex::MultiFab const& src = *face_fillpatch.b[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.growntilebox(dst.nGrowVect());
         const auto out = dst.array(mfi);
         const auto in = src.const_array(mfi);
         const int d = dir;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real logical_area = cartesian_face_area_scale(d, dx);
            const amrex::Real physical_flux = in(i, j, k, 0) * logical_area;
            out(i, j, k, 0) = ct_face_b_from_flux(physical_flux, d, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
         });
      }
      }
   }
}

inline void fill_face_fillpatch_from_face_b(
   FaceMagneticField& face_fillpatch,
   FaceMagneticField const& face_b,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
   fill_face_fillpatch_from_face_b(
      face_fillpatch,
      face_b,
      geom,
      mesh_coordinate_code(cfg.mesh_coordinate),
      mesh_axes_layout(cfg)
   );
}

inline void fill_face_b_from_face_fillpatch(
   FaceMagneticField& face_b,
   FaceMagneticField const& face_fillpatch,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
   fill_face_b_from_face_fillpatch(
      face_b,
      face_fillpatch,
      geom,
      mesh_coordinate_code(cfg.mesh_coordinate),
      mesh_axes_layout(cfg)
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ct_edge_line_integral(
   amrex::Real edge_e,
   int edge_component,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   const amrex::Real length = mesh_edge_length_scale(edge_component, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
   return generated::emf_line_integral(edge_e, length);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> ct_coordinate_triplet_at_location(
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k,
   bool nodal0,
   bool nodal1,
   bool nodal2
) noexcept
{
   return mesh_coordinate_triplet_at_location(
      mesh_coordinate, mesh_axes, problo, dx, i, j, k, nodal0, nodal1, nodal2
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ct_component_scale_factor(
   int component,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::Real q1,
   amrex::Real q2,
   amrex::Real q3
) noexcept
{
   const auto scale = coordinate_scale_factors(mesh_coordinate, q1, q2, q3);
   const int axis = mesh_vector_axis_for_component(mesh_coordinate, mesh_axes, component);
   if (mesh_coordinate == MeshCylindrical) {
      if (axis == MeshAxisR) {
         return scale[0];
      }
      if (axis == MeshAxisPhi) {
         return scale[1];
      }
      if (axis == MeshAxisZ) {
         return scale[2];
      }
      return 1.0;
   }
   if (axis == MeshAxisX || axis == MeshAxisR) {
      return scale[0];
   }
   if (axis == MeshAxisY || axis == MeshAxisTheta) {
      return scale[1];
   }
   if (axis == MeshAxisZ || axis == MeshAxisPhi) {
      return scale[2];
   }
   return 1.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ct_component_scale_at_location(
   int component,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k,
   bool nodal0,
   bool nodal1,
   bool nodal2
) noexcept
{
   const auto q = ct_coordinate_triplet_at_location(mesh_coordinate, mesh_axes, problo, dx, i, j, k, nodal0, nodal1, nodal2);
   return ct_component_scale_factor(component, mesh_coordinate, mesh_axes, q[0], q[1], q[2]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ct_component_scale_at_face(
   int component,
   int face_dir,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   return ct_component_scale_at_location(
      component,
      mesh_coordinate,
      mesh_axes,
      problo,
      dx,
      i,
      j,
      k,
      face_dir == 0,
#if (AMREX_SPACEDIM >= 2)
      face_dir == 1,
#else
      false,
#endif
#if (AMREX_SPACEDIM == 3)
      face_dir == 2
#else
      false
#endif
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ct_component_scale_at_edge(
   int component,
   int edge_component,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   return ct_component_scale_at_location(
      component,
      mesh_coordinate,
      mesh_axes,
      problo,
      dx,
      i,
      j,
      k,
      edge_component != 0,
#if (AMREX_SPACEDIM >= 2)
      edge_component != 1,
#else
      false,
#endif
#if (AMREX_SPACEDIM == 3)
      edge_component != 2
#else
      false
#endif
   );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ct_scaled_face_b(
   amrex::Array4<amrex::Real const> const& b,
   int face_dir,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   return b(i, j, k, 0) * ct_component_scale_at_face(face_dir, face_dir, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ct_backward_difference_scaled_face_b(
   amrex::Array4<amrex::Real const> const& b,
   int face_dir,
   int derivative_dir,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
   int il = i;
   int jl = j;
   int kl = k;
   if (derivative_dir == 0) {
      --il;
   }
#if (AMREX_SPACEDIM >= 2)
   if (derivative_dir == 1) {
      --jl;
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (derivative_dir == 2) {
      --kl;
   }
#endif
   return (
      ct_scaled_face_b(b, face_dir, mesh_coordinate, mesh_axes, problo, dx, i, j, k)
      - ct_scaled_face_b(b, face_dir, mesh_coordinate, mesh_axes, problo, dx, il, jl, kl)
   ) / dx[derivative_dir];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ct_edge_current_from_face_b(
   amrex::Array4<amrex::Real const> const& b0,
   amrex::Array4<amrex::Real const> const& b1,
#if (AMREX_SPACEDIM == 3)
   amrex::Array4<amrex::Real const> const& b2,
#endif
   int component,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   int i,
   int j,
   int k
) noexcept
{
#if (AMREX_SPACEDIM == 2)
   amrex::ignore_unused(component);
   const amrex::Real h0 = ct_component_scale_at_edge(0, 2, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
   const amrex::Real h1 = ct_component_scale_at_edge(1, 2, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
   const amrex::Real inv_h0h1 = 1.0 / amrex::max(amrex::Math::abs(h0 * h1), 1.0e-300);
   const amrex::Real d_h1_b1_dq0 =
      ct_backward_difference_scaled_face_b(b1, 1, 0, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
   const amrex::Real d_h0_b0_dq1 =
      ct_backward_difference_scaled_face_b(b0, 0, 1, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
   return generated::curl_orthogonal_3(inv_h0h1, d_h1_b1_dq0, d_h0_b0_dq1);
#elif (AMREX_SPACEDIM == 3)
   if (component == 0) {
      const amrex::Real h1 = ct_component_scale_at_edge(1, 0, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
      const amrex::Real h2 = ct_component_scale_at_edge(2, 0, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
      const amrex::Real inv_h1h2 = 1.0 / amrex::max(amrex::Math::abs(h1 * h2), 1.0e-300);
      const amrex::Real d_h2_b2_dq1 =
         ct_backward_difference_scaled_face_b(b2, 2, 1, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
      const amrex::Real d_h1_b1_dq2 =
         ct_backward_difference_scaled_face_b(b1, 1, 2, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
      return generated::curl_orthogonal_1(inv_h1h2, d_h2_b2_dq1, d_h1_b1_dq2);
   }
   if (component == 1) {
      const amrex::Real h2 = ct_component_scale_at_edge(2, 1, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
      const amrex::Real h0 = ct_component_scale_at_edge(0, 1, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
      const amrex::Real inv_h2h0 = 1.0 / amrex::max(amrex::Math::abs(h2 * h0), 1.0e-300);
      const amrex::Real d_h0_b0_dq2 =
         ct_backward_difference_scaled_face_b(b0, 0, 2, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
      const amrex::Real d_h2_b2_dq0 =
         ct_backward_difference_scaled_face_b(b2, 2, 0, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
      return generated::curl_orthogonal_2(inv_h2h0, d_h0_b0_dq2, d_h2_b2_dq0);
   }
   const amrex::Real h0 = ct_component_scale_at_edge(0, 2, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
   const amrex::Real h1 = ct_component_scale_at_edge(1, 2, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
   const amrex::Real inv_h0h1 = 1.0 / amrex::max(amrex::Math::abs(h0 * h1), 1.0e-300);
   const amrex::Real d_h1_b1_dq0 =
      ct_backward_difference_scaled_face_b(b1, 1, 0, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
   const amrex::Real d_h0_b0_dq1 =
      ct_backward_difference_scaled_face_b(b0, 0, 1, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
   return generated::curl_orthogonal_3(inv_h0h1, d_h1_b1_dq0, d_h0_b0_dq1);
#else
   amrex::ignore_unused(b0, b1, component, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
   return 0.0;
#endif
}

template <bool ResistiveOnly>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real edge_cell_emf_component(
   amrex::Array4<amrex::Real const> const& s,
   int i,
   int j,
   int k,
   int component,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real eta
) noexcept
{
   if constexpr (ResistiveOnly) {
      return cell_resistive_emf_component(s, i, j, k, component, dx, eta);
   }
   return cell_emf_component(s, i, j, k, component, dx, eta);
}

template <bool ResistiveOnly>
inline void fill_edge_electric_field_impl(
   EdgeElectricField& edge_e,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   amrex::Real eta
)
{
   const auto dx = geom.CellSizeArray();
   for (auto& component : edge_e.e) {
      if (component) {
         component->setVal(0.0);
      }
   }
#if (AMREX_SPACEDIM == 2)
   amrex::MultiFab& ez = *edge_e.e[2];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(ez, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto e = ez.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         e(i, j, k, 0) = 0.25 * (
            edge_cell_emf_component<ResistiveOnly>(s, i - 1, j - 1, k, 2, dx, eta)
            + edge_cell_emf_component<ResistiveOnly>(s, i, j - 1, k, 2, dx, eta)
            + edge_cell_emf_component<ResistiveOnly>(s, i - 1, j, k, 2, dx, eta)
            + edge_cell_emf_component<ResistiveOnly>(s, i, j, k, 2, dx, eta)
         );
      });
   }
   }
#elif (AMREX_SPACEDIM == 3)
   for (int component = 0; component < 3; ++component) {
      amrex::MultiFab& edge = *edge_e.e[component];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(edge, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto s = state.const_array(mfi);
         const auto e = edge.array(mfi);
         const int c = component;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (c == 0) {
               e(i, j, k, 0) = 0.25 * (
                  edge_cell_emf_component<ResistiveOnly>(s, i, j - 1, k - 1, 0, dx, eta)
                  + edge_cell_emf_component<ResistiveOnly>(s, i, j, k - 1, 0, dx, eta)
                  + edge_cell_emf_component<ResistiveOnly>(s, i, j - 1, k, 0, dx, eta)
                  + edge_cell_emf_component<ResistiveOnly>(s, i, j, k, 0, dx, eta)
               );
            } else if (c == 1) {
               e(i, j, k, 0) = 0.25 * (
                  edge_cell_emf_component<ResistiveOnly>(s, i - 1, j, k - 1, 1, dx, eta)
                  + edge_cell_emf_component<ResistiveOnly>(s, i, j, k - 1, 1, dx, eta)
                  + edge_cell_emf_component<ResistiveOnly>(s, i - 1, j, k, 1, dx, eta)
                  + edge_cell_emf_component<ResistiveOnly>(s, i, j, k, 1, dx, eta)
               );
            } else {
               e(i, j, k, 0) = 0.25 * (
                  edge_cell_emf_component<ResistiveOnly>(s, i - 1, j - 1, k, 2, dx, eta)
                  + edge_cell_emf_component<ResistiveOnly>(s, i, j - 1, k, 2, dx, eta)
                  + edge_cell_emf_component<ResistiveOnly>(s, i - 1, j, k, 2, dx, eta)
                  + edge_cell_emf_component<ResistiveOnly>(s, i, j, k, 2, dx, eta)
               );
            }
         });
      }
      }
   }
#else
   (void)edge_e;
   (void)state;
   (void)geom;
   (void)eta;
#endif
}

template <typename EdgeField>
inline void fill_edge_current_density_from_cells(
   EdgeField& edge_j,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom
)
{
   const auto dx = geom.CellSizeArray();
   set_edge_field_value(edge_j, 0.0);
#if (AMREX_SPACEDIM == 2)
   amrex::MultiFab& jz = *mutable_edge_component(edge_j, 2);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(jz, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto out = jz.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         out(i, j, k, 0) = 0.25 * (
            cell_current_z(s, i - 1, j - 1, k, dx)
            + cell_current_z(s, i, j - 1, k, dx)
            + cell_current_z(s, i - 1, j, k, dx)
            + cell_current_z(s, i, j, k, dx)
         );
      });
   }
   }
#elif (AMREX_SPACEDIM == 3)
   for (int component = 0; component < 3; ++component) {
      amrex::MultiFab& edge = *mutable_edge_component(edge_j, component);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(edge, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto s = state.const_array(mfi);
         const auto out = edge.array(mfi);
         const int c = component;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (c == 0) {
               out(i, j, k, 0) = 0.25 * (
                  cell_current_x(s, i, j - 1, k - 1, dx)
                  + cell_current_x(s, i, j, k - 1, dx)
                  + cell_current_x(s, i, j - 1, k, dx)
                  + cell_current_x(s, i, j, k, dx)
               );
            } else if (c == 1) {
               out(i, j, k, 0) = 0.25 * (
                  cell_current_y(s, i - 1, j, k - 1, dx)
                  + cell_current_y(s, i, j, k - 1, dx)
                  + cell_current_y(s, i - 1, j, k, dx)
                  + cell_current_y(s, i, j, k, dx)
               );
            } else {
               out(i, j, k, 0) = 0.25 * (
                  cell_current_z(s, i - 1, j - 1, k, dx)
                  + cell_current_z(s, i, j - 1, k, dx)
                  + cell_current_z(s, i - 1, j, k, dx)
                  + cell_current_z(s, i, j, k, dx)
               );
            }
         });
      }
      }
   }
#else
   (void)edge_j;
   (void)state;
   (void)geom;
#endif
}

template <bool StoreResistiveEmf, typename EdgeField>
inline void fill_edge_curl_b_from_faces(
   EdgeField& edge_out,
   FaceMagneticField const& face_b,
   amrex::Geometry const& geom,
   amrex::Real eta,
   int mesh_coordinate,
   MeshAxes const& mesh_axes
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   set_edge_field_value(edge_out, 0.0);
#if (AMREX_SPACEDIM == 2)
   amrex::MultiFab& edge_z = *mutable_edge_component(edge_out, 2);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(edge_z, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto b1_face = face_b.b[0]->const_array(mfi);
      const auto b2_face = face_b.b[1]->const_array(mfi);
      const auto out = edge_z.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real current = ct_edge_current_from_face_b(
            b1_face,
            b2_face,
            2,
            mesh_coordinate,
            mesh_axes,
            problo,
            dx,
            i,
            j,
            k
         );
         if constexpr (StoreResistiveEmf) {
            out(i, j, k, 0) = generated::resistive_emf(eta, current);
         } else {
            out(i, j, k, 0) = current;
         }
      });
   }
   }
#elif (AMREX_SPACEDIM == 3)
   for (int component = 0; component < 3; ++component) {
      amrex::MultiFab& edge = *mutable_edge_component(edge_out, component);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(edge, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto b1_face = face_b.b[0]->const_array(mfi);
         const auto b2_face = face_b.b[1]->const_array(mfi);
         const auto b3_face = face_b.b[2]->const_array(mfi);
         const auto out = edge.array(mfi);
         const int c = component;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real current = ct_edge_current_from_face_b(
               b1_face,
               b2_face,
               b3_face,
               c,
               mesh_coordinate,
               mesh_axes,
               problo,
               dx,
               i,
               j,
               k
            );
            if constexpr (StoreResistiveEmf) {
               out(i, j, k, 0) = generated::resistive_emf(eta, current);
            } else {
               out(i, j, k, 0) = current;
            }
         });
      }
      }
   }
#else
   (void)edge_out;
   (void)face_b;
   (void)geom;
   (void)eta;
   (void)mesh_coordinate;
   (void)mesh_axes;
#endif
}

template <typename EdgeField>
inline void fill_edge_current_density_from_faces(
   EdgeField& edge_j,
   FaceMagneticField const& face_b,
   amrex::Geometry const& geom,
   int mesh_coordinate,
   MeshAxes const& mesh_axes
)
{
   fill_edge_curl_b_from_faces<false>(edge_j, face_b, geom, 1.0, mesh_coordinate, mesh_axes);
}

template <typename EdgeField>
inline void fill_edge_current_density_from_faces(
   EdgeField& edge_j,
   FaceMagneticField const& face_b,
   amrex::Geometry const& geom
)
{
   fill_edge_current_density_from_faces(
      edge_j,
      face_b,
      geom,
      MeshCartesian,
      mesh_default_axes_layout(MeshCartesian)
   );
}

template <typename EdgeField>
inline void fill_edge_current_density_from_faces(
   EdgeField& edge_j,
   FaceMagneticField const& face_b,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
   fill_edge_current_density_from_faces(edge_j, face_b, geom, mesh_coordinate_code(cfg.mesh_coordinate), mesh_axes_layout(cfg));
}

template <typename EdgeField>
inline void fill_ct_resistive_emf_from_face_b(
   EdgeField& edge_e,
   FaceMagneticField const& face_b,
   amrex::Geometry const& geom,
   amrex::Real eta,
   int mesh_coordinate,
   MeshAxes const& mesh_axes
)
{
   fill_edge_curl_b_from_faces<true>(edge_e, face_b, geom, eta, mesh_coordinate, mesh_axes);
}

template <typename EdgeField>
inline void fill_ct_resistive_emf_from_face_b(
   EdgeField& edge_e,
   FaceMagneticField const& face_b,
   amrex::Geometry const& geom,
   amrex::Real eta
)
{
   fill_ct_resistive_emf_from_face_b(
      edge_e,
      face_b,
      geom,
      eta,
      MeshCartesian,
      mesh_default_axes_layout(MeshCartesian)
   );
}

template <typename EdgeField>
inline void fill_ct_resistive_emf_from_face_b(
   EdgeField& edge_e,
   FaceMagneticField const& face_b,
   amrex::Geometry const& geom,
   amrex::Real eta,
   RunConfig const& cfg
)
{
   fill_ct_resistive_emf_from_face_b(
      edge_e,
      face_b,
      geom,
      eta,
      mesh_coordinate_code(cfg.mesh_coordinate),
      mesh_axes_layout(cfg)
   );
}

inline void add_scaled_edge_field(
   EdgeElectricField& dst,
   EdgeElectricField const& src,
   amrex::Real scale
)
{
   if (scale == 0.0) {
      return;
   }
   for (int component = 0; component < 3; ++component) {
      if (!dst.e[component] || !src.e[component]) {
         continue;
      }
      amrex::MultiFab::Saxpy(*dst.e[component], scale, *src.e[component], 0, 0, 1, 0);
   }
}

inline void fill_edge_electric_field_for_output(
   EdgeElectricField& edge_e,
   amrex::MultiFab const& state,
   FaceMagneticField const* face_b,
   EdgeElectricField const* edge_j,
   amrex::Geometry const& geom,
   amrex::Real eta,
   RunConfig const& cfg
)
{
   fill_edge_electric_field_impl<false>(edge_e, state, geom, 0.0);
   if (eta == 0.0) {
      enforce_edge_electric_boundary(edge_e, geom, cfg);
      return;
   }

   EdgeElectricField edge_j_work;
   EdgeElectricField const* current = edge_j;
   if (current == nullptr) {
      allocate_edge_electric_field(edge_j_work, state.boxArray(), state.DistributionMap());
      if (face_b != nullptr) {
         fill_edge_current_density_from_faces(edge_j_work, *face_b, geom, cfg);
      } else {
         fill_edge_current_density_from_cells(edge_j_work, state, geom);
      }
      enforce_edge_electric_periodicity(edge_j_work, geom);
      current = &edge_j_work;
   }
   add_scaled_edge_field(edge_e, *current, eta);
   enforce_edge_electric_boundary(edge_e, geom, cfg);
}

inline void fill_edge_electric_field_for_output(
   EdgeElectricField& edge_e,
   amrex::MultiFab const& state,
   FaceMagneticField const* face_b,
   EdgeElectricField const* edge_j,
   amrex::Geometry const& geom,
   amrex::Real eta
)
{
   RunConfig cfg;
   fill_edge_electric_field_for_output(edge_e, state, face_b, edge_j, geom, eta, cfg);
}

inline void fill_edge_electric_field(
   EdgeElectricField& edge_e,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   amrex::Real eta
)
{
   fill_edge_electric_field_impl<false>(edge_e, state, geom, eta);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ct_index_in_box(amrex::IntVect const& lo, amrex::IntVect const& hi, int i, int j, int k) noexcept
{
   bool in_box = (i >= lo[0] && i <= hi[0]);
#if (AMREX_SPACEDIM >= 2)
   in_box = in_box && (j >= lo[1] && j <= hi[1]);
#endif
#if (AMREX_SPACEDIM == 3)
   in_box = in_box && (k >= lo[2] && k <= hi[2]);
#else
   (void)k;
#endif
   return in_box;
}

template <int Axis>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ct_axis_index(int i, int j, int k) noexcept
{
   if constexpr (Axis == 0) {
      return i;
   } else if constexpr (Axis == 1) {
      return j;
   } else {
      return k;
   }
}

template <int Axis>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ct_step_negative_along_axis(int& i, int& j, int& k) noexcept
{
   if constexpr (Axis == 0) {
      --i;
   } else if constexpr (Axis == 1) {
      --j;
   } else {
      --k;
   }
}

template <int SampleAxis, int Sign, int MagneticComponent>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void add_ct_flux_emf_contribution(
   amrex::Real& sum,
   int& count,
   amrex::Array4<amrex::Real const> const& flux,
   amrex::Array4<int const> const& owner,
   amrex::IntVect const& flux_lo,
   amrex::IntVect const& flux_hi,
   amrex::IntVect const& dom_lo,
   amrex::IntVect const& dom_hi,
   int i,
   int j,
   int k
) noexcept
{
   constexpr int magnetic_index = B1 + MagneticComponent;
   const amrex::Real sign = static_cast<amrex::Real>(Sign);
   if (
      ct_axis_index<SampleAxis>(i, j, k) <= dom_hi[SampleAxis]
      && ct_index_in_box(flux_lo, flux_hi, i, j, k)
      && owner(i, j, k, 0) != 0
   ) {
      sum += sign * flux(i, j, k, magnetic_index);
      ++count;
   }

   int im = i;
   int jm = j;
   int km = k;
   ct_step_negative_along_axis<SampleAxis>(im, jm, km);
   if (
      ct_axis_index<SampleAxis>(i, j, k) > dom_lo[SampleAxis]
      && ct_index_in_box(flux_lo, flux_hi, im, jm, km)
      && owner(im, jm, km, 0) != 0
   ) {
      sum += sign * flux(im, jm, km, magnetic_index);
      ++count;
   }
}

template <
   int EdgeComponent,
   int FaceDir0,
   int MagneticComponent0,
   int SampleAxis0,
   int Sign0,
   int FaceDir1,
   int MagneticComponent1,
   int SampleAxis1,
   int Sign1>
inline void fill_edge_electric_component_from_mhd_fluxes(
   EdgeElectricField& edge_e,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   std::array<amrex::iMultiFab const*, AMREX_SPACEDIM> const& flux_owner,
   amrex::Geometry const& geom
)
{
   AMREX_ALWAYS_ASSERT(edge_e.e[EdgeComponent] != nullptr);
   AMREX_ALWAYS_ASSERT(fluxes[FaceDir0] != nullptr);
   AMREX_ALWAYS_ASSERT(fluxes[FaceDir1] != nullptr);
   AMREX_ALWAYS_ASSERT(flux_owner[FaceDir0] != nullptr);
   AMREX_ALWAYS_ASSERT(flux_owner[FaceDir1] != nullptr);

   amrex::MultiFab& edge_component = *edge_e.e[EdgeComponent];
   edge_component.setVal(0.0);
   amrex::MultiFab edge_count(edge_component.boxArray(), edge_component.DistributionMap(), 1, 0);
   edge_count.setVal(0.0);
   const amrex::Box domain = geom.Domain();
   const auto dom_lo = domain.smallEnd();
   const auto dom_hi = domain.bigEnd();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(edge_component, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const amrex::Box& flux0_box = (*fluxes[FaceDir0])[mfi].box();
      const amrex::Box& flux1_box = (*fluxes[FaceDir1])[mfi].box();
      const auto flux0_lo = flux0_box.smallEnd();
      const auto flux0_hi = flux0_box.bigEnd();
      const auto flux1_lo = flux1_box.smallEnd();
      const auto flux1_hi = flux1_box.bigEnd();
      const auto flux0 = fluxes[FaceDir0]->const_array(mfi);
      const auto flux1 = fluxes[FaceDir1]->const_array(mfi);
      const auto owner0 = flux_owner[FaceDir0]->const_array(mfi);
      const auto owner1 = flux_owner[FaceDir1]->const_array(mfi);
      const auto e = edge_component.array(mfi);
      const auto c = edge_count.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         amrex::Real sum = 0.0;
         int count = 0;
         add_ct_flux_emf_contribution<SampleAxis0, Sign0, MagneticComponent0>(
            sum,
            count,
            flux0,
            owner0,
            flux0_lo,
            flux0_hi,
            dom_lo,
            dom_hi,
            i,
            j,
            k
         );
         add_ct_flux_emf_contribution<SampleAxis1, Sign1, MagneticComponent1>(
            sum,
            count,
            flux1,
            owner1,
            flux1_lo,
            flux1_hi,
            dom_lo,
            dom_hi,
            i,
            j,
            k
         );
         e(i, j, k, 0) = sum;
         c(i, j, k, 0) = static_cast<amrex::Real>(count);
      });
   }
   }

   edge_component.SumBoundary(geom.periodicity(), true);
   edge_count.SumBoundary(geom.periodicity(), true);
   edge_component.OverrideSync(geom.periodicity());
   edge_count.OverrideSync(geom.periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(edge_component, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto e = edge_component.array(mfi);
      const auto c = edge_count.const_array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real count = c(i, j, k, 0);
         e(i, j, k, 0) = count > 0.0 ? e(i, j, k, 0) / count : 0.0;
      });
   }
   }
}

inline bool ct_flux_derived_emf_is_available(std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes)
{
#if (AMREX_SPACEDIM >= 2)
   bool available = true;
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      available = available && (fluxes[direction] != nullptr);
   }
   return available;
#else
   (void)fluxes;
   return false;
#endif
}

inline std::array<std::unique_ptr<amrex::iMultiFab>, AMREX_SPACEDIM> make_mhd_flux_owner_masks_for_ct_edge_emf(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::Geometry const& geom
)
{
   std::array<std::unique_ptr<amrex::iMultiFab>, AMREX_SPACEDIM> owner_masks;
#if (AMREX_SPACEDIM >= 2)
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      if (fluxes[direction] != nullptr) {
         owner_masks[direction] = fluxes[direction]->OwnerMask(geom.periodicity());
      }
   }
#else
   (void)fluxes;
   (void)geom;
#endif
   return owner_masks;
}

inline std::array<amrex::iMultiFab const*, AMREX_SPACEDIM> raw_owner_mask_ptrs(
   std::array<std::unique_ptr<amrex::iMultiFab>, AMREX_SPACEDIM> const& owner_masks
)
{
   std::array<amrex::iMultiFab const*, AMREX_SPACEDIM> ptrs{};
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      ptrs[direction] = owner_masks[direction].get();
   }
   return ptrs;
}

inline void fill_edge_electric_field_from_mhd_fluxes(
   EdgeElectricField& edge_e,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   std::array<amrex::iMultiFab const*, AMREX_SPACEDIM> const& flux_owner,
   amrex::Geometry const& geom
)
{
   // Magnetic Riemann fluxes are local orthonormal fluxes. Coordinate factors enter
   // later through ct_edge_line_integral/apply_edge_electric_curl.
   for (int component = 0; component < 3; ++component) {
      if (edge_e.e[component]) {
         edge_e.e[component]->setVal(0.0);
      }
   }
#if (AMREX_SPACEDIM == 2)
   fill_edge_electric_component_from_mhd_fluxes<2, 0, 1, 1, -1, 1, 0, 0, +1>(edge_e, fluxes, flux_owner, geom);
#elif (AMREX_SPACEDIM == 3)
   fill_edge_electric_component_from_mhd_fluxes<0, 1, 2, 2, -1, 2, 1, 1, +1>(edge_e, fluxes, flux_owner, geom);
   fill_edge_electric_component_from_mhd_fluxes<1, 2, 0, 0, -1, 0, 2, 2, +1>(edge_e, fluxes, flux_owner, geom);
   fill_edge_electric_component_from_mhd_fluxes<2, 0, 1, 1, -1, 1, 0, 0, +1>(edge_e, fluxes, flux_owner, geom);
#else
   (void)edge_e;
   (void)fluxes;
   (void)geom;
#endif
}

inline void fill_edge_electric_field_from_mhd_fluxes(
   EdgeElectricField& edge_e,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::Geometry const& geom
)
{
   const auto flux_owner_storage = make_mhd_flux_owner_masks_for_ct_edge_emf(fluxes, geom);
   fill_edge_electric_field_from_mhd_fluxes(edge_e, fluxes, raw_owner_mask_ptrs(flux_owner_storage), geom);
}

inline void fill_resistive_edge_electric_field(
   EdgeElectricField& edge_e,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   amrex::Real eta
)
{
   fill_edge_electric_field_impl<true>(edge_e, state, geom, eta);
}

inline void enforce_edge_electric_periodicity(EdgeElectricField& edge_e, amrex::Geometry const& geom)
{
   for (int component_index = 0; component_index < 3; ++component_index) {
      auto& component = edge_e.e[component_index];
      if (component) {
         component->FillBoundary(geom.periodicity());
         component->OverrideSync(geom.periodicity());
         component->EnforcePeriodicity(geom.periodicity());
      }
   }
}

inline void enforce_edge_electric_origin_boundary(
   EdgeElectricField& edge_e,
   amrex::Geometry const& geom,
   BoundaryData const& bc
)
{
   const int radial_dir = mesh_direction_for_axis(bc.mesh_axes, MeshAxisR);
   if (radial_dir < 0 || bc.origin_lo[radial_dir] == 0) {
      return;
   }

   for (int component = 0; component < 3; ++component) {
      if (!edge_e.e[component]) {
         continue;
      }
      const int component_axis = mesh_vector_axis_for_component(bc.mesh_coordinate, bc.mesh_axes, component);
      if (component_axis == MeshAxisR) {
         continue;
      }

      amrex::MultiFab& edge_component = *edge_e.e[component];
      const amrex::Box edge_domain = amrex::convert(geom.Domain(), edge_index_type(component));
      const int origin_index = edge_domain.smallEnd(radial_dir);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(edge_component, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto e = edge_component.array(mfi);
         const int d = radial_dir;
         const int o = origin_index;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            bool on_origin = (d == 0 && i == o);
#if (AMREX_SPACEDIM >= 2)
            on_origin = on_origin || (d == 1 && j == o);
#endif
#if (AMREX_SPACEDIM == 3)
            on_origin = on_origin || (d == 2 && k == o);
#endif
            if (on_origin) {
               e(i, j, k, 0) = 0.0;
            }
         });
      }
      }
   }
}

inline void enforce_edge_electric_boundary(
   EdgeElectricField& edge_e,
   amrex::Geometry const& geom,
   BoundaryData const& bc
)
{
   enforce_edge_electric_periodicity(edge_e, geom);
   enforce_edge_electric_origin_boundary(edge_e, geom, bc);
}

inline void enforce_edge_electric_boundary(
   EdgeElectricField& edge_e,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
   enforce_edge_electric_boundary(edge_e, geom, make_boundary_data(cfg));
}

inline void apply_edge_electric_curl(
   FaceMagneticField& face_b,
   EdgeElectricField& edge_e,
   amrex::Geometry const& geom,
   amrex::Real dt,
   int mesh_coordinate = MeshCartesian,
   MeshAxes mesh_axes = mesh_default_axes_layout(MeshCartesian)
)
{
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
#if (AMREX_SPACEDIM == 2)
   const auto ez = edge_e.e[2]->const_arrays();
   for (int dir = 0; dir < 2; ++dir) {
      amrex::MultiFab& face = *face_b.b[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(face, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto b = face.array(mfi);
         const auto e = edge_e.e[2]->const_array(mfi);
         const int d = dir;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real old_flux =
               ct_face_flux(b(i, j, k, 0), d, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            if (d == 0) {
               const amrex::Real ez_hi =
                  ct_edge_line_integral(e(i, j + 1, k, 0), 2, mesh_coordinate, mesh_axes, problo, dx, i, j + 1, k);
               const amrex::Real ez_lo =
                  ct_edge_line_integral(e(i, j, k, 0), 2, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
               const amrex::Real new_flux = generated::ct_flux_update_2d_dir0(old_flux, dt, ez_hi, ez_lo);
               b(i, j, k, 0) = ct_face_b_from_flux(new_flux, d, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            } else {
               const amrex::Real ez_hi =
                  ct_edge_line_integral(e(i + 1, j, k, 0), 2, mesh_coordinate, mesh_axes, problo, dx, i + 1, j, k);
               const amrex::Real ez_lo =
                  ct_edge_line_integral(e(i, j, k, 0), 2, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
               const amrex::Real new_flux = generated::ct_flux_update_2d_dir1(old_flux, dt, ez_hi, ez_lo);
               b(i, j, k, 0) = ct_face_b_from_flux(new_flux, d, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            }
         });
      }
      }
   }
   (void)ez;
#elif (AMREX_SPACEDIM == 3)
   for (int dir = 0; dir < 3; ++dir) {
      amrex::MultiFab& face = *face_b.b[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(face, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto b = face.array(mfi);
         const auto ex = edge_e.e[0]->const_array(mfi);
         const auto ey = edge_e.e[1]->const_array(mfi);
         const auto ez = edge_e.e[2]->const_array(mfi);
         const int d = dir;
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real old_flux =
               ct_face_flux(b(i, j, k, 0), d, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            if (d == 0) {
               const amrex::Real ez_hi =
                  ct_edge_line_integral(ez(i, j + 1, k, 0), 2, mesh_coordinate, mesh_axes, problo, dx, i, j + 1, k);
               const amrex::Real ez_lo =
                  ct_edge_line_integral(ez(i, j, k, 0), 2, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
               const amrex::Real ey_hi =
                  ct_edge_line_integral(ey(i, j, k + 1, 0), 1, mesh_coordinate, mesh_axes, problo, dx, i, j, k + 1);
               const amrex::Real ey_lo =
                  ct_edge_line_integral(ey(i, j, k, 0), 1, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
               const amrex::Real new_flux = generated::ct_flux_update_3d_dir0(
                  old_flux,
                  dt,
                  ez_hi,
                  ez_lo,
                  ey_hi,
                  ey_lo
               );
               b(i, j, k, 0) = ct_face_b_from_flux(new_flux, d, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            } else if (d == 1) {
               const amrex::Real ex_hi =
                  ct_edge_line_integral(ex(i, j, k + 1, 0), 0, mesh_coordinate, mesh_axes, problo, dx, i, j, k + 1);
               const amrex::Real ex_lo =
                  ct_edge_line_integral(ex(i, j, k, 0), 0, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
               const amrex::Real ez_hi =
                  ct_edge_line_integral(ez(i + 1, j, k, 0), 2, mesh_coordinate, mesh_axes, problo, dx, i + 1, j, k);
               const amrex::Real ez_lo =
                  ct_edge_line_integral(ez(i, j, k, 0), 2, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
               const amrex::Real new_flux = generated::ct_flux_update_3d_dir1(
                  old_flux,
                  dt,
                  ex_hi,
                  ex_lo,
                  ez_hi,
                  ez_lo
               );
               b(i, j, k, 0) = ct_face_b_from_flux(new_flux, d, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            } else {
               const amrex::Real ey_hi =
                  ct_edge_line_integral(ey(i + 1, j, k, 0), 1, mesh_coordinate, mesh_axes, problo, dx, i + 1, j, k);
               const amrex::Real ey_lo =
                  ct_edge_line_integral(ey(i, j, k, 0), 1, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
               const amrex::Real ex_hi =
                  ct_edge_line_integral(ex(i, j + 1, k, 0), 0, mesh_coordinate, mesh_axes, problo, dx, i, j + 1, k);
               const amrex::Real ex_lo =
                  ct_edge_line_integral(ex(i, j, k, 0), 0, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
               const amrex::Real new_flux = generated::ct_flux_update_3d_dir2(
                  old_flux,
                  dt,
                  ey_hi,
                  ey_lo,
                  ex_hi,
                  ex_lo
               );
               b(i, j, k, 0) = ct_face_b_from_flux(new_flux, d, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            }
         });
      }
      }
   }
#else
   (void)face_b;
   (void)edge_e;
   (void)geom;
   (void)dt;
#endif
}

inline void apply_edge_electric_curl(
   FaceMagneticField& face_b,
   EdgeElectricField& edge_e,
   amrex::Geometry const& geom,
   amrex::Real dt,
   RunConfig const& cfg
)
{
   apply_edge_electric_curl(
      face_b,
      edge_e,
      geom,
      dt,
      mesh_coordinate_code(cfg.mesh_coordinate),
      mesh_axes_layout(cfg)
   );
}

inline void advance_ct_magnetic_field(
   FaceMagneticField& face_b,
   EdgeElectricField& edge_e,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   amrex::Real dt,
   amrex::Real eta,
   RunConfig const& cfg
)
{
   fill_edge_electric_field(edge_e, state, geom, eta);
   enforce_edge_electric_boundary(edge_e, geom, cfg);
   apply_edge_electric_curl(face_b, edge_e, geom, dt, cfg);
}

inline void advance_ct_resistive_magnetic_field(
   FaceMagneticField& face_b,
   EdgeElectricField& edge_e,
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   amrex::Real dt,
   amrex::Real eta,
   RunConfig const& cfg
)
{
   fill_resistive_edge_electric_field(edge_e, state, geom, eta);
   enforce_edge_electric_boundary(edge_e, geom, cfg);
   apply_edge_electric_curl(face_b, edge_e, geom, dt, cfg);
}

inline void advance_ct_resistive_magnetic_field_from_faces(
   FaceMagneticField& face_b,
   EdgeElectricField& edge_e,
   amrex::Geometry const& geom,
   amrex::Real dt,
   amrex::Real eta,
   RunConfig const& cfg
)
{
   fill_ct_resistive_emf_from_face_b(edge_e, face_b, geom, eta, cfg);
   enforce_edge_electric_boundary(edge_e, geom, cfg);
   apply_edge_electric_curl(face_b, edge_e, geom, dt, cfg);
}

#if (AMREX_SPACEDIM == 2)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ct_inactive_magnetic_resistive_flux(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int face_dir,
   int inactive_component,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real eta
) noexcept
{
   int il = i;
   int jl = j;
   int ir = i;
   int jr = j;
   if (face_dir == 0) {
      --il;
   } else {
      --jl;
   }

   const amrex::Real h_inactive_left = ct_component_scale_at_location(
      inactive_component, mesh_coordinate, mesh_axes, problo, dx, il, jl, k, false, false, false
   );
   const amrex::Real h_inactive_right = ct_component_scale_at_location(
      inactive_component, mesh_coordinate, mesh_axes, problo, dx, ir, jr, k, false, false, false
   );
   const amrex::Real scaled_left = h_inactive_left * state(il, jl, k, B1 + inactive_component);
   const amrex::Real scaled_right = h_inactive_right * state(ir, jr, k, B1 + inactive_component);
   const amrex::Real d_scaled_b = (scaled_right - scaled_left) / dx[face_dir];

   const int tangent_dir = 1 - face_dir;
   const amrex::Real h_normal = ct_component_scale_at_location(
      face_dir,
      mesh_coordinate,
      mesh_axes,
      problo,
      dx,
      i,
      j,
      k,
      face_dir == 0,
      face_dir == 1,
      false
   );
   const amrex::Real h_tangent = ct_component_scale_at_location(
      tangent_dir,
      mesh_coordinate,
      mesh_axes,
      problo,
      dx,
      i,
      j,
      k,
      face_dir == 0,
      face_dir == 1,
      false
   );
   const amrex::Real h_inactive_face = ct_component_scale_at_location(
      inactive_component,
      mesh_coordinate,
      mesh_axes,
      problo,
      dx,
      i,
      j,
      k,
      face_dir == 0,
      face_dir == 1,
      false
   );
   const amrex::Real denominator = amrex::max(
      amrex::Math::abs(h_inactive_face * h_normal),
      static_cast<amrex::Real>(1.0e-300)
   );
   return generated::resistive_emf_scaled_gradient(eta, h_tangent, d_scaled_b, denominator);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ct_inactive_magnetic_resistive_rhs(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k,
   int inactive_component,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
   amrex::Real eta
) noexcept
{
   const amrex::Real flux0_hi = ct_inactive_magnetic_resistive_flux(
      state, i + 1, j, k, 0, inactive_component, mesh_coordinate, mesh_axes, problo, dx, eta
   );
   const amrex::Real flux0_lo = ct_inactive_magnetic_resistive_flux(
      state, i, j, k, 0, inactive_component, mesh_coordinate, mesh_axes, problo, dx, eta
   );
   const amrex::Real flux1_hi = ct_inactive_magnetic_resistive_flux(
      state, i, j + 1, k, 1, inactive_component, mesh_coordinate, mesh_axes, problo, dx, eta
   );
   const amrex::Real flux1_lo = ct_inactive_magnetic_resistive_flux(
      state, i, j, k, 1, inactive_component, mesh_coordinate, mesh_axes, problo, dx, eta
   );

   const amrex::Real h0 = ct_component_scale_at_location(
      0, mesh_coordinate, mesh_axes, problo, dx, i, j, k, false, false, false
   );
   const amrex::Real h1 = ct_component_scale_at_location(
      1, mesh_coordinate, mesh_axes, problo, dx, i, j, k, false, false, false
   );
   const amrex::Real active_area = amrex::max(
      amrex::Math::abs(h0 * h1),
      static_cast<amrex::Real>(1.0e-300)
   );
   return ((flux0_hi - flux0_lo) / dx[0] + (flux1_hi - flux1_lo) / dx[1]) / active_area;
}
#endif

inline void explicit_ct_inactive_magnetic_resistivity_update(
   amrex::MultiFab& out,
   amrex::MultiFab const& base,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Real dt
)
{
#if (AMREX_SPACEDIM == 2)
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes mesh_axes = mesh_axes_layout(cfg);
   const amrex::Real eta = cfg.resistivity;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(base, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto o = out.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         for (int inactive_component = AMREX_SPACEDIM; inactive_component < 3; ++inactive_component) {
            const int state_component = B1 + inactive_component;
            const amrex::Real rhs = ct_inactive_magnetic_resistive_rhs(
               b, i, j, k, inactive_component, mesh_coordinate, mesh_axes, problo, dx, eta
            );
            o(i, j, k, state_component) = generated::explicit_diffusion_update(
               b(i, j, k, state_component),
               dt,
               rhs
            );
         }
      });
   }
   }
#else
   amrex::ignore_unused(out, base, geom, cfg, dt);
#endif
}

} // namespace pyfluid_amrex
