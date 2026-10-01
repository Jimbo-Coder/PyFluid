#include "pyfluid_amrex/ct_edge_flux_register.hpp"

#if (AMREX_SPACEDIM >= 2)
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Orientation.H>

#include "pyfluid_amrex/kernels_generated/vector/ct.hpp"
#include "pyfluid_amrex/metric_provider.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

#if (AMREX_SPACEDIM == 3)
static constexpr std::array<amrex::IntVect, 3> EdgeIndexType{
   amrex::IntVect(0, 1, 1),
   amrex::IntVect(1, 0, 1),
   amrex::IntVect(1, 1, 0)
};
#endif

CtEdgeFluxRegister::CtEdgeFluxRegister(
   amrex::BoxArray const& fine_ba,
   amrex::BoxArray const& coarse_ba,
   amrex::DistributionMapping const& fine_dm,
   amrex::DistributionMapping const& coarse_dm,
   amrex::Geometry const& fine_geom,
   amrex::Geometry const& coarse_geom,
   int ncomp
)
{
   define(fine_ba, coarse_ba, fine_dm, coarse_dm, fine_geom, coarse_geom, ncomp);
}

void CtEdgeFluxRegister::define(
   amrex::BoxArray const& fine_ba,
   amrex::BoxArray const& coarse_ba,
   amrex::DistributionMapping const& fine_dm,
   amrex::DistributionMapping const& coarse_dm,
   amrex::Geometry const& fine_geom,
   amrex::Geometry const& coarse_geom,
   int ncomp
)
{
   fine_geom_ = fine_geom;
   coarse_geom_ = coarse_geom;
   ratio_ = fine_geom.Domain().size() / coarse_geom.Domain().size();
   AMREX_ALWAYS_ASSERT(fine_geom.Domain() == amrex::refine(coarse_geom.Domain(), ratio_));
   ncomp_ = ncomp;

   has_coarse_fine_.define(coarse_ba, coarse_dm);
   for (amrex::MFIter mfi(has_coarse_fine_, amrex::MFItInfo().DisableDeviceSync()); mfi.isValid(); ++mfi) {
      has_coarse_fine_[mfi] = 0;
   }

#if (AMREX_SPACEDIM == 3)
   for (int component = 0; component < AMREX_SPACEDIM; ++component) {
      coarse_edge_line_[component].define(amrex::convert(coarse_ba, EdgeIndexType[component]), coarse_dm, ncomp, 0);
   }
   for (amrex::OrientationIter oit; oit.isValid(); ++oit) {
      const amrex::Orientation face = oit();
      const int direction = face.coordDir();
      int count = 0;
      for (int component = 0; component < AMREX_SPACEDIM; ++component) {
         if (component == direction) {
            continue;
         }
         amrex::BoxArray edge_ba(
            amrex::convert(fine_ba, amrex::IntVect(0)),
            amrex::BATransformer(face, amrex::IndexType(EdgeIndexType[component]), 0, 1, 0)
         );
         edge_ba.coarsen(ratio_);
         fine_edge_line_[face][count++].define(edge_ba, fine_dm, ncomp, 0);
      }
   }

   for (int component = 0; component < AMREX_SPACEDIM; ++component) {
      amrex::Array<amrex::MultiFab const*, 4> fine_edges;
      int count = 0;
      for (amrex::OrientationIter oit; oit.isValid(); ++oit) {
         const amrex::Orientation face = oit();
         const int direction = face.coordDir();
         if (direction == component) {
            continue;
         }
         const int local_component = (component < direction) ? component : component - 1;
         fine_edges[count++] = &fine_edge_line_[face][local_component];
      }
      for (int idx = 0; idx < 4; ++idx) {
         amrex::LayoutData<int> has_cf;
         auto mask = amrex::makeFineMask(
            coarse_edge_line_[component],
            *fine_edges[idx],
            amrex::IntVect(0),
            amrex::IntVect(1),
            coarse_geom_.periodicity(),
            0,
            1,
            has_cf
         );
         if (idx == 0) {
            fine_mask_[component] = std::move(mask);
         } else {
            amrex::Add(fine_mask_[component], mask, 0, 0, 1, 0);
         }
         for (amrex::MFIter mfi(has_coarse_fine_, amrex::MFItInfo().DisableDeviceSync()); mfi.isValid(); ++mfi) {
            has_coarse_fine_[mfi] += has_cf[mfi];
         }
      }
   }
#else
   coarse_edge_line_.define(amrex::convert(coarse_ba, amrex::IntVect(1)), coarse_dm, ncomp, 0);

   for (amrex::OrientationIter oit; oit.isValid(); ++oit) {
      const amrex::Orientation face = oit();
      amrex::BoxArray edge_ba(
         amrex::convert(fine_ba, amrex::IntVect(0)),
         amrex::BATransformer(face, amrex::IndexType::TheNodeType(), 0, 1, 0)
      );
      edge_ba.coarsen(ratio_);
      fine_edge_line_[face].define(edge_ba, fine_dm, ncomp, 0);
   }

   for (amrex::OrientationIter oit; oit.isValid(); ++oit) {
      const amrex::Orientation face = oit();
      amrex::LayoutData<int> has_cf;
      auto mask = amrex::makeFineMask(
         coarse_edge_line_,
         fine_edge_line_[face],
         amrex::IntVect(0),
         amrex::IntVect(1),
         coarse_geom_.periodicity(),
         0,
         1,
         has_cf
      );
      if (int(face) == 0) {
         fine_mask_ = std::move(mask);
      } else {
         amrex::Add(fine_mask_, mask, 0, 0, 1, 0);
      }
      for (amrex::MFIter mfi(has_coarse_fine_, amrex::MFItInfo().DisableDeviceSync()); mfi.isValid(); ++mfi) {
         has_coarse_fine_[mfi] += has_cf[mfi];
      }
   }
#endif
}

void CtEdgeFluxRegister::reset()
{
#if (AMREX_SPACEDIM == 3)
   for (auto& mf : coarse_edge_line_) {
      const auto arrays = mf.arrays();
      amrex::ParallelFor(mf, amrex::IntVect(0), mf.nComp(), [=] AMREX_GPU_DEVICE(
         int box_no,
         int i,
         int j,
         int k,
         int n
      ) noexcept {
         arrays[box_no](i, j, k, n) = 0.0_rt;
      });
   }
   for (auto& face_edges : fine_edge_line_) {
      for (auto& mf : face_edges) {
#ifdef AMREX_USE_GPU
         const auto arrays = mf.arrays();
         amrex::ParallelFor(mf, amrex::IntVect(0), mf.nComp(), [=] AMREX_GPU_DEVICE(
            int box_no,
            int i,
            int j,
            int k,
            int n
         ) noexcept {
            arrays[box_no](i, j, k, n) = 0.0_rt;
         });
#else
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
         for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
            mf[mfi].template setVal<amrex::RunOn::Host>(0.0_rt);
         }
#endif
      }
   }
#else
   {
      const auto arrays = coarse_edge_line_.arrays();
      amrex::ParallelFor(coarse_edge_line_, amrex::IntVect(0), coarse_edge_line_.nComp(), [=] AMREX_GPU_DEVICE(
         int box_no,
         int i,
         int j,
         int k,
         int n
      ) noexcept {
         arrays[box_no](i, j, k, n) = 0.0_rt;
      });
   }
   for (auto& mf : fine_edge_line_) {
#ifdef AMREX_USE_GPU
      const auto arrays = mf.arrays();
      amrex::ParallelFor(mf, amrex::IntVect(0), mf.nComp(), [=] AMREX_GPU_DEVICE(
         int box_no,
         int i,
         int j,
         int k,
         int n
      ) noexcept {
         arrays[box_no](i, j, k, n) = 0.0_rt;
      });
#else
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
      for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
         mf[mfi].template setVal<amrex::RunOn::Host>(0.0_rt);
      }
#endif
   }
#endif
   amrex::Gpu::synchronize();
}

#if (AMREX_SPACEDIM == 3)
void CtEdgeFluxRegister::CrseAdd(
   amrex::MFIter const& mfi,
   amrex::Array<amrex::FArrayBox const*, 3> const& edge_line,
   amrex::Real dt
)
{
   AMREX_ASSERT(mfi.validbox() == mfi.tilebox());
   if (!has_coarse_fine_[mfi]) {
      return;
   }
   const auto dst0 = coarse_edge_line_[0].array(mfi);
   const auto dst1 = coarse_edge_line_[1].array(mfi);
   const auto dst2 = coarse_edge_line_[2].array(mfi);
   const auto src0 = edge_line[0]->const_array();
   const auto src1 = edge_line[1]->const_array();
   const auto src2 = edge_line[2]->const_array();
   amrex::ParallelFor(
      amrex::convert(mfi.validbox(), EdgeIndexType[0]),
      ncomp_,
      [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
         dst0(i, j, k, n) += dt * src0(i, j, k, n);
      },
      amrex::convert(mfi.validbox(), EdgeIndexType[1]),
      ncomp_,
      [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
         dst1(i, j, k, n) += dt * src1(i, j, k, n);
      },
      amrex::convert(mfi.validbox(), EdgeIndexType[2]),
      ncomp_,
      [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
         dst2(i, j, k, n) += dt * src2(i, j, k, n);
      }
   );
}

void CtEdgeFluxRegister::FineAdd(
   amrex::MFIter const& mfi,
   amrex::Array<amrex::FArrayBox const*, 3> const& edge_line,
   amrex::Real dt
)
{
   AMREX_ASSERT(mfi.validbox() == mfi.tilebox());
   const auto ratio = ratio_;
   for (int component = 0; component < AMREX_SPACEDIM; ++component) {
      const auto src = edge_line[component]->const_array();
      for (amrex::OrientationIter oit; oit.isValid(); ++oit) {
         const amrex::Orientation face = oit();
         const int direction = face.coordDir();
         if (direction == component) {
            continue;
         }
         const int local_component = (component < direction) ? component : component - 1;
         const auto dst = fine_edge_line_[face][local_component].array(mfi);
         AMREX_ASSERT(edge_line[component]->box().ixType() == fine_edge_line_[face][local_component].ixType());
         const auto offset = amrex::IntVect::TheDimensionVector(component).dim3();
         amrex::ParallelFor(amrex::Box(dst), ncomp_, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            const int ii = ratio[0] * i;
            const int jj = ratio[1] * j;
            const int kk = ratio[2] * k;
            for (int rr = 0; rr < ratio[component]; ++rr) {
               dst(i, j, k, n) += dt * src(ii + offset.x * rr, jj + offset.y * rr, kk + offset.z * rr, n);
            }
         });
      }
   }
}
#else
void CtEdgeFluxRegister::CrseAdd(amrex::MFIter const& mfi, amrex::FArrayBox const& edge_line, amrex::Real dt)
{
   AMREX_ASSERT(mfi.validbox() == mfi.tilebox());
   if (!has_coarse_fine_[mfi]) {
      return;
   }
   const auto dst = coarse_edge_line_.array(mfi);
   const auto src = edge_line.const_array();
   amrex::ParallelFor(amrex::convert(mfi.validbox(), amrex::IntVect(1)), ncomp_, [=] AMREX_GPU_DEVICE(
      int i,
      int j,
      int k,
      int n
   ) noexcept {
      dst(i, j, k, n) += dt * src(i, j, k, n);
   });
}

void CtEdgeFluxRegister::FineAdd(amrex::MFIter const& mfi, amrex::FArrayBox const& edge_line, amrex::Real dt)
{
   AMREX_ASSERT(mfi.validbox() == mfi.tilebox());
   const auto ratio = ratio_;
   const auto src = edge_line.const_array();
   for (amrex::OrientationIter oit; oit.isValid(); ++oit) {
      const amrex::Orientation face = oit();
      const auto dst = fine_edge_line_[face].array(mfi);
      amrex::ParallelFor(amrex::Box(dst), ncomp_, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
         const int ii = ratio[0] * i;
         const int jj = ratio[1] * j;
         dst(i, j, k, n) += dt * src(ii, jj, k, n);
      });
   }
}
#endif

void CtEdgeFluxRegister::Reflux(
   amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> const& coarse_face_b,
   int mesh_coordinate,
   MeshAxes mesh_axes
) const
{
   const auto problo = coarse_geom_.ProbLoArray();
   const auto dx = coarse_geom_.CellSizeArray();
#if (AMREX_SPACEDIM == 3)
   amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> fine_on_coarse_edge;
   for (int component = 0; component < AMREX_SPACEDIM; ++component) {
      fine_on_coarse_edge[component].define(
         coarse_edge_line_[component].boxArray(),
         coarse_edge_line_[component].DistributionMap(),
         ncomp_,
         0
      );
      for (amrex::OrientationIter oit; oit.isValid(); ++oit) {
         const amrex::Orientation face = oit();
         const int direction = face.coordDir();
         if (direction == component) {
            continue;
         }
         const int local_component = (component < direction) ? component : component - 1;
         fine_on_coarse_edge[component].ParallelCopy(fine_edge_line_[face][local_component], coarse_geom_.periodicity());
      }
   }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(*coarse_face_b[0]); mfi.isValid(); ++mfi) {
      if (!has_coarse_fine_[mfi]) {
         continue;
      }
      const amrex::Box xbx = amrex::convert(mfi.validbox(), coarse_face_b[0]->ixType());
      const amrex::Box ybx = amrex::convert(mfi.validbox(), coarse_face_b[1]->ixType());
      const amrex::Box zbx = amrex::convert(mfi.validbox(), coarse_face_b[2]->ixType());
      const auto xmask = fine_mask_[0].const_array(mfi);
      const auto ymask = fine_mask_[1].const_array(mfi);
      const auto zmask = fine_mask_[2].const_array(mfi);
      const auto bx_face = coarse_face_b[0]->array(mfi);
      const auto by_face = coarse_face_b[1]->array(mfi);
      const auto bz_face = coarse_face_b[2]->array(mfi);
      const auto coarse_ex = coarse_edge_line_[0].const_array(mfi);
      const auto coarse_ey = coarse_edge_line_[1].const_array(mfi);
      const auto coarse_ez = coarse_edge_line_[2].const_array(mfi);
      const auto fine_ex = fine_on_coarse_edge[0].const_array(mfi);
      const auto fine_ey = fine_on_coarse_edge[1].const_array(mfi);
      const auto fine_ez = fine_on_coarse_edge[2].const_array(mfi);
      amrex::ParallelFor(
         xbx,
         ncomp_,
         [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            const amrex::Real d_ey_lo = ymask(i, j, k) ? fine_ey(i, j, k, n) - coarse_ey(i, j, k, n) : 0.0_rt;
            const amrex::Real d_ey_hi = ymask(i, j, k + 1) ? fine_ey(i, j, k + 1, n) - coarse_ey(i, j, k + 1, n) : 0.0_rt;
            const amrex::Real d_ez_lo = zmask(i, j, k) ? fine_ez(i, j, k, n) - coarse_ez(i, j, k, n) : 0.0_rt;
            const amrex::Real d_ez_hi = zmask(i, j + 1, k) ? fine_ez(i, j + 1, k, n) - coarse_ez(i, j + 1, k, n) : 0.0_rt;
            const amrex::Real area = mesh_face_area_scale(0, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            amrex::Real face_flux = generated::magnetic_face_flux(bx_face(i, j, k, n), area);
            face_flux = generated::ct_flux_update_3d_dir0(face_flux, 1.0_rt, d_ez_hi, d_ez_lo, d_ey_hi, d_ey_lo);
            bx_face(i, j, k, n) = area > 0.0_rt ? generated::magnetic_face_component(face_flux, area) : 0.0_rt;
         },
         ybx,
         ncomp_,
         [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            const amrex::Real d_ex_lo = xmask(i, j, k) ? fine_ex(i, j, k, n) - coarse_ex(i, j, k, n) : 0.0_rt;
            const amrex::Real d_ex_hi = xmask(i, j, k + 1) ? fine_ex(i, j, k + 1, n) - coarse_ex(i, j, k + 1, n) : 0.0_rt;
            const amrex::Real d_ez_lo = zmask(i, j, k) ? fine_ez(i, j, k, n) - coarse_ez(i, j, k, n) : 0.0_rt;
            const amrex::Real d_ez_hi = zmask(i + 1, j, k) ? fine_ez(i + 1, j, k, n) - coarse_ez(i + 1, j, k, n) : 0.0_rt;
            const amrex::Real area = mesh_face_area_scale(1, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            amrex::Real face_flux = generated::magnetic_face_flux(by_face(i, j, k, n), area);
            face_flux = generated::ct_flux_update_3d_dir1(face_flux, 1.0_rt, d_ex_hi, d_ex_lo, d_ez_hi, d_ez_lo);
            by_face(i, j, k, n) = area > 0.0_rt ? generated::magnetic_face_component(face_flux, area) : 0.0_rt;
         },
         zbx,
         ncomp_,
         [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            const amrex::Real d_ex_lo = xmask(i, j, k) ? fine_ex(i, j, k, n) - coarse_ex(i, j, k, n) : 0.0_rt;
            const amrex::Real d_ex_hi = xmask(i, j + 1, k) ? fine_ex(i, j + 1, k, n) - coarse_ex(i, j + 1, k, n) : 0.0_rt;
            const amrex::Real d_ey_lo = ymask(i, j, k) ? fine_ey(i, j, k, n) - coarse_ey(i, j, k, n) : 0.0_rt;
            const amrex::Real d_ey_hi = ymask(i + 1, j, k) ? fine_ey(i + 1, j, k, n) - coarse_ey(i + 1, j, k, n) : 0.0_rt;
            const amrex::Real area = mesh_face_area_scale(2, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            amrex::Real face_flux = generated::magnetic_face_flux(bz_face(i, j, k, n), area);
            face_flux = generated::ct_flux_update_3d_dir2(face_flux, 1.0_rt, d_ey_hi, d_ey_lo, d_ex_hi, d_ex_lo);
            bz_face(i, j, k, n) = area > 0.0_rt ? generated::magnetic_face_component(face_flux, area) : 0.0_rt;
         }
      );
   }
#else
   amrex::MultiFab fine_on_coarse_edge(coarse_edge_line_.boxArray(), coarse_edge_line_.DistributionMap(), ncomp_, 0);
   for (amrex::OrientationIter oit; oit.isValid(); ++oit) {
      const amrex::Orientation face = oit();
      fine_on_coarse_edge.ParallelCopy(fine_edge_line_[face], coarse_geom_.periodicity());
   }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(*coarse_face_b[0]); mfi.isValid(); ++mfi) {
      if (!has_coarse_fine_[mfi]) {
         continue;
      }
      const amrex::Box xbx = amrex::convert(mfi.validbox(), coarse_face_b[0]->ixType());
      const amrex::Box ybx = amrex::convert(mfi.validbox(), coarse_face_b[1]->ixType());
      const auto mask = fine_mask_.const_array(mfi);
      const auto bx_face = coarse_face_b[0]->array(mfi);
      const auto by_face = coarse_face_b[1]->array(mfi);
      const auto coarse_ez = coarse_edge_line_.const_array(mfi);
      const auto fine_ez = fine_on_coarse_edge.const_array(mfi);
      amrex::ParallelFor(
         xbx,
         ncomp_,
         [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            const amrex::Real d_ez_lo = mask(i, j, k) ? fine_ez(i, j, k, n) - coarse_ez(i, j, k, n) : 0.0_rt;
            const amrex::Real d_ez_hi = mask(i, j + 1, k) ? fine_ez(i, j + 1, k, n) - coarse_ez(i, j + 1, k, n) : 0.0_rt;
            const amrex::Real area = mesh_face_area_scale(0, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            amrex::Real face_flux = generated::magnetic_face_flux(bx_face(i, j, k, n), area);
            face_flux = generated::ct_flux_update_2d_dir0(face_flux, 1.0_rt, d_ez_hi, d_ez_lo);
            bx_face(i, j, k, n) = area > 0.0_rt ? generated::magnetic_face_component(face_flux, area) : 0.0_rt;
         },
         ybx,
         ncomp_,
         [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            const amrex::Real d_ez_lo = mask(i, j, k) ? fine_ez(i, j, k, n) - coarse_ez(i, j, k, n) : 0.0_rt;
            const amrex::Real d_ez_hi = mask(i + 1, j, k) ? fine_ez(i + 1, j, k, n) - coarse_ez(i + 1, j, k, n) : 0.0_rt;
            const amrex::Real area = mesh_face_area_scale(1, mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            amrex::Real face_flux = generated::magnetic_face_flux(by_face(i, j, k, n), area);
            face_flux = generated::ct_flux_update_2d_dir1(face_flux, 1.0_rt, d_ez_hi, d_ez_lo);
            by_face(i, j, k, n) = area > 0.0_rt ? generated::magnetic_face_component(face_flux, area) : 0.0_rt;
         }
      );
   }
#endif
}

} // namespace pyfluid_amrex
#endif
