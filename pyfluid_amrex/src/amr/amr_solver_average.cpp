#include "pyfluid_amrex/amr_solver_base.hpp"
#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/patch_metrics.hpp"

namespace pyfluid_amrex {

void AmrSolverBase::project_metric_conservative_prolongation(
   int fine_lev,
   amrex::MultiFab const& coarse_state,
   amrex::MultiFab& fine_state
) const
{
   const bool use_curvilinear = mesh_is_curvilinear();
   const bool use_mapped_patches = mixed_patch_system_enabled() && !mhd_enabled();
   const bool transform_state_vectors = use_curvilinear && !mhd_enabled();
   if (fine_lev <= 0 || (!use_curvilinear && !use_mapped_patches)) {
      return;
   }

   const int coarse_lev = fine_lev - 1;
   const int ncomp = num_state_components();
   const amrex::IntVect ratio = refRatio(coarse_lev);
   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   const auto problo = Geom(fine_lev).ProbLoArray();
   const auto dx = Geom(fine_lev).CellSizeArray();
   const auto coarse_problo = Geom(coarse_lev).ProbLoArray();
   const auto coarse_dx = Geom(coarse_lev).CellSizeArray();
   const PatchGridSystem patch_grid_system = use_mapped_patches
      ? production_patch_grid_system_for_level(fine_lev)
      : PatchGridSystem{};
   const PatchLayout patch_layout = use_mapped_patches
      ? production_patch_layout_for_level(fine_lev)
      : PatchLayout{};

   amrex::BoxArray coarse_boxes = fine_state.boxArray();
   coarse_boxes.coarsen(ratio);
   amrex::MultiFab correction(coarse_boxes, fine_state.DistributionMap(), ncomp + 1, 0);
   correction.setVal(0.0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(correction, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& coarse_box = mfi.tilebox();
      const PatchDescriptor layout_patch = use_mapped_patches
         ? patch_layout_descriptor_containing_box(patch_layout, amrex::refine(mfi.validbox(), ratio))
         : PatchDescriptor{};
      const PatchGridDescriptor grid_patch = use_mapped_patches
         ? patch_grid_descriptor_for_layout_patch(patch_grid_system, layout_patch)
         : PatchGridDescriptor{};
      const auto sums = correction.array(mfi);
      const auto fine = fine_state.const_array(mfi);
      const int rx = ratio[0];
#if (AMREX_SPACEDIM >= 2)
      const int ry = ratio[1];
#else
      const int ry = 1;
#endif
#if (AMREX_SPACEDIM == 3)
      const int rz = ratio[2];
#else
      const int rz = 1;
#endif
      amrex::ParallelFor(coarse_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         amrex::Real weight_sum = 0.0;
         amrex::Real weighted_sum[MaxStateComponents]{};
         for (int kk = 0; kk < rz; ++kk) {
            const int fk = k * rz + kk;
            for (int jj = 0; jj < ry; ++jj) {
               const int fj = j * ry + jj;
               for (int ii = 0; ii < rx; ++ii) {
                  const int fi = i * rx + ii;
                  amrex::Real weight = 1.0;
                  if (use_mapped_patches) {
                     const PatchCoordinate local =
                        patch_layout_cell_center_coordinate(grid_patch, layout_patch, fi, fj, fk);
                     weight = patch_cell_volume(grid_patch, local);
                  } else {
                     weight = curvilinear_cell_volume_scale(coordinate, axes, problo, dx, fi, fj, fk);
                  }
                  for (int n = 0; n < ncomp; ++n) {
                     weighted_sum[n] += weight * fine(fi, fj, fk, n);
                  }
                  if (transform_state_vectors) {
                     const auto fine_q = mesh_coordinate_triplet_at_location(
                        coordinate, axes, problo, dx, fi, fj, fk, false, false, false
                     );
                     const StateVector3 momentum = mesh_basis_vector_to_cartesian(
                        coordinate,
                        axes,
                        fine_q,
                        {fine(fi, fj, fk, MOM1), fine(fi, fj, fk, MOM2), fine(fi, fj, fk, MOM3)}
                     );
                     weighted_sum[MOM1] += weight * (momentum.x - fine(fi, fj, fk, MOM1));
                     weighted_sum[MOM2] += weight * (momentum.y - fine(fi, fj, fk, MOM2));
                     weighted_sum[MOM3] += weight * (momentum.z - fine(fi, fj, fk, MOM3));
                  }
                  weight_sum += weight;
               }
            }
         }
         for (int n = 0; n < ncomp; ++n) {
            sums(i, j, k, n) = weighted_sum[n];
         }
         sums(i, j, k, ncomp) = weight_sum;
      });
   }
   }

   amrex::MultiFab coarse_target(coarse_boxes, fine_state.DistributionMap(), ncomp, 0);
   coarse_target.ParallelCopy(
      coarse_state,
      0,
      0,
      ncomp,
      0,
      0,
      Geom(coarse_lev).periodicity()
   );

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(correction, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& coarse_box = mfi.tilebox();
      const auto delta = correction.array(mfi);
      const auto target = coarse_target.const_array(mfi);
      amrex::ParallelFor(coarse_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real weight_sum = delta(i, j, k, ncomp);
         amrex::Real target_values[MaxStateComponents]{};
         for (int n = 0; n < ncomp; ++n) {
            target_values[n] = target(i, j, k, n);
         }
         if (transform_state_vectors) {
            const auto coarse_q = mesh_coordinate_triplet_at_location(
               coordinate, axes, coarse_problo, coarse_dx, i, j, k, false, false, false
            );
            const StateVector3 momentum = mesh_basis_vector_to_cartesian(
               coordinate,
               axes,
               coarse_q,
               {target(i, j, k, MOM1), target(i, j, k, MOM2), target(i, j, k, MOM3)}
            );
            target_values[MOM1] = momentum.x;
            target_values[MOM2] = momentum.y;
            target_values[MOM3] = momentum.z;
         }
         for (int n = 0; n < ncomp; ++n) {
            delta(i, j, k, n) = target_values[n] - delta(i, j, k, n) / weight_sum;
         }
      });
   }
   }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(fine_state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& fine_box = mfi.tilebox();
      const auto fine = fine_state.array(mfi);
      const auto delta = correction.const_array(mfi);
      const int rx = ratio[0];
#if (AMREX_SPACEDIM >= 2)
      const int ry = ratio[1];
#else
      const int ry = 1;
#endif
#if (AMREX_SPACEDIM == 3)
      const int rz = ratio[2];
#else
      const int rz = 1;
#endif
      amrex::ParallelFor(fine_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const int ci = amrex::coarsen(i, rx);
         const int cj = amrex::coarsen(j, ry);
         const int ck = amrex::coarsen(k, rz);
         for (int n = 0; n < ncomp; ++n) {
            if (!transform_state_vectors || n < MOM1 || n > MOM3) {
               fine(i, j, k, n) += delta(ci, cj, ck, n);
            }
         }
         if (transform_state_vectors) {
            const auto fine_q = mesh_coordinate_triplet_at_location(
               coordinate,
               axes,
               problo,
               dx,
               i,
               j,
               k,
               false,
               false,
               false
            );
            StateVector3 momentum = mesh_basis_vector_to_cartesian(
               coordinate,
               axes,
               fine_q,
               {fine(i, j, k, MOM1), fine(i, j, k, MOM2), fine(i, j, k, MOM3)}
            );
            momentum.x += delta(ci, cj, ck, MOM1);
            momentum.y += delta(ci, cj, ck, MOM2);
            momentum.z += delta(ci, cj, ck, MOM3);
            momentum = cartesian_vector_to_mesh_basis(coordinate, axes, fine_q, momentum);
            fine(i, j, k, MOM1) = momentum.x;
            fine(i, j, k, MOM2) = momentum.y;
            fine(i, j, k, MOM3) = momentum.z;
         }
      });
   }
   }
}

void AmrSolverBase::average_down_state()
{
   for (int lev = finestLevel(); lev > 0; --lev) {
      average_down_state_level(lev);
   }
}

void AmrSolverBase::average_down_state_level(int fine_lev)
{
   if (fine_lev <= 0 || fine_lev > finestLevel()) {
      return;
   }
   if (
      (!mhd_enabled() && solid_active(fine_lev))
      || mesh_is_curvilinear()
      || (mixed_patch_system_enabled() && !mhd_enabled())
   ) {
      average_down_state_level_weighted_hydro(fine_lev);
      return;
   }
   amrex::average_down(
      *fluid_state_.state()[fine_lev],
      *fluid_state_.state()[fine_lev - 1],
      Geom(fine_lev),
      Geom(fine_lev - 1),
      0,
      num_state_components(),
      refRatio(fine_lev - 1)
   );
}

void AmrSolverBase::average_down_state_level_weighted_hydro(int fine_lev)
{
   if (fine_lev <= 0 || fine_lev > finestLevel()) {
      return;
   }
   const int coarse_lev = fine_lev - 1;
   const int ncomp = num_state_components();
   const amrex::IntVect ratio = refRatio(coarse_lev);
   const bool use_solid_mask = solid_active(fine_lev);
   const bool use_volume_fraction = use_solid_mask && solid_volume_fraction_[fine_lev] != nullptr;
   const bool use_curvilinear = mesh_is_curvilinear();
   const bool use_mapped_patches = mixed_patch_system_enabled() && !mhd_enabled();
   const bool transform_state_vectors = use_curvilinear && !mhd_enabled();
   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   const auto problo = Geom(fine_lev).ProbLoArray();
   const auto dx = Geom(fine_lev).CellSizeArray();
   const auto coarse_problo = Geom(coarse_lev).ProbLoArray();
   const auto coarse_dx = Geom(coarse_lev).CellSizeArray();
   const PatchGridSystem patch_grid_system = use_mapped_patches
      ? production_patch_grid_system_for_level(fine_lev)
      : PatchGridSystem{};
   const PatchLayout patch_layout = use_mapped_patches
      ? production_patch_layout_for_level(fine_lev)
      : PatchLayout{};
   amrex::BoxArray crse_fine_ba = fluid_state_.state()[fine_lev]->boxArray();
   crse_fine_ba.coarsen(ratio);
   amrex::MultiFab crse_from_fine(
      crse_fine_ba,
      fluid_state_.state()[fine_lev]->DistributionMap(),
      ncomp + 1,
      0
   );
   crse_from_fine.setVal(0.0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(crse_from_fine, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const PatchDescriptor layout_patch = use_mapped_patches
         ? patch_layout_descriptor_containing_box(patch_layout, amrex::refine(mfi.validbox(), ratio))
         : PatchDescriptor{};
      const PatchGridDescriptor grid_patch = use_mapped_patches
         ? patch_grid_descriptor_for_layout_patch(patch_grid_system, layout_patch)
         : PatchGridDescriptor{};
      const auto coarse = crse_from_fine.array(mfi);
      const auto fine = fluid_state_.state()[fine_lev]->const_array(mfi);
      const auto solid = use_solid_mask ? solid_mask_[fine_lev]->const_array(mfi) : amrex::Array4<int const>{};
      const auto vf = use_volume_fraction
         ? solid_volume_fraction_[fine_lev]->const_array(mfi)
         : amrex::Array4<amrex::Real const>{};
      const int rx = ratio[0];
#if (AMREX_SPACEDIM >= 2)
      const int ry = ratio[1];
#else
      const int ry = 1;
#endif
#if (AMREX_SPACEDIM == 3)
      const int rz = ratio[2];
#else
      const int rz = 1;
#endif
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         amrex::Real weight_sum = 0.0;
         amrex::Real sum[MaxStateComponents]{};
         for (int kk = 0; kk < rz; ++kk) {
            const int fk = k * rz + kk;
            for (int jj = 0; jj < ry; ++jj) {
               const int fj = j * ry + jj;
               for (int ii = 0; ii < rx; ++ii) {
                  const int fi = i * rx + ii;
                  if (use_solid_mask && solid(fi, fj, fk, 0) != 0) {
                     continue;
                  }
                  amrex::Real weight = 1.0;
                  if (use_mapped_patches) {
                     const PatchCoordinate local =
                        patch_layout_cell_center_coordinate(grid_patch, layout_patch, fi, fj, fk);
                     weight = patch_cell_volume(grid_patch, local);
                  } else if (use_curvilinear) {
                     weight = curvilinear_cell_volume_scale(coordinate, axes, problo, dx, fi, fj, fk);
                  }
                  if (use_volume_fraction) {
                     weight *= vf(fi, fj, fk, 0);
                  }
                  if (weight <= 0.0 || !amrex::Math::isfinite(weight)) {
                     continue;
                  }
                  for (int n = 0; n < ncomp; ++n) {
                     sum[n] += weight * fine(fi, fj, fk, n);
                  }
                  if (transform_state_vectors) {
                     const auto fine_q = mesh_coordinate_triplet_at_location(
                        coordinate, axes, problo, dx, fi, fj, fk, false, false, false
                     );
                     const StateVector3 momentum = mesh_basis_vector_to_cartesian(
                        coordinate,
                        axes,
                        fine_q,
                        {fine(fi, fj, fk, MOM1), fine(fi, fj, fk, MOM2), fine(fi, fj, fk, MOM3)}
                     );
                     sum[MOM1] += weight * (momentum.x - fine(fi, fj, fk, MOM1));
                     sum[MOM2] += weight * (momentum.y - fine(fi, fj, fk, MOM2));
                     sum[MOM3] += weight * (momentum.z - fine(fi, fj, fk, MOM3));
                  }
                  weight_sum += weight;
               }
            }
         }
         for (int n = 0; n < ncomp; ++n) {
            coarse(i, j, k, n) = sum[n];
         }
         coarse(i, j, k, ncomp) = weight_sum;
      });
   }
   }

   amrex::MultiFab overlay(
      fluid_state_.state()[fine_lev - 1]->boxArray(),
      fluid_state_.state()[fine_lev - 1]->DistributionMap(),
      ncomp + 1,
      0
   );
   overlay.setVal(0.0);
   overlay.ParallelCopy(
      crse_from_fine,
      0,
      0,
      ncomp + 1,
      0,
      0,
      Geom(coarse_lev).periodicity(),
      amrex::FabArrayBase::ADD
   );

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(*fluid_state_.state()[fine_lev - 1], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto coarse = fluid_state_.state()[fine_lev - 1]->array(mfi);
      const auto avg = overlay.const_array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (avg(i, j, k, ncomp) <= 0.0) {
            return;
         }
         const amrex::Real inv_weight = 1.0 / avg(i, j, k, ncomp);
         for (int n = 0; n < ncomp; ++n) {
            coarse(i, j, k, n) = avg(i, j, k, n) * inv_weight;
         }
         if (transform_state_vectors) {
            const auto coarse_q = mesh_coordinate_triplet_at_location(
               coordinate, axes, coarse_problo, coarse_dx, i, j, k, false, false, false
            );
            StateVector3 momentum{
               coarse(i, j, k, MOM1), coarse(i, j, k, MOM2), coarse(i, j, k, MOM3)
            };
            momentum = cartesian_vector_to_mesh_basis(coordinate, axes, coarse_q, momentum);
            coarse(i, j, k, MOM1) = momentum.x;
            coarse(i, j, k, MOM2) = momentum.y;
            coarse(i, j, k, MOM3) = momentum.z;
         }
      });
   }
   }
}

void AmrSolverBase::average_down_face_b()
{
   if (!ct_enabled(cfg_)) {
      return;
   }
   apply_all_face_b_boundaries();
   for (int lev = finestLevel(); lev > 0; --lev) {
      average_down_face_b_level(lev);
   }
}

void AmrSolverBase::average_down_face_b_level(int fine_lev)
{
   if (!ct_enabled(cfg_) || fine_lev <= 0 || fine_lev > finestLevel()) {
      return;
   }
   if (mesh_is_curvilinear()) {
      average_down_face_b_level_metric(fine_lev);
      return;
   }
   amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> fine;
   amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> coarse;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      fine[dir] = face_b_[fine_lev].b[dir].get();
      coarse[dir] = face_b_[fine_lev - 1].b[dir].get();
   }
   apply_face_b_boundary(face_b_[fine_lev], Geom(fine_lev), boundary_data_);
   amrex::average_down_faces(fine, coarse, refRatio(fine_lev - 1), 0);
   apply_face_b_boundary(face_b_[fine_lev - 1], Geom(fine_lev - 1), boundary_data_);
}

void AmrSolverBase::average_down_face_b_level_metric(int fine_lev)
{
   if (!ct_enabled(cfg_) || fine_lev <= 0 || fine_lev > finestLevel()) {
      return;
   }
   const int coarse_lev = fine_lev - 1;
   const amrex::IntVect ratio = refRatio(coarse_lev);
   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   const auto fine_problo = Geom(fine_lev).ProbLoArray();
   const auto fine_dx = Geom(fine_lev).CellSizeArray();
   const auto coarse_problo = Geom(coarse_lev).ProbLoArray();
   const auto coarse_dx = Geom(coarse_lev).CellSizeArray();

   apply_face_b_boundary(face_b_[fine_lev], Geom(fine_lev), boundary_data_);

   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::MultiFab const& fine_face = *face_b_[fine_lev].b[dir];
      amrex::MultiFab& coarse_face = *face_b_[coarse_lev].b[dir];

      amrex::BoxArray crse_fine_ba = fine_face.boxArray();
      crse_fine_ba.coarsen(ratio);
      amrex::MultiFab crse_from_fine(crse_fine_ba, fine_face.DistributionMap(), 2, 0);
      crse_from_fine.setVal(0.0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(crse_from_fine, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto fine = fine_face.const_array(mfi);
         const auto avg = crse_from_fine.array(mfi);
         const int d = dir;
         const int rx = ratio[0];
#if (AMREX_SPACEDIM >= 2)
         const int ry = ratio[1];
#else
         const int ry = 1;
#endif
#if (AMREX_SPACEDIM == 3)
         const int rz = ratio[2];
#else
         const int rz = 1;
#endif
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real flux_sum = 0.0;
            int fine_faces = 0;
            const int ni = d == 0 ? 1 : rx;
#if (AMREX_SPACEDIM >= 2)
            const int nj = d == 1 ? 1 : ry;
#else
            const int nj = 1;
#endif
#if (AMREX_SPACEDIM == 3)
            const int nk = d == 2 ? 1 : rz;
#else
            const int nk = 1;
#endif
            for (int kk = 0; kk < nk; ++kk) {
               const int fk = k * rz + (d == 2 ? 0 : kk);
               for (int jj = 0; jj < nj; ++jj) {
                  const int fj = j * ry + (d == 1 ? 0 : jj);
                  for (int ii = 0; ii < ni; ++ii) {
                     const int fi = i * rx + (d == 0 ? 0 : ii);
                     const amrex::Real fine_area =
                        mesh_face_area_scale(d, coordinate, axes, fine_problo, fine_dx, fi, fj, fk);
                     flux_sum += generated::magnetic_face_flux(fine(fi, fj, fk, 0), fine_area);
                     ++fine_faces;
                  }
               }
            }
            const amrex::Real coarse_area =
               mesh_face_area_scale(d, coordinate, axes, coarse_problo, coarse_dx, i, j, k);
            if (fine_faces > 0 && coarse_area > 0.0) {
               avg(i, j, k, 0) = generated::magnetic_face_component(flux_sum, coarse_area);
               avg(i, j, k, 1) = 1.0;
            }
         });
      }
      }

      amrex::MultiFab overlay(coarse_face.boxArray(), coarse_face.DistributionMap(), 2, 0);
      overlay.setVal(0.0);
      overlay.ParallelCopy(crse_from_fine, 0, 0, 2, 0, 0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(coarse_face, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto coarse = coarse_face.array(mfi);
         const auto avg = overlay.const_array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (avg(i, j, k, 1) > 0.0) {
               coarse(i, j, k, 0) = avg(i, j, k, 0);
            }
         });
      }
      }
   }

   apply_face_b_boundary(face_b_[coarse_lev], Geom(coarse_lev), boundary_data_);
}

void AmrSolverBase::apply_all_face_b_boundaries()
{
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      apply_face_b_boundary(face_b_[lev], Geom(lev), boundary_data_);
   }
}

void AmrSolverBase::sync_all_cell_b_from_faces(AmrSolverBase::LevelData& data, bool preserve_pressure)
{
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      sync_cell_b_from_faces(*data[lev], face_b_[lev], Geom(lev), preserve_pressure);
      apply_boundary(*data[lev], Geom(lev), level_times_[lev]);
   }
}

} // namespace pyfluid_amrex
