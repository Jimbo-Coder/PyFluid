#include "pyfluid_amrex/boundary.hpp"
#include "pyfluid_amrex/boundary_point.hpp"
#include "pyfluid_amrex/patch_domain.hpp"

#include <AMReX_BC_TYPES.H>
#include <AMReX_BoxList.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>

namespace pyfluid_amrex {

using namespace amrex::literals;

int amrex_state_bc_type(BoundaryKind kind, int component, int dir, bool mhd)
{
   if (kind == BoundaryKind::Periodic) {
      return amrex::BCType::int_dir;
   }
   if (kind == BoundaryKind::Inflow) {
      return amrex::BCType::ext_dir;
   }
   if (kind == BoundaryKind::Analytic) {
      return amrex::BCType::ext_dir;
   }
   if (kind == BoundaryKind::Outflow || kind == BoundaryKind::Diode) {
      return amrex::BCType::foextrap;
   }
   if (kind == BoundaryKind::Origin) {
      return amrex::BCType::reflect_even;
   }
   if (kind == BoundaryKind::Reflecting || kind == BoundaryKind::Symmetry) {
      const bool normal_momentum =
         (dir == 0 && component == MOM1)
#if (AMREX_SPACEDIM >= 2)
         || (dir == 1 && component == MOM2)
#endif
#if (AMREX_SPACEDIM == 3)
         || (dir == 2 && component == MOM3)
#endif
         ;
      if (normal_momentum) {
         return amrex::BCType::reflect_odd;
      }
      if (kind == BoundaryKind::Symmetry && mhd) {
         const bool magnetic_component = component == B1 || component == B2 || component == B3;
         const bool normal_magnetic =
            (dir == 0 && component == B1)
#if (AMREX_SPACEDIM >= 2)
            || (dir == 1 && component == B2)
#endif
#if (AMREX_SPACEDIM == 3)
            || (dir == 2 && component == B3)
#endif
            ;
         if (magnetic_component && !normal_magnetic) {
            return amrex::BCType::reflect_odd;
         }
      }
      return amrex::BCType::reflect_even;
   }
   return amrex::BCType::int_dir;
}

int amrex_face_b_bc_type(BoundaryKind kind, int face_dir, int boundary_dir)
{
   if (kind == BoundaryKind::Periodic) {
      return amrex::BCType::int_dir;
   }
   if (kind == BoundaryKind::Inflow) {
      return amrex::BCType::ext_dir;
   }
   if (kind == BoundaryKind::Analytic) {
      return amrex::BCType::ext_dir;
   }
   if (kind == BoundaryKind::Outflow || kind == BoundaryKind::Diode) {
      return amrex::BCType::foextrap;
   }
   if (kind == BoundaryKind::Origin) {
      return amrex::BCType::reflect_even;
   }
   if (kind == BoundaryKind::Reflecting) {
      return amrex::BCType::reflect_even;
   }
   if (kind == BoundaryKind::Symmetry) {
      return face_dir == boundary_dir ? amrex::BCType::reflect_even : amrex::BCType::reflect_odd;
   }
   return amrex::BCType::int_dir;
}

template <int NComp>
void apply_state_boundary(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   BoundaryData const& bc,
   amrex::Real time
)
{
   state.FillBoundary(geom.periodicity());
   validate_origin_boundary_geometry(bc, geom);
   if (boundary_data_all_periodic(bc)) {
      return;
   }

   const amrex::Box domain = geom.Domain();
   const auto lo = amrex::lbound(domain);
   const auto hi = amrex::ubound(domain);
   const auto is_periodic = geom.isPeriodicArray();
   const int ng = state.nGrow();
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
   const int analytic_bc = static_cast<int>(BoundaryKind::Analytic);
   const int diode_bc = static_cast<int>(BoundaryKind::Diode);
   const int origin_bc = static_cast<int>(BoundaryKind::Origin);
   const int nonlocal_origin_dir = needs_radial_antipodal_origin_boundary(bc)
      ? mesh_direction_for_axis(bc.mesh_axes, MeshAxisR) : -1;
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	      amrex::Box grown = mfi.growntilebox(ng);
	      grown &= state[mfi].box();
	      amrex::BoxList ghost_boxes = amrex::boxDiff(grown, domain);
	      const auto s = state.array(mfi);
	      for (amrex::Box const& bx : ghost_boxes) {
	      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         bool outside = !is_periodic[0] && (i < lo.x || i > hi.x);
#if (AMREX_SPACEDIM >= 2)
         outside = outside || (!is_periodic[1] && (j < lo.y || j > hi.y));
#endif
#if (AMREX_SPACEDIM == 3)
         outside = outside || (!is_periodic[2] && (k < lo.z || k > hi.z));
#endif
         if (!outside) {
            return;
         }

         // The nonlocal owner completes these ghosts. On a partial AMR level,
         // preserve the hierarchy-filled values between those completions (for
         // example the fixed boundary data of an implicit diffusion iteration).
         if ((nonlocal_origin_dir == 0 && i < lo.x)
#if AMREX_SPACEDIM >= 2
             || (nonlocal_origin_dir == 1 && j < lo.y)
#endif
#if AMREX_SPACEDIM == 3
             || (nonlocal_origin_dir == 2 && k < lo.z)
#endif
         ) {
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

         const bool touches_inflow = (xlo && xlo_bc == inflow_bc) || (xhi && xhi_bc == inflow_bc)
#if (AMREX_SPACEDIM >= 2)
            || (ylo && ylo_bc == inflow_bc) || (yhi && yhi_bc == inflow_bc)
#endif
#if (AMREX_SPACEDIM == 3)
            || (zlo && zlo_bc == inflow_bc) || (zhi && zhi_bc == inflow_bc)
#endif
            ;

         int analytic_side = -1;
         if (xlo && xlo_bc == analytic_bc) {
            analytic_side = 0;
         } else if (xhi && xhi_bc == analytic_bc) {
            analytic_side = 1;
         }
#if (AMREX_SPACEDIM >= 2)
         else if (ylo && ylo_bc == analytic_bc) {
            analytic_side = 2;
         } else if (yhi && yhi_bc == analytic_bc) {
            analytic_side = 3;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         else if (zlo && zlo_bc == analytic_bc) {
            analytic_side = 4;
         } else if (zhi && zhi_bc == analytic_bc) {
            analytic_side = 5;
         }
#endif

         int ii = is_periodic[0] ? i : (x_reflect ? mirrored_index(i, lo.x, hi.x) : clamp_index(i, lo.x, hi.x));
#if (AMREX_SPACEDIM >= 2)
         int jj = is_periodic[1] ? j : (y_reflect ? mirrored_index(j, lo.y, hi.y) : clamp_index(j, lo.y, hi.y));
#else
         const int jj = 0;
#endif
#if (AMREX_SPACEDIM == 3)
         int kk = is_periodic[2] ? k : (z_reflect ? mirrored_index(k, lo.z, hi.z) : clamp_index(k, lo.z, hi.z));
#else
         const int kk = 0;
#endif
         bool fixed_inflow = false;
         if (touches_inflow) {
            int boundary_dir = xlo || xhi ? 0 : -1;
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
            fixed_inflow = inflow_profile_is_fixed(bc, boundary_dir, x, y, z);
            if (fixed_inflow) {
               for (int q = 0; q < NComp; ++q) {
                  s(i, j, k, q) = inflow_state_component(bc, q, boundary_dir, x, y, z);
               }
            }
         }
         bool fixed_analytic = false;
         int analytic_action = AnalyticBoundaryCopy;
         int analytic_normal_dir = -1;
         if (analytic_side >= 0) {
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
            analytic_action = analytic_boundary_action(bc, analytic_side, x, y, z, time);
            analytic_normal_dir = analytic_side / 2;
            if (analytic_action == AnalyticBoundaryLeftState || analytic_action == AnalyticBoundaryRightState) {
               for (int q = 0; q < NComp; ++q) {
                  s(i, j, k, q) = analytic_shock_state_component(
                     bc, q, analytic_action == AnalyticBoundaryLeftState
                  );
               }
               fixed_analytic = true;
            } else if (analytic_action == AnalyticBoundaryReflect) {
               if (analytic_normal_dir == 0) {
                  ii = mirrored_index(i, lo.x, hi.x);
               }
#if (AMREX_SPACEDIM >= 2)
               if (analytic_normal_dir == 1) {
                  jj = mirrored_index(j, lo.y, hi.y);
               }
#endif
#if (AMREX_SPACEDIM == 3)
               if (analytic_normal_dir == 2) {
                  kk = mirrored_index(k, lo.z, hi.z);
               }
#endif
            }
         }
         if (!fixed_inflow && !fixed_analytic) {
            for (int q = 0; q < NComp; ++q) {
               s(i, j, k, q) = s(ii, jj, kk, q);
            }
         }
         if (analytic_action == AnalyticBoundaryReflect && analytic_normal_dir >= 0) {
            s(i, j, k, MOM1 + analytic_normal_dir) = -s(i, j, k, MOM1 + analytic_normal_dir);
         }
         if (x_origin) {
            for (int q = 0; q < NComp; ++q) {
               s(i, j, k, q) *= origin_state_parity(bc, 0, q);
            }
         } else if (x_reflect) {
            s(i, j, k, MOM1) = -s(i, j, k, MOM1);
         }
#if (AMREX_SPACEDIM >= 2)
         if (y_origin) {
            for (int q = 0; q < NComp; ++q) {
               s(i, j, k, q) *= origin_state_parity(bc, 1, q);
            }
         } else if (y_reflect) {
            s(i, j, k, MOM2) = -s(i, j, k, MOM2);
         }
#endif
#if (AMREX_SPACEDIM == 3)
         if (z_origin) {
            for (int q = 0; q < NComp; ++q) {
               s(i, j, k, q) *= origin_state_parity(bc, 2, q);
            }
         } else if (z_reflect) {
            s(i, j, k, MOM3) = -s(i, j, k, MOM3);
         }
#endif
         if constexpr (has_mhd_components<NComp>()) {
            if (x_symmetry) {
               s(i, j, k, B2) = -s(i, j, k, B2);
               s(i, j, k, B3) = -s(i, j, k, B3);
            }
#if (AMREX_SPACEDIM >= 2)
            if (y_symmetry) {
               s(i, j, k, B1) = -s(i, j, k, B1);
               s(i, j, k, B3) = -s(i, j, k, B3);
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (z_symmetry) {
               s(i, j, k, B1) = -s(i, j, k, B1);
               s(i, j, k, B2) = -s(i, j, k, B2);
            }
#endif
         }
         if ((xlo && xlo_bc == diode_bc && s(i, j, k, MOM1) > 0.0_rt)
            || (xhi && xhi_bc == diode_bc && s(i, j, k, MOM1) < 0.0_rt)) {
            s(i, j, k, MOM1) = 0.0_rt;
         }
#if (AMREX_SPACEDIM >= 2)
         if ((ylo && ylo_bc == diode_bc && s(i, j, k, MOM2) > 0.0_rt)
            || (yhi && yhi_bc == diode_bc && s(i, j, k, MOM2) < 0.0_rt)) {
            s(i, j, k, MOM2) = 0.0_rt;
         }
#endif
#if (AMREX_SPACEDIM == 3)
         if ((zlo && zlo_bc == diode_bc && s(i, j, k, MOM3) > 0.0_rt)
            || (zhi && zhi_bc == diode_bc && s(i, j, k, MOM3) < 0.0_rt)) {
            s(i, j, k, MOM3) = 0.0_rt;
         }
#endif
	      });
	      }
	   }
	   }
}


template void apply_state_boundary<NVAR_HYDRO>(amrex::MultiFab&, amrex::Geometry const&, BoundaryData const&, amrex::Real);
template void apply_state_boundary<NVAR_HYDRO + 1>(amrex::MultiFab&, amrex::Geometry const&, BoundaryData const&, amrex::Real);
template void apply_state_boundary<NVAR_MHD>(amrex::MultiFab&, amrex::Geometry const&, BoundaryData const&, amrex::Real);
template void apply_state_boundary<NVAR_MHD + 1>(amrex::MultiFab&, amrex::Geometry const&, BoundaryData const&, amrex::Real);

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool patch_union_fab_contains(
   amrex::Dim3 const& lo,
   amrex::Dim3 const& hi,
   int i,
   int j,
   int k
) noexcept
{
   bool contains = i >= lo.x && i <= hi.x;
#if (AMREX_SPACEDIM >= 2)
   contains = contains && j >= lo.y && j <= hi.y;
#endif
#if (AMREX_SPACEDIM == 3)
   contains = contains && k >= lo.z && k <= hi.z;
#endif
   return contains;
}

void apply_patch_union_reflecting_boundary(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   PatchDomainBoxes const& patch_domain
)
{
   const int ng = state.nGrow();
   if (ng <= 0) {
      return;
   }

   amrex::iMultiFab valid_mask(state.boxArray(), state.DistributionMap(), 1, state.nGrowVect());
   valid_mask.setVal(0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(valid_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto valid = valid_mask.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         valid(i, j, k, 0) = 1;
      });
   }
   }
   valid_mask.FillBoundary(geom.periodicity());

   const amrex::Box domain = geom.Domain();
   const auto domain_lo = amrex::lbound(domain);
   const auto domain_hi = amrex::ubound(domain);
   const int ncomp = state.nComp();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      amrex::Box grown = mfi.growntilebox(ng);
      grown &= state[mfi].box();
      if (!grown.ok()) {
         continue;
      }
      const auto s = state.array(mfi);
      const auto valid = valid_mask.const_array(mfi);
      const auto fab_lo = amrex::lbound(state[mfi].box());
      const auto fab_hi = amrex::ubound(state[mfi].box());
      amrex::ParallelFor(grown, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (
            !domain.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))
            || patch_domain_contains_cell(patch_domain, i, j, k)
         ) {
            return;
         }

         int reflected_axis = -1;
         int src_i = i;
         int src_j = j;
         int src_k = k;
         for (int d = 1; d <= ng && reflected_axis < 0; ++d) {
            const int candidate_i = i - d;
            if (
               patch_union_fab_contains(fab_lo, fab_hi, candidate_i, j, k)
               && valid(candidate_i, j, k, 0) != 0
               && patch_domain_contains_cell(patch_domain, candidate_i, j, k)
            ) {
               reflected_axis = 0;
               src_i = i - (2 * d - 1);
               if (
                  !patch_union_fab_contains(fab_lo, fab_hi, src_i, j, k)
                  || !patch_domain_contains_cell(patch_domain, src_i, j, k)
                  || valid(src_i, j, k, 0) == 0
               ) {
                  src_i = candidate_i;
               }
            }
            const int candidate_hi_i = i + d;
            if (
               reflected_axis < 0
               && patch_union_fab_contains(fab_lo, fab_hi, candidate_hi_i, j, k)
               && valid(candidate_hi_i, j, k, 0) != 0
               && patch_domain_contains_cell(patch_domain, candidate_hi_i, j, k)
            ) {
               reflected_axis = 0;
               src_i = i + (2 * d - 1);
               if (
                  !patch_union_fab_contains(fab_lo, fab_hi, src_i, j, k)
                  || !patch_domain_contains_cell(patch_domain, src_i, j, k)
                  || valid(src_i, j, k, 0) == 0
               ) {
                  src_i = candidate_hi_i;
               }
            }
         }
#if (AMREX_SPACEDIM >= 2)
         for (int d = 1; d <= ng && reflected_axis < 0; ++d) {
            const int candidate_j = j - d;
            if (
               patch_union_fab_contains(fab_lo, fab_hi, i, candidate_j, k)
               && valid(i, candidate_j, k, 0) != 0
               && patch_domain_contains_cell(patch_domain, i, candidate_j, k)
            ) {
               reflected_axis = 1;
               src_j = j - (2 * d - 1);
               if (
                  !patch_union_fab_contains(fab_lo, fab_hi, i, src_j, k)
                  || !patch_domain_contains_cell(patch_domain, i, src_j, k)
                  || valid(i, src_j, k, 0) == 0
               ) {
                  src_j = candidate_j;
               }
            }
            const int candidate_hi_j = j + d;
            if (
               reflected_axis < 0
               && patch_union_fab_contains(fab_lo, fab_hi, i, candidate_hi_j, k)
               && valid(i, candidate_hi_j, k, 0) != 0
               && patch_domain_contains_cell(patch_domain, i, candidate_hi_j, k)
            ) {
               reflected_axis = 1;
               src_j = j + (2 * d - 1);
               if (
                  !patch_union_fab_contains(fab_lo, fab_hi, i, src_j, k)
                  || !patch_domain_contains_cell(patch_domain, i, src_j, k)
                  || valid(i, src_j, k, 0) == 0
               ) {
                  src_j = candidate_hi_j;
               }
            }
         }
#endif
#if (AMREX_SPACEDIM == 3)
         for (int d = 1; d <= ng && reflected_axis < 0; ++d) {
            const int candidate_k = k - d;
            if (
               patch_union_fab_contains(fab_lo, fab_hi, i, j, candidate_k)
               && valid(i, j, candidate_k, 0) != 0
               && patch_domain_contains_cell(patch_domain, i, j, candidate_k)
            ) {
               reflected_axis = 2;
               src_k = k - (2 * d - 1);
               if (
                  !patch_union_fab_contains(fab_lo, fab_hi, i, j, src_k)
                  || !patch_domain_contains_cell(patch_domain, i, j, src_k)
                  || valid(i, j, src_k, 0) == 0
               ) {
                  src_k = candidate_k;
               }
            }
            const int candidate_hi_k = k + d;
            if (
               reflected_axis < 0
               && patch_union_fab_contains(fab_lo, fab_hi, i, j, candidate_hi_k)
               && valid(i, j, candidate_hi_k, 0) != 0
               && patch_domain_contains_cell(patch_domain, i, j, candidate_hi_k)
            ) {
               reflected_axis = 2;
               src_k = k + (2 * d - 1);
               if (
                  !patch_union_fab_contains(fab_lo, fab_hi, i, j, src_k)
                  || !patch_domain_contains_cell(patch_domain, i, j, src_k)
                  || valid(i, j, src_k, 0) == 0
               ) {
                  src_k = candidate_hi_k;
               }
            }
         }
#endif
         if (reflected_axis < 0) {
            return;
         }
         for (int q = 0; q < ncomp; ++q) {
            s(i, j, k, q) = s(src_i, src_j, src_k, q);
         }
         if (reflected_axis == 0) {
            s(i, j, k, MOM1) = -s(i, j, k, MOM1);
         }
#if (AMREX_SPACEDIM >= 2)
         if (reflected_axis == 1) {
            s(i, j, k, MOM2) = -s(i, j, k, MOM2);
         }
#endif
#if (AMREX_SPACEDIM == 3)
         if (reflected_axis == 2) {
            s(i, j, k, MOM3) = -s(i, j, k, MOM3);
         }
#endif
      });
   }
   }
}

void apply_configured_state_boundary(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   BoundaryData const& bc,
   amrex::Real time
)
{
   const int ncomp = state.nComp();
   if (bc.mhd != 0) {
      if (ncomp == NVAR_MHD + 1) {
         apply_state_boundary<NVAR_MHD + 1>(state, geom, bc, time);
      } else if (ncomp == NVAR_MHD) {
         apply_state_boundary<NVAR_MHD>(state, geom, bc, time);
      } else {
         amrex::Abort("MHD state component count does not match pyfluid.passive_scalars");
      }
   } else {
      if (ncomp == NVAR_HYDRO + 1) {
         apply_state_boundary<NVAR_HYDRO + 1>(state, geom, bc, time);
      } else if (ncomp == NVAR_HYDRO) {
         apply_state_boundary<NVAR_HYDRO>(state, geom, bc, time);
      } else {
         amrex::Abort("hydro state component count does not match pyfluid.passive_scalars");
      }
   }
   apply_radial_antipodal_origin_boundary(state, geom, bc);
}


} // namespace pyfluid_amrex
