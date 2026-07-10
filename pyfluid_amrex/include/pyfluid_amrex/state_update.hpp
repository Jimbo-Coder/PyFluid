#pragma once

#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include "pyfluid_amrex/mask_regions.hpp"

namespace pyfluid_amrex {

template <int NComp>
inline void copy_plus_scaled_components(
   amrex::MultiFab& out,
   amrex::MultiFab const& base,
   amrex::MultiFab const& increment,
   amrex::Real scale
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(base, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto inc = increment.const_array(mfi);
      const auto o = out.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         for (int q = 0; q < NComp; ++q) {
            o(i, j, k, q) = b(i, j, k, q) + scale * inc(i, j, k, q);
         }
      });
   }
   }
}

template <int NComp>
inline void copy_plus_scaled_components_masked(
   amrex::MultiFab& out,
   amrex::MultiFab const& base,
   amrex::MultiFab const& increment,
   amrex::iMultiFab const& solid_mask,
   amrex::Real scale,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(base); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.validbox();
      const auto b = base.const_array(mfi);
	      const auto inc = increment.const_array(mfi);
	      const auto solid = solid_mask.const_array(mfi);
			      const auto o = out.array(mfi);
			      const int kind = cached_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
				      if (kind == MaskKindNonzero) {
				         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				            for (int q = 0; q < NComp; ++q) {
				               o(i, j, k, q) = b(i, j, k, q);
				            }
				         });
				         continue;
				      }
	      if (kind == MaskKindZero) {
	         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	            for (int q = 0; q < NComp; ++q) {
	               o(i, j, k, q) = b(i, j, k, q) + scale * inc(i, j, k, q);
	            }
	         });
	         continue;
	      }
	      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	         if (solid(i, j, k, 0) != 0) {
	            for (int q = 0; q < NComp; ++q) {
               o(i, j, k, q) = b(i, j, k, q);
            }
            return;
         }
         for (int q = 0; q < NComp; ++q) {
            o(i, j, k, q) = b(i, j, k, q) + scale * inc(i, j, k, q);
         }
      });
   }
   }
}

template <int NComp>
inline void add_scaled_components(amrex::MultiFab& out, amrex::MultiFab const& increment, amrex::Real scale)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(out, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.tilebox();
      const auto inc = increment.const_array(mfi);
      const auto o = out.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         for (int q = 0; q < NComp; ++q) {
            o(i, j, k, q) += scale * inc(i, j, k, q);
         }
      });
   }
	   }
}

template <int NComp>
inline void copy_components_masked(
   amrex::MultiFab& out,
   amrex::MultiFab const& input,
   amrex::iMultiFab const& solid_mask,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
	   for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.validbox();
	      const auto in = input.const_array(mfi);
	      const auto solid = solid_mask.const_array(mfi);
		      const auto o = out.array(mfi);
		      const int kind = cached_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
			      if (kind == MaskKindNonzero) {
			         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			            for (int q = 0; q < NComp; ++q) {
			               o(i, j, k, q) = in(i, j, k, q);
			            }
			         });
			         continue;
			      }
	      if (kind == MaskKindZero) {
	         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            for (int q = 0; q < NComp; ++q) {
               o(i, j, k, q) = in(i, j, k, q);
            }
         });
         continue;
      }
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (solid(i, j, k, 0) != 0) {
            return;
         }
         for (int q = 0; q < NComp; ++q) {
            o(i, j, k, q) = in(i, j, k, q);
         }
      });
   }
   }
}

template <int NComp>
inline void add_scaled_components_masked(
   amrex::MultiFab& out,
   amrex::MultiFab const& increment,
   amrex::iMultiFab const& solid_mask,
   amrex::Real scale,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.validbox();
	      const auto inc = increment.const_array(mfi);
	      const auto solid = solid_mask.const_array(mfi);
	      const auto o = out.array(mfi);
	      const int kind = cached_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
	      if (kind == MaskKindNonzero) {
	         continue;
	      }
	      if (kind == MaskKindZero) {
	         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	            for (int q = 0; q < NComp; ++q) {
	               o(i, j, k, q) += scale * inc(i, j, k, q);
	            }
	         });
	         continue;
	      }
	      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	         if (solid(i, j, k, 0) != 0) {
	            return;
         }
         for (int q = 0; q < NComp; ++q) {
            o(i, j, k, q) += scale * inc(i, j, k, q);
         }
      });
   }
   }
}

template <int NComp>
inline void copy_two_plus_scaled_components(
   amrex::MultiFab& out_a,
   amrex::MultiFab& out_b,
   amrex::MultiFab const& base,
   amrex::MultiFab const& increment,
   amrex::Real scale_a,
   amrex::Real scale_b
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(base, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto inc = increment.const_array(mfi);
      const auto a = out_a.array(mfi);
      const auto o = out_b.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         for (int q = 0; q < NComp; ++q) {
            const amrex::Real value = b(i, j, k, q);
            const amrex::Real rhs = inc(i, j, k, q);
            a(i, j, k, q) = value + scale_a * rhs;
            o(i, j, k, q) = value + scale_b * rhs;
         }
      });
   }
   }
}

template <int NComp>
inline void copy_two_plus_scaled_components_masked(
   amrex::MultiFab& out_a,
   amrex::MultiFab& out_b,
   amrex::MultiFab const& base,
   amrex::MultiFab const& increment,
   amrex::iMultiFab const& solid_mask,
   amrex::Real scale_a,
   amrex::Real scale_b,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(base); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.validbox();
      const auto b = base.const_array(mfi);
      const auto inc = increment.const_array(mfi);
	      const auto solid = solid_mask.const_array(mfi);
	      const auto a = out_a.array(mfi);
		      const auto o = out_b.array(mfi);
		      const int kind = cached_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
		      if (kind == MaskKindNonzero) {
		         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		            for (int q = 0; q < NComp; ++q) {
		               const amrex::Real value = b(i, j, k, q);
		               a(i, j, k, q) = value;
		               o(i, j, k, q) = value;
		            }
		         });
		         continue;
		      }
	      if (kind == MaskKindZero) {
	         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	            for (int q = 0; q < NComp; ++q) {
	               const amrex::Real value = b(i, j, k, q);
	               const amrex::Real rhs = inc(i, j, k, q);
	               a(i, j, k, q) = value + scale_a * rhs;
	               o(i, j, k, q) = value + scale_b * rhs;
	            }
	         });
	         continue;
	      }
	      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	         for (int q = 0; q < NComp; ++q) {
	            const amrex::Real value = b(i, j, k, q);
            if (solid(i, j, k, 0) != 0) {
               a(i, j, k, q) = value;
               o(i, j, k, q) = value;
            } else {
               const amrex::Real rhs = inc(i, j, k, q);
               a(i, j, k, q) = value + scale_a * rhs;
               o(i, j, k, q) = value + scale_b * rhs;
            }
         }
      });
   }
   }
}

template <int NComp>
inline void add_scaled_and_copy_plus_scaled_components(
   amrex::MultiFab& accumulator,
   amrex::MultiFab& scratch,
   amrex::MultiFab const& base,
   amrex::MultiFab const& increment,
   amrex::Real accumulator_scale,
   amrex::Real scratch_scale
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(accumulator, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto inc = increment.const_array(mfi);
      const auto acc = accumulator.array(mfi);
      const auto s = scratch.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         for (int q = 0; q < NComp; ++q) {
            const amrex::Real rhs = inc(i, j, k, q);
            acc(i, j, k, q) += accumulator_scale * rhs;
            s(i, j, k, q) = b(i, j, k, q) + scratch_scale * rhs;
         }
      });
   }
   }
}

template <int NComp>
inline void add_scaled_and_copy_plus_scaled_components_masked(
   amrex::MultiFab& accumulator,
   amrex::MultiFab& scratch,
   amrex::MultiFab const& base,
   amrex::MultiFab const& increment,
   amrex::iMultiFab const& solid_mask,
   amrex::Real accumulator_scale,
   amrex::Real scratch_scale,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(accumulator); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.validbox();
      const auto b = base.const_array(mfi);
      const auto inc = increment.const_array(mfi);
	      const auto solid = solid_mask.const_array(mfi);
	      const auto acc = accumulator.array(mfi);
		      const auto s = scratch.array(mfi);
		      const int kind = cached_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
		      if (kind == MaskKindNonzero) {
		         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		            for (int q = 0; q < NComp; ++q) {
		               s(i, j, k, q) = b(i, j, k, q);
		            }
		         });
		         continue;
		      }
	      if (kind == MaskKindZero) {
	         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	            for (int q = 0; q < NComp; ++q) {
	               const amrex::Real rhs = inc(i, j, k, q);
	               acc(i, j, k, q) += accumulator_scale * rhs;
	               s(i, j, k, q) = b(i, j, k, q) + scratch_scale * rhs;
	            }
	         });
	         continue;
	      }
	      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	         for (int q = 0; q < NComp; ++q) {
	            if (solid(i, j, k, 0) != 0) {
               s(i, j, k, q) = b(i, j, k, q);
            } else {
               const amrex::Real rhs = inc(i, j, k, q);
               acc(i, j, k, q) += accumulator_scale * rhs;
               s(i, j, k, q) = b(i, j, k, q) + scratch_scale * rhs;
            }
         }
      });
   }
   }
}

template <int NComp>
inline void linear_combination_plus_scaled_components(
   amrex::MultiFab& out,
   amrex::MultiFab const& base,
   amrex::MultiFab const& input,
   amrex::MultiFab const& increment,
   amrex::Real base_weight,
   amrex::Real input_weight,
   amrex::Real increment_scale
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(out, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.tilebox();
      const auto b = base.const_array(mfi);
      const auto s = input.const_array(mfi);
      const auto inc = increment.const_array(mfi);
      const auto o = out.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         for (int q = 0; q < NComp; ++q) {
            o(i, j, k, q) = base_weight * b(i, j, k, q) + input_weight * s(i, j, k, q) + increment_scale * inc(i, j, k, q);
         }
      });
   }
   }
}

template <int NComp>
inline void linear_combination_plus_scaled_components_masked(
   amrex::MultiFab& out,
   amrex::MultiFab const& base,
   amrex::MultiFab const& input,
   amrex::MultiFab const& increment,
   amrex::iMultiFab const& solid_mask,
   amrex::Real base_weight,
   amrex::Real input_weight,
   amrex::Real increment_scale,
   amrex::Vector<int> const* solid_valid_kind = nullptr
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
	   {
	   for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi) {
	      const amrex::Box& bx = mfi.validbox();
      const auto b = base.const_array(mfi);
      const auto s = input.const_array(mfi);
	      const auto inc = increment.const_array(mfi);
	      const auto solid = solid_mask.const_array(mfi);
	      const auto o = out.array(mfi);
	      const int kind = cached_mask_kind(solid_valid_kind, mfi, solid_mask, bx);
		      if (kind == MaskKindNonzero) {
		         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		            for (int q = 0; q < NComp; ++q) {
		               o(i, j, k, q) = b(i, j, k, q);
		            }
		         });
		         continue;
		      }
	      if (kind == MaskKindZero) {
	         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	            for (int q = 0; q < NComp; ++q) {
	               o(i, j, k, q) =
	                  base_weight * b(i, j, k, q) + input_weight * s(i, j, k, q) + increment_scale * inc(i, j, k, q);
	            }
	         });
	         continue;
	      }
	      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	         for (int q = 0; q < NComp; ++q) {
	            if (solid(i, j, k, 0) != 0) {
               o(i, j, k, q) = b(i, j, k, q);
            } else {
               o(i, j, k, q) =
                  base_weight * b(i, j, k, q) + input_weight * s(i, j, k, q) + increment_scale * inc(i, j, k, q);
            }
         }
      });
   }
   }
}


} // namespace pyfluid_amrex
