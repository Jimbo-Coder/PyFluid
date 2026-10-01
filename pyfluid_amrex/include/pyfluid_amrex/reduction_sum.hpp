#pragma once

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

#include <cmath>
#include <type_traits>

namespace pyfluid_amrex {

// Neumaier accumulation retains the residual of additions without introducing
// FP64 arithmetic into an FP32 kernel. Addition also merges partial sums, so
// the same type works in AMReX's serial, thread and device reduction trees.
class CompensatedSum {
public:
   AMREX_GPU_HOST_DEVICE
   constexpr CompensatedSum(amrex::Real value = 0) noexcept : sum_(value) {}

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   CompensatedSum& operator+=(CompensatedSum other) noexcept
   {
      // Copy the pair before mutation so self-addition has value semantics.
      add(other.sum_);
      add(other.correction_);
      return *this;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   explicit operator amrex::Real() const noexcept { return sum_ + correction_; }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   friend CompensatedSum operator+(CompensatedSum left, CompensatedSum const& right) noexcept
   {
      left += right;
      return left;
   }

private:
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   void add(amrex::Real value) noexcept
   {
      const amrex::Real next = sum_ + value;
      correction_ += std::abs(sum_) >= std::abs(value)
         ? (sum_ - next) + value : (value - next) + sum_;
      sum_ = next;
   }

   amrex::Real sum_ = 0;
   amrex::Real correction_ = 0;
};

// Retain the existing FP64 summation path. Both selected types fit AMReX's
// eight-byte-per-lane SYCL reduction scratch storage.
using ReductionSum = std::conditional_t<sizeof(amrex::Real) == 4, CompensatedSum, amrex::Real>;
static_assert(sizeof(ReductionSum) <= sizeof(unsigned long long));

} // namespace pyfluid_amrex
