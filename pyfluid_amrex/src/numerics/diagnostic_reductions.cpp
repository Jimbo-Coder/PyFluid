#include "pyfluid_amrex/diagnostic_reductions.hpp"
#include "pyfluid_amrex/reduction_sum.hpp"

#include <AMReX_Reduce.H>

#include <limits>

namespace pyfluid_amrex {

amrex::Real diagnostic_component_sum(amrex::MultiFab const& fields, int component)
{
   amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
   amrex::ReduceData<ReductionSum> reduce_data(reduce_ops);
   using ReduceTuple = decltype(reduce_data)::Type;
   const auto values = fields.const_arrays();
   reduce_ops.eval(fields, amrex::IntVect(0), reduce_data,
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> ReduceTuple {
         return {values[box_no](i, j, k, component)};
      });
   amrex::Real total = static_cast<amrex::Real>(amrex::get<0>(reduce_data.value()));
   amrex::ParallelDescriptor::ReduceRealSum(total);
   return total;
}

CellMaximum diagnostic_masked_maximum(
   amrex::MultiFab const& fields,
   amrex::iMultiFab const& mask,
   amrex::Box const& domain,
   int component
)
{
   AMREX_ALWAYS_ASSERT(fields.boxArray() == mask.boxArray());
   AMREX_ALWAYS_ASSERT(fields.DistributionMap() == mask.DistributionMap());
   const auto values = fields.const_arrays();
   const auto masks = mask.const_arrays();
   constexpr amrex::Real infinity = std::numeric_limits<amrex::Real>::infinity();
   amrex::ReduceOps<amrex::ReduceOpMax> maximum_ops;
   amrex::ReduceData<amrex::Real> maximum_data(maximum_ops);
   using MaximumTuple = decltype(maximum_data)::Type;
   maximum_ops.eval(fields, amrex::IntVect(0), maximum_data,
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> MaximumTuple {
         if (masks[box_no](i, j, k, 0) == 0) return {-infinity};
         const amrex::Real value = values[box_no](i, j, k, component);
         return {amrex::Math::isfinite(value) ? value : infinity};
      });
   const amrex::Real maximum = amrex::get<0>(maximum_data.value());
   if (!std::isfinite(maximum)) return {maximum, -1};

   // Keep Real and index reductions separate: an FP64 value plus an index
   // does not fit the standard SYCL per-lane reduction scratch value.
   constexpr amrex::Long no_index = std::numeric_limits<amrex::Long>::max();
   amrex::ReduceOps<amrex::ReduceOpMin> index_ops;
   amrex::ReduceData<amrex::Long> index_data(index_ops);
   using IndexTuple = decltype(index_data)::Type;
   index_ops.eval(fields, amrex::IntVect(0), index_data,
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> IndexTuple {
         if (masks[box_no](i, j, k, 0) == 0
             || values[box_no](i, j, k, component) != maximum) return {no_index};
         return {domain.index(amrex::IntVect(AMREX_D_DECL(i, j, k)))};
      });
   const amrex::Long cell_index = amrex::get<0>(index_data.value());
   return cell_index == no_index ? CellMaximum{-infinity, -1}
                                 : CellMaximum{maximum, cell_index};
}

} // namespace pyfluid_amrex
