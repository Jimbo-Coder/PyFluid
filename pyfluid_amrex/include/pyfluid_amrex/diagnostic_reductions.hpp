#pragma once

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

namespace pyfluid_amrex {

// Sum valid cells of one already-weighted diagnostic component over all ranks.
// FP32 accumulates a compensation residual, then returns the selected Real type.
amrex::Real diagnostic_component_sum(amrex::MultiFab const& fields, int component);

struct CellMaximum {
   amrex::Real value;
   amrex::Long cell_index;
};

// Local-rank maximum over unmasked valid cells. Ties choose the smallest
// domain-relative cell index. Empty/nonfinite results have cell_index=-1;
// a nonfinite unmasked value propagates +infinity for collective rejection.
CellMaximum diagnostic_masked_maximum(
   amrex::MultiFab const& fields,
   amrex::iMultiFab const& mask,
   amrex::Box const& domain,
   int component
);

} // namespace pyfluid_amrex
