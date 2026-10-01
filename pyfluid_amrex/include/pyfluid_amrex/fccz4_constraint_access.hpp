#pragma once

#include "pyfluid_amrex/fccz4_state_access.hpp"
#include "pyfluid_amrex/kernels_generated/fccz4/geometry.hpp"
#include "pyfluid_amrex/kernels_generated/fccz4/constraints.hpp"

namespace pyfluid_amrex {

// Adapt physical constraints to the existing diagnostic layout. Components
// are in the runtime unit basis; the reduction owns norm/volume conventions.
template <typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cConstraints evaluate_fccz4_constraint_layout(
   Array const& values, int i, int j, int k, Z4cSpatialDerivatives const& derivatives,
   ReferenceChartPoint const& chart, EulerianMatter const& matter) noexcept
{
   const auto point = gather_fccz4_point(values, i, j, k, derivatives, chart);
   const auto geometry = generated::fccz4_geometry(point);
   const auto result = generated::fccz4_constraints(point, canonical_matter(matter, chart), geometry);
   // Legacy z_tilde diagnostics mean (Lambda - Delta)/2, not physical Z_i.
   return {result.theta,
           geometry.z_contravariant[chart.canonical[0]] / point.chi.value,
           geometry.z_contravariant[chart.canonical[1]] / point.chi.value,
           geometry.z_contravariant[chart.canonical[2]] / point.chi.value, result.hamiltonian,
           result.momentum[chart.canonical[0]], result.momentum[chart.canonical[1]],
           result.momentum[chart.canonical[2]], result.log_determinant, result.trace_a};
}

} // namespace pyfluid_amrex
