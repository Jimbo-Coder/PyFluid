#pragma once

#include "pyfluid_amrex/relativistic_state.hpp"

namespace pyfluid_amrex {

template <typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_eulerian_matter(
   Array const& output, int i, int j, int k, EulerianMatter const& matter
) noexcept
{
   output(i, j, k, EULERIAN_MATTER_RHO) = matter.rho;
   output(i, j, k, EULERIAN_MATTER_MOMENTUM1) = matter.momentum1;
   output(i, j, k, EULERIAN_MATTER_MOMENTUM2) = matter.momentum2;
   output(i, j, k, EULERIAN_MATTER_MOMENTUM3) = matter.momentum3;
   output(i, j, k, EULERIAN_MATTER_STRESS11) = matter.stress11;
   output(i, j, k, EULERIAN_MATTER_STRESS12) = matter.stress12;
   output(i, j, k, EULERIAN_MATTER_STRESS13) = matter.stress13;
   output(i, j, k, EULERIAN_MATTER_STRESS22) = matter.stress22;
   output(i, j, k, EULERIAN_MATTER_STRESS23) = matter.stress23;
   output(i, j, k, EULERIAN_MATTER_STRESS33) = matter.stress33;
}

template <typename Array>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
EulerianMatter load_eulerian_matter(Array const& input, int i, int j, int k) noexcept
{
   return {
      input(i, j, k, EULERIAN_MATTER_RHO),
      input(i, j, k, EULERIAN_MATTER_MOMENTUM1),
      input(i, j, k, EULERIAN_MATTER_MOMENTUM2),
      input(i, j, k, EULERIAN_MATTER_MOMENTUM3),
      input(i, j, k, EULERIAN_MATTER_STRESS11),
      input(i, j, k, EULERIAN_MATTER_STRESS12),
      input(i, j, k, EULERIAN_MATTER_STRESS13),
      input(i, j, k, EULERIAN_MATTER_STRESS22),
      input(i, j, k, EULERIAN_MATTER_STRESS23),
      input(i, j, k, EULERIAN_MATTER_STRESS33),
   };
}

} // namespace pyfluid_amrex
