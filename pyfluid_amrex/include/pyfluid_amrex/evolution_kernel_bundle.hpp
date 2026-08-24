#pragma once

namespace pyfluid_amrex {

class AmrSolver;

struct EvolutionKernelBundleKey {
   int num_components = 0;
   bool ct_enabled = false;
};

using EvolutionKernelEntry = void (*)(AmrSolver&);

EvolutionKernelEntry find_evolution_kernel_entry(
   EvolutionKernelBundleKey key
) noexcept;

} // namespace pyfluid_amrex
