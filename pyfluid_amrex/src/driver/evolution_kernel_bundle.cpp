#include "pyfluid_amrex/evolution_kernel_bundle.hpp"

#include "pyfluid_amrex/build_options.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

namespace evolution_kernel_bundles {

void hydro(AmrSolver& solver);
void hydro_passive(AmrSolver& solver);
void mhd(AmrSolver& solver);
void mhd_ct(AmrSolver& solver);
void mhd_passive(AmrSolver& solver);
void mhd_passive_ct(AmrSolver& solver);

} // namespace evolution_kernel_bundles

namespace {

struct EvolutionKernelBundle {
   EvolutionKernelBundleKey key;
   EvolutionKernelEntry entry = nullptr;
};

constexpr bool same_key(
   EvolutionKernelBundleKey lhs,
   EvolutionKernelBundleKey rhs
) noexcept
{
   return lhs.num_components == rhs.num_components
      && lhs.ct_enabled == rhs.ct_enabled;
}

constexpr EvolutionKernelBundle compiled_evolution_kernel_bundles[] = {
   {{NVAR_HYDRO, false}, evolution_kernel_bundles::hydro},
#if PYFLUID_AMREX_ENABLE_HYDRO
   {{NVAR_HYDRO + 1, false}, evolution_kernel_bundles::hydro_passive},
#endif
#if PYFLUID_AMREX_ENABLE_MHD
   {{NVAR_MHD, false}, evolution_kernel_bundles::mhd},
#if PYFLUID_AMREX_ENABLE_CT
   {{NVAR_MHD, true}, evolution_kernel_bundles::mhd_ct},
#endif
   {{NVAR_MHD + 1, false}, evolution_kernel_bundles::mhd_passive},
#if PYFLUID_AMREX_ENABLE_CT
   {{NVAR_MHD + 1, true}, evolution_kernel_bundles::mhd_passive_ct},
#endif
#endif
};

} // namespace

EvolutionKernelEntry find_evolution_kernel_entry(
   EvolutionKernelBundleKey key
) noexcept
{
   for (auto const& bundle : compiled_evolution_kernel_bundles) {
      if (same_key(bundle.key, key)) {
         return bundle.entry;
      }
   }
   return nullptr;
}

} // namespace pyfluid_amrex
