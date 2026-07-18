#include "pyfluid_amrex/amr_solver_internal.hpp"

namespace pyfluid_amrex::evolution_kernel_bundles {

void mhd_passive_ct(AmrSolver& solver)
{
   solver.evolve_configured_eos_bundle<NVAR_MHD + 1, true>();
}

} // namespace pyfluid_amrex::evolution_kernel_bundles
