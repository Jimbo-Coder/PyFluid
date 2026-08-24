#include "amr/amr_solver_internal.hpp"

namespace pyfluid_amrex::evolution_kernel_bundles {

void mhd_ct(AmrSolver& solver)
{
   solver.evolve_configured_eos_bundle<NVAR_MHD, true>();
}

} // namespace pyfluid_amrex::evolution_kernel_bundles
