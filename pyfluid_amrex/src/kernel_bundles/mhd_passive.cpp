#include "amr/amr_solver_internal.hpp"

namespace pyfluid_amrex::evolution_kernel_bundles {

void mhd_passive(AmrSolver& solver)
{
   solver.evolve_configured_eos_bundle<NVAR_MHD + 1, false>();
}

} // namespace pyfluid_amrex::evolution_kernel_bundles
