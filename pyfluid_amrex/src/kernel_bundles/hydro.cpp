#include "amr/amr_solver_internal.hpp"

namespace pyfluid_amrex::evolution_kernel_bundles {

void hydro(AmrSolver& solver)
{
   solver.evolve_configured_eos_bundle<NVAR_HYDRO, false>();
}

} // namespace pyfluid_amrex::evolution_kernel_bundles
