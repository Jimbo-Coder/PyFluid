#include "pyfluid_amrex/amr_solver_internal.hpp"

namespace pyfluid_amrex::evolution_kernel_bundles {

void hydro_passive(AmrSolver& solver)
{
   solver.evolve_configured_eos_bundle<NVAR_HYDRO + 1, false>();
}

} // namespace pyfluid_amrex::evolution_kernel_bundles
