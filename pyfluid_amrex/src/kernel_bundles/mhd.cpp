#include "pyfluid_amrex/amr_solver_internal.hpp"

namespace pyfluid_amrex::evolution_kernel_bundles {

void mhd(AmrSolver& solver)
{
   solver.evolve_configured_eos_bundle<NVAR_MHD, false>();
}

void rmhd(AmrSolver& solver)
{
   solver.evolve_ideal_gas_bundle<NVAR_MHD, false>();
}

} // namespace pyfluid_amrex::evolution_kernel_bundles
