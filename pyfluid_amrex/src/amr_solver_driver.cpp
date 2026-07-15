#include "pyfluid_amrex/amr_solver.hpp"

#include "pyfluid_amrex/amr_solver_internal.hpp"

namespace pyfluid_amrex {

void run_amr_solver(RunConfig const& cfg)
{
   AmrSolver solver(cfg);
   solver.evolve();
}

} // namespace pyfluid_amrex
