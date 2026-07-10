#include "pyfluid_amrex/solver.hpp"

#include "pyfluid_amrex/amr_solver.hpp"

namespace pyfluid_amrex {

void run_solver(RunConfig const& cfg)
{
   run_amr_solver(cfg);
}

} // namespace pyfluid_amrex
