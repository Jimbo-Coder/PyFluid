#include <AMReX.H>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/solver.hpp"

int main(int argc, char* argv[])
{
   amrex::Initialize(argc, argv);
   {
      const pyfluid_amrex::RunConfig cfg = pyfluid_amrex::read_config();
      pyfluid_amrex::run_solver(cfg);
   }
   amrex::Finalize();
   return 0;
}
