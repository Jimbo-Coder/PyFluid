#include <AMReX.H>

#include <iostream>

#include "pyfluid_amrex/amr_solver.hpp"
#include "pyfluid_amrex/config.hpp"

int main(int argc, char* argv[])
{
   if (argc != 2) {
      std::cerr << "usage: " << argv[0] << " <input.in>\n"
                << "solver parameters belong in the input file; command-line overrides are unsupported\n";
      return 2;
   }

   amrex::system::abort_on_unused_inputs = true;
   amrex::Initialize(argc, argv);
   {
      if (!amrex::system::abort_on_unused_inputs) {
         amrex::Abort("amrex.abort_on_unused_inputs must remain enabled");
      }
      const pyfluid_amrex::RunConfig cfg = pyfluid_amrex::read_config();
      pyfluid_amrex::run_amr_solver(cfg);
   }
   amrex::Finalize();
   return 0;
}
