#include <AMReX.H>
#include <AMReX_Print.H>

#include <iostream>
#include <sstream>
#include <string_view>

#include "pyfluid_amrex/amr_solver.hpp"
#include "pyfluid_amrex/config.hpp"
#include "framework/build_capabilities.hpp"

int main(int argc, char* argv[])
{
   if (argc == 2 && std::string_view(argv[1]) == "--build-capabilities") {
      pyfluid_amrex::write_build_capabilities(std::cout);
      return 0;
   }
   const bool check_build = argc == 3
      && std::string_view(argv[1]) == "--check-build";
   if (argc != 2 && !check_build) {
      std::cerr << "usage: " << argv[0] << " <input.in>\n"
                << "       " << argv[0] << " --build-capabilities\n"
                << "       " << argv[0] << " --check-build <input.in>\n"
                << "solver parameters belong in the input file; command-line overrides are unsupported\n";
      return 2;
   }

   int input_argc = 2;
   char* input_argv[] = {argv[0], argv[check_build ? 2 : 1], nullptr};
   char** input_arguments = input_argv;
   amrex::system::abort_on_unused_inputs = !check_build;
   amrex::Initialize(input_argc, input_arguments);
   {
      if (!check_build && !amrex::system::abort_on_unused_inputs) {
         amrex::Abort("amrex.abort_on_unused_inputs must remain enabled");
      }
      // A build preflight does not construct AmrCore or consume its geometry
      // keys. The complete run retains strict unused-input validation.
      if (check_build) amrex::system::abort_on_unused_inputs = false;
      const pyfluid_amrex::RunConfig cfg = check_build
         ? pyfluid_amrex::read_build_config() : pyfluid_amrex::read_config();
      pyfluid_amrex::validate_build_config(cfg);
      if (check_build) {
         amrex::Print() << "PASS selected executable supports requested build choices\n";
      } else {
         std::ostringstream capabilities;
         pyfluid_amrex::write_build_capabilities(capabilities);
         amrex::Print() << "PyFluid build capabilities " << capabilities.str();
         pyfluid_amrex::run_amr_solver(cfg);
      }
   }
   amrex::Finalize();
   return 0;
}
