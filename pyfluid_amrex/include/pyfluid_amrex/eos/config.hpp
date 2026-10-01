#pragma once

#include <string>

#include "pyfluid_amrex/eos/types.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct RunConfig;

struct EOSIdentity {
   std::string backend;
   std::string fingerprint;
   std::string source_directory;
   amrex::Real baryon_mass_mev = 0.0_rt;
   int source_points = 0;
   int retained_points = 0;
   bool table_truncated_for_causality = false;
};

EOSParams eos_params(RunConfig const& cfg);
EOSIdentity eos_identity(RunConfig const& cfg);
void prepare_eos(RunConfig& cfg);

} // namespace pyfluid_amrex
