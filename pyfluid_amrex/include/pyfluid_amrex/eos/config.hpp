#pragma once

#include "pyfluid_amrex/eos/types.hpp"

namespace pyfluid_amrex {

struct RunConfig;

EOSParams eos_params(RunConfig const& cfg);

} // namespace pyfluid_amrex
