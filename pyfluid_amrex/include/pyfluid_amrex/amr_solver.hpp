#pragma once

namespace pyfluid_amrex {

struct RunConfig;

void run_amr_solver(RunConfig const& cfg);

} // namespace pyfluid_amrex
