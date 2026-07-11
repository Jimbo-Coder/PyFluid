#pragma once

#include <AMReX_ParmParse.H>

#include <string>

namespace pyfluid_amrex {

struct RunConfig;

void initialize_registered_parameters(RunConfig& cfg);
void query_registered_method_parameters(amrex::ParmParse& pp, RunConfig& cfg);
void query_registered_amr_parameters(amrex::ParmParse& pp, RunConfig& cfg);
void sync_registered_amr_parameters(amrex::ParmParse& pp, RunConfig const& cfg);
void validate_registered_parameters(RunConfig const& cfg);
void write_run_parameter_file(std::string const& path);

} // namespace pyfluid_amrex
