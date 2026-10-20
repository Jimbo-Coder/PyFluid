#pragma once

#include <iosfwd>

namespace pyfluid_amrex {

struct RunConfig;

RunConfig read_build_config();
void validate_build_config(RunConfig const& cfg);
void write_build_capabilities(std::ostream& output);

} // namespace pyfluid_amrex
