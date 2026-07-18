#pragma once

#include <AMReX_Vector.H>

#include "pyfluid_amrex/fields.hpp"

namespace pyfluid_amrex {

GridFieldSetInfo const& z4c_state_grid_fields();
amrex::Vector<std::string> z4c_state_grid_field_names();

} // namespace pyfluid_amrex
