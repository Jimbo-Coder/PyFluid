#pragma once

#include <AMReX_Vector.H>

#include "pyfluid_amrex/fields.hpp"

namespace pyfluid_amrex {

enum Z4cComponent : int {
   Z4C_CHI = 0,
   Z4C_GAMMA_TILDE11,
   Z4C_GAMMA_TILDE12,
   Z4C_GAMMA_TILDE13,
   Z4C_GAMMA_TILDE22,
   Z4C_GAMMA_TILDE23,
   Z4C_GAMMA_TILDE33,
   Z4C_K_HAT,
   Z4C_A_TILDE11,
   Z4C_A_TILDE12,
   Z4C_A_TILDE13,
   Z4C_A_TILDE22,
   Z4C_A_TILDE23,
   Z4C_A_TILDE33,
   Z4C_GAMMA_TILDE1,
   Z4C_GAMMA_TILDE2,
   Z4C_GAMMA_TILDE3,
   Z4C_THETA,
   Z4C_LAPSE,
   Z4C_SHIFT1,
   Z4C_SHIFT2,
   Z4C_SHIFT3,
   Z4C_NUM_COMPONENTS,
};

inline constexpr int Z4C_GAMMA_TILDE_COMPONENTS[6] = {
   Z4C_GAMMA_TILDE11,
   Z4C_GAMMA_TILDE12,
   Z4C_GAMMA_TILDE13,
   Z4C_GAMMA_TILDE22,
   Z4C_GAMMA_TILDE23,
   Z4C_GAMMA_TILDE33,
};

inline constexpr int Z4C_A_TILDE_COMPONENTS[6] = {
   Z4C_A_TILDE11,
   Z4C_A_TILDE12,
   Z4C_A_TILDE13,
   Z4C_A_TILDE22,
   Z4C_A_TILDE23,
   Z4C_A_TILDE33,
};

inline constexpr int Z4C_GAMMA_TILDE_VECTOR_COMPONENTS[3] = {
   Z4C_GAMMA_TILDE1,
   Z4C_GAMMA_TILDE2,
   Z4C_GAMMA_TILDE3,
};

inline constexpr int Z4C_SHIFT_COMPONENTS[3] = {
   Z4C_SHIFT1,
   Z4C_SHIFT2,
   Z4C_SHIFT3,
};

GridFieldSetInfo const& z4c_state_grid_fields();
amrex::Vector<std::string> z4c_state_grid_field_names();

} // namespace pyfluid_amrex
