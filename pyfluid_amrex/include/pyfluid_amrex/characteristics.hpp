#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace pyfluid_amrex {

struct HydroNormalPrimitive {
   amrex::Real rho = 0.0;
   amrex::Real vn = 0.0;
   amrex::Real vt1 = 0.0;
   amrex::Real vt2 = 0.0;
   amrex::Real pressure = 0.0;
};

struct HydroCharacteristics {
   amrex::Real entropy = 0.0;
   amrex::Real shear1 = 0.0;
   amrex::Real shear2 = 0.0;
   amrex::Real sound_minus = 0.0;
   amrex::Real sound_plus = 0.0;
};

struct HydroCharacteristicBasis {
   amrex::Real rho = 0.0;
   amrex::Real inv_rho = 0.0;
   amrex::Real sound_speed = 0.0;
   amrex::Real sound2 = 0.0;
   amrex::Real inv_sound_speed = 0.0;
   amrex::Real inv_sound2 = 0.0;
};

struct MHDReducedPrimitive {
   amrex::Real rho = 0.0;
   amrex::Real vn = 0.0;
   amrex::Real vt1 = 0.0;
   amrex::Real vt2 = 0.0;
   amrex::Real pressure = 0.0;
   amrex::Real bt1 = 0.0;
   amrex::Real bt2 = 0.0;
};

struct MHDCharacteristics {
   amrex::Real fast_minus = 0.0;
   amrex::Real alfven_minus = 0.0;
   amrex::Real slow_minus = 0.0;
   amrex::Real entropy = 0.0;
   amrex::Real slow_plus = 0.0;
   amrex::Real alfven_plus = 0.0;
   amrex::Real fast_plus = 0.0;
};

struct MHDCharacteristicBasis {
   amrex::Real rho = 0.0;
   amrex::Real inv_rho = 0.0;
   amrex::Real sqrt_rho = 0.0;
   amrex::Real inv_sqrt_rho = 0.0;
   amrex::Real sound_speed = 0.0;
   amrex::Real inv_sound_speed = 0.0;
   amrex::Real sound2 = 0.0;
   amrex::Real inv_sound2 = 0.0;
   amrex::Real fast_speed = 0.0;
   amrex::Real slow_speed = 0.0;
   amrex::Real alpha_f = 0.0;
   amrex::Real alpha_s = 0.0;
   amrex::Real beta_t1 = 0.0;
   amrex::Real beta_t2 = 0.0;
   amrex::Real normal_sign = 1.0;
   amrex::Real velocity_denom = 0.0;
   amrex::Real inv_velocity_denom = 0.0;
};

} // namespace pyfluid_amrex
