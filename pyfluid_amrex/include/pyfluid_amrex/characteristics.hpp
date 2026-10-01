#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace pyfluid_amrex {

using namespace amrex::literals;

struct HydroNormalPrimitive {
   amrex::Real rho = 0.0_rt;
   amrex::Real vn = 0.0_rt;
   amrex::Real vt1 = 0.0_rt;
   amrex::Real vt2 = 0.0_rt;
   amrex::Real pressure = 0.0_rt;
};

struct HydroCharacteristics {
   amrex::Real entropy = 0.0_rt;
   amrex::Real shear1 = 0.0_rt;
   amrex::Real shear2 = 0.0_rt;
   amrex::Real sound_minus = 0.0_rt;
   amrex::Real sound_plus = 0.0_rt;
};

struct HydroCharacteristicBasis {
   amrex::Real rho = 0.0_rt;
   amrex::Real inv_rho = 0.0_rt;
   amrex::Real sound_speed = 0.0_rt;
   amrex::Real sound2 = 0.0_rt;
   amrex::Real inv_sound_speed = 0.0_rt;
   amrex::Real inv_sound2 = 0.0_rt;
};

struct MHDReducedPrimitive {
   amrex::Real rho = 0.0_rt;
   amrex::Real vn = 0.0_rt;
   amrex::Real vt1 = 0.0_rt;
   amrex::Real vt2 = 0.0_rt;
   amrex::Real pressure = 0.0_rt;
   amrex::Real bt1 = 0.0_rt;
   amrex::Real bt2 = 0.0_rt;
};

struct MHDCharacteristics {
   amrex::Real fast_minus = 0.0_rt;
   amrex::Real alfven_minus = 0.0_rt;
   amrex::Real slow_minus = 0.0_rt;
   amrex::Real entropy = 0.0_rt;
   amrex::Real slow_plus = 0.0_rt;
   amrex::Real alfven_plus = 0.0_rt;
   amrex::Real fast_plus = 0.0_rt;
};

struct MHDCharacteristicBasis {
   amrex::Real rho = 0.0_rt;
   amrex::Real inv_rho = 0.0_rt;
   amrex::Real sqrt_rho = 0.0_rt;
   amrex::Real inv_sqrt_rho = 0.0_rt;
   amrex::Real sound_speed = 0.0_rt;
   amrex::Real inv_sound_speed = 0.0_rt;
   amrex::Real sound2 = 0.0_rt;
   amrex::Real inv_sound2 = 0.0_rt;
   amrex::Real fast_speed = 0.0_rt;
   amrex::Real slow_speed = 0.0_rt;
   amrex::Real alpha_f = 0.0_rt;
   amrex::Real alpha_s = 0.0_rt;
   amrex::Real beta_t1 = 0.0_rt;
   amrex::Real beta_t2 = 0.0_rt;
   amrex::Real normal_sign = 1.0_rt;
   amrex::Real velocity_denom = 0.0_rt;
   amrex::Real inv_velocity_denom = 0.0_rt;
};

} // namespace pyfluid_amrex
