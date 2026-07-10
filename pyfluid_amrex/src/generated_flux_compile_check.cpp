#include <AMReX.H>

#include "pyfluid_amrex/kernels_generated/scalar_ops.hpp"
#include "pyfluid_amrex/kernels_generated/vector_ops.hpp"
#include "pyfluid_amrex/kernels_generated/coordinate_metrics.hpp"
#include "pyfluid_amrex/kernels_generated/ideal_mhd_flux_cartesian.hpp"
#include "pyfluid_amrex/kernels_generated/patch_maps.hpp"
#include "pyfluid_amrex/kernels_generated/riemann_ops.hpp"
#include "pyfluid_amrex/metric_provider.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace {

pyfluid_amrex::HydroState hydro_flux_reference(
   pyfluid_amrex::HydroState const& state,
   amrex::Real pressure,
   int direction
)
{
   const amrex::Real momentum[3] = {state.mom1, state.mom2, state.mom3};
   const amrex::Real vn = momentum[direction] / state.rho;
   return {
      state.rho * vn,
      state.mom1 * vn + (direction == 0 ? pressure : 0.0),
      state.mom2 * vn + (direction == 1 ? pressure : 0.0),
      state.mom3 * vn + (direction == 2 ? pressure : 0.0),
      (state.energy + pressure) * vn,
   };
}

pyfluid_amrex::MHDState mhd_flux_reference(
   pyfluid_amrex::MHDState const& state,
   amrex::Real pressure,
   int direction
)
{
   const pyfluid_amrex::StateVector3 velocity = {
      state.mom1 / state.rho,
      state.mom2 / state.rho,
      state.mom3 / state.rho,
   };
   const pyfluid_amrex::StateVector3 magnetic = {
      state.b1,
      state.b2,
      state.b3,
   };
   const amrex::Real vn = direction == 0 ? velocity.x : direction == 1 ? velocity.y : velocity.z;
   const amrex::Real bn = direction == 0 ? magnetic.x : direction == 1 ? magnetic.y : magnetic.z;
   const amrex::Real b2 = state.b1 * state.b1 + state.b2 * state.b2 + state.b3 * state.b3;
   const amrex::Real ptot = pressure + 0.5 * b2;
   const amrex::Real vdotb = (state.mom1 * state.b1 + state.mom2 * state.b2 + state.mom3 * state.b3) / state.rho;
   pyfluid_amrex::MHDState flux{
      state.rho * vn,
      state.mom1 * vn - state.b1 * bn,
      state.mom2 * vn - state.b2 * bn,
      state.mom3 * vn - state.b3 * bn,
      (state.energy + ptot) * vn - bn * vdotb,
      state.b1 * vn - bn * velocity.x,
      state.b2 * vn - bn * velocity.y,
      state.b3 * vn - bn * velocity.z,
   };
   if (direction == 0) {
      flux.mom1 += ptot;
      flux.b1 = 0.0;
   } else if (direction == 1) {
      flux.mom2 += ptot;
      flux.b2 = 0.0;
   } else {
      flux.mom3 += ptot;
      flux.b3 = 0.0;
   }
   return flux;
}

bool close_enough(amrex::Real actual, amrex::Real expected)
{
   const amrex::Real scale = std::max<amrex::Real>(1.0, std::abs(expected));
   return std::isfinite(actual) && std::abs(actual - expected) <= 8.0e-15 * scale;
}

bool compare_hydro(pyfluid_amrex::HydroState const& actual, pyfluid_amrex::HydroState const& expected)
{
   return close_enough(actual.rho, expected.rho)
      && close_enough(actual.mom1, expected.mom1)
      && close_enough(actual.mom2, expected.mom2)
      && close_enough(actual.mom3, expected.mom3)
      && close_enough(actual.energy, expected.energy);
}

bool compare_mhd(pyfluid_amrex::MHDState const& actual, pyfluid_amrex::MHDState const& expected)
{
   return close_enough(actual.rho, expected.rho)
      && close_enough(actual.mom1, expected.mom1)
      && close_enough(actual.mom2, expected.mom2)
      && close_enough(actual.mom3, expected.mom3)
      && close_enough(actual.energy, expected.energy)
      && close_enough(actual.b1, expected.b1)
      && close_enough(actual.b2, expected.b2)
      && close_enough(actual.b3, expected.b3);
}

pyfluid_amrex::HydroState hydro_llf_reference(
   pyfluid_amrex::HydroState const& flux_l,
   pyfluid_amrex::HydroState const& flux_r,
   pyfluid_amrex::HydroState const& state_l,
   pyfluid_amrex::HydroState const& state_r,
   amrex::Real wavespeed
)
{
   return {
      0.5 * (flux_l.rho + flux_r.rho) - 0.5 * wavespeed * (state_r.rho - state_l.rho),
      0.5 * (flux_l.mom1 + flux_r.mom1) - 0.5 * wavespeed * (state_r.mom1 - state_l.mom1),
      0.5 * (flux_l.mom2 + flux_r.mom2) - 0.5 * wavespeed * (state_r.mom2 - state_l.mom2),
      0.5 * (flux_l.mom3 + flux_r.mom3) - 0.5 * wavespeed * (state_r.mom3 - state_l.mom3),
      0.5 * (flux_l.energy + flux_r.energy) - 0.5 * wavespeed * (state_r.energy - state_l.energy),
   };
}

pyfluid_amrex::HydroState hydro_hll_reference(
   pyfluid_amrex::HydroState const& flux_l,
   pyfluid_amrex::HydroState const& flux_r,
   pyfluid_amrex::HydroState const& state_l,
   pyfluid_amrex::HydroState const& state_r,
   amrex::Real s_left,
   amrex::Real s_right
)
{
   const amrex::Real inv = 1.0 / (s_right - s_left);
   return {
      (s_right * flux_l.rho - s_left * flux_r.rho + s_left * s_right * (state_r.rho - state_l.rho)) * inv,
      (s_right * flux_l.mom1 - s_left * flux_r.mom1 + s_left * s_right * (state_r.mom1 - state_l.mom1)) * inv,
      (s_right * flux_l.mom2 - s_left * flux_r.mom2 + s_left * s_right * (state_r.mom2 - state_l.mom2)) * inv,
      (s_right * flux_l.mom3 - s_left * flux_r.mom3 + s_left * s_right * (state_r.mom3 - state_l.mom3)) * inv,
      (s_right * flux_l.energy - s_left * flux_r.energy + s_left * s_right * (state_r.energy - state_l.energy)) * inv,
   };
}

pyfluid_amrex::MHDState mhd_llf_reference(
   pyfluid_amrex::MHDState const& flux_l,
   pyfluid_amrex::MHDState const& flux_r,
   pyfluid_amrex::MHDState const& state_l,
   pyfluid_amrex::MHDState const& state_r,
   amrex::Real wavespeed
)
{
   return {
      0.5 * (flux_l.rho + flux_r.rho) - 0.5 * wavespeed * (state_r.rho - state_l.rho),
      0.5 * (flux_l.mom1 + flux_r.mom1) - 0.5 * wavespeed * (state_r.mom1 - state_l.mom1),
      0.5 * (flux_l.mom2 + flux_r.mom2) - 0.5 * wavespeed * (state_r.mom2 - state_l.mom2),
      0.5 * (flux_l.mom3 + flux_r.mom3) - 0.5 * wavespeed * (state_r.mom3 - state_l.mom3),
      0.5 * (flux_l.energy + flux_r.energy) - 0.5 * wavespeed * (state_r.energy - state_l.energy),
      0.5 * (flux_l.b1 + flux_r.b1) - 0.5 * wavespeed * (state_r.b1 - state_l.b1),
      0.5 * (flux_l.b2 + flux_r.b2) - 0.5 * wavespeed * (state_r.b2 - state_l.b2),
      0.5 * (flux_l.b3 + flux_r.b3) - 0.5 * wavespeed * (state_r.b3 - state_l.b3),
   };
}

pyfluid_amrex::MHDState mhd_hll_reference(
   pyfluid_amrex::MHDState const& flux_l,
   pyfluid_amrex::MHDState const& flux_r,
   pyfluid_amrex::MHDState const& state_l,
   pyfluid_amrex::MHDState const& state_r,
   amrex::Real s_left,
   amrex::Real s_right
)
{
   const amrex::Real inv = 1.0 / (s_right - s_left);
   return {
      (s_right * flux_l.rho - s_left * flux_r.rho + s_left * s_right * (state_r.rho - state_l.rho)) * inv,
      (s_right * flux_l.mom1 - s_left * flux_r.mom1 + s_left * s_right * (state_r.mom1 - state_l.mom1)) * inv,
      (s_right * flux_l.mom2 - s_left * flux_r.mom2 + s_left * s_right * (state_r.mom2 - state_l.mom2)) * inv,
      (s_right * flux_l.mom3 - s_left * flux_r.mom3 + s_left * s_right * (state_r.mom3 - state_l.mom3)) * inv,
      (s_right * flux_l.energy - s_left * flux_r.energy + s_left * s_right * (state_r.energy - state_l.energy)) * inv,
      (s_right * flux_l.b1 - s_left * flux_r.b1 + s_left * s_right * (state_r.b1 - state_l.b1)) * inv,
      (s_right * flux_l.b2 - s_left * flux_r.b2 + s_left * s_right * (state_r.b2 - state_l.b2)) * inv,
      (s_right * flux_l.b3 - s_left * flux_r.b3 + s_left * s_right * (state_r.b3 - state_l.b3)) * inv,
   };
}

amrex::Real laplacian_reference(
   int dim,
   amrex::Real center,
   amrex::Real xm,
   amrex::Real xp,
   amrex::Real ym,
   amrex::Real yp,
   amrex::Real zm,
   amrex::Real zp,
   amrex::Real inv_dx2,
   amrex::Real inv_dy2,
   amrex::Real inv_dz2
)
{
   amrex::Real value = (xm - 2.0 * center + xp) * inv_dx2;
   if (dim >= 2) {
      value += (ym - 2.0 * center + yp) * inv_dy2;
   }
   if (dim == 3) {
      value += (zm - 2.0 * center + zp) * inv_dz2;
   }
   return value;
}

amrex::Real relaxed_reference(
   int dim,
   amrex::Real base,
   amrex::Real xm,
   amrex::Real xp,
   amrex::Real ym,
   amrex::Real yp,
   amrex::Real zm,
   amrex::Real zp,
   amrex::Real cx,
   amrex::Real cy,
   amrex::Real cz
)
{
   amrex::Real numerator = base + cx * (xm + xp);
   amrex::Real denominator = 1.0 + 2.0 * cx;
   if (dim >= 2) {
      numerator += cy * (ym + yp);
      denominator += 2.0 * cy;
   }
   if (dim == 3) {
      numerator += cz * (zm + zp);
      denominator += 2.0 * cz;
   }
   return numerator / denominator;
}

amrex::Real residual_reference(
   int dim,
   amrex::Real base,
   amrex::Real center,
   amrex::Real xm,
   amrex::Real xp,
   amrex::Real ym,
   amrex::Real yp,
   amrex::Real zm,
   amrex::Real zp,
   amrex::Real cx,
   amrex::Real cy,
   amrex::Real cz
)
{
   amrex::Real value = center - base + cx * (2.0 * center - xm - xp);
   if (dim >= 2) {
      value += cy * (2.0 * center - ym - yp);
   }
   if (dim == 3) {
      value += cz * (2.0 * center - zm - zp);
   }
   return value;
}

} // namespace

int main()
{
   const std::array<pyfluid_amrex::HydroState, 4> hydro_states{{
      {1.2, 0.3, -0.2, 0.1, 2.5},
      {0.7, -1.4, 0.5, 0.25, 5.0},
      {3.0, 1.1, 2.2, -0.9, 12.0},
      {10.0, -0.02, 0.03, -0.04, 50.0},
   }};
   const std::array<pyfluid_amrex::MHDState, 4> mhd_states{{
      {1.2, 0.3, -0.2, 0.1, 2.5, 0.4, -0.1, 0.2},
      {0.7, -1.4, 0.5, 0.25, 5.0, -0.6, 0.2, 0.7},
      {3.0, 1.1, 2.2, -0.9, 12.0, 0.0, 1.3, -0.8},
      {10.0, -0.02, 0.03, -0.04, 50.0, 2.0, -1.5, 0.1},
   }};
   const std::array<amrex::Real, 4> pressures{{0.7, 2.5, 0.05, 20.0}};
   for (int n = 0; n < 4; ++n) {
      for (int direction = 0; direction < 3; ++direction) {
         pyfluid_amrex::HydroState hydro_flux{};
         pyfluid_amrex::MHDState mhd_flux{};
         if (direction == 0) {
            hydro_flux = pyfluid_amrex::generated::hydro_flux_orthonormal<0>(hydro_states[n], pressures[n]);
            mhd_flux = pyfluid_amrex::generated::mhd_flux_orthonormal<0>(mhd_states[n], pressures[n]);
         } else if (direction == 1) {
            hydro_flux = pyfluid_amrex::generated::hydro_flux_orthonormal<1>(hydro_states[n], pressures[n]);
            mhd_flux = pyfluid_amrex::generated::mhd_flux_orthonormal<1>(mhd_states[n], pressures[n]);
         } else {
            hydro_flux = pyfluid_amrex::generated::hydro_flux_orthonormal<2>(hydro_states[n], pressures[n]);
            mhd_flux = pyfluid_amrex::generated::mhd_flux_orthonormal<2>(mhd_states[n], pressures[n]);
         }
         if (!compare_hydro(hydro_flux, hydro_flux_reference(hydro_states[n], pressures[n], direction))) {
            std::cerr << "generated hydro flux mismatch state=" << n << " direction=" << direction << "\n";
            return 1;
         }
         if (!compare_mhd(mhd_flux, mhd_flux_reference(mhd_states[n], pressures[n], direction))) {
            std::cerr << "generated mhd flux mismatch state=" << n << " direction=" << direction << "\n";
            return 1;
         }
      }
   }
   {
      const pyfluid_amrex::HydroState state_l{1.1, 0.2, -0.3, 0.4, 2.7};
      const pyfluid_amrex::HydroState state_r{0.8, -0.5, 0.6, -0.1, 3.2};
      const pyfluid_amrex::HydroState flux_l{0.4, 1.2, -0.8, 0.25, 2.1};
      const pyfluid_amrex::HydroState flux_r{-0.2, 0.7, 0.9, -1.1, 1.6};
      constexpr amrex::Real wavespeed = 2.3;
      constexpr amrex::Real s_left = -1.7;
      constexpr amrex::Real s_right = 2.9;
      if (!compare_hydro(
             pyfluid_amrex::generated::hydro_llf_flux(flux_l, flux_r, state_l, state_r, wavespeed),
             hydro_llf_reference(flux_l, flux_r, state_l, state_r, wavespeed)
          )) {
         std::cerr << "generated hydro LLF flux mismatch\n";
         return 1;
      }
      if (!compare_hydro(
             pyfluid_amrex::generated::hydro_hll_flux(flux_l, flux_r, state_l, state_r, s_left, s_right),
             hydro_hll_reference(flux_l, flux_r, state_l, state_r, s_left, s_right)
          )) {
         std::cerr << "generated hydro HLL flux mismatch\n";
         return 1;
      }
   }
   {
      const pyfluid_amrex::MHDState state_l{1.1, 0.2, -0.3, 0.4, 2.7, 0.05, -0.4, 0.7};
      const pyfluid_amrex::MHDState state_r{0.8, -0.5, 0.6, -0.1, 3.2, -0.2, 0.3, -0.6};
      const pyfluid_amrex::MHDState flux_l{0.4, 1.2, -0.8, 0.25, 2.1, -0.9, 0.1, 0.8};
      const pyfluid_amrex::MHDState flux_r{-0.2, 0.7, 0.9, -1.1, 1.6, 0.6, -0.5, 0.2};
      constexpr amrex::Real wavespeed = 2.3;
      constexpr amrex::Real s_left = -1.7;
      constexpr amrex::Real s_right = 2.9;
      if (!compare_mhd(
             pyfluid_amrex::generated::mhd_llf_flux(flux_l, flux_r, state_l, state_r, wavespeed),
             mhd_llf_reference(flux_l, flux_r, state_l, state_r, wavespeed)
          )) {
         std::cerr << "generated MHD LLF flux mismatch\n";
         return 1;
      }
      if (!compare_mhd(
             pyfluid_amrex::generated::mhd_hll_flux(flux_l, flux_r, state_l, state_r, s_left, s_right),
             mhd_hll_reference(flux_l, flux_r, state_l, state_r, s_left, s_right)
          )) {
         std::cerr << "generated MHD HLL flux mismatch\n";
         return 1;
      }
   }
   constexpr amrex::Real center = 1.4;
   constexpr amrex::Real base = -0.2;
   constexpr amrex::Real xm = 0.5;
   constexpr amrex::Real xp = 2.1;
   constexpr amrex::Real ym = -1.2;
   constexpr amrex::Real yp = 0.9;
   constexpr amrex::Real zm = 0.3;
   constexpr amrex::Real zp = -0.7;
   constexpr amrex::Real inv_dx2 = 2.0;
   constexpr amrex::Real inv_dy2 = 3.0;
   constexpr amrex::Real inv_dz2 = 5.0;
   constexpr amrex::Real cx = 0.11;
   constexpr amrex::Real cy = 0.07;
   constexpr amrex::Real cz = 0.03;
   for (int dim = 1; dim <= 3; ++dim) {
      amrex::Real laplacian{};
      amrex::Real relaxed{};
      amrex::Real residual{};
      if (dim == 1) {
         laplacian = pyfluid_amrex::generated::laplacian_cartesian<1>(
            center, xm, xp, ym, yp, zm, zp, inv_dx2, inv_dy2, inv_dz2
         );
         relaxed = pyfluid_amrex::generated::implicit_relaxed_cartesian<1>(
            base, xm, xp, ym, yp, zm, zp, cx, cy, cz
         );
         residual = pyfluid_amrex::generated::implicit_residual_cartesian<1>(
            base, center, xm, xp, ym, yp, zm, zp, cx, cy, cz
         );
      } else if (dim == 2) {
         laplacian = pyfluid_amrex::generated::laplacian_cartesian<2>(
            center, xm, xp, ym, yp, zm, zp, inv_dx2, inv_dy2, inv_dz2
         );
         relaxed = pyfluid_amrex::generated::implicit_relaxed_cartesian<2>(
            base, xm, xp, ym, yp, zm, zp, cx, cy, cz
         );
         residual = pyfluid_amrex::generated::implicit_residual_cartesian<2>(
            base, center, xm, xp, ym, yp, zm, zp, cx, cy, cz
         );
      } else {
         laplacian = pyfluid_amrex::generated::laplacian_cartesian<3>(
            center, xm, xp, ym, yp, zm, zp, inv_dx2, inv_dy2, inv_dz2
         );
         relaxed = pyfluid_amrex::generated::implicit_relaxed_cartesian<3>(
            base, xm, xp, ym, yp, zm, zp, cx, cy, cz
         );
         residual = pyfluid_amrex::generated::implicit_residual_cartesian<3>(
            base, center, xm, xp, ym, yp, zm, zp, cx, cy, cz
         );
      }
      if (!close_enough(laplacian, laplacian_reference(dim, center, xm, xp, ym, yp, zm, zp, inv_dx2, inv_dy2, inv_dz2))) {
         std::cerr << "generated laplacian mismatch dim=" << dim << "\n";
         return 1;
      }
      if (!close_enough(relaxed, relaxed_reference(dim, base, xm, xp, ym, yp, zm, zp, cx, cy, cz))) {
         std::cerr << "generated implicit relaxed mismatch dim=" << dim << "\n";
         return 1;
      }
      if (!close_enough(residual, residual_reference(dim, base, center, xm, xp, ym, yp, zm, zp, cx, cy, cz))) {
         std::cerr << "generated implicit residual mismatch dim=" << dim << "\n";
         return 1;
      }
   }
   const amrex::Real update = pyfluid_amrex::generated::explicit_diffusion_update(1.25, 0.125, -3.0);
   if (!close_enough(update, 1.25 + 0.125 * -3.0)) {
      std::cerr << "generated explicit diffusion update mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::curl_x_cartesian(3.0, -2.0), 5.0)) {
      std::cerr << "generated cartesian curl x mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::curl_y_cartesian(0.5, 2.5), -2.0)) {
      std::cerr << "generated cartesian curl y mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::curl_z_cartesian(-1.5, 4.0), -5.5)) {
      std::cerr << "generated cartesian curl z mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::divergence_cartesian(1.0, -2.0, 0.25), -0.75)) {
      std::cerr << "generated cartesian divergence mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::curl_orthogonal_1(0.25, 7.0, -1.0), 2.0)) {
      std::cerr << "generated orthogonal curl mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::divergence_orthogonal(0.5, 1.0, 2.0, -0.5), 1.25)) {
      std::cerr << "generated orthogonal divergence mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::ideal_emf_x(2.0, -3.0, 0.25, 4.0), -8.75)) {
      std::cerr << "generated ideal emf x mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::ideal_emf_y(-3.0, 2.0, 4.0, 0.25), 8.75)) {
      std::cerr << "generated ideal emf y mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::ideal_emf_z(2.0, -3.0, 0.25, 4.0), -8.75)) {
      std::cerr << "generated ideal emf z mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::total_emf(-2.0, 0.5, 3.0), -0.5)) {
      std::cerr << "generated total emf mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::resistive_emf_scaled_gradient(0.5, 2.0, -3.0, 4.0), -0.75)) {
      std::cerr << "generated scaled-gradient resistive emf mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::magnetic_energy_density_from_components(1.0, 2.0, 3.0), 7.0)) {
      std::cerr << "generated magnetic energy mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::plasma_beta_from_pressure(4.0, 1.0e-300, 1.0, 2.0, 1.0), 4.0 / 3.0)) {
      std::cerr << "generated plasma beta mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cartesian_cell_volume(2.0, 3.0, 4.0), 24.0)) {
      std::cerr << "generated cartesian cell volume mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cartesian_face_area_x(3.0, 4.0), 12.0)) {
      std::cerr << "generated cartesian face area mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_volume_scale(2.5, 0.7, 0.0), 2.5)) {
      std::cerr << "generated cylindrical volume scale mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_radial_shell_volume_scale(2.0, 3.0), 2.5)) {
      std::cerr << "generated cylindrical radial shell volume mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_radial_pressure_source_factor(2.0, 3.0), 0.4)) {
      std::cerr << "generated cylindrical radial pressure source mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_phiz_cell_volume_scale(2.0, 3.0, 0.5), 3.0)) {
      std::cerr << "generated cylindrical phi-z volume mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_phiz_axial_face_area_scale(2.0, 0.5), 1.0)) {
      std::cerr << "generated cylindrical phi-z axial face area mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_phiz_azimuthal_face_area_scale(3.0), 3.0)) {
      std::cerr << "generated cylindrical phi-z azimuthal face area mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::spherical_scale_phi(2.0, 0.5, 0.0), 2.0 * std::sin(0.5))) {
      std::cerr << "generated spherical scale phi mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::spherical_radial_shell_volume_scale(2.0, 3.0), 19.0 / 3.0)) {
      std::cerr << "generated spherical radial shell volume mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::spherical_radial_pressure_source_factor(2.0, 3.0), 15.0 / 19.0)) {
      std::cerr << "generated spherical radial pressure source mismatch\n";
      return 1;
   }
   if (!close_enough(
          pyfluid_amrex::generated::spherical_thetaphi_cell_volume_scale(2.0, 0.25, 0.75, 0.5),
          2.0 * (std::cos(0.25) - std::cos(0.75))
       )) {
      std::cerr << "generated spherical theta-phi volume mismatch\n";
      return 1;
   }
   if (!close_enough(
          pyfluid_amrex::generated::spherical_thetaphi_theta_face_area_scale(2.0, 0.25, 0.5),
          std::sin(0.25)
       )) {
      std::cerr << "generated spherical theta-phi theta face area mismatch\n";
      return 1;
   }
   if (!close_enough(
          pyfluid_amrex::generated::spherical_thetaphi_azimuthal_face_area_scale(2.0, 0.25, 0.75),
          1.0
       )) {
      std::cerr << "generated spherical theta-phi azimuthal face area mismatch\n";
      return 1;
   }
   {
      constexpr amrex::Real radial_inner = 1.0;
      constexpr amrex::Real radial_outer = 4.0;
      const auto point = pyfluid_amrex::generated::thornburg_shell_local_to_global_axis0(
         0.2,
         -0.1,
         2.0,
         0.3,
         -0.2,
         0.4,
         -1.0,
         radial_inner,
         radial_outer
      );
      const amrex::Real tan_q1 = std::tan(0.2);
      const amrex::Real tan_q2 = std::tan(-0.1);
      const amrex::Real denom2 = 1.0 + tan_q1 * tan_q1 + tan_q2 * tan_q2;
      const amrex::Real primary = -2.0 / std::sqrt(denom2);
      if (
         !close_enough(point.q1, 0.3 + primary)
         || !close_enough(point.q2, -0.2 + primary * tan_q2)
         || !close_enough(point.q3, 0.4 + primary * tan_q1)
      ) {
         std::cerr << "generated Thornburg patch-map transform mismatch\n";
         return 1;
      }
      const auto jacobian = pyfluid_amrex::generated::thornburg_shell_jacobian_axis0(
         0.2,
         -0.1,
         2.0,
         -1.0,
         radial_inner,
         radial_outer
      );
      constexpr amrex::Real eps = 1.0e-6;
      const auto point_plus = pyfluid_amrex::generated::thornburg_shell_local_to_global_axis0(
         0.2,
         -0.1 + eps,
         2.0,
         0.3,
         -0.2,
         0.4,
         -1.0,
         radial_inner,
         radial_outer
      );
      const auto point_minus = pyfluid_amrex::generated::thornburg_shell_local_to_global_axis0(
         0.2,
         -0.1 - eps,
         2.0,
         0.3,
         -0.2,
         0.4,
         -1.0,
         radial_inner,
         radial_outer
      );
      const amrex::Real expected_j22 = (point_plus.q2 - point_minus.q2) / (2.0 * eps);
      if (std::abs(jacobian.j22 - expected_j22) > 1.0e-9) {
         std::cerr << "generated Thornburg patch-map Jacobian mismatch\n";
         return 1;
      }
   }
   {
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{};
      dx[0] = 0.5;
#if (AMREX_SPACEDIM >= 2)
      dx[1] = 0.25;
#endif
#if (AMREX_SPACEDIM == 3)
      dx[2] = 0.125;
#endif
      const pyfluid_amrex::CartesianUniformMetric metric = pyfluid_amrex::make_cartesian_uniform_metric(dx);
#if (AMREX_SPACEDIM == 1)
      const amrex::Real expected_volume = 0.5;
      const amrex::Real expected_area_x = 1.0;
#elif (AMREX_SPACEDIM == 2)
      const amrex::Real expected_volume = 0.5 * 0.25;
      const amrex::Real expected_area_x = 0.25;
#else
      const amrex::Real expected_volume = 0.5 * 0.25 * 0.125;
      const amrex::Real expected_area_x = 0.25 * 0.125;
#endif
      if (!close_enough(metric.cell_volume, expected_volume) || !close_enough(metric.face_area[0], expected_area_x)) {
         std::cerr << "cartesian metric provider mismatch\n";
         return 1;
      }
      const auto cart_scale = pyfluid_amrex::coordinate_scale_factors<pyfluid_amrex::MeshCartesian>(2.0, 0.5, 0.0);
      if (!close_enough(cart_scale[0], 1.0) || !close_enough(cart_scale[1], 1.0) || !close_enough(cart_scale[2], 1.0)) {
         std::cerr << "cartesian coordinate scale provider mismatch\n";
         return 1;
      }
      const auto sph_scale = pyfluid_amrex::coordinate_scale_factors<pyfluid_amrex::MeshSpherical>(2.0, 0.5, 0.0);
      if (!close_enough(sph_scale[2], 2.0 * std::sin(0.5))) {
         std::cerr << "spherical coordinate scale provider mismatch\n";
         return 1;
      }
   }
   return 0;
}
