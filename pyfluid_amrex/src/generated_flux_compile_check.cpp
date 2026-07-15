#include <AMReX.H>

#include "pyfluid_amrex/kernels_generated/eos.hpp"
#include "pyfluid_amrex/kernels_generated/characteristic_transforms.hpp"
#include "pyfluid_amrex/kernels_generated/scalar_ops.hpp"
#include "pyfluid_amrex/kernels_generated/vector_ops.hpp"
#include "pyfluid_amrex/kernels_generated/coordinate_metrics.hpp"
#include "pyfluid_amrex/kernels_generated/ideal_mhd_flux_cartesian.hpp"
#include "pyfluid_amrex/kernels_generated/patch_maps.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic_fluid.hpp"
#include "pyfluid_amrex/kernels_generated/riemann_ops.hpp"
#include "pyfluid_amrex/c2p_policy.hpp"
#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rhd_system.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

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

template <std::size_t N>
amrex::Real scaled_max_error(
   std::array<amrex::Real, N> const& actual,
   std::array<amrex::Real, N> const& expected
)
{
   amrex::Real error = 0.0;
   for (std::size_t component = 0; component < N; ++component) {
      const amrex::Real scale = std::max({
         amrex::Real{1.0},
         std::abs(actual[component]),
         std::abs(expected[component]),
      });
      error = std::max(error, std::abs(actual[component] - expected[component]) / scale);
   }
   return error;
}

std::array<amrex::Real, 5> components(pyfluid_amrex::HydroPrimitive const& state)
{
   return {state.rho, state.v1, state.v2, state.v3, state.pressure};
}

std::array<amrex::Real, 5> components(pyfluid_amrex::HydroState const& state)
{
   return {state.rho, state.mom1, state.mom2, state.mom3, state.energy};
}

std::array<amrex::Real, 8> components(pyfluid_amrex::MHDPrimitive const& state)
{
   return {state.rho, state.v1, state.v2, state.v3, state.pressure, state.b1, state.b2, state.b3};
}

std::array<amrex::Real, 8> components(pyfluid_amrex::MHDState const& state)
{
   return {state.rho, state.mom1, state.mom2, state.mom3, state.energy, state.b1, state.b2, state.b3};
}

std::array<amrex::Real, 6> components(pyfluid_amrex::RHDPrimitive const& state)
{
   return {state.rho, state.eps, state.pressure, state.v1, state.v2, state.v3};
}

std::array<amrex::Real, 5> components(pyfluid_amrex::RHDConserved const& state)
{
   return {state.dens, state.mom1, state.mom2, state.mom3, state.tau};
}

std::array<amrex::Real, 9> components(pyfluid_amrex::RMHDPrimitive const& state)
{
   return {
      state.rho,
      state.eps,
      state.pressure,
      state.v1,
      state.v2,
      state.v3,
      state.b1,
      state.b2,
      state.b3,
   };
}

std::array<amrex::Real, 8> components(pyfluid_amrex::RMHDConserved const& state)
{
   return {
      state.dens,
      state.mom1,
      state.mom2,
      state.mom3,
      state.tau,
      state.b1,
      state.b2,
      state.b3,
   };
}

template <int EOSMethod>
bool check_newtonian_round_trip(
   pyfluid_amrex::HydroPrimitive primitive,
   pyfluid_amrex::MHDPrimitive mhd_primitive,
   pyfluid_amrex::EOSParams const& eos
)
{
   if constexpr (EOSMethod == pyfluid_amrex::EOSPolytrope) {
      primitive.pressure = pyfluid_amrex::eos_cold_pressure<EOSMethod>(primitive.rho, eos);
      mhd_primitive.pressure = pyfluid_amrex::eos_cold_pressure<EOSMethod>(mhd_primitive.rho, eos);
   }

   const auto conserved = pyfluid_amrex::conserved_from_primitive<EOSMethod>(primitive, eos);
   const auto recovered_primitive = pyfluid_amrex::primitive_from_conserved<EOSMethod>(conserved, eos);
   const auto conserved_round_trip =
      pyfluid_amrex::conserved_from_primitive<EOSMethod>(recovered_primitive, eos);

   const auto mhd_conserved = pyfluid_amrex::conserved_from_primitive<EOSMethod>(mhd_primitive, eos);
   const auto recovered_mhd_primitive =
      pyfluid_amrex::primitive_from_conserved<EOSMethod>(mhd_conserved, eos);
   const auto mhd_conserved_round_trip =
      pyfluid_amrex::conserved_from_primitive<EOSMethod>(recovered_mhd_primitive, eos);

   auto independent_conserved = conserved;
   auto independent_mhd_conserved = mhd_conserved;
   if constexpr (EOSMethod != pyfluid_amrex::EOSPolytrope) {
      independent_conserved.mom1 *= -0.81;
      independent_conserved.mom2 += 0.23 * independent_conserved.rho;
      independent_conserved.energy += 0.37 * std::max(amrex::Real{1.0}, independent_conserved.rho);
      independent_mhd_conserved.mom1 *= -0.81;
      independent_mhd_conserved.mom2 += 0.23 * independent_mhd_conserved.rho;
      independent_mhd_conserved.energy += 0.37 * std::max(amrex::Real{1.0}, independent_mhd_conserved.rho);
   }
   const auto primitive_from_independent =
      pyfluid_amrex::primitive_from_conserved<EOSMethod>(independent_conserved, eos);
   const auto independent_conserved_round_trip =
      pyfluid_amrex::conserved_from_primitive<EOSMethod>(primitive_from_independent, eos);
   const auto mhd_primitive_from_independent =
      pyfluid_amrex::primitive_from_conserved<EOSMethod>(independent_mhd_conserved, eos);
   const auto independent_mhd_conserved_round_trip =
      pyfluid_amrex::conserved_from_primitive<EOSMethod>(mhd_primitive_from_independent, eos);

   constexpr amrex::Real tolerance = 256.0 * std::numeric_limits<amrex::Real>::epsilon();
   const std::array<amrex::Real, 6> errors{
      scaled_max_error(components(recovered_primitive), components(primitive)),
      scaled_max_error(components(conserved_round_trip), components(conserved)),
      scaled_max_error(components(recovered_mhd_primitive), components(mhd_primitive)),
      scaled_max_error(components(mhd_conserved_round_trip), components(mhd_conserved)),
      scaled_max_error(components(independent_conserved_round_trip), components(independent_conserved)),
      scaled_max_error(components(independent_mhd_conserved_round_trip), components(independent_mhd_conserved)),
   };
   if (*std::max_element(errors.begin(), errors.end()) > tolerance) {
      std::cerr << "Newtonian round-trip residuals EOS=" << EOSMethod;
      for (amrex::Real const error : errors) {
         std::cerr << ' ' << error;
      }
      std::cerr << '\n';
      return false;
   }
   return true;
}

bool validate_newtonian_round_trips()
{
   pyfluid_amrex::EOSParams eos{};
   eos.gamma = 1.4;
   eos.polytropic_k = 0.73;
   eos.polytropic_gamma = 1.65;
   eos.thermal_gamma = 1.37;
   const std::array<pyfluid_amrex::HydroPrimitive, 5> hydro_corpus{{
      {0.071, 2.3, -0.47, 0.19, 0.013},
      {0.82, -1.7, 0.93, -0.31, 4.6},
      {1.0, 0.0, 0.0, 0.0, 0.41},
      {7.4, 0.017, -3.2, 1.1, 13.0},
      {19.0, -0.004, 0.007, -0.009, 0.26},
   }};
   const std::array<pyfluid_amrex::MHDPrimitive, 5> mhd_corpus{{
      {0.071, 2.3, -0.47, 0.19, 0.013, 4.0, -0.07, 0.8},
      {0.82, -1.7, 0.93, -0.31, 4.6, -2.1, 3.7, -0.4},
      {1.0, 0.0, 0.0, 0.0, 0.41, 0.0, 0.0, 0.0},
      {7.4, 0.017, -3.2, 1.1, 13.0, 0.003, -8.0, 2.5},
      {19.0, -0.004, 0.007, -0.009, 0.26, 15.0, 0.2, -0.5},
   }};
   for (std::size_t sample = 0; sample < hydro_corpus.size(); ++sample) {
      if (!check_newtonian_round_trip<pyfluid_amrex::EOSIdealGas>(
             hydro_corpus[sample], mhd_corpus[sample], eos
          )
          || !check_newtonian_round_trip<pyfluid_amrex::EOSPolytrope>(
             hydro_corpus[sample], mhd_corpus[sample], eos
          )
          || !check_newtonian_round_trip<pyfluid_amrex::EOSHybridPolytrope>(
             hydro_corpus[sample], mhd_corpus[sample], eos
          )) {
         std::cerr << "Newtonian primitive/conserved round trip failed for sample=" << sample << "\n";
         return false;
      }
   }
   return true;
}

bool validate_srhd_round_trips()
{
   constexpr amrex::Real gamma = 1.4;
   constexpr amrex::Real tolerance = 2.0e-11;
   const pyfluid_amrex::SpacetimeMetric minkowski{};
   std::array<pyfluid_amrex::RHDPrimitive, 7> primitive_corpus{{
      {1.0e-3, 1.0e-4, 0.0, 0.0, 0.0, 0.0},
      {0.071, 0.23, 0.0, 0.31, -0.17, 0.09},
      {0.82, 4.7, 0.0, -0.71, 0.23, -0.11},
      {1.0, 0.7, 0.0, 0.0, 0.0, 0.0},
      {7.4, 0.017, 0.0, 0.82, -0.31, 0.18},
      {19.0, 13.0, 0.0, -0.91, 0.21, -0.07},
      {0.3, 2.1, 0.0, 0.965, 0.08, -0.03},
   }};

   amrex::Real worst_primitive_error = 0.0;
   amrex::Real worst_conserved_error = 0.0;
   for (std::size_t sample = 0; sample < primitive_corpus.size(); ++sample) {
      auto primitive = primitive_corpus[sample];
      primitive.pressure = pyfluid_amrex::generated::relativistic_ideal_gas_pressure(
         primitive.rho,
         primitive.eps,
         gamma
      );
      const auto conserved =
         pyfluid_amrex::generated::rhd_primitive_to_conserved(primitive, minkowski);
      const auto recovered = pyfluid_amrex::recover_srhd_palenzuela_1d(
         conserved,
         gamma
      );
      if (!recovered.attempt.succeeded()) {
         std::cerr << "SRHD primitive round trip failed sample=" << sample
                   << " status=" << static_cast<int>(recovered.attempt.status)
                   << " residual=" << recovered.attempt.residual << '\n';
         return false;
      }
      const auto conserved_round_trip =
         pyfluid_amrex::generated::rhd_primitive_to_conserved(
            recovered.primitive,
            minkowski
         );
      worst_primitive_error = std::max(
         worst_primitive_error,
         scaled_max_error(components(recovered.primitive), components(primitive))
      );
      worst_conserved_error = std::max(
         worst_conserved_error,
         scaled_max_error(components(conserved_round_trip), components(conserved))
      );

      auto independent_conserved = conserved;
      independent_conserved.mom1 *= -0.83;
      independent_conserved.mom2 += 0.07 * independent_conserved.dens;
      independent_conserved.mom3 *= 0.91;
      independent_conserved.tau += 0.19 * (
         independent_conserved.dens + independent_conserved.tau
      );
      const auto independent_recovered =
         pyfluid_amrex::recover_srhd_palenzuela_1d(independent_conserved, gamma);
      if (!independent_recovered.attempt.succeeded()) {
         std::cerr << "SRHD conserved round trip failed sample=" << sample
                   << " status=" << static_cast<int>(independent_recovered.attempt.status)
                   << " residual=" << independent_recovered.attempt.residual << '\n';
         return false;
      }
      const auto independent_round_trip =
         pyfluid_amrex::generated::rhd_primitive_to_conserved(
            independent_recovered.primitive,
            minkowski
         );
      worst_conserved_error = std::max(
         worst_conserved_error,
         scaled_max_error(
            components(independent_round_trip),
            components(independent_conserved)
         )
      );

      const amrex::Real velocity_squared = primitive.v1 * primitive.v1
         + primitive.v2 * primitive.v2 + primitive.v3 * primitive.v3;
      const amrex::Real lorentz_factor = 1.0 / std::sqrt(1.0 - velocity_squared);
      const amrex::Real x = (1.0 + gamma * primitive.eps) * lorentz_factor;
      const amrex::Real step = 1.0e-6 * std::max(amrex::Real{1.0}, x);
      const amrex::Real finite_difference = (
         pyfluid_amrex::srhd_palenzuela_residual(conserved, x + step, gamma)
         - pyfluid_amrex::srhd_palenzuela_residual(conserved, x - step, gamma)
      ) / (2.0 * step);
      const amrex::Real analytic =
         pyfluid_amrex::srhd_palenzuela_residual_derivative(conserved, x, gamma);
      if (!std::isfinite(analytic)
          || std::abs(analytic - finite_difference)
             > 2.0e-6 * std::max(amrex::Real{1.0}, std::abs(finite_difference))) {
         std::cerr << "SRHD generated residual derivative mismatch sample=" << sample
                   << " analytic=" << analytic
                   << " finite_difference=" << finite_difference << '\n';
         return false;
      }
   }

   pyfluid_amrex::SpacetimeMetric curved_metric{};
   curved_metric.lapse = 0.83;
   curved_metric.shift1 = 0.07;
   curved_metric.shift2 = -0.03;
   curved_metric.shift3 = 0.02;
   curved_metric.gamma12 = 0.1;
   curved_metric.inv_gamma11 = 1.0 / 0.99;
   curved_metric.inv_gamma12 = -0.1 / 0.99;
   curved_metric.inv_gamma22 = 1.0 / 0.99;
   curved_metric.sqrt_det_gamma = std::sqrt(0.99);
   for (std::size_t sample = 0; sample < primitive_corpus.size(); ++sample) {
      auto primitive = primitive_corpus[sample];
      primitive.pressure = pyfluid_amrex::generated::relativistic_ideal_gas_pressure(
         primitive.rho, primitive.eps, gamma
      );
      const auto conserved = pyfluid_amrex::generated::rhd_primitive_to_conserved(
         primitive, curved_metric
      );
      const auto recovered = pyfluid_amrex::recover_srhd_palenzuela_1d(
         conserved, curved_metric, gamma
      );
      if (!recovered.attempt.succeeded()) {
         std::cerr << "Curved RHD recovery failed sample=" << sample
                   << " status=" << static_cast<int>(recovered.attempt.status)
                   << " residual=" << recovered.attempt.residual << '\n';
         return false;
      }
      const auto conserved_round_trip =
         pyfluid_amrex::generated::rhd_primitive_to_conserved(
            recovered.primitive, curved_metric
         );
      worst_primitive_error = std::max(
         worst_primitive_error,
         scaled_max_error(components(recovered.primitive), components(primitive))
      );
      worst_conserved_error = std::max(
         worst_conserved_error,
         scaled_max_error(components(conserved_round_trip), components(conserved))
      );
   }

   if (worst_primitive_error > tolerance || worst_conserved_error > tolerance) {
      std::cerr << "SRHD round-trip residuals primitive=" << worst_primitive_error
                << " conserved=" << worst_conserved_error << '\n';
      return false;
   }
   return true;
}

bool validate_fixed_spacetime_provider()
{
   using pyfluid_amrex::FixedSpacetimeMetric;
   using pyfluid_amrex::FixedSpacetimeParameters;
   using pyfluid_amrex::fixed_spacetime_geometry;

   const FixedSpacetimeParameters rindler{
      FixedSpacetimeMetric::Rindler,
      0.11,
      -0.07,
      0.03,
      0.2,
      -0.1,
      0.4,
      1.0,
   };
   const auto rindler_geometry = fixed_spacetime_geometry(
      rindler, 0.8, -0.4, 0.6, 0.0
   );
   const amrex::Real expected_lapse = 1.0
      + rindler.acceleration1 * (0.8 - rindler.origin1)
      + rindler.acceleration2 * (-0.4 - rindler.origin2)
      + rindler.acceleration3 * (0.6 - rindler.origin3);
   if (!close_enough(rindler_geometry.metric.lapse, expected_lapse)
       || !close_enough(
          rindler_geometry.derivatives.d1_lapse, rindler.acceleration1
       )
       || !close_enough(
          rindler_geometry.derivatives.d2_lapse, rindler.acceleration2
       )
       || !close_enough(
          rindler_geometry.derivatives.d3_lapse, rindler.acceleration3
       )) {
      std::cerr << "Rindler fixed-spacetime provider mismatch\n";
      return false;
   }

   pyfluid_amrex::RHDPrimitive rest{1.0, 0.5, 0.2, 0.0, 0.0, 0.0};
   pyfluid_amrex::SpacetimeMetricDerivatives acceleration{};
   acceleration.d1_lapse = 0.1;
   const auto source = pyfluid_amrex::generated::rhd_geometric_source(
      rest, pyfluid_amrex::minkowski_spacetime_metric(), acceleration
   );
   const amrex::Real eulerian_energy = rest.rho * (1.0 + rest.eps);
   if (!close_enough(source.dens, 0.0)
       || !close_enough(source.mom1, -0.1 * eulerian_energy)
       || !close_enough(source.mom2, 0.0)
       || !close_enough(source.mom3, 0.0)
       || !close_enough(source.tau, 0.0)) {
      std::cerr << "Rindler geometric source mismatch\n";
      return false;
   }

   const FixedSpacetimeParameters schwarzschild{
      FixedSpacetimeMetric::SchwarzschildIsotropic,
      0.0,
      0.0,
      0.0,
      -0.3,
      0.2,
      -0.1,
      0.4,
   };
   constexpr amrex::Real position1 = 1.7;
   constexpr amrex::Real position2 = 0.6;
   constexpr amrex::Real position3 = -0.25;
   const auto schwarzschild_geometry = fixed_spacetime_geometry(
      schwarzschild, position1, position2, position3, 0.0
   );
   const auto& metric = schwarzschild_geometry.metric;
   if (!close_enough(metric.gamma11 * metric.inv_gamma11, 1.0)
       || !close_enough(metric.gamma22 * metric.inv_gamma22, 1.0)
       || !close_enough(metric.gamma33 * metric.inv_gamma33, 1.0)
       || !close_enough(
          metric.sqrt_det_gamma * metric.sqrt_det_gamma,
          metric.gamma11 * metric.gamma22 * metric.gamma33
       )) {
      std::cerr << "isotropic-Schwarzschild metric identity mismatch\n";
      return false;
   }
   constexpr amrex::Real step = 1.0e-6;
   const auto plus = fixed_spacetime_geometry(
      schwarzschild, position1 + step, position2, position3, 0.0
   );
   const auto minus = fixed_spacetime_geometry(
      schwarzschild, position1 - step, position2, position3, 0.0
   );
   const amrex::Real lapse_derivative = (
      plus.metric.lapse - minus.metric.lapse
   ) / (2.0 * step);
   const amrex::Real gamma_derivative = (
      plus.metric.gamma11 - minus.metric.gamma11
   ) / (2.0 * step);
   if (std::abs(lapse_derivative - schwarzschild_geometry.derivatives.d1_lapse)
          > 2.0e-9
       || std::abs(gamma_derivative - schwarzschild_geometry.derivatives.d1_gamma11)
          > 2.0e-9) {
      std::cerr << "isotropic-Schwarzschild metric derivative mismatch\n";
      return false;
   }
   return true;
}

bool validate_srmhd_round_trips()
{
   constexpr amrex::Real gamma = 1.4;
   constexpr amrex::Real tolerance = 5.0e-10;
   const pyfluid_amrex::SpacetimeMetric minkowski{};
   std::array<pyfluid_amrex::RMHDPrimitive, 8> primitive_corpus{{
      {1.0e-3, 1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.071, 0.23, 0.0, 0.31, -0.17, 0.09, 0.04, -0.02, 0.01},
      {0.82, 4.7, 0.0, -0.71, 0.23, -0.11, -2.1, 3.7, -0.4},
      {1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 8.0, -3.0, 1.5},
      {7.4, 0.017, 0.0, 0.82, -0.31, 0.18, 0.003, -8.0, 2.5},
      {19.0, 13.0, 0.0, -0.91, 0.21, -0.07, 15.0, 0.2, -0.5},
      {0.3, 2.1, 0.0, 0.965, 0.08, -0.03, 0.9, -1.4, 0.7},
      {2.7, 0.08, 0.0, -0.44, -0.51, 0.37, -9.0, 4.0, 6.0},
   }};

   amrex::Real worst_primitive_error = 0.0;
   amrex::Real worst_conserved_error = 0.0;
   for (std::size_t sample = 0; sample < primitive_corpus.size(); ++sample) {
      auto primitive = primitive_corpus[sample];
      primitive.pressure = pyfluid_amrex::generated::relativistic_ideal_gas_pressure(
         primitive.rho,
         primitive.eps,
         gamma
      );
      const auto conserved =
         pyfluid_amrex::generated::rmhd_primitive_to_conserved(primitive, minkowski);
      const auto recovered = pyfluid_amrex::recover_srmhd_palenzuela_1d<true>(
         conserved,
         gamma
      );
      if (!recovered.attempt.succeeded()) {
         std::cerr << "SRMHD primitive round trip failed sample=" << sample
                   << " status=" << static_cast<int>(recovered.attempt.status)
                   << " residual=" << recovered.attempt.residual << '\n';
         return false;
      }
      const auto conserved_round_trip =
         pyfluid_amrex::generated::rmhd_primitive_to_conserved(
            recovered.primitive,
            minkowski
         );
      worst_primitive_error = std::max(
         worst_primitive_error,
         scaled_max_error(components(recovered.primitive), components(primitive))
      );
      worst_conserved_error = std::max(
         worst_conserved_error,
         scaled_max_error(components(conserved_round_trip), components(conserved))
      );
      const amrex::Real velocity_squared = primitive.v1 * primitive.v1
         + primitive.v2 * primitive.v2 + primitive.v3 * primitive.v3;
      const amrex::Real lorentz_factor = 1.0 / std::sqrt(1.0 - velocity_squared);
      const pyfluid_amrex::RMHDEntropyInput entropy{
         lorentz_factor * primitive.pressure
            / std::pow(primitive.rho, gamma - 1.0),
      };
      const auto default_policy = pyfluid_amrex::recover_srmhd<true>(
         conserved, gamma
      );
      if (!default_policy.succeeded()
          || default_policy.attempt.method != pyfluid_amrex::C2PMethod::RePrimAnd
          || default_policy.attempt_count != 1) {
         std::cerr << "SRMHD default policy ordering failed sample=" << sample
                   << '\n';
         return false;
      }
      const std::array independent_methods{
         pyfluid_amrex::recover_srmhd_reprimand<true>(conserved, gamma),
         pyfluid_amrex::recover_srmhd_noble_2d<true>(
            conserved, primitive, gamma
         ),
         pyfluid_amrex::recover_srmhd<true>(conserved, primitive, gamma),
         pyfluid_amrex::recover_srmhd_palenzuela_1d_entropy<true>(
            conserved, entropy, gamma
         ),
         pyfluid_amrex::recover_srmhd_noble_1d_entropy<true>(
            conserved, primitive, entropy, gamma
         ),
         default_policy,
      };
      for (const auto& method_recovery : independent_methods) {
         if (!method_recovery.succeeded()) {
            std::cerr << "SRMHD alternate recovery failed sample=" << sample
                      << " method=" << static_cast<int>(method_recovery.attempt.method)
                      << " status=" << static_cast<int>(method_recovery.attempt.status)
                      << " residual=" << method_recovery.attempt.residual << '\n';
            return false;
         }
         const auto method_round_trip =
            pyfluid_amrex::generated::rmhd_primitive_to_conserved(
               method_recovery.primitive,
               minkowski
            );
         worst_primitive_error = std::max(
            worst_primitive_error,
            scaled_max_error(
               components(method_recovery.primitive), components(primitive)
            )
         );
         worst_conserved_error = std::max(
            worst_conserved_error,
            scaled_max_error(components(method_round_trip), components(conserved))
         );
      }

      auto independent_conserved = conserved;
      independent_conserved.mom1 *= 0.997;
      independent_conserved.mom2 += 0.001 * independent_conserved.dens;
      independent_conserved.mom3 *= 1.002;
      independent_conserved.tau += 0.003 * (
         independent_conserved.dens + independent_conserved.tau
      );
      const auto independent_recovered =
         pyfluid_amrex::recover_srmhd_palenzuela_1d<true>(
            independent_conserved, gamma
         );
      if (!independent_recovered.attempt.succeeded()) {
         std::cerr << "SRMHD conserved round trip failed sample=" << sample
                   << " status=" << static_cast<int>(independent_recovered.attempt.status)
                   << " residual=" << independent_recovered.attempt.residual << '\n';
         return false;
      }
      const auto independent_round_trip =
         pyfluid_amrex::generated::rmhd_primitive_to_conserved(
            independent_recovered.primitive,
            minkowski
         );
      worst_conserved_error = std::max(
         worst_conserved_error,
         scaled_max_error(
            components(independent_round_trip),
            components(independent_conserved)
         )
      );
      const auto independent_reprimand =
         pyfluid_amrex::recover_srmhd_reprimand<true>(
            independent_conserved, gamma
         );
      const auto independent_policy =
         pyfluid_amrex::recover_srmhd<true>(independent_conserved, gamma);
      for (const auto& method_recovery : std::array{
              independent_reprimand,
              independent_policy,
           }) {
         if (!method_recovery.succeeded()) {
            std::cerr << "SRMHD alternate conserved recovery failed sample=" << sample
                      << " method=" << static_cast<int>(method_recovery.attempt.method)
                      << " status=" << static_cast<int>(method_recovery.attempt.status)
                      << " residual=" << method_recovery.attempt.residual << '\n';
            return false;
         }
         const auto method_round_trip =
            pyfluid_amrex::generated::rmhd_primitive_to_conserved(
               method_recovery.primitive,
               minkowski
            );
         worst_conserved_error = std::max(
            worst_conserved_error,
            scaled_max_error(
               components(method_round_trip), components(independent_conserved)
            )
         );
      }

      const amrex::Real x = (1.0 + gamma * primitive.eps) * lorentz_factor;
      const amrex::Real step = 1.0e-6 * std::max(amrex::Real{1.0}, x);
      const amrex::Real finite_difference = (
         pyfluid_amrex::srmhd_palenzuela_residual(conserved, x + step, gamma)
         - pyfluid_amrex::srmhd_palenzuela_residual(conserved, x - step, gamma)
      ) / (2.0 * step);
      const amrex::Real analytic =
         pyfluid_amrex::srmhd_palenzuela_residual_derivative(conserved, x, gamma);
      if (!std::isfinite(analytic)
          || std::abs(analytic - finite_difference)
             > 3.0e-6 * std::max(amrex::Real{1.0}, std::abs(finite_difference))) {
         std::cerr << "SRMHD generated residual derivative mismatch sample=" << sample
                   << " analytic=" << analytic
                   << " finite_difference=" << finite_difference << '\n';
         return false;
      }

      constexpr pyfluid_amrex::ColdPolytropeEquationOfState cold_eos{
         0.2, 1.8,
      };
      auto cold_primitive = primitive;
      cold_primitive.pressure = cold_eos.polytropic_k * std::pow(
         cold_primitive.rho, cold_eos.polytropic_gamma
      );
      cold_primitive.eps = cold_primitive.pressure / (
         cold_primitive.rho * (cold_eos.polytropic_gamma - 1.0)
      );
      const auto cold_conserved =
         pyfluid_amrex::generated::rmhd_primitive_to_conserved(
            cold_primitive, minkowski
         );
      const auto cold_recovered = pyfluid_amrex::recover_srmhd_font_1d<true>(
         cold_conserved, cold_eos
      );
      if (!cold_recovered.succeeded()) {
         std::cerr << "SRMHD cold recovery failed sample=" << sample
                   << " status=" << static_cast<int>(cold_recovered.attempt.status)
                   << " residual=" << cold_recovered.attempt.residual << '\n';
         return false;
      }
      worst_primitive_error = std::max(
         worst_primitive_error,
         scaled_max_error(
            components(cold_recovered.primitive), components(cold_primitive)
         )
      );
   }

   pyfluid_amrex::SpacetimeMetric curved_metric{};
   curved_metric.lapse = 0.83;
   curved_metric.shift1 = 0.07;
   curved_metric.shift2 = -0.03;
   curved_metric.shift3 = 0.02;
   curved_metric.gamma12 = 0.1;
   curved_metric.inv_gamma11 = 1.0 / 0.99;
   curved_metric.inv_gamma12 = -0.1 / 0.99;
   curved_metric.inv_gamma22 = 1.0 / 0.99;
   curved_metric.sqrt_det_gamma = std::sqrt(0.99);
   for (std::size_t sample = 0; sample < primitive_corpus.size(); ++sample) {
      auto primitive = primitive_corpus[sample];
      primitive.pressure = pyfluid_amrex::generated::relativistic_ideal_gas_pressure(
         primitive.rho, primitive.eps, gamma
      );
      const auto conserved =
         pyfluid_amrex::generated::rmhd_primitive_to_conserved(
            primitive, curved_metric
         );
      const amrex::Real velocity_squared =
         pyfluid_amrex::rmhd_velocity_squared(primitive, curved_metric);
      const amrex::Real lorentz_factor = 1.0 / std::sqrt(
         1.0 - velocity_squared
      );
      const pyfluid_amrex::RMHDEntropyInput entropy{
         curved_metric.sqrt_det_gamma * lorentz_factor * primitive.pressure
            / std::pow(primitive.rho, gamma - 1.0),
      };
      const std::array recoveries{
         pyfluid_amrex::recover_srmhd_palenzuela_1d<true>(
            conserved, curved_metric, gamma
         ),
         pyfluid_amrex::recover_srmhd_reprimand<true>(
            conserved, curved_metric, gamma
         ),
         pyfluid_amrex::recover_srmhd_noble_2d<true>(
            conserved, curved_metric, primitive, gamma
         ),
         pyfluid_amrex::recover_srmhd_palenzuela_1d_entropy<true>(
            conserved, curved_metric, entropy, gamma
         ),
         pyfluid_amrex::recover_srmhd_noble_1d_entropy<true>(
            conserved, curved_metric, primitive, entropy, gamma
         ),
         pyfluid_amrex::recover_srmhd<true>(conserved, curved_metric, gamma),
      };
      for (const auto& recovered : recoveries) {
         if (!recovered.succeeded()) {
            std::cerr << "Curved RMHD recovery failed sample=" << sample
                      << " method=" << static_cast<int>(recovered.attempt.method)
                      << " status=" << static_cast<int>(recovered.attempt.status)
                      << " residual=" << recovered.attempt.residual << '\n';
            return false;
         }
         const auto conserved_round_trip =
            pyfluid_amrex::generated::rmhd_primitive_to_conserved(
               recovered.primitive, curved_metric
            );
         worst_primitive_error = std::max(
            worst_primitive_error,
            scaled_max_error(
               components(recovered.primitive), components(primitive)
            )
         );
         worst_conserved_error = std::max(
            worst_conserved_error,
            scaled_max_error(
               components(conserved_round_trip), components(conserved)
            )
         );
      }
   }

   if (worst_primitive_error > tolerance || worst_conserved_error > tolerance) {
      std::cerr << "SRMHD round-trip residuals primitive=" << worst_primitive_error
                << " conserved=" << worst_conserved_error << '\n';
      return false;
   }

   return true;
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
   if (!validate_newtonian_round_trips()
       || !validate_srhd_round_trips()
       || !validate_srmhd_round_trips()
       || !validate_fixed_spacetime_provider()) {
      return 1;
   }
   static_assert(pyfluid_amrex::valid_spacetime_evolution(
      pyfluid_amrex::PhysicalSystem::HD,
      pyfluid_amrex::SpacetimeEvolution::None
   ));
   static_assert(pyfluid_amrex::valid_spacetime_evolution(
      pyfluid_amrex::PhysicalSystem::RHD,
      pyfluid_amrex::SpacetimeEvolution::Fixed
   ));
   static_assert(!pyfluid_amrex::valid_spacetime_evolution(
      pyfluid_amrex::PhysicalSystem::RHD,
      pyfluid_amrex::SpacetimeEvolution::None
   ));
   static_assert(pyfluid_amrex::ideal_rmhd_c2p_policy[0]
                 == pyfluid_amrex::C2PMethod::RePrimAnd);
   static_assert(pyfluid_amrex::ideal_rmhd_c2p_policy[1]
                 == pyfluid_amrex::C2PMethod::Palenzuela1D);
   static_assert(pyfluid_amrex::c2p_method_capabilities(
      pyfluid_amrex::C2PMethod::RePrimAnd
   ).implemented);
   static_assert(pyfluid_amrex::c2p_method_capabilities(
      pyfluid_amrex::C2PMethod::Noble2D
   ).requires_seed);
   static_assert(pyfluid_amrex::c2p_method_capabilities(
      pyfluid_amrex::C2PMethod::Palenzuela1DEntropy
   ).requires_entropy);
   static_assert(pyfluid_amrex::c2p_method_capabilities(
      pyfluid_amrex::C2PMethod::Font1D
   ).requires_cold_equation_of_state);
   {
      const pyfluid_amrex::RHDPrimitive primitive{1.3, 0.2, 0.4, 0.2, -0.1, 0.05};
      const pyfluid_amrex::SpacetimeMetric metric{};
      const amrex::Real velocity_squared =
         primitive.v1 * primitive.v1 + primitive.v2 * primitive.v2 + primitive.v3 * primitive.v3;
      const amrex::Real lorentz_factor = 1.0 / std::sqrt(1.0 - velocity_squared);
      const amrex::Real enthalpy = 1.0 + primitive.eps + primitive.pressure / primitive.rho;
      const pyfluid_amrex::RHDConserved expected{
         primitive.rho * lorentz_factor,
         primitive.rho * enthalpy * lorentz_factor * lorentz_factor * primitive.v1,
         primitive.rho * enthalpy * lorentz_factor * lorentz_factor * primitive.v2,
         primitive.rho * enthalpy * lorentz_factor * lorentz_factor * primitive.v3,
         primitive.rho * enthalpy * lorentz_factor * lorentz_factor
            - primitive.pressure - primitive.rho * lorentz_factor,
      };
      const auto conserved = pyfluid_amrex::generated::rhd_primitive_to_conserved(primitive, metric);
      if (!close_enough(conserved.dens, expected.dens)
          || !close_enough(conserved.mom1, expected.mom1)
          || !close_enough(conserved.mom2, expected.mom2)
          || !close_enough(conserved.mom3, expected.mom3)
          || !close_enough(conserved.tau, expected.tau)) {
         std::cerr << "generated RHD primitive-to-conserved mismatch\n";
         return 1;
      }
      const auto flux1 = pyfluid_amrex::generated::rhd_flux<0>(primitive, metric);
      const auto flux2 = pyfluid_amrex::generated::rhd_flux<1>(primitive, metric);
      const auto flux3 = pyfluid_amrex::generated::rhd_flux<2>(primitive, metric);
      if (!close_enough(flux1.dens, expected.dens * primitive.v1)
          || !close_enough(flux1.mom1, expected.mom1 * primitive.v1 + primitive.pressure)
          || !close_enough(flux2.mom2, expected.mom2 * primitive.v2 + primitive.pressure)
          || !close_enough(flux3.tau, expected.tau * primitive.v3 + primitive.pressure * primitive.v3)) {
         std::cerr << "generated RHD directional flux mismatch\n";
         return 1;
      }
      const pyfluid_amrex::RHDWaveSpeeds speeds =
         pyfluid_amrex::generated::srhd_wave_speeds<0>(
            primitive, metric, 5.0 / 3.0
         );
      if (!std::isfinite(speeds.minus) || !std::isfinite(speeds.plus)
          || speeds.minus > speeds.plus || speeds.minus < -1.0 || speeds.plus > 1.0) {
         std::cerr << "generated SRHD wave-speed bounds mismatch\n";
         return 1;
      }
      const pyfluid_amrex::EOSParams eos{};
      const pyfluid_amrex::RHDPrimitive consistent{
         1.3,
         0.2,
         (eos.gamma - 1.0) * 1.3 * 0.2,
         0.2,
         -0.1,
         0.05,
      };
      const auto physical_flux = pyfluid_amrex::generated::rhd_flux<0>(consistent, metric);
      const auto hll_flux = pyfluid_amrex::rhd_hll_flux<0>(
         consistent, consistent, eos, metric
      );
      if (!close_enough(hll_flux.dens, physical_flux.dens)
          || !close_enough(hll_flux.mom1, physical_flux.mom1)
          || !close_enough(hll_flux.mom2, physical_flux.mom2)
          || !close_enough(hll_flux.mom3, physical_flux.mom3)
          || !close_enough(hll_flux.tau, physical_flux.tau)) {
         std::cerr << "SRHD HLL consistency mismatch\n";
         return 1;
      }
   }
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
   if (!close_enough(pyfluid_amrex::generated::ideal_gas_pressure_from_density_energy(3.0, 5.0 / 3.0), 2.0)) {
      std::cerr << "generated ideal-gas pressure mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::ideal_gas_internal_energy_density_from_pressure(2.0, 5.0 / 3.0), 3.0)) {
      std::cerr << "generated ideal-gas energy mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::polytrope_cold_pressure(2.0, 0.5, 2.0), 2.0)) {
      std::cerr << "generated polytropic pressure mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::hybrid_polytrope_pressure_from_density_energy(2.0, 4.0, 0.5, 2.0, 1.5), 3.0)) {
      std::cerr << "generated hybrid-polytrope pressure mismatch\n";
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
      const amrex::Real radial_fraction = (2.0 - radial_inner) / (radial_outer - radial_inner);
      const amrex::Real blend = 1.0 - radial_fraction + radial_fraction / std::sqrt(denom2);
      const amrex::Real primary = -2.0 * blend;
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
