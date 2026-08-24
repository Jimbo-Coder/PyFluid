#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/flux.hpp"
#include "pyfluid_amrex/kernels_generated/riemann_ops.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_riemann.hpp"

#include <AMReX_Algorithm.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

namespace {

using namespace pyfluid_amrex;

template <int Direction>
RMHDWaveSpeeds ideal_srmhd_wave_speeds(
   RMHDPrimitive const& primitive,
   SpacetimeMetric const& metric,
   amrex::Real gamma
) noexcept
{
   return generated::srmhd_wave_speeds<Direction>(
      primitive,
      metric,
      generated::relativistic_ideal_gas_sound_speed_squared(
         primitive.eps, gamma
      ),
      generated::relativistic_ideal_gas_specific_enthalpy(
         primitive.eps, gamma
      )
   );
}

bool close_enough(
   amrex::Real actual,
   amrex::Real expected,
   amrex::Real tolerance = 2.0e-11
)
{
   const amrex::Real scale = amrex::max(
      1.0,
      amrex::max(amrex::Math::abs(actual), amrex::Math::abs(expected))
   );
   return amrex::Math::abs(actual - expected) <= tolerance * scale;
}

bool require_close(
   std::string_view label,
   amrex::Real actual,
   amrex::Real expected,
   amrex::Real tolerance = 2.0e-11
)
{
   if (close_enough(actual, expected, tolerance)) {
      return true;
   }
   std::cerr << label << ": got " << actual << ", expected " << expected
             << '\n';
   return false;
}

bool require_state_close(
   std::string_view label,
   const SRMHDState& actual,
   const SRMHDState& expected,
   amrex::Real tolerance = 2.0e-11
)
{
   bool passed = true;
   passed = require_close(label, actual.dens, expected.dens, tolerance) && passed;
   passed = require_close(label, actual.momn, expected.momn, tolerance) && passed;
   passed = require_close(label, actual.momt1, expected.momt1, tolerance) && passed;
   passed = require_close(label, actual.momt2, expected.momt2, tolerance) && passed;
   passed = require_close(label, actual.energy, expected.energy, tolerance) && passed;
   passed = require_close(label, actual.bn, expected.bn, tolerance) && passed;
   passed = require_close(label, actual.bt1, expected.bt1, tolerance) && passed;
   passed = require_close(label, actual.bt2, expected.bt2, tolerance) && passed;
   return passed;
}

bool require_flux_close(
   std::string_view label,
   const RMHDConserved& actual,
   const RMHDConserved& expected,
   amrex::Real tolerance = 2.0e-11
)
{
   bool passed = true;
   passed = require_close(label, actual.dens, expected.dens, tolerance) && passed;
   passed = require_close(label, actual.mom1, expected.mom1, tolerance) && passed;
   passed = require_close(label, actual.mom2, expected.mom2, tolerance) && passed;
   passed = require_close(label, actual.mom3, expected.mom3, tolerance) && passed;
   passed = require_close(label, actual.tau, expected.tau, tolerance) && passed;
   passed = require_close(label, actual.b1, expected.b1, tolerance) && passed;
   passed = require_close(label, actual.b2, expected.b2, tolerance) && passed;
   passed = require_close(label, actual.b3, expected.b3, tolerance) && passed;
   return passed;
}

RMHDPrimitive primitive(
   amrex::Real rho,
   amrex::Real pressure,
   amrex::Real gamma,
   amrex::Real v1,
   amrex::Real v2,
   amrex::Real v3,
   amrex::Real b1,
   amrex::Real b2,
   amrex::Real b3
)
{
   return {
      rho,
      pressure / ((gamma - 1.0) * rho),
      pressure,
      v1,
      v2,
      v3,
      b1,
      b2,
      b3,
   };
}

RMHDConserved hll_flux(
   const RMHDPrimitive& left,
   const RMHDPrimitive& right,
   amrex::Real gamma
)
{
   const SpacetimeMetric metric = minkowski_spacetime_metric();
   const RMHDConserved state_l = generated::rmhd_primitive_to_conserved(
      left, metric
   );
   const RMHDConserved state_r = generated::rmhd_primitive_to_conserved(
      right, metric
   );
   const RMHDConserved flux_l = generated::rmhd_flux<0>(left, metric);
   const RMHDConserved flux_r = generated::rmhd_flux<0>(right, metric);
   const RMHDWaveSpeeds waves_l = ideal_srmhd_wave_speeds<0>(
      left, metric, gamma
   );
   const RMHDWaveSpeeds waves_r = ideal_srmhd_wave_speeds<0>(
      right, metric, gamma
   );
   const amrex::Real speed_l = amrex::min(
      0.0, amrex::min(waves_l.minus, waves_r.minus)
   );
   const amrex::Real speed_r = amrex::max(
      0.0, amrex::max(waves_l.plus, waves_r.plus)
   );
   if (speed_l >= 0.0) {
      return flux_l;
   }
   if (speed_r <= 0.0) {
      return flux_r;
   }
   return generated::rmhd_hll_flux(
      flux_l, flux_r, state_l, state_r, speed_l, speed_r
   );
}

bool read_primitive(amrex::Real gamma, RMHDPrimitive& value)
{
   amrex::Real rho;
   amrex::Real pressure;
   amrex::Real v1;
   amrex::Real v2;
   amrex::Real v3;
   amrex::Real b1;
   amrex::Real b2;
   amrex::Real b3;
   if (!(std::cin >> rho >> pressure >> v1 >> v2 >> v3 >> b1 >> b2 >> b3)) {
      return false;
   }
   value = primitive(rho, pressure, gamma, v1, v2, v3, b1, b2, b3);
   return true;
}

void write_conserved(const RMHDConserved& value)
{
   std::cout << '\t' << value.dens << '\t' << value.mom1 << '\t'
             << value.mom2 << '\t' << value.mom3 << '\t' << value.tau
             << '\t' << value.b1 << '\t' << value.b2 << '\t' << value.b3;
}

void write_primitive(const RMHDPrimitive& value)
{
   std::cout << '\t' << value.rho << '\t' << value.eps << '\t'
             << value.pressure << '\t' << value.v1 << '\t' << value.v2
             << '\t' << value.v3 << '\t' << value.b1 << '\t' << value.b2
             << '\t' << value.b3;
}

RMHDConserved coordinate_state(const SRMHDState& value)
{
   return {
      value.dens,
      value.momn,
      value.momt1,
      value.momt2,
      value.energy - value.dens,
      value.bn,
      value.bt1,
      value.bt2,
   };
}

int run_flux_table()
{
   std::cout << std::setprecision(17);
   std::int64_t row_id;
   amrex::Real gamma;
   while (std::cin >> row_id >> gamma) {
      RMHDPrimitive left{};
      RMHDPrimitive right{};
      if (!read_primitive(gamma, left) || !read_primitive(gamma, right)) {
         std::cerr << "incomplete primitive pair for row " << row_id << '\n';
         return 2;
      }
      const EOSParams eos{.gamma = gamma};
      const SpacetimeMetric metric = minkowski_spacetime_metric();
      const RMHDConserved hll = hll_flux(left, right, gamma);
      const RMHDRiemannFluxResult hllc = rmhd_hllc_flux<0>(
         left, right, eos, metric
      );
      const RMHDRiemannFluxResult hlld = rmhd_hlld_flux<0>(
         left, right, eos, metric
      );
      std::cout << row_id;
      write_conserved(hll);
      write_conserved(hllc.flux);
      std::cout << '\t' << static_cast<int>(hllc.used_hll);
      write_conserved(hlld.flux);
      std::cout << '\t' << static_cast<int>(hlld.used_hll) << '\n';
   }
   return std::cin.eof() ? 0 : 2;
}

int run_primitive_table()
{
   std::cout << std::setprecision(17);
   std::int64_t row_id;
   amrex::Real gamma;
   while (std::cin >> row_id >> gamma) {
      RMHDPrimitive value{};
      if (!read_primitive(gamma, value)) {
         std::cerr << "incomplete primitive state for row " << row_id << '\n';
         return 2;
      }
      const SpacetimeMetric metric = minkowski_spacetime_metric();
      std::cout << row_id;
      write_conserved(generated::rmhd_primitive_to_conserved(value, metric));
      write_conserved(generated::rmhd_flux<0>(value, metric));
      std::cout << '\n';
   }
   return std::cin.eof() ? 0 : 2;
}

int run_c2p_table()
{
   std::cout << std::setprecision(17);
   std::int64_t row_id;
   amrex::Real gamma;
   while (std::cin >> row_id >> gamma) {
      RMHDConserved conserved{};
      if (!(std::cin >> conserved.dens >> conserved.mom1 >> conserved.mom2
            >> conserved.mom3 >> conserved.tau >> conserved.b1
            >> conserved.b2 >> conserved.b3)) {
         std::cerr << "incomplete conserved state for row " << row_id << '\n';
         return 2;
      }
      const SpacetimeMetric metric = minkowski_spacetime_metric();
      const RMHDRecoveryResult recovered = recover_srmhd(
         conserved, metric, gamma
      );
      const amrex::Real conserved_residual = recovered.succeeded()
         ? rmhd_conserved_residual(recovered.primitive, conserved, metric)
         : recovered.attempt.conserved_residual;
      std::cout << row_id << '\t' << static_cast<int>(recovered.succeeded())
                << '\t' << static_cast<int>(recovered.attempt.status)
                << '\t' << recovered.attempt_count << '\t'
                << recovered.attempt.iterations << '\t'
                << recovered.attempt.residual << '\t'
                << conserved_residual;
      write_primitive(recovered.primitive);
      std::cout << '\n' << std::flush;
   }
   return std::cin.eof() ? 0 : 2;
}

int run_hlld_fan_table()
{
   std::cout << std::setprecision(17);
   std::int64_t row_id;
   amrex::Real gamma;
   while (std::cin >> row_id >> gamma) {
      RMHDPrimitive left{};
      RMHDPrimitive right{};
      if (!read_primitive(gamma, left) || !read_primitive(gamma, right)) {
         std::cerr << "incomplete primitive pair for row " << row_id << '\n';
         return 2;
      }
      const amrex::Real shared_bn = 0.5 * (left.b1 + right.b1);
      left.b1 = shared_bn;
      right.b1 = shared_bn;
      const SRMHDState state_l = srmhd_state_from_primitive(left);
      const SRMHDState state_r = srmhd_state_from_primitive(right);
      const SRMHDState flux_l = srmhd_flux_from_primitive(left);
      const SRMHDState flux_r = srmhd_flux_from_primitive(right);
      const RMHDWaveSpeeds waves_l = ideal_srmhd_wave_speeds<0>(
         left, minkowski_spacetime_metric(), gamma
      );
      const RMHDWaveSpeeds waves_r = ideal_srmhd_wave_speeds<0>(
         right, minkowski_spacetime_metric(), gamma
      );
      const amrex::Real speed_l = amrex::min(waves_l.minus, waves_r.minus);
      const amrex::Real speed_r = amrex::max(waves_l.plus, waves_r.plus);
      const SRMHDHLLDSolution solution = srmhd_hlld_pressure_solution(
         left, right, state_l, state_r, flux_l, flux_r,
         speed_l, speed_r, shared_bn
      );
      SRMHDHLLDContactState contact_l{};
      SRMHDHLLDContactState contact_r{};
      amrex::Real contact_speed = 0.0;
      bool valid = solution.valid;
      if (valid) {
         const SRMHDHLLDEvaluation& fan = solution.evaluation;
         contact_l = srmhd_hlld_contact_state(
            fan.fast_l, fan.contact_velocity_l, fan.contact_magnetic,
            solution.total_pressure
         );
         contact_r = srmhd_hlld_contact_state(
            fan.fast_r, fan.contact_velocity_r, fan.contact_magnetic,
            solution.total_pressure
         );
         contact_speed = 0.5 * (
            fan.contact_velocity_l.n + fan.contact_velocity_r.n
         );
         valid = contact_l.valid && contact_r.valid;
      }
      std::cout << row_id << '\t' << static_cast<int>(valid);
      if (!valid) {
         for (int component = 0; component < 39; ++component) {
            std::cout << "\t0";
         }
         std::cout << '\n' << std::flush;
         continue;
      }
      const SRMHDHLLDEvaluation& fan = solution.evaluation;
      std::cout << '\t' << speed_l << '\t' << fan.fast_l.alfven_speed
                << '\t' << contact_speed << '\t'
                << fan.fast_r.alfven_speed << '\t' << speed_r
                << '\t' << solution.total_pressure << '\t'
                << fan.residual;
      write_conserved(coordinate_state(fan.fast_l.state));
      write_conserved(coordinate_state(contact_l.state));
      write_conserved(coordinate_state(contact_r.state));
      write_conserved(coordinate_state(fan.fast_r.state));
      std::cout << '\n';
   }
   return std::cin.eof() ? 0 : 2;
}

int run_hlld_evaluation_table()
{
   std::cout << std::setprecision(17);
   std::int64_t row_id;
   amrex::Real gamma;
   while (std::cin >> row_id >> gamma) {
      RMHDPrimitive left{};
      RMHDPrimitive right{};
      if (!read_primitive(gamma, left) || !read_primitive(gamma, right)) {
         std::cerr << "incomplete primitive pair for row " << row_id << '\n';
         return 2;
      }
      amrex::Real speed_l;
      amrex::Real speed_r;
      amrex::Real total_pressure;
      if (!(std::cin >> speed_l >> speed_r >> total_pressure)) {
         std::cerr << "incomplete HLLD parameters for row " << row_id << '\n';
         return 2;
      }
      const amrex::Real shared_bn = 0.5 * (left.b1 + right.b1);
      left.b1 = shared_bn;
      right.b1 = shared_bn;
      const SRMHDState state_l = srmhd_state_from_primitive(left);
      const SRMHDState state_r = srmhd_state_from_primitive(right);
      const SRMHDState flux_l = srmhd_flux_from_primitive(left);
      const SRMHDState flux_r = srmhd_flux_from_primitive(right);
      const SRMHDHLLDEvaluation evaluation = srmhd_hlld_evaluate(
         state_l, state_r, flux_l, flux_r, speed_l, speed_r,
         shared_bn, total_pressure
      );
      SRMHDHLLDContactState contact_l{};
      SRMHDHLLDContactState contact_r{};
      amrex::Real contact_speed = 0.0;
      if (evaluation.valid) {
         contact_l = srmhd_hlld_contact_state(
            evaluation.fast_l,
            evaluation.contact_velocity_l,
            evaluation.contact_magnetic,
            total_pressure
         );
         contact_r = srmhd_hlld_contact_state(
            evaluation.fast_r,
            evaluation.contact_velocity_r,
            evaluation.contact_magnetic,
            total_pressure
         );
         contact_speed = 0.5 * (
            evaluation.contact_velocity_l.n
            + evaluation.contact_velocity_r.n
         );
      }
      const bool contacts_valid = contact_l.valid && contact_r.valid;
      const bool physical = srmhd_hlld_physical_solution(
         evaluation, speed_l, speed_r, total_pressure
      ) && contacts_valid;
      std::cout << row_id << '\t' << static_cast<int>(evaluation.valid)
                << '\t' << static_cast<int>(physical) << '\t'
                << static_cast<int>(contacts_valid);
      if (!evaluation.valid || !contacts_valid) {
         for (int component = 0; component < 39; ++component) {
            std::cout << "\t0";
         }
         std::cout << '\n' << std::flush;
         continue;
      }
      std::cout << '\t' << speed_l << '\t'
                << evaluation.fast_l.alfven_speed << '\t'
                << contact_speed << '\t'
                << evaluation.fast_r.alfven_speed << '\t' << speed_r
                << '\t' << total_pressure << '\t' << evaluation.residual;
      write_conserved(coordinate_state(evaluation.fast_l.state));
      write_conserved(coordinate_state(contact_l.state));
      write_conserved(coordinate_state(contact_r.state));
      write_conserved(coordinate_state(evaluation.fast_r.state));
      std::cout << '\n' << std::flush;
   }
   return std::cin.eof() ? 0 : 2;
}

SpacetimeMetric nontrivial_metric()
{
   SpacetimeMetric metric{};
   metric.lapse = 0.91;
   metric.shift1 = 0.07;
   metric.shift2 = -0.04;
   metric.shift3 = 0.03;
   metric.gamma11 = 1.44;
   metric.gamma12 = 0.12;
   metric.gamma13 = -0.03;
   metric.gamma22 = 0.85;
   metric.gamma23 = 0.05;
   metric.gamma33 = 1.21;
   const amrex::Real determinant =
      metric.gamma11 * (
         metric.gamma22 * metric.gamma33 - metric.gamma23 * metric.gamma23
      )
      - metric.gamma12 * (
         metric.gamma12 * metric.gamma33 - metric.gamma13 * metric.gamma23
      )
      + metric.gamma13 * (
         metric.gamma12 * metric.gamma23 - metric.gamma13 * metric.gamma22
      );
   metric.inv_gamma11 = (
      metric.gamma22 * metric.gamma33 - metric.gamma23 * metric.gamma23
   ) / determinant;
   metric.inv_gamma12 = (
      metric.gamma13 * metric.gamma23 - metric.gamma12 * metric.gamma33
   ) / determinant;
   metric.inv_gamma13 = (
      metric.gamma12 * metric.gamma23 - metric.gamma13 * metric.gamma22
   ) / determinant;
   metric.inv_gamma22 = (
      metric.gamma11 * metric.gamma33 - metric.gamma13 * metric.gamma13
   ) / determinant;
   metric.inv_gamma23 = (
      metric.gamma12 * metric.gamma13 - metric.gamma11 * metric.gamma23
   ) / determinant;
   metric.inv_gamma33 = (
      metric.gamma11 * metric.gamma22 - metric.gamma12 * metric.gamma12
   ) / determinant;
   metric.sqrt_det_gamma = std::sqrt(determinant);
   return metric;
}

template <int Direction>
bool validate_frame()
{
   const SpacetimeMetric metric = nontrivial_metric();
   const RMHDFaceFrame frame = rmhd_face_frame<Direction>(metric);
   bool passed = true;
   passed = require_close(
      "frame normal norm", rmhd_spatial_dot(frame.basis_n, frame.basis_n, metric), 1.0
   ) && passed;
   passed = require_close(
      "frame tangent1 norm", rmhd_spatial_dot(frame.basis_t1, frame.basis_t1, metric), 1.0
   ) && passed;
   passed = require_close(
      "frame tangent2 norm", rmhd_spatial_dot(frame.basis_t2, frame.basis_t2, metric), 1.0
   ) && passed;
   passed = require_close(
      "frame normal tangent1", rmhd_spatial_dot(frame.basis_n, frame.basis_t1, metric), 0.0
   ) && passed;
   passed = require_close(
      "frame normal tangent2", rmhd_spatial_dot(frame.basis_n, frame.basis_t2, metric), 0.0
   ) && passed;
   passed = require_close(
      "frame tangent cross", rmhd_spatial_dot(frame.basis_t1, frame.basis_t2, metric), 0.0
   ) && passed;
   return passed;
}

template <int Direction>
bool validate_identical_curved_flux()
{
   constexpr amrex::Real gamma = 5.0 / 3.0;
   const EOSParams eos{.gamma = gamma};
   const SpacetimeMetric metric = nontrivial_metric();
   const RMHDPrimitive state = primitive(
      1.2, 0.8, gamma, 0.08, -0.06, 0.04, 0.7, -0.3, 0.2
   );
   const RMHDConserved expected = generated::rmhd_flux<Direction>(state, metric);
   const RMHDRiemannFluxResult hllc = rmhd_hllc_flux<Direction>(
      state, state, eos, metric
   );
   const RMHDRiemannFluxResult hlld = rmhd_hlld_flux<Direction>(
      state, state, eos, metric
   );
   bool passed = !hllc.used_hll && !hlld.used_hll;
   if (!passed) {
      std::cerr << "identical curved state unexpectedly used HLL\n";
   }
   passed = require_flux_close("curved HLLC identity", hllc.flux, expected) && passed;
   passed = require_flux_close("curved HLLD identity", hlld.flux, expected) && passed;
   return passed;
}

template <int Direction>
bool validate_curved_hll_equivalence()
{
   constexpr amrex::Real gamma = 5.0 / 3.0;
   const SpacetimeMetric metric = nontrivial_metric();
   const RMHDPrimitive left = primitive(
      1.2, 0.8, gamma, 0.08, -0.06, 0.04, 0.7, -0.3, 0.2
   );
   const RMHDPrimitive right = primitive(
      0.7, 0.35, gamma, -0.11, 0.03, -0.05, 0.7, -0.3, 0.2
   );
   const RMHDConserved state_l = generated::rmhd_primitive_to_conserved(
      left, metric
   );
   const RMHDConserved state_r = generated::rmhd_primitive_to_conserved(
      right, metric
   );
   const RMHDConserved flux_l = generated::rmhd_flux<Direction>(left, metric);
   const RMHDConserved flux_r = generated::rmhd_flux<Direction>(right, metric);
   const RMHDWaveSpeeds waves_l = ideal_srmhd_wave_speeds<Direction>(
      left, metric, gamma
   );
   const RMHDWaveSpeeds waves_r = ideal_srmhd_wave_speeds<Direction>(
      right, metric, gamma
   );
   const amrex::Real speed_l = amrex::min(
      0.0, amrex::min(waves_l.minus, waves_r.minus)
   );
   const amrex::Real speed_r = amrex::max(
      0.0, amrex::max(waves_l.plus, waves_r.plus)
   );
   const RMHDConserved expected = generated::rmhd_hll_flux(
      flux_l, flux_r, state_l, state_r, speed_l, speed_r
   );

   const RMHDFaceFrame frame = rmhd_face_frame<Direction>(metric);
   RMHDPrimitive local_l = rmhd_primitive_to_face_frame(left, frame);
   RMHDPrimitive local_r = rmhd_primitive_to_face_frame(right, frame);
   const amrex::Real shared_bn = 0.5 * (local_l.b1 + local_r.b1);
   local_l.b1 = shared_bn;
   local_r.b1 = shared_bn;
   const SRMHDState local_state_l = srmhd_state_from_primitive(local_l);
   const SRMHDState local_state_r = srmhd_state_from_primitive(local_r);
   const SRMHDState local_flux_l = srmhd_flux_from_primitive(local_l);
   const SRMHDState local_flux_r = srmhd_flux_from_primitive(local_r);
   const RMHDWaveSpeeds local_waves_l = ideal_srmhd_wave_speeds<0>(
      local_l, minkowski_spacetime_metric(), gamma
   );
   const RMHDWaveSpeeds local_waves_r = ideal_srmhd_wave_speeds<0>(
      local_r, minkowski_spacetime_metric(), gamma
   );
   const SRMHDRiemannSample sample = srmhd_hll_sample(
      local_state_l,
      local_state_r,
      local_flux_l,
      local_flux_r,
      amrex::min(local_waves_l.minus, local_waves_r.minus),
      amrex::max(local_waves_l.plus, local_waves_r.plus),
      frame.interface_speed
   );
   const RMHDConserved actual = rmhd_face_sample_to_coordinate_flux<Direction>(
      sample, frame, metric
   );
   return require_flux_close(
      "curved local/direct HLL equivalence", actual, expected, 2.0e-10
   );
}

bool validate_hll_algebra()
{
   const SRMHDState left{1.0, 2.0, 3.0, 4.0, 5.0, 0.5, -0.2, 0.3};
   const SRMHDState right{2.0, -1.0, 0.5, 3.0, 7.0, 0.5, 0.6, -0.1};
   const SRMHDState flux_l{-0.2, 1.1, -2.0, 0.4, 3.0, 0.0, 0.7, -0.8};
   const SRMHDState flux_r{0.6, -0.5, 1.0, -0.9, 2.0, 0.0, -0.3, 0.2};
   constexpr amrex::Real speed_l = -0.7;
   constexpr amrex::Real speed_r = 0.9;
   const SRMHDState actual = srmhd_hll_state(
      left, right, flux_l, flux_r, speed_l, speed_r
   );
   const SRMHDState expected = srmhd_state_add_scaled(
      srmhd_state_add_scaled({}, speed_r, right),
      -speed_l,
      left
   );
   const SRMHDState with_flux = srmhd_state_add_scaled(
      srmhd_state_add_scaled(expected, 1.0, flux_l), -1.0, flux_r
   );
   const SRMHDState normalized = srmhd_state_add_scaled(
      {}, 1.0 / (speed_r - speed_l), with_flux
   );
   return require_state_close("HLL state algebra", actual, normalized);
}

bool validate_contact()
{
   constexpr amrex::Real gamma = 5.0 / 3.0;
   const RMHDPrimitive left = primitive(
      10.0, 1.0, gamma, 0.0, 0.7, 0.2, 5.0, 1.0, 0.5
   );
   const RMHDPrimitive right = primitive(
      1.0, 1.0, gamma, 0.0, 0.7, 0.2, 5.0, 1.0, 0.5
   );
   const SRMHDState expected_flux = srmhd_flux_from_primitive(left);
   const SRMHDRiemannSample hllc = srmhd_hllc_sample(left, right, gamma, 0.0);
   const SRMHDRiemannSample hlld = srmhd_hlld_sample(left, right, gamma, 0.0);
   bool passed = !hllc.used_hll && !hlld.used_hll;
   if (!passed) {
      const SRMHDState state_l = srmhd_state_from_primitive(left);
      const SRMHDState state_r = srmhd_state_from_primitive(right);
      const SRMHDState flux_l = srmhd_flux_from_primitive(left);
      const SRMHDState flux_r = srmhd_flux_from_primitive(right);
      const RMHDWaveSpeeds waves_l = ideal_srmhd_wave_speeds<0>(
         left, minkowski_spacetime_metric(), gamma
      );
      const RMHDWaveSpeeds waves_r = ideal_srmhd_wave_speeds<0>(
         right, minkowski_spacetime_metric(), gamma
      );
      const amrex::Real speed_l = amrex::min(waves_l.minus, waves_r.minus);
      const amrex::Real speed_r = amrex::max(waves_l.plus, waves_r.plus);
      const amrex::Real exact_pressure = srmhd_total_pressure(left);
      const SRMHDHLLDEvaluation exact = srmhd_hlld_evaluate(
         state_l, state_r, flux_l, flux_r, speed_l, speed_r, left.b1,
         exact_pressure
      );
      std::cerr << "isolated contact unexpectedly used HLL: hllc="
                << hllc.used_hll << " hlld=" << hlld.used_hll
                << " p=" << exact_pressure << " eval=" << exact.valid
                << " residual=" << exact.residual
                << " vaL=" << exact.fast_l.alfven_speed
                << " vaR=" << exact.fast_r.alfven_speed
                << " vcL=" << exact.contact_velocity_l.n
                << " vcR=" << exact.contact_velocity_r.n
                << " vfastL=" << exact.fast_l.velocity.n << ','
                << exact.fast_l.velocity.t1 << ',' << exact.fast_l.velocity.t2
                << " vfastR=" << exact.fast_r.velocity.n << ','
                << exact.fast_r.velocity.t1 << ',' << exact.fast_r.velocity.t2
                << " Bc=" << exact.contact_magnetic.n << ','
                << exact.contact_magnetic.t1 << ','
                << exact.contact_magnetic.t2 << '\n';
   }
   passed = require_state_close(
      "HLLC isolated contact", hllc.flux, expected_flux, 2.0e-9
   ) && passed;
   passed = require_state_close(
      "HLLD isolated contact", hlld.flux, expected_flux, 2.0e-9
   ) && passed;
   return passed;
}

bool validate_rotational_wave()
{
   constexpr amrex::Real gamma = 5.0 / 3.0;
   const RMHDPrimitive left = primitive(
      1.0, 1.0, gamma, 0.4, -0.3, 0.5, 2.4, 1.0, -1.6
   );
   const RMHDPrimitive right = primitive(
      1.0, 1.0, gamma, 0.377347, -0.482389, 0.424190,
      2.4, -0.1, -2.178213
   );
   const SRMHDState state_l = srmhd_state_from_primitive(left);
   const SRMHDState state_r = srmhd_state_from_primitive(right);
   const SRMHDState flux_l = srmhd_flux_from_primitive(left);
   const SRMHDState flux_r = srmhd_flux_from_primitive(right);
   const amrex::Real wave_speed = (flux_r.bt1 - flux_l.bt1)
      / (state_r.bt1 - state_l.bt1);
   const SRMHDState expected_flux = wave_speed >= 0.0 ? flux_l : flux_r;
   const SRMHDRiemannSample hlld = srmhd_hlld_sample(left, right, gamma, 0.0);
   if (hlld.used_hll) {
      const RMHDWaveSpeeds waves_l = ideal_srmhd_wave_speeds<0>(
         left, minkowski_spacetime_metric(), gamma
      );
      const RMHDWaveSpeeds waves_r = ideal_srmhd_wave_speeds<0>(
         right, minkowski_spacetime_metric(), gamma
      );
      const SRMHDHLLDEvaluation exact = srmhd_hlld_evaluate(
         state_l, state_r, flux_l, flux_r,
         amrex::min(waves_l.minus, waves_r.minus),
         amrex::max(waves_l.plus, waves_r.plus), left.b1,
         srmhd_total_pressure(left)
      );
      std::cerr << "isolated rotational wave unexpectedly used HLL: pL="
                << srmhd_total_pressure(left) << " pR="
                << srmhd_total_pressure(right) << " eval=" << exact.valid
                << " residual=" << exact.residual
                << " vaL=" << exact.fast_l.alfven_speed
                << " vaR=" << exact.fast_r.alfven_speed
                << " vcL=" << exact.contact_velocity_l.n
                << " vcR=" << exact.contact_velocity_r.n << '\n';
      return false;
   }
   return require_state_close(
      "HLLD isolated rotational wave", hlld.flux, expected_flux, 2.0e-5
   );
}

bool validate_zero_normal_degeneracy()
{
   constexpr amrex::Real gamma = 5.0 / 3.0;
   const RMHDPrimitive left = primitive(
      1.0, 1.0, gamma, 0.1, 0.2, -0.1, 0.0, 1.0, 0.5
   );
   const RMHDPrimitive right = primitive(
      0.8, 0.7, gamma, -0.05, -0.1, 0.15, 0.0, -0.4, 0.2
   );
   const SRMHDRiemannSample hllc = srmhd_hllc_sample(left, right, gamma, 0.0);
   const SRMHDRiemannSample hlld = srmhd_hlld_sample(left, right, gamma, 0.0);
   if (hllc.used_hll || hlld.used_hll) {
      std::cerr << "zero-normal-field HLLC degeneracy unexpectedly used HLL\n";
      return false;
   }
   return require_state_close(
      "HLLD zero-normal degeneracy", hlld.flux, hllc.flux
   );
}

amrex::Real state_max_abs(const SRMHDState& state)
{
   amrex::Real scale = 0.0;
   scale = amrex::max(scale, amrex::Math::abs(state.dens));
   scale = amrex::max(scale, amrex::Math::abs(state.momn));
   scale = amrex::max(scale, amrex::Math::abs(state.momt1));
   scale = amrex::max(scale, amrex::Math::abs(state.momt2));
   scale = amrex::max(scale, amrex::Math::abs(state.energy));
   scale = amrex::max(scale, amrex::Math::abs(state.bn));
   scale = amrex::max(scale, amrex::Math::abs(state.bt1));
   scale = amrex::max(scale, amrex::Math::abs(state.bt2));
   return scale;
}

amrex::Real relative_rh_residual(
   const SRMHDState& state_l,
   const SRMHDState& state_r,
   const SRMHDState& flux_l,
   const SRMHDState& flux_r,
   amrex::Real wave_speed
)
{
   const SRMHDState residual = srmhd_state_add_scaled(
      srmhd_state_difference(flux_r, flux_l),
      -wave_speed,
      srmhd_state_difference(state_r, state_l)
   );
   const SRMHDState state_jump = srmhd_state_difference(state_r, state_l);
   const amrex::Real scale = amrex::max(
      1.0,
      amrex::max(
         amrex::max(state_max_abs(flux_l), state_max_abs(flux_r)),
         amrex::Math::abs(wave_speed) * state_max_abs(state_jump)
      )
   );
   return state_max_abs(residual) / scale;
}

amrex::Real random_unit(std::uint64_t& state)
{
   state = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
   return static_cast<amrex::Real>(state >> 11)
      * (1.0 / 9007199254740992.0);
}

bool validate_random_fans()
{
   constexpr amrex::Real gamma = 5.0 / 3.0;
   std::uint64_t generator = UINT64_C(0x5f3759df12345678);
   int accepted_hllc = 0;
   int accepted_hlld = 0;
   int hll_fallbacks = 0;
   amrex::Real maximum_hllc_contact_residual = 0.0;
   amrex::Real maximum_contact_residual = 0.0;
   for (int sample_index = 0; sample_index < 2000; ++sample_index) {
      const auto signed_random = [&]() {
         return 2.0 * random_unit(generator) - 1.0;
      };
      const amrex::Real normal_magnetic = 3.0 * signed_random();
      const auto make_primitive = [&]() {
         const amrex::Real rho = std::pow(10.0, -3.0 + 5.0 * random_unit(generator));
         const amrex::Real pressure = std::pow(
            10.0, -3.0 + 5.0 * random_unit(generator)
         );
         return primitive(
            rho,
            pressure,
            gamma,
            0.5 * signed_random(),
            0.5 * signed_random(),
            0.5 * signed_random(),
            normal_magnetic,
            3.0 * signed_random(),
            3.0 * signed_random()
         );
      };
      const RMHDPrimitive left = make_primitive();
      const RMHDPrimitive right = make_primitive();
      const SRMHDRiemannSample hllc = srmhd_hllc_sample(
         left, right, gamma, 0.0
      );
      const SRMHDRiemannSample hlld = srmhd_hlld_sample(
         left, right, gamma, 0.0
      );
      if (!finite_srmhd_state(hllc.state) || !finite_srmhd_state(hllc.flux)
          || !finite_srmhd_state(hlld.state) || !finite_srmhd_state(hlld.flux)) {
         std::cerr << "random RMHD fan returned a nonfinite sample\n";
         return false;
      }

      const SRMHDState state_l = srmhd_state_from_primitive(left);
      const SRMHDState state_r = srmhd_state_from_primitive(right);
      const SRMHDState flux_l = srmhd_flux_from_primitive(left);
      const SRMHDState flux_r = srmhd_flux_from_primitive(right);
      const RMHDWaveSpeeds waves_l = ideal_srmhd_wave_speeds<0>(
         left, minkowski_spacetime_metric(), gamma
      );
      const RMHDWaveSpeeds waves_r = ideal_srmhd_wave_speeds<0>(
         right, minkowski_spacetime_metric(), gamma
      );
      const amrex::Real speed_l = amrex::min(waves_l.minus, waves_r.minus);
      const amrex::Real speed_r = amrex::max(waves_l.plus, waves_r.plus);
      const SRMHDHLLCStates hllc_fan = srmhd_hllc_intermediate_states(
         left, right, state_l, state_r, flux_l, flux_r, speed_l, speed_r
      );
      if (hllc_fan.valid) {
         maximum_hllc_contact_residual = amrex::max(
            maximum_hllc_contact_residual,
            relative_rh_residual(
               hllc_fan.state_l,
               hllc_fan.state_r,
               hllc_fan.flux_l,
               hllc_fan.flux_r,
               hllc_fan.contact_speed
            )
         );
         ++accepted_hllc;
      }
      if (hlld.used_hll) {
         ++hll_fallbacks;
         continue;
      }

      const SRMHDHLLDSolution solution = srmhd_hlld_pressure_solution(
         left,
         right,
         state_l,
         state_r,
         flux_l,
         flux_r,
         speed_l,
         speed_r,
         normal_magnetic
      );
      if (!solution.valid) {
         continue;
      }
      const SRMHDHLLDContactState contact_l = srmhd_hlld_contact_state(
         solution.evaluation.fast_l,
         solution.evaluation.contact_velocity_l,
         solution.evaluation.contact_magnetic,
         solution.total_pressure
      );
      const SRMHDHLLDContactState contact_r = srmhd_hlld_contact_state(
         solution.evaluation.fast_r,
         solution.evaluation.contact_velocity_r,
         solution.evaluation.contact_magnetic,
         solution.total_pressure
      );
      if (!contact_l.valid || !contact_r.valid) {
         continue;
      }
      const amrex::Real contact_speed = 0.5 * (
         solution.evaluation.contact_velocity_l.n
         + solution.evaluation.contact_velocity_r.n
      );
      maximum_contact_residual = amrex::max(
         maximum_contact_residual,
         relative_rh_residual(
            contact_l.state,
            contact_r.state,
            contact_l.flux,
            contact_r.flux,
            contact_speed
         )
      );
      ++accepted_hlld;
   }
   if (accepted_hllc < 100 || accepted_hlld < 100
       || maximum_hllc_contact_residual > 2.0e-6
       || maximum_contact_residual > 2.0e-6) {
      std::cerr << "random RMHD fan audit failed: accepted_hllc="
                << accepted_hllc << " HLLC_contact_RH="
                << maximum_hllc_contact_residual << " accepted_hlld="
                << accepted_hlld
                << " HLL_fallbacks=" << hll_fallbacks
                << " HLLD_contact_RH=" << maximum_contact_residual << '\n';
      return false;
   }
   return true;
}

} // namespace

int main(int argc, char** argv)
{
   if (argc == 2) {
      const std::string mode = argv[1];
      if (mode == "--flux-table") {
         return run_flux_table();
      }
      if (mode == "--primitive-table") {
         return run_primitive_table();
      }
      if (mode == "--c2p-table") {
         return run_c2p_table();
      }
      if (mode == "--hlld-fan-table") {
         return run_hlld_fan_table();
      }
      if (mode == "--hlld-evaluation-table") {
         return run_hlld_evaluation_table();
      }
   }
   if (argc != 1) {
      std::cerr << "usage: " << argv[0]
                << " [--flux-table|--primitive-table|--c2p-table"
                << "|--hlld-fan-table|--hlld-evaluation-table]\n";
      return 2;
   }
   bool passed = true;
   passed = validate_hll_algebra() && passed;
   passed = validate_frame<0>() && passed;
   passed = validate_frame<1>() && passed;
   passed = validate_frame<2>() && passed;
   passed = validate_identical_curved_flux<0>() && passed;
   passed = validate_identical_curved_flux<1>() && passed;
   passed = validate_identical_curved_flux<2>() && passed;
   passed = validate_curved_hll_equivalence<0>() && passed;
   passed = validate_curved_hll_equivalence<1>() && passed;
   passed = validate_curved_hll_equivalence<2>() && passed;
   passed = validate_contact() && passed;
   passed = validate_rotational_wave() && passed;
   passed = validate_zero_normal_degeneracy() && passed;
   passed = validate_random_fans() && passed;
   if (!passed) {
      return 1;
   }
   std::cout << "RMHD HLLC/HLLD formula and tetrad checks passed\n";
   return 0;
}
