#include "relativistic_vacuum_check.hpp"

#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/flux.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/sources.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "relativity/rhd_system.hpp"
#include "relativity/relativistic_atmosphere.hpp"

#include <array>
#include <cmath>
#include <iostream>

namespace {

using namespace amrex::literals;

bool zero_matter(pyfluid_amrex::RHDConserved const& state)
{
   return state.dens == 0.0_rt && state.mom1 == 0.0_rt && state.mom2 == 0.0_rt
      && state.mom3 == 0.0_rt && state.tau == 0.0_rt;
}

bool zero_matter(pyfluid_amrex::RMHDConserved const& state)
{
   return zero_matter(pyfluid_amrex::RHDConserved{
      state.dens, state.mom1, state.mom2, state.mom3, state.tau
   }) && state.b1 == 0.0_rt && state.b2 == 0.0_rt && state.b3 == 0.0_rt;
}

bool zero_matter(pyfluid_amrex::StressEnergyTensor const& state)
{
   return state.t00 == 0.0_rt && state.t01 == 0.0_rt && state.t02 == 0.0_rt
      && state.t03 == 0.0_rt && state.t11 == 0.0_rt && state.t12 == 0.0_rt
      && state.t13 == 0.0_rt && state.t22 == 0.0_rt && state.t23 == 0.0_rt
      && state.t33 == 0.0_rt;
}

bool zero_matter(pyfluid_amrex::EulerianMatter const& state)
{
   return state.rho == 0.0_rt && state.momentum1 == 0.0_rt
      && state.momentum2 == 0.0_rt && state.momentum3 == 0.0_rt
      && state.stress11 == 0.0_rt && state.stress12 == 0.0_rt
      && state.stress13 == 0.0_rt && state.stress22 == 0.0_rt
      && state.stress23 == 0.0_rt && state.stress33 == 0.0_rt;
}

bool finite_matter(pyfluid_amrex::RHDConserved const& state)
{
   return std::isfinite(state.dens) && std::isfinite(state.mom1)
      && std::isfinite(state.mom2) && std::isfinite(state.mom3)
      && std::isfinite(state.tau);
}

} // namespace

bool validate_relativistic_vacuum_algebra()
{
   using namespace pyfluid_amrex;
   std::array<FixedSpacetimeParameters, 3> parameters{};
   parameters[1].metric = FixedSpacetimeMetric::Rindler;
   parameters[1].acceleration1 = 0.03_rt;
   parameters[1].acceleration2 = -0.02_rt;
   parameters[2].metric = FixedSpacetimeMetric::KerrSchild;
   parameters[2].mass = 1.0_rt;
   parameters[2].dimensionless_spin3 = 0.7_rt;
   const std::array<std::array<amrex::Real, 3>, 2> velocities{{
      {0.0_rt, 0.0_rt, 0.0_rt}, {0.1_rt, -0.08_rt, 0.03_rt},
   }};
   int checked = 0;
   for (auto const& spacetime : parameters) {
      const auto geometry = fixed_spacetime_geometry(
         spacetime, 4.0_rt, 2.0_rt, 1.0_rt, 0.0_rt
      );
      const auto& metric = geometry.metric;
      for (auto const& velocity : velocities) {
         for (amrex::Real eps : {0.0_rt, 0.3_rt}) {
            // Specific energy/velocity are undefined in vacuum. Any finite,
            // causal representatives must give exactly zero matter tensors.
            const RHDPrimitive fluid{
               0.0_rt, eps, 0.0_rt, velocity[0], velocity[1], velocity[2]
            };
            const RMHDPrimitive unmagnetized{
               fluid.rho, fluid.eps, fluid.pressure,
               fluid.v1, fluid.v2, fluid.v3, 0.0_rt, 0.0_rt, 0.0_rt
            };
            const std::array<bool, 14> checks{{
               zero_matter(generated::rhd_primitive_to_conserved(fluid, metric)),
               zero_matter(generated::rhd_flux<0>(fluid, metric)),
               zero_matter(generated::rhd_flux<1>(fluid, metric)),
               zero_matter(generated::rhd_flux<2>(fluid, metric)),
               zero_matter(generated::rhd_stress_energy(fluid, metric)),
               zero_matter(generated::rhd_eulerian_matter(fluid, metric)),
               zero_matter(generated::rhd_geometric_source(
                  fluid, metric, geometry.derivatives)),
               zero_matter(generated::rmhd_primitive_to_conserved(unmagnetized, metric)),
               zero_matter(generated::rmhd_flux<0>(unmagnetized, metric)),
               zero_matter(generated::rmhd_flux<1>(unmagnetized, metric)),
               zero_matter(generated::rmhd_flux<2>(unmagnetized, metric)),
               zero_matter(generated::rmhd_stress_energy(unmagnetized, metric)),
               zero_matter(generated::rmhd_eulerian_matter(unmagnetized, metric)),
               zero_matter(generated::rmhd_geometric_source(
                  unmagnetized, metric, geometry.derivatives)),
            }};
            for (std::size_t index = 0; index < checks.size(); ++index) {
               if (!checks[index]) {
                  std::cerr << "zero-matter Valencia algebra failed sample="
                            << checked << " kernel=" << index << '\n';
                  return false;
               }
            }
            ++checked;
         }
      }
   }
   EOSParams eos;
   eos.method = EOSIdealGas;
   eos.gamma = 5.0_rt / 3.0_rt;
   const RHDConserved vacuum{};
   const RHDRecoveryResult recovered_ideal = recover_rhd<EOSIdealGas>(
      vacuum, minkowski_spacetime_metric(), eos
   );
   const RHDRecoveryResult recovered_runtime = recover_rhd<EOSRuntime>(
      vacuum, minkowski_spacetime_metric(), eos
   );
   if (!recovered_ideal.attempt.succeeded()
       || !recovered_runtime.attempt.succeeded()
       || !canonical_rhd_vacuum(recovered_ideal.primitive)
       || !canonical_rhd_vacuum(recovered_runtime.primitive)) {
      std::cerr << "canonical vacuum did not recover to the canonical "
                   "zero primitive\n";
      return false;
   }
   const RHDRecoveryResult invalid_nonzero = recover_rhd<EOSIdealGas>(
      {0.0_rt, 1.0_rt, 0.0_rt, 0.0_rt, 0.0_rt}, minkowski_spacetime_metric(), eos
   );
   if (invalid_nonzero.attempt.succeeded()) {
      std::cerr << "invalid nonzero matter was misclassified as vacuum\n";
      return false;
   }
   // Cold positive-density matter belongs to the physical state space even
   // though it has zero pressure. It must not be erased as canonical vacuum.
   // Exercise the same round trip used by accepted-state validation, including
   // the very small density exposed at a spherical coarse-fine vacuum edge.
   for (const amrex::Real density : {amrex::Real(1.0_rt), amrex::Real(1.0e-21_rt), amrex::Real(1.0e-90_rt)}) {
      const RHDConserved cold{density, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
      const auto metric = minkowski_spacetime_metric();
      const auto recovered = recover_rhd<EOSRuntime>(cold, metric, eos);
      const auto reconstructed = generated::rhd_primitive_to_conserved(
         recovered.primitive, metric
      );
      if (!recovered.attempt.succeeded()
          || !admissible_rhd_primitive(recovered.primitive, metric)
          || canonical_rhd_vacuum(recovered.primitive)
          || reconstructed.dens != density || reconstructed.tau != 0.0_rt) {
         std::cerr << "cold matter failed the shared physical-state contract: D="
                   << density << " status=" << static_cast<int>(recovered.attempt.status)
                   << " eps=" << recovered.primitive.eps
                   << " reconstructed_tau=" << reconstructed.tau << "\n";
         return false;
      }
      const auto negative_energy = recover_rhd<EOSRuntime>(
         {density, 0.0_rt, 0.0_rt, 0.0_rt, -0.1_rt * density}, metric, eos
      );
      if (negative_energy.attempt.succeeded()) {
         std::cerr << "negative internal energy passed cold-state recovery\n";
         return false;
      }
   }
   if (relativistic_specific_internal_energy_from_pressure<EOSIdealGas>(
          0.0_rt, 0.0_rt, eos
       ) != 0.0_rt
       || relativistic_specific_enthalpy<EOSIdealGas>(0.0_rt, 0.0_rt, eos) != 1.0_rt
       || relativistic_sound_speed_squared<EOSIdealGas>(0.0_rt, 0.0_rt, eos)
          != 0.0_rt) {
      std::cerr << "canonical vacuum EoS extension is not exact\n";
      return false;
   }
   const RHDPrimitive matter{1.0_rt, 0.3_rt, 0.2_rt, 0.0_rt, 0.0_rt, 0.0_rt};
   const RHDPrimitive empty{};
   const RHDConserved llf_expansion = rhd_llf_flux<0, EOSIdealGas>(
      matter, empty, eos, minkowski_spacetime_metric()
   );
   const RHDConserved hll_expansion = rhd_hll_flux<0, EOSIdealGas>(
      matter, empty, eos, minkowski_spacetime_metric()
   );
   const RHDConserved empty_flux = rhd_llf_flux<0, EOSIdealGas>(
      empty, empty, eos, minkowski_spacetime_metric()
   );
   if (!finite_matter(llf_expansion) || !finite_matter(hll_expansion)
       || !(llf_expansion.dens > 0.0_rt) || !(hll_expansion.dens > 0.0_rt)
       || !zero_matter(empty_flux)) {
      std::cerr << "matter-vacuum Riemann flux is not finite and causal\n";
      return false;
   }
   std::array<amrex::Real, NVAR_HYDRO> projected_storage{};
   const amrex::Array4<amrex::Real> projected(
      projected_storage.data(),
      amrex::Dim3{0, 0, 0},
      amrex::Dim3{1, 1, 1},
      NVAR_HYDRO
   );
   const RHDConserved matter_conserved = generated::rhd_primitive_to_conserved(
      matter, minkowski_spacetime_metric()
   );
   store_rhd_conserved(projected, 0, 0, 0, matter_conserved);
   RelativisticRecoveryCellEvents projection_events;
   const auto projected_recovery = recover_rhd_with_vacuum_exterior<
      EOSRuntime
   >(
      projected,
      0,
      0,
      0,
      eos,
      minkowski_spacetime_metric(),
      &projection_events
   );
   const RHDConserved projected_matter = load_rhd_conserved(
      projected, 0, 0, 0
   );
   if (!projected_recovery.attempt.succeeded()
       || canonical_rhd_vacuum(projected_matter)
       || projection_events.atmosphere_reset_count != 0
       || std::abs(projected_matter.dens - matter_conserved.dens) > 1.0e-14_rt
       || std::abs(projected_matter.tau - matter_conserved.tau) > 1.0e-14_rt) {
      std::cerr << "vacuum exterior projection erased valid matter: D="
                << projected_matter.dens << " tau=" << projected_matter.tau
                << " status="
                << static_cast<int>(projected_recovery.attempt.status)
                << " resets=" << projection_events.atmosphere_reset_count
                << '\n';
      return false;
   }
   std::cout << "zero-matter Valencia algebra: " << checked
             << " samples, 14 kernels each, exact zero; recovery/EoS/"
                "one-sided flux checks passed\n";
   return true;
}
