#include "pyfluid_amrex/kernels_generated/eos.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/c2p_rmhd.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"

#include <AMReX_REAL.H>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

using pyfluid_amrex::RMHDConserved;
using pyfluid_amrex::RMHDPrimitive;
using pyfluid_amrex::RMHDRecoveryResult;

constexpr amrex::Real gamma = 1.4;
constexpr pyfluid_amrex::ColdPolytropeEquationOfState cold_eos{0.2, 1.8};
constexpr std::size_t corpus_size = 4096;
constexpr int timed_repetitions = 32;

struct Corpus {
   std::array<RMHDConserved, corpus_size> conserved{};
   std::array<RMHDPrimitive, corpus_size> seeds{};
   std::array<pyfluid_amrex::RMHDEntropyInput, corpus_size> entropy{};
};

Corpus make_corpus(bool stress, bool cold = false)
{
   Corpus corpus{};
   const auto metric = pyfluid_amrex::minkowski_spacetime_metric();
   for (std::size_t index = 0; index < corpus.conserved.size(); ++index) {
      const amrex::Real phase = static_cast<amrex::Real>(index);
      const amrex::Real maximum_speed = stress ? 0.9995 : 0.93;
      const amrex::Real speed = 0.02 + (maximum_speed - 0.02) * (
         0.5 + 0.5 * std::sin(0.017 * phase)
      );
      const amrex::Real direction1 = std::sin(0.031 * phase);
      const amrex::Real direction2 = std::cos(0.047 * phase);
      const amrex::Real direction3 = std::sin(0.071 * phase + 0.3);
      const amrex::Real direction_norm = std::sqrt(
         direction1 * direction1 + direction2 * direction2
         + direction3 * direction3
      );
      RMHDPrimitive primitive{};
      const amrex::Real density_log_min = stress ? -9.0 : -4.0;
      const amrex::Real density_log_span = stress ? 15.0 : 7.0;
      const amrex::Real energy_log_min = stress ? -11.0 : -7.0;
      const amrex::Real energy_log_span = stress ? 18.0 : 10.0;
      primitive.rho = std::exp(
         density_log_min
         + density_log_span * (0.5 + 0.5 * std::sin(0.013 * phase))
      );
      if (cold) {
         primitive.pressure = cold_eos.polytropic_k * std::pow(
            primitive.rho, cold_eos.polytropic_gamma
         );
         primitive.eps = primitive.pressure / (
            primitive.rho * (cold_eos.polytropic_gamma - 1.0)
         );
      } else {
         primitive.eps = std::exp(
            energy_log_min
            + energy_log_span * (0.5 + 0.5 * std::cos(0.019 * phase))
         );
         primitive.pressure =
            pyfluid_amrex::generated::relativistic_ideal_gas_pressure(
               primitive.rho, primitive.eps, gamma
            );
      }
      primitive.v1 = speed * direction1 / direction_norm;
      primitive.v2 = speed * direction2 / direction_norm;
      primitive.v3 = speed * direction3 / direction_norm;
      const amrex::Real magnetic_scale = std::exp(
         (stress ? -8.0 : -5.0)
         + (stress ? 16.0 : 8.0)
            * (0.5 + 0.5 * std::sin(0.023 * phase + 0.7))
      );
      primitive.b1 = magnetic_scale * std::cos(0.029 * phase);
      primitive.b2 = magnetic_scale * std::sin(0.037 * phase);
      primitive.b3 = magnetic_scale * std::cos(0.053 * phase + 0.2);
      corpus.conserved[index] = pyfluid_amrex::generated::rmhd_primitive_to_conserved(
         primitive, metric
      );
      const amrex::Real velocity_squared = primitive.v1 * primitive.v1
         + primitive.v2 * primitive.v2 + primitive.v3 * primitive.v3;
      const amrex::Real lorentz_factor = 1.0 / std::sqrt(
         1.0 - velocity_squared
      );
      corpus.entropy[index].conserved_entropy = lorentz_factor
         * primitive.pressure / std::pow(primitive.rho, gamma - 1.0);
      RMHDPrimitive seed = primitive;
      seed.rho *= 1.01;
      seed.eps *= 1.02;
      seed.pressure = cold
         ? cold_eos.polytropic_k
            * std::pow(seed.rho, cold_eos.polytropic_gamma)
         : pyfluid_amrex::generated::relativistic_ideal_gas_pressure(
              seed.rho, seed.eps, gamma
           );
      seed.v1 *= 0.99;
      seed.v2 *= 0.99;
      seed.v3 *= 0.99;
      corpus.seeds[index] = seed;
   }
   return corpus;
}

template <typename Recover>
void benchmark_candidate(
   std::string_view name,
   const Corpus& corpus,
   Recover&& recover
)
{
   std::size_t successes = 0;
   std::size_t iterations = 0;
   std::array<std::size_t, 8> status_counts{};
   amrex::Real max_residual = 0.0;
   amrex::Real max_conserved_residual = 0.0;
   amrex::Real checksum = 0.0;
   std::size_t failure_reports = 0;
   for (std::size_t index = 0; index < corpus.conserved.size(); ++index) {
      const auto& conserved = corpus.conserved[index];
      const RMHDRecoveryResult result = recover(index, conserved);
      successes += static_cast<std::size_t>(result.succeeded());
      ++status_counts[static_cast<std::size_t>(result.attempt.status)];
      iterations += static_cast<std::size_t>(result.attempt.iterations);
      max_residual = std::max(max_residual, result.attempt.residual);
      if (result.succeeded()) {
         max_conserved_residual = std::max(
            max_conserved_residual,
            pyfluid_amrex::rmhd_conserved_residual(result.primitive, conserved)
         );
      }
      checksum += result.primitive.rho + result.primitive.pressure;
      if (!result.succeeded() && failure_reports < 3) {
         std::cerr << name << " failure sample=" << index
                   << " status=" << static_cast<int>(result.attempt.status)
                   << " iterations=" << result.attempt.iterations
                   << " residual=" << result.attempt.residual
                   << " dens=" << conserved.dens << " tau=" << conserved.tau
                   << " mom=(" << conserved.mom1 << ',' << conserved.mom2 << ','
                   << conserved.mom3 << ") b=(" << conserved.b1 << ','
                   << conserved.b2 << ',' << conserved.b3 << ") primitive=("
                   << result.primitive.rho << ',' << result.primitive.eps << ','
                   << result.primitive.pressure << ',' << result.primitive.v1 << ','
                   << result.primitive.v2 << ',' << result.primitive.v3 << ")\n";
         ++failure_reports;
      }
   }

   const auto start = std::chrono::steady_clock::now();
   for (int repetition = 0; repetition < timed_repetitions; ++repetition) {
      for (std::size_t index = 0; index < corpus.conserved.size(); ++index) {
         const RMHDRecoveryResult result = recover(index, corpus.conserved[index]);
         checksum += result.primitive.rho + result.primitive.pressure;
      }
   }
   const auto stop = std::chrono::steady_clock::now();
   const double seconds = std::chrono::duration<double>(stop - start).count();
   const double recoveries = static_cast<double>(corpus.conserved.size()) * timed_repetitions;
   const double million_recoveries_per_second = recoveries / seconds / 1.0e6;
   const double mean_iterations = static_cast<double>(iterations) / corpus.conserved.size();

   std::cout << name << '\t' << successes << '\t' << corpus.conserved.size() << '\t'
             << std::setprecision(10) << mean_iterations << '\t' << max_residual
             << '\t' << max_conserved_residual << '\t'
             << million_recoveries_per_second << '\t' << checksum;
   for (const std::size_t count : status_counts) {
      std::cout << '\t' << count;
   }
   std::cout << '\n';
}

void report_stress_complementarity(const Corpus& corpus)
{
   bool reported_reprimand_only = false;
   bool reported_palenzuela_only = false;
   for (std::size_t index = 0; index < corpus.conserved.size(); ++index) {
      const auto& conserved = corpus.conserved[index];
      const auto palenzuela = pyfluid_amrex::recover_srmhd_palenzuela_1d<true>(
         conserved, gamma
      );
      const auto reprimand = pyfluid_amrex::recover_srmhd_reprimand<true>(
         conserved, gamma
      );
      if (!reported_reprimand_only && !palenzuela.succeeded()
          && reprimand.succeeded()) {
         std::cout << std::setprecision(17)
                   << "reprimand_only_stress_sample\t" << index << '\t'
                   << conserved.dens << '\t' << conserved.mom1 << '\t'
                   << conserved.mom2 << '\t' << conserved.mom3 << '\t'
                   << conserved.tau << '\t' << conserved.b1 << '\t'
                   << conserved.b2 << '\t' << conserved.b3 << '\n';
         reported_reprimand_only = true;
      }
      if (!reported_palenzuela_only && palenzuela.succeeded()
          && !reprimand.succeeded()) {
         std::cout << std::setprecision(17)
                   << "palenzuela_only_stress_sample\t" << index << '\t'
                   << conserved.dens << '\t' << conserved.mom1 << '\t'
                   << conserved.mom2 << '\t' << conserved.mom3 << '\t'
                   << conserved.tau << '\t' << conserved.b1 << '\t'
                   << conserved.b2 << '\t' << conserved.b3 << '\n';
         reported_palenzuela_only = true;
      }
      if (reported_reprimand_only && reported_palenzuela_only) {
         return;
      }
   }
   if (!reported_reprimand_only) {
      std::cout << "reprimand_only_stress_sample\tnone\n";
   }
   if (!reported_palenzuela_only) {
      std::cout << "palenzuela_only_stress_sample\tnone\n";
   }
}

} // namespace

int main()
{
   const auto regular_corpus = make_corpus(false);
   const auto stress_corpus = make_corpus(true);
   const auto cold_corpus = make_corpus(false, true);
   const auto cold_stress_corpus = make_corpus(true, true);
   pyfluid_amrex::EOSParams ideal_eos{};
   ideal_eos.method = pyfluid_amrex::EOSIdealGas;
   ideal_eos.gamma = gamma;
   std::cout << "method\tsuccesses\ttotal\tmean_iterations\tmax_root_residual"
                "\tmax_conserved_residual"
                "\tmillion_recoveries_per_second\tchecksum\tsuccess\tinvalid_conserved"
                "\tinvalid_eos\tno_bracket\tnonfinite\tdid_not_converge"
                "\tunphysical\tconserved_mismatch\n";
   benchmark_candidate("palenzuela_1d", regular_corpus, [](std::size_t, const RMHDConserved& conserved) {
      return pyfluid_amrex::recover_srmhd_palenzuela_1d(conserved, gamma);
   });
   benchmark_candidate("reprimand", regular_corpus, [](std::size_t, const RMHDConserved& conserved) {
      return pyfluid_amrex::recover_srmhd_reprimand(conserved, gamma);
   });
   benchmark_candidate("noble_2d", regular_corpus, [&regular_corpus](std::size_t index, const RMHDConserved& conserved) {
      return pyfluid_amrex::recover_srmhd_noble_2d(
         conserved, regular_corpus.seeds[index], gamma
      );
   });
   benchmark_candidate("palenzuela_1d_entropy", regular_corpus, [&regular_corpus](std::size_t index, const RMHDConserved& conserved) {
      return pyfluid_amrex::recover_srmhd_palenzuela_1d_entropy(
         conserved, regular_corpus.entropy[index], gamma
      );
   });
   benchmark_candidate("noble_1d_entropy", regular_corpus, [&regular_corpus](std::size_t index, const RMHDConserved& conserved) {
      return pyfluid_amrex::recover_srmhd_noble_1d_entropy(
         conserved,
         regular_corpus.seeds[index],
         regular_corpus.entropy[index],
         gamma
      );
   });
   benchmark_candidate("font_1d_cold", cold_corpus, [](std::size_t, const RMHDConserved& conserved) {
      return pyfluid_amrex::recover_srmhd_font_1d(conserved, cold_eos);
   });
   benchmark_candidate("ordered_policy", regular_corpus, [](std::size_t, const RMHDConserved& conserved) {
      return pyfluid_amrex::recover_srmhd(conserved, gamma);
   });
   benchmark_candidate(
      "runtime_eos_generic",
      regular_corpus,
      [&ideal_eos](std::size_t, const RMHDConserved& conserved) {
         return pyfluid_amrex::recover_srmhd_runtime_energy(
            conserved,
            pyfluid_amrex::minkowski_spacetime_metric(),
            ideal_eos
         );
      }
   );
   benchmark_candidate(
      "runtime_eos_selected",
      regular_corpus,
      [&ideal_eos](std::size_t, const RMHDConserved& conserved) {
         return pyfluid_amrex::recover_srmhd(
            conserved,
            pyfluid_amrex::minkowski_spacetime_metric(),
            ideal_eos
         );
      }
   );
   benchmark_candidate("palenzuela_1d_stress", stress_corpus, [](std::size_t, const RMHDConserved& conserved) {
      return pyfluid_amrex::recover_srmhd_palenzuela_1d(conserved, gamma);
   });
   benchmark_candidate("reprimand_stress", stress_corpus, [](std::size_t, const RMHDConserved& conserved) {
      return pyfluid_amrex::recover_srmhd_reprimand(conserved, gamma);
   });
   benchmark_candidate("noble_2d_stress", stress_corpus, [&stress_corpus](std::size_t index, const RMHDConserved& conserved) {
      return pyfluid_amrex::recover_srmhd_noble_2d(
         conserved, stress_corpus.seeds[index], gamma
      );
   });
   benchmark_candidate("palenzuela_1d_entropy_stress", stress_corpus, [&stress_corpus](std::size_t index, const RMHDConserved& conserved) {
      return pyfluid_amrex::recover_srmhd_palenzuela_1d_entropy(
         conserved, stress_corpus.entropy[index], gamma
      );
   });
   benchmark_candidate("noble_1d_entropy_stress", stress_corpus, [&stress_corpus](std::size_t index, const RMHDConserved& conserved) {
      return pyfluid_amrex::recover_srmhd_noble_1d_entropy(
         conserved,
         stress_corpus.seeds[index],
         stress_corpus.entropy[index],
         gamma
      );
   });
   benchmark_candidate("font_1d_cold_stress", cold_stress_corpus, [](std::size_t, const RMHDConserved& conserved) {
      return pyfluid_amrex::recover_srmhd_font_1d(conserved, cold_eos);
   });
   report_stress_complementarity(stress_corpus);
   return 0;
}
