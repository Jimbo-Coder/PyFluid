#include "pyfluid_amrex/multipoles.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/generated/diagnostic_catalog.hpp"

namespace pyfluid_amrex {
namespace {

amrex::Real factorial_ratio_log(int numerator, int denominator)
{
   return std::lgamma(static_cast<amrex::Real>(numerator) + 1.0)
      - std::lgamma(static_cast<amrex::Real>(denominator) + 1.0);
}

amrex::Real binomial(int n, int k)
{
   if (k < 0 || k > n) {
      return 0.0;
   }
   return std::exp(
      std::lgamma(static_cast<amrex::Real>(n) + 1.0)
      - std::lgamma(static_cast<amrex::Real>(k) + 1.0)
      - std::lgamma(static_cast<amrex::Real>(n - k) + 1.0)
   );
}

amrex::Real parity_sign(int exponent)
{
   return exponent % 2 == 0 ? 1.0 : -1.0;
}

void validate_complex_samples(
   ParametricSurface const& surface,
   std::vector<amrex::Real> const& real_values,
   std::vector<amrex::Real> const& imaginary_values
)
{
   if (real_values.size() != surface.points().size()
       || imaginary_values.size() != surface.points().size()) {
      throw std::invalid_argument(
         "complex surface samples must contain one real and imaginary value per point"
      );
   }
}

} // namespace

std::complex<amrex::Real> spin_weighted_spherical_harmonic(
   int spin_weight,
   int ell,
   int m,
   amrex::Real theta,
   amrex::Real phi
)
{
   if (ell < 0 || std::abs(spin_weight) > ell || std::abs(m) > ell) {
      throw std::invalid_argument("spin-weighted harmonic requires ell >= |s| and ell >= |m|");
   }
   const amrex::Real log_normalization = 0.5 * (
      factorial_ratio_log(ell + m, ell + spin_weight)
      + factorial_ratio_log(ell - m, ell - spin_weight)
      + std::log((2.0 * ell + 1.0) / (4.0 * constants::pi))
   );
   const amrex::Real normalization = parity_sign(m) * std::exp(log_normalization);
   const amrex::Real cosine = std::cos(0.5 * theta);
   const amrex::Real sine = std::sin(0.5 * theta);
   amrex::Real sum = 0.0;
   const int first = std::max(m - spin_weight, 0);
   const int last = std::min(ell + m, ell - spin_weight);
   for (int index = first; index <= last; ++index) {
      sum += parity_sign(ell - index - spin_weight)
         * binomial(ell - spin_weight, index)
         * binomial(ell + spin_weight, index + spin_weight - m)
         * std::pow(cosine, 2 * index + spin_weight - m)
         * std::pow(sine, 2 * (ell - index) + m - spin_weight);
   }
   return normalization * sum * std::complex<amrex::Real>(
      std::cos(m * phi), std::sin(m * phi)
   );
}

std::vector<MultipoleMode> decompose_spin_weighted_field(
   ParametricSurface const& surface,
   std::vector<amrex::Real> const& real_values,
   std::vector<amrex::Real> const& imaginary_values,
   std::string field,
   int spin_weight,
   int ell_min,
   int ell_max
)
{
   validate_complex_samples(surface, real_values, imaginary_values);
   ell_min = std::max(ell_min, std::abs(spin_weight));
   if (ell_max < ell_min) {
      throw std::invalid_argument("multipole ell range is empty");
   }
   std::vector<MultipoleMode> modes;
   constexpr ComplexModeDiagnosticInfo mode_info = ComplexModeDiagnosticCatalog[
      static_cast<std::size_t>(
         ComplexModeDiagnosticId::Psi4SpinWeightedSurface
      )
   ];
   for (int ell = ell_min; ell <= ell_max; ++ell) {
      for (int m = -ell; m <= ell; ++m) {
         std::complex<amrex::Real> integral{};
         for (std::size_t point = 0; point < surface.points().size(); ++point) {
            const SurfaceQuadraturePoint& quadrature = surface.points()[point];
            const auto harmonic = spin_weighted_spherical_harmonic(
               spin_weight, ell, m, quadrature.theta, quadrature.phi
            );
            integral += quadrature.solid_angle_weight * std::conj(harmonic)
               * std::complex<amrex::Real>(real_values[point], imaginary_values[point]);
         }
         modes.push_back({
            field,
            std::string(mode_info.basis),
            spin_weight,
            ell,
            m,
            integral,
            surface.radius(),
         });
      }
   }
   return modes;
}

std::vector<MultipoleMode> decompose_azimuthal_field(
   ParametricSurface const& surface,
   std::vector<amrex::Real> const& values,
   std::string field,
   int m_max
)
{
   if (values.size() != surface.points().size()) {
      throw std::invalid_argument("azimuthal surface samples must contain one value per point");
   }
   if (m_max < 0) {
      throw std::invalid_argument("azimuthal mode maximum must be non-negative");
   }
   std::vector<MultipoleMode> modes;
   constexpr ComplexModeDiagnosticInfo mode_info = ComplexModeDiagnosticCatalog[
      static_cast<std::size_t>(
         ComplexModeDiagnosticId::ScalarAzimuthalSurface
      )
   ];
   modes.reserve(static_cast<std::size_t>(m_max + 1));
   const amrex::Real radius_squared = surface.radius() * surface.radius();
   for (int m = 0; m <= m_max; ++m) {
      std::complex<amrex::Real> integral{};
      for (std::size_t point = 0; point < surface.points().size(); ++point) {
         const SurfaceQuadraturePoint& quadrature = surface.points()[point];
         const std::complex<amrex::Real> phase(
            std::cos(m * quadrature.phi), -std::sin(m * quadrature.phi)
         );
         integral += radius_squared * quadrature.solid_angle_weight
            * values[point] * phase;
      }
      modes.push_back({
         field,
         std::string(mode_info.basis),
         0,
         std::nullopt,
         m,
         integral,
         surface.radius(),
      });
   }
   return modes;
}

amrex::Real surface_complex_l2(
   ParametricSurface const& surface,
   std::vector<amrex::Real> const& real_values,
   std::vector<amrex::Real> const& imaginary_values
)
{
   validate_complex_samples(surface, real_values, imaginary_values);
   amrex::Real norm_squared = 0.0;
   for (std::size_t point = 0; point < surface.points().size(); ++point) {
      norm_squared += surface.points()[point].solid_angle_weight
         * (real_values[point] * real_values[point]
            + imaginary_values[point] * imaginary_values[point]);
   }
   return std::sqrt(norm_squared);
}

void write_multipole_header(
   std::ofstream& output,
   int step,
   amrex::Real time
)
{
   output << "# diagnostic=multipoles start_step=" << step
          << " start_time=" << std::setprecision(17) << time << '\n';
   output << "step\ttime\tradius\tfield\tbasis\tspin_weight\tell\tm\treal\timag\n";
}

void append_multipole_rows(
   std::ofstream& output,
   int step,
   amrex::Real time,
   std::vector<MultipoleMode> const& modes
)
{
   for (MultipoleMode const& mode : modes) {
      output << step << '\t' << std::setprecision(17) << time << '\t';
      if (mode.radius.has_value()) {
         output << *mode.radius;
      } else {
         output << "none";
      }
      output << '\t' << mode.field << '\t' << mode.basis << '\t'
             << mode.spin_weight << '\t';
      if (mode.ell.has_value()) {
         output << *mode.ell;
      } else {
         output << "none";
      }
      output << '\t' << mode.m << '\t' << mode.value.real() << '\t'
             << mode.value.imag() << '\n';
   }
}

} // namespace pyfluid_amrex
