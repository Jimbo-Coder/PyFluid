#include "pyfluid_amrex/eos/tabulated.hpp"

#include <AMReX_Gpu.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace pyfluid_amrex::eos {
namespace {

constexpr amrex::Real gravitational_constant_cgs = 6.67430e-8;
constexpr amrex::Real speed_of_light_cgs = 2.99792458e10;
constexpr amrex::Real solar_mass_parameter_cgs = 1.3271244e26;
constexpr amrex::Real solar_mass_length_cgs =
   solar_mass_parameter_cgs / (speed_of_light_cgs * speed_of_light_cgs);
constexpr amrex::Real pressure_cgs_to_code =
   gravitational_constant_cgs * solar_mass_length_cgs * solar_mass_length_cgs
   / (
      speed_of_light_cgs * speed_of_light_cgs
      * speed_of_light_cgs * speed_of_light_cgs
   );
constexpr amrex::Real mev_per_fm3_to_cgs = 1.602176634e33;
constexpr amrex::Real mev_per_fm3_to_code =
   mev_per_fm3_to_cgs * pressure_cgs_to_code;
constexpr amrex::Real causal_margin = 1.0e-8;
constexpr std::string_view compose_backend = "compose_cold_barotrope_v1";

struct IndexedGrid {
   int lower = 0;
   int upper = -1;
   std::vector<amrex::Real> values;
};

struct ComposeThermodynamicRow {
   amrex::Real pressure_per_baryon = 0.0;
   amrex::Real total_energy_offset = 0.0;
   bool present = false;
};

std::string trim(std::string_view text)
{
   const auto begin = text.find_first_not_of(" \t\r\n");
   if (begin == std::string_view::npos) {
      return {};
   }
   const auto end = text.find_last_not_of(" \t\r\n");
   return std::string(text.substr(begin, end - begin + 1));
}

std::map<std::string, std::string> read_manifest(
   std::filesystem::path const& path
)
{
   std::ifstream input(path);
   if (!input.good()) {
      throw std::runtime_error(
         "a CompOSE cold table requires the provenance manifest "
         + path.string()
      );
   }
   std::map<std::string, std::string> values;
   for (std::string line; std::getline(input, line);) {
      const std::string stripped = trim(line);
      if (stripped.empty() || stripped.front() == '#') {
         continue;
      }
      const auto equal = stripped.find('=');
      if (equal == std::string::npos) {
         throw std::runtime_error("invalid EoS manifest line: " + line);
      }
      const std::string key = trim(
         std::string_view(stripped).substr(0, equal)
      );
      std::string value = trim(
         std::string_view(stripped).substr(equal + 1)
      );
      if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
         value = value.substr(1, value.size() - 2);
      }
      if (key.empty() || !values.emplace(key, value).second) {
         throw std::runtime_error("invalid duplicate EoS manifest key: " + key);
      }
   }
   return values;
}

void require_manifest_value(
   std::map<std::string, std::string> const& manifest,
   std::string const& key,
   std::string const& expected
)
{
   const auto found = manifest.find(key);
   if (found == manifest.end() || found->second != expected) {
      throw std::runtime_error(
         "EoS manifest requires " + key + " = \"" + expected + "\""
      );
   }
}

std::string table_fingerprint(
   std::vector<amrex::Real> const& log_density,
   std::vector<amrex::Real> const& log_pressure,
   std::vector<amrex::Real> const& log_enthalpy,
   amrex::Real low_density_gamma,
   amrex::Real high_density_sound_speed_squared,
   amrex::Real baryon_mass_mev,
   int source_points
)
{
   std::ostringstream canonical;
   canonical << compose_backend << '|' << source_points << '|'
             << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
             << std::scientific << low_density_gamma << '|'
             << high_density_sound_speed_squared << '|' << baryon_mass_mev;
   for (std::size_t index = 0; index < log_density.size(); ++index) {
      canonical << '|' << log_density[index] << ',' << log_pressure[index]
                << ',' << log_enthalpy[index];
   }
   std::uint64_t hash = 14695981039346656037ULL;
   for (unsigned char byte : canonical.str()) {
      hash ^= byte;
      hash *= 1099511628211ULL;
   }
   std::ostringstream encoded;
   encoded << std::hex << std::setfill('0') << std::setw(16) << hash;
   return encoded.str();
}

IndexedGrid read_indexed_grid(std::filesystem::path const& path)
{
   std::ifstream input(path);
   if (!input.good()) {
      throw std::runtime_error("unable to open CompOSE grid file: " + path.string());
   }
   IndexedGrid grid;
   if (!(input >> grid.lower >> grid.upper) || grid.upper < grid.lower) {
      throw std::runtime_error("invalid CompOSE index range in " + path.string());
   }
   grid.values.resize(static_cast<std::size_t>(grid.upper - grid.lower + 1));
   for (amrex::Real& value : grid.values) {
      if (!(input >> value) || !std::isfinite(value)) {
         throw std::runtime_error("invalid CompOSE grid value in " + path.string());
      }
   }
   amrex::Real extra = 0.0;
   if (input >> extra) {
      throw std::runtime_error("unexpected extra CompOSE grid value in " + path.string());
   }
   return grid;
}

amrex::Real interval_sound_speed_squared(
   amrex::Real log_density_left,
   amrex::Real log_density_right,
   amrex::Real log_pressure_left,
   amrex::Real log_pressure_right,
   amrex::Real log_enthalpy_left,
   amrex::Real log_enthalpy_right,
   amrex::Real fraction
)
{
   const amrex::Real inverse_width = 1.0
      / (log_density_right - log_density_left);
   const amrex::Real pressure_slope =
      (log_pressure_right - log_pressure_left) * inverse_width;
   const amrex::Real enthalpy_slope =
      (log_enthalpy_right - log_enthalpy_left) * inverse_width;
   const amrex::Real log_density =
      log_density_left + fraction * (log_density_right - log_density_left);
   const amrex::Real log_pressure =
      log_pressure_left + fraction * (log_pressure_right - log_pressure_left);
   const amrex::Real log_enthalpy =
      log_enthalpy_left + fraction * (log_enthalpy_right - log_enthalpy_left);
   const amrex::Real density = std::exp(log_density);
   const amrex::Real pressure = std::exp(log_pressure);
   const amrex::Real enthalpy = std::exp(log_enthalpy);
   const amrex::Real numerator = pressure_slope * pressure;
   const amrex::Real denominator =
      density * enthalpy * (1.0 + enthalpy_slope) - numerator;
   return numerator / denominator;
}

int causal_prefix_size(
   std::vector<amrex::Real> const& log_density,
   std::vector<amrex::Real> const& log_pressure,
   std::vector<amrex::Real> const& log_enthalpy
)
{
   for (int right = 1; right < static_cast<int>(log_density.size()); ++right) {
      // Log-linear p and h make p/(rho h) monotone on an interval, and the
      // resulting c_s^2 is monotone in that ratio. Its extrema are therefore
      // attained at the two interval endpoints.
      for (int sample = 0; sample <= 1; ++sample) {
         const amrex::Real sound_speed_squared = interval_sound_speed_squared(
            log_density[right - 1],
            log_density[right],
            log_pressure[right - 1],
            log_pressure[right],
            log_enthalpy[right - 1],
            log_enthalpy[right],
            static_cast<amrex::Real>(sample)
         );
         if (!std::isfinite(sound_speed_squared)
             || sound_speed_squared < 0.0
             || sound_speed_squared >= 1.0) {
            return right;
         }
      }
   }
   return static_cast<int>(log_density.size());
}

} // namespace

TabulatedEOSStorage::TabulatedEOSStorage(
   std::string source_directory,
   std::vector<amrex::Real> log_density,
   std::vector<amrex::Real> log_pressure,
   std::vector<amrex::Real> log_specific_enthalpy,
   amrex::Real low_density_gamma,
   amrex::Real high_density_sound_speed_squared,
   amrex::Real baryon_mass_mev,
   int source_points
)
   : source_directory_(std::move(source_directory)),
     log_density_(log_density.size()),
     log_pressure_(log_pressure.size()),
     log_specific_enthalpy_(log_specific_enthalpy.size()),
     low_density_gamma_(low_density_gamma),
     high_density_sound_speed_squared_(high_density_sound_speed_squared),
     baryon_mass_mev_(baryon_mass_mev),
     source_points_(source_points)
{
   backend_ = std::string(compose_backend);
   fingerprint_ = table_fingerprint(
      log_density, log_pressure, log_specific_enthalpy,
      low_density_gamma, high_density_sound_speed_squared,
      baryon_mass_mev, source_points
   );
   amrex::Gpu::copy(
      amrex::Gpu::hostToDevice,
      log_density.begin(),
      log_density.end(),
      log_density_.begin()
   );
   amrex::Gpu::copy(
      amrex::Gpu::hostToDevice,
      log_pressure.begin(),
      log_pressure.end(),
      log_pressure_.begin()
   );
   amrex::Gpu::copy(
      amrex::Gpu::hostToDevice,
      log_specific_enthalpy.begin(),
      log_specific_enthalpy.end(),
      log_specific_enthalpy_.begin()
   );
   amrex::Gpu::streamSynchronize();
}

std::string const& TabulatedEOSStorage::backend() const noexcept
{
   return backend_;
}

std::string const& TabulatedEOSStorage::fingerprint() const noexcept
{
   return fingerprint_;
}

TabulatedEOSView TabulatedEOSStorage::view() const noexcept
{
   return {
      static_cast<int>(log_density_.size()),
      log_density_.data(),
      log_pressure_.data(),
      log_specific_enthalpy_.data(),
      low_density_gamma_,
      high_density_sound_speed_squared_,
   };
}

std::string const& TabulatedEOSStorage::source_directory() const noexcept
{
   return source_directory_;
}

amrex::Real TabulatedEOSStorage::baryon_mass_mev() const noexcept
{
   return baryon_mass_mev_;
}

int TabulatedEOSStorage::source_points() const noexcept
{
   return source_points_;
}

int TabulatedEOSStorage::retained_points() const noexcept
{
   return static_cast<int>(log_density_.size());
}

bool TabulatedEOSStorage::was_truncated_for_causality() const noexcept
{
   return retained_points() < source_points_;
}

std::shared_ptr<TabulatedEOSStorage const> load_compose_cold_table(
   std::string const& source_directory
)
{
   const std::filesystem::path directory = std::filesystem::absolute(
      std::filesystem::path(source_directory)
   ).lexically_normal();
   const auto manifest = read_manifest(directory / "table.toml");
   require_manifest_value(manifest, "schema_version", "1");
   require_manifest_value(
      manifest, "table_kind", "cold_beta_equilibrium"
   );
   require_manifest_value(
      manifest, "high_density_policy", "causal_extend"
   );
   require_manifest_value(
      manifest, "rest_mass_normalization", "matched_vacuum_enthalpy"
   );
   const IndexedGrid temperature_grid = read_indexed_grid(directory / "eos.t");
   const IndexedGrid charge_grid = read_indexed_grid(directory / "eos.yq");
   if (temperature_grid.values.size() != 1
       || charge_grid.values.size() != 1
       || std::abs(temperature_grid.values.front()) > 1.0e-14
       || std::abs(charge_grid.values.front()) > 1.0e-14) {
      throw std::runtime_error(
         "the cold beta-equilibrium reader requires singleton zero T/Yq axes"
      );
   }
   const IndexedGrid density_grid = read_indexed_grid(directory / "eos.nb");
   if (density_grid.values.size() < 2) {
      throw std::runtime_error("a cold CompOSE table requires at least two density points");
   }
   for (std::size_t index = 0; index < density_grid.values.size(); ++index) {
      if (!(density_grid.values[index] > 0.0)
          || (index > 0
              && !(density_grid.values[index] > density_grid.values[index - 1]))) {
         throw std::runtime_error("CompOSE baryon densities must be positive and increasing");
      }
   }

   std::ifstream thermo_input(directory / "eos.thermo");
   if (!thermo_input.good()) {
      throw std::runtime_error(
         "unable to open CompOSE thermodynamic file: "
         + (directory / "eos.thermo").string()
      );
   }
   amrex::Real neutron_mass_mev = 0.0;
   amrex::Real proton_mass_mev = 0.0;
   int lepton_flag = 0;
   if (!(thermo_input >> neutron_mass_mev >> proton_mass_mev >> lepton_flag)
       || !(neutron_mass_mev > 0.0) || !(proton_mass_mev > 0.0)) {
      throw std::runtime_error("invalid CompOSE eos.thermo header");
   }
   std::string ignored_header_remainder;
   std::getline(thermo_input, ignored_header_remainder);

   std::vector<ComposeThermodynamicRow> rows(density_grid.values.size());
   std::string line;
   while (std::getline(thermo_input, line)) {
      if (line.find_first_not_of(" \t\r") == std::string::npos
          || line.find_first_not_of(" \t\r") == line.find('#')) {
         continue;
      }
      std::istringstream row_input(line);
      int temperature_index = 0;
      int density_index = 0;
      int charge_index = 0;
      std::array<amrex::Real, 7> quantities{};
      int additional_pairs = 0;
      if (!(row_input >> temperature_index >> density_index >> charge_index
            >> quantities[0] >> quantities[1] >> quantities[2]
            >> quantities[3] >> quantities[4] >> quantities[5]
            >> quantities[6] >> additional_pairs)) {
         throw std::runtime_error("invalid CompOSE eos.thermo row: " + line);
      }
      if (temperature_index != temperature_grid.lower
          || charge_index != charge_grid.lower || additional_pairs < 0) {
         throw std::runtime_error(
            "CompOSE cold row uses an unexpected T/Yq index or quantity count"
         );
      }
      for (int additional = 0; additional < additional_pairs; ++additional) {
         amrex::Real ignored_quantity = 0.0;
         if (!(row_input >> ignored_quantity)
             || !std::isfinite(ignored_quantity)) {
            throw std::runtime_error(
               "invalid additional CompOSE thermodynamic quantity"
            );
         }
      }
      std::string extra_token;
      if (row_input >> extra_token) {
         throw std::runtime_error(
            "unexpected trailing CompOSE thermodynamic data"
         );
      }
      const int offset = density_index - density_grid.lower;
      if (offset < 0 || offset >= static_cast<int>(rows.size())) {
         throw std::runtime_error("CompOSE eos.thermo density index is out of range");
      }
      if (rows[offset].present) {
         throw std::runtime_error(
            "only one cold beta-equilibrium row per density is supported"
         );
      }
      if (!std::isfinite(quantities[0]) || !std::isfinite(quantities[6])) {
         throw std::runtime_error("nonfinite required CompOSE thermodynamic quantity");
      }
      rows[offset] = {quantities[0], quantities[6], true};
   }
   if (std::any_of(rows.begin(), rows.end(), [](auto const& row) {
          return !row.present;
       })) {
      throw std::runtime_error("CompOSE eos.thermo does not cover every density index");
   }

   std::vector<amrex::Real> pressure_mev_fm3(rows.size());
   std::vector<amrex::Real> energy_mev_fm3(rows.size());
   for (std::size_t index = 0; index < rows.size(); ++index) {
      const amrex::Real baryon_density = density_grid.values[index];
      pressure_mev_fm3[index] = baryon_density * rows[index].pressure_per_baryon;
      energy_mev_fm3[index] = baryon_density * neutron_mass_mev
         * (1.0 + rows[index].total_energy_offset);
      if (!(pressure_mev_fm3[index] > 0.0)
          || !(energy_mev_fm3[index] > pressure_mev_fm3[index])) {
         throw std::runtime_error(
            "CompOSE cold pressure and total energy must be positive with e > p"
         );
      }
   }

   const amrex::Real low_density_gamma = std::log(
      pressure_mev_fm3[1] / pressure_mev_fm3[0]
   ) / std::log(density_grid.values[1] / density_grid.values[0]);
   if (!(low_density_gamma > 1.0) || !std::isfinite(low_density_gamma)) {
      throw std::runtime_error(
         "CompOSE low-density pressure does not admit a vacuum polytropic extension"
      );
   }
   const amrex::Real baryon_mass_mev = (
      energy_mev_fm3[0]
      - pressure_mev_fm3[0] / (low_density_gamma - 1.0)
   ) / density_grid.values[0];
   if (!(baryon_mass_mev > 0.0) || !std::isfinite(baryon_mass_mev)) {
      throw std::runtime_error("invalid CompOSE baryon-mass normalization");
   }

   std::vector<amrex::Real> log_density(rows.size());
   std::vector<amrex::Real> log_pressure(rows.size());
   std::vector<amrex::Real> log_specific_enthalpy(rows.size());
   for (std::size_t index = 0; index < rows.size(); ++index) {
      const amrex::Real density = density_grid.values[index]
         * baryon_mass_mev * mev_per_fm3_to_code;
      const amrex::Real pressure = pressure_mev_fm3[index] * mev_per_fm3_to_code;
      const amrex::Real energy = energy_mev_fm3[index] * mev_per_fm3_to_code;
      const amrex::Real specific_enthalpy = (energy + pressure) / density;
      if (!(specific_enthalpy > 1.0) || !std::isfinite(specific_enthalpy)) {
         throw std::runtime_error(
            "CompOSE normalization must give specific enthalpy greater than one"
         );
      }
      log_density[index] = std::log(density);
      log_pressure[index] = std::log(pressure);
      log_specific_enthalpy[index] = std::log(specific_enthalpy);
      if (index > 0
          && !(log_specific_enthalpy[index] > log_specific_enthalpy[index - 1])) {
         throw std::runtime_error(
            "CompOSE cold specific enthalpy must increase with density"
         );
      }
   }

   const int source_points = static_cast<int>(rows.size());
   const int retained_points = causal_prefix_size(
      log_density, log_pressure, log_specific_enthalpy
   );
   if (retained_points < 2) {
      throw std::runtime_error("CompOSE table has no causal two-point prefix");
   }
   log_density.resize(retained_points);
   log_pressure.resize(retained_points);
   log_specific_enthalpy.resize(retained_points);
   amrex::Real high_density_sound_speed_squared = interval_sound_speed_squared(
      log_density[retained_points - 2],
      log_density[retained_points - 1],
      log_pressure[retained_points - 2],
      log_pressure[retained_points - 1],
      log_specific_enthalpy[retained_points - 2],
      log_specific_enthalpy[retained_points - 1],
      1.0
   );
   high_density_sound_speed_squared = std::clamp(
      high_density_sound_speed_squared,
      std::numeric_limits<amrex::Real>::epsilon(),
      1.0 - causal_margin
   );

   return std::make_shared<TabulatedEOSStorage>(
      directory.string(),
      std::move(log_density),
      std::move(log_pressure),
      std::move(log_specific_enthalpy),
      low_density_gamma,
      high_density_sound_speed_squared,
      baryon_mass_mev,
      source_points
   );
}

} // namespace pyfluid_amrex::eos
