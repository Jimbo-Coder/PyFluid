#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/eos/presets.hpp"
#include "pyfluid_amrex/initial_data/tov.hpp"

namespace {

std::vector<amrex::Real> logarithmic_densities(
   amrex::Real minimum,
   amrex::Real maximum,
   int count
)
{
   if (!(minimum > 0.0) || !(maximum > minimum) || count < 2) {
      amrex::Abort(
         "tov_sequence requires 0 < central_density_min < central_density_max and points >= 2"
      );
   }
   std::vector<amrex::Real> densities(count);
   const amrex::Real log_minimum = std::log(minimum);
   const amrex::Real log_maximum = std::log(maximum);
   for (int point = 0; point < count; ++point) {
      const amrex::Real weight = static_cast<amrex::Real>(point) / (count - 1);
      densities[point] = std::exp(
         log_minimum + weight * (log_maximum - log_minimum)
      );
   }
   return densities;
}

pyfluid_amrex::EOSParams sequence_eos(
   std::string const& preset,
   pyfluid_amrex::RunConfig const& config
)
{
   if (preset == "custom") {
      return pyfluid_amrex::eos_params(config);
   }
   pyfluid_amrex::EOSParams parameters =
      pyfluid_amrex::eos::make_read_piecewise_polytrope(
         pyfluid_amrex::eos::piecewise_polytrope_preset(preset),
         config.thermal_gamma,
         config.rho_floor,
         config.pressure_floor
      );
   parameters.gamma = config.gamma;
   return parameters;
}

std::string sequence_label(
   std::string const& preset,
   pyfluid_amrex::RunConfig const& config
)
{
   if (preset != "custom") {
      return {};
   }
   if (!config.polytropic_density_breaks.empty()
       || !config.polytropic_segment_gammas.empty()) {
      return "custom PWP";
   }
   std::ostringstream label;
   label << "Gamma=" << config.polytropic_gamma
         << ", K=" << config.polytropic_k;
   return label.str();
}

} // namespace

int main(int argc, char* argv[])
{
   if (argc != 2) {
      std::cerr << "usage: " << argv[0] << " <tov_sequence.in>\n";
      return 2;
   }
   amrex::system::abort_on_unused_inputs = true;
   amrex::Initialize(argc, argv);
   {
      const pyfluid_amrex::RunConfig config = pyfluid_amrex::read_config();
      amrex::ParmParse sequence_parameters("tov_sequence");
      amrex::Real minimum_density = 7.0e-4;
      amrex::Real maximum_density = 5.0e-3;
      int point_count = 80;
      std::string output_directory = "runs/current/tov/sequences";
      amrex::Vector<std::string> presets;
      sequence_parameters.query("central_density_min", minimum_density);
      sequence_parameters.query("central_density_max", maximum_density);
      sequence_parameters.query("points", point_count);
      sequence_parameters.query("output_dir", output_directory);
      sequence_parameters.queryarr("eos_presets", presets);
      if (presets.empty()) {
         presets.push_back(config.eos_preset);
      }
      const std::vector<amrex::Real> densities = logarithmic_densities(
         minimum_density, maximum_density, point_count
      );
      std::filesystem::create_directories(output_directory);
      for (std::string const& preset : presets) {
         pyfluid_amrex::TovSolveOptions options{
            densities.front(),
            config.tov_absolute_tolerance,
            config.tov_relative_tolerance,
            config.tov_max_steps,
         };
         const std::vector<pyfluid_amrex::TovSolution> solutions =
            pyfluid_amrex::solve_tov_sequence(
               sequence_eos(preset, config), densities, options
         );
         const std::string output_path = output_directory + "/" + preset + ".tsv";
         pyfluid_amrex::write_tov_sequence(
            output_path, sequence_label(preset, config), solutions
         );
         auto const maximum = std::max_element(
            solutions.begin(),
            solutions.end(),
            [](auto const& left, auto const& right) {
               return left.observables().gravitational_mass
                  < right.observables().gravitational_mass;
            }
         );
         amrex::Print() << "TOV sequence " << preset
                        << ": points=" << solutions.size()
                        << " Mmax=" << maximum->observables().gravitational_mass
                        << " R_at_Mmax=" << maximum->observables().areal_radius
                        << " rho_c_at_Mmax=" << maximum->observables().central_density
                        << " output=" << output_path << "\n";
      }
   }
   amrex::Finalize();
   return 0;
}
