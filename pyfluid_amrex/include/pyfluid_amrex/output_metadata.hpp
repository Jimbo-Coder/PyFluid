#pragma once

#include <AMReX_REAL.H>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pyfluid_amrex/config.hpp"

#ifdef AMREX_USE_HDF5
#include <hdf5.h>
#endif

namespace pyfluid_amrex {

using OutputMetadataEntry = std::pair<std::string, std::string>;
using OutputMetadataEntries = std::vector<OutputMetadataEntry>;

inline std::string output_metadata_real_string(amrex::Real value)
{
   std::ostringstream out;
   out << std::setprecision(17) << value;
   return out.str();
}

inline std::string output_metadata_real_vector_string(
   std::vector<amrex::Real> const& values
)
{
   std::ostringstream out;
   out << std::setprecision(17);
   for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0) {
         out << ' ';
      }
      out << values[index];
   }
   return out.str();
}

inline OutputMetadataEntries output_metadata_records(
   std::string_view source,
   RunConfig const& cfg,
   int step,
   amrex::Real time,
   std::string_view kind = "",
   int finest_level = -1,
   std::string_view compression = "",
   OutputMetadataEntries const& extras = {}
)
{
   OutputMetadataEntries entries;
   entries.emplace_back("source", std::string(source));
   if (!kind.empty()) {
      entries.emplace_back("kind", std::string(kind));
   }
   entries.emplace_back("step", std::to_string(step));
   entries.emplace_back("time", output_metadata_real_string(time));
   entries.emplace_back("output_dir", cfg.output_dir);
   entries.emplace_back("run_name", cfg.run_name);
   entries.emplace_back("physics", cfg.physics);
   entries.emplace_back("spacetime_evolution", cfg.spacetime_evolution);
   entries.emplace_back("spacetime_matter_coupling", cfg.spacetime_matter_coupling);
   entries.emplace_back("spacetime_fixed_metric", cfg.spacetime_fixed_metric);
   entries.emplace_back("spacetime_initial_data", cfg.spacetime_initial_data);
   entries.emplace_back(
      "spacetime_black_hole_interior", cfg.spacetime_black_hole_interior
   );
   entries.emplace_back(
      "spacetime_black_hole_fill_radius",
      output_metadata_real_string(cfg.spacetime_black_hole_fill_radius)
   );
   entries.emplace_back("initial_data_magnetic", cfg.initial_data_magnetic);
   entries.emplace_back("initial_data_spacetime", cfg.initial_data_spacetime);
   entries.emplace_back("initial_data_lapse", cfg.initial_data_lapse);
   entries.emplace_back("initial_data_shift", cfg.initial_data_shift);
   entries.emplace_back("initial_data_atmosphere", cfg.initial_data_atmosphere);
   entries.emplace_back(
      "initial_data_atmosphere_density",
      output_metadata_real_string(cfg.initial_data_atmosphere_density)
   );
   entries.emplace_back(
      "initial_data_atmosphere_pressure",
      output_metadata_real_string(cfg.initial_data_atmosphere_pressure)
   );
   entries.emplace_back(
      "initial_data_atmosphere_transition_radius",
      output_metadata_real_string(cfg.initial_data_atmosphere_transition_radius)
   );
   entries.emplace_back(
      "initial_data_atmosphere_power",
      output_metadata_real_string(cfg.initial_data_atmosphere_power)
   );
   entries.emplace_back(
      "initial_data_disk_axis1",
      output_metadata_real_string(cfg.initial_data_disk_axis1)
   );
   entries.emplace_back(
      "initial_data_disk_axis2",
      output_metadata_real_string(cfg.initial_data_disk_axis2)
   );
   entries.emplace_back(
      "initial_data_disk_axis3",
      output_metadata_real_string(cfg.initial_data_disk_axis3)
   );
   entries.emplace_back("spacetime_lapse_gauge", cfg.spacetime_lapse_gauge);
   entries.emplace_back("spacetime_shift_gauge", cfg.spacetime_shift_gauge);
   entries.emplace_back("spacetime_amplitude", output_metadata_real_string(cfg.spacetime_amplitude));
   entries.emplace_back("spacetime_wave_number", output_metadata_real_string(cfg.spacetime_wave_number));
   entries.emplace_back("spacetime_kappa1", output_metadata_real_string(cfg.spacetime_kappa1));
   entries.emplace_back("spacetime_kappa2", output_metadata_real_string(cfg.spacetime_kappa2));
   entries.emplace_back("spacetime_shift_eta", output_metadata_real_string(cfg.spacetime_shift_eta));
   entries.emplace_back("spacetime_slow_start_time", output_metadata_real_string(cfg.spacetime_slow_start_time));
   entries.emplace_back("spacetime_ko_dissipation", output_metadata_real_string(cfg.spacetime_ko_dissipation));
   entries.emplace_back("spacetime_chi_floor", output_metadata_real_string(cfg.spacetime_chi_floor));
   entries.emplace_back("spacetime_lapse_floor", output_metadata_real_string(cfg.spacetime_lapse_floor));
   entries.emplace_back("spacetime_acceleration1", output_metadata_real_string(cfg.spacetime_acceleration1));
   entries.emplace_back("spacetime_acceleration2", output_metadata_real_string(cfg.spacetime_acceleration2));
   entries.emplace_back("spacetime_acceleration3", output_metadata_real_string(cfg.spacetime_acceleration3));
   entries.emplace_back("spacetime_origin1", output_metadata_real_string(cfg.spacetime_origin1));
   entries.emplace_back("spacetime_origin2", output_metadata_real_string(cfg.spacetime_origin2));
   entries.emplace_back("spacetime_origin3", output_metadata_real_string(cfg.spacetime_origin3));
   entries.emplace_back("spacetime_mass", output_metadata_real_string(cfg.spacetime_mass));
   entries.emplace_back(
      "spacetime_dimensionless_spin1",
      output_metadata_real_string(cfg.spacetime_dimensionless_spin1)
   );
   entries.emplace_back(
      "spacetime_dimensionless_spin2",
      output_metadata_real_string(cfg.spacetime_dimensionless_spin2)
   );
   entries.emplace_back(
      "spacetime_dimensionless_spin3",
      output_metadata_real_string(cfg.spacetime_dimensionless_spin3)
   );
   entries.emplace_back("tov_central_density", output_metadata_real_string(cfg.tov_central_density));
   entries.emplace_back("tov_absolute_tolerance", output_metadata_real_string(cfg.tov_absolute_tolerance));
   entries.emplace_back("tov_relative_tolerance", output_metadata_real_string(cfg.tov_relative_tolerance));
   entries.emplace_back("tov_max_steps", std::to_string(cfg.tov_max_steps));
   entries.emplace_back("initial_data_rotation_law", cfg.initial_data_rotation_law);
   entries.emplace_back("initial_data_angular_velocity", output_metadata_real_string(cfg.initial_data_angular_velocity));
   entries.emplace_back("initial_data_differential_rotation_length", output_metadata_real_string(cfg.initial_data_differential_rotation_length));
   entries.emplace_back("initial_data_rotation_power_q", output_metadata_real_string(cfg.initial_data_rotation_power_q));
   entries.emplace_back("initial_data_rotation_axis1", output_metadata_real_string(cfg.initial_data_rotation_axis1));
   entries.emplace_back("initial_data_rotation_axis2", output_metadata_real_string(cfg.initial_data_rotation_axis2));
   entries.emplace_back("initial_data_rotation_axis3", output_metadata_real_string(cfg.initial_data_rotation_axis3));
   entries.emplace_back("initial_data_rotation_center1", output_metadata_real_string(cfg.initial_data_rotation_center1));
   entries.emplace_back("initial_data_rotation_center2", output_metadata_real_string(cfg.initial_data_rotation_center2));
   entries.emplace_back("initial_data_rotation_center3", output_metadata_real_string(cfg.initial_data_rotation_center3));
   entries.emplace_back("initial_data_matter_relaxation", output_metadata_real_string(cfg.initial_data_matter_relaxation));
   entries.emplace_back("initial_data_matter_relative_tolerance", output_metadata_real_string(cfg.initial_data_matter_relative_tolerance));
   entries.emplace_back("initial_data_matter_max_iterations", std::to_string(cfg.initial_data_matter_max_iterations));
   entries.emplace_back("initial_data_rotation_continuation_steps", std::to_string(cfg.initial_data_rotation_continuation_steps));
   entries.emplace_back("eos", cfg.eos);
   entries.emplace_back("polytropic_k", output_metadata_real_string(cfg.polytropic_k));
   entries.emplace_back("polytropic_gamma", output_metadata_real_string(cfg.polytropic_gamma));
   entries.emplace_back(
      "polytropic_density_breaks",
      output_metadata_real_vector_string(cfg.polytropic_density_breaks)
   );
   entries.emplace_back(
      "polytropic_segment_gammas",
      output_metadata_real_vector_string(cfg.polytropic_segment_gammas)
   );
   entries.emplace_back("thermal_gamma", output_metadata_real_string(cfg.thermal_gamma));
   if (finest_level >= 0) {
      entries.emplace_back("finest_level", std::to_string(finest_level));
   }
   if (!compression.empty()) {
      entries.emplace_back("compression", std::string(compression));
   }
   for (auto const& [key, value] : extras) {
      entries.emplace_back(key, value);
   }
   return entries;
}

inline void append_output_metadata(
   std::ofstream& out,
   std::string_view source,
   RunConfig const& cfg,
   int step,
   amrex::Real time,
   std::string_view kind = "",
   int finest_level = -1,
   std::string_view compression = "",
   OutputMetadataEntries const& extras = {}
)
{
   out << "# PyFluid output metadata\n";
   for (auto const& [key, value] :
        output_metadata_records(source, cfg, step, time, kind, finest_level, compression, extras)) {
      out << "# " << key << "=" << value << "\n";
   }
}

#ifdef AMREX_USE_HDF5
inline bool write_output_metadata_hdf5_attributes(
   std::string const& hdf5_path,
   std::string_view source,
   RunConfig const& cfg,
   int step,
   amrex::Real time,
   std::string_view kind = "",
   int finest_level = -1,
   std::string_view compression = "",
   OutputMetadataEntries const& extras = {}
)
{
   const hid_t file_id = H5Fopen(hdf5_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
   if (file_id < 0) {
      return false;
   }

   const auto entries = output_metadata_records(source, cfg, step, time, kind, finest_level, compression, extras);
   const hid_t attr_space = H5Screate(H5S_SCALAR);
   if (attr_space < 0) {
      H5Fclose(file_id);
      return false;
    }

   for (auto const& [key, value] : entries) {
      if (H5Aexists(file_id, key.c_str()) > 0) {
         H5Adelete(file_id, key.c_str());
      }
      const hid_t attr_type = H5Tcopy(H5T_C_S1);
      if (attr_type < 0) {
         continue;
      }
      const std::string text = value;
      H5Tset_size(attr_type, text.size() + 1);
      H5Tset_strpad(attr_type, H5T_STR_NULLTERM);
      const hid_t attr = H5Acreate2(
         file_id,
         key.c_str(),
         attr_type,
         attr_space,
         H5P_DEFAULT,
         H5P_DEFAULT
      );
      if (attr >= 0) {
         H5Awrite(attr, attr_type, text.c_str());
         H5Aclose(attr);
      }
      H5Tclose(attr_type);
   }
   H5Sclose(attr_space);
   H5Fclose(file_id);
   return true;
}
#endif

} // namespace pyfluid_amrex
