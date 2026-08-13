#include "pyfluid_amrex/checkpoint.hpp"
#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/output_metadata.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/multilevel_field_io.hpp"

#include <AMReX_FileSystem.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_PlotFileUtilHDF5.H>
#include <AMReX_VisMF.H>

#include <hdf5.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <array>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace pyfluid_amrex {

namespace {

constexpr const char* CheckpointLevelPrefix = "Level_";
constexpr const char* CheckpointHeaderName = "Header";
constexpr const char* CheckpointHeaderLabel = "PyFluid AMReX checkpoint v4";
constexpr int CheckpointSchemaVersion = 7;

using CheckpointMetadata = std::map<std::string, std::string>;

std::string trimmed(std::string_view text)
{
   const auto begin = text.find_first_not_of(" \t\r\n");
   if (begin == std::string_view::npos) {
      return {};
   }
   const auto end = text.find_last_not_of(" \t\r\n");
   return std::string(text.substr(begin, end - begin + 1));
}

CheckpointMetadata parse_checkpoint_metadata(std::string const& checkpoint_name)
{
   const std::filesystem::path metadata = std::filesystem::path(checkpoint_name) / "Header.meta";
   std::ifstream input(metadata);
   if (!input.good()) {
      amrex::Abort("checkpoint metadata file is required: " + metadata.string());
   }

   CheckpointMetadata entries;
   for (std::string line; std::getline(input, line);) {
      if (line.empty() || line[0] != '#') {
         continue;
      }
      const auto equal = line.find('=');
      if (equal == std::string::npos) {
         continue;
      }
      const auto key = trimmed(std::string_view(line).substr(1, equal - 1));
      const auto value = trimmed(std::string_view(line).substr(equal + 1));
      if (!key.empty()) {
         entries[key] = value;
      }
   }
   return entries;
}

std::string int_vect_string(amrex::IntVect const& value)
{
   std::ostringstream out;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (dir != 0) {
         out << ',';
      }
      out << value[dir];
   }
   return out.str();
}

std::string geometry_domain_string(amrex::Geometry const& geometry)
{
   std::ostringstream out;
   out << geometry.Domain();
   return out.str();
}

std::string geometry_real_vector_string(amrex::Geometry const& geometry, bool upper)
{
   std::ostringstream out;
   out.precision(17);
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (dir != 0) {
         out << ',';
      }
      out << (upper ? geometry.ProbHi(dir) : geometry.ProbLo(dir));
   }
   return out.str();
}

std::string geometry_periodicity_string(amrex::Geometry const& geometry)
{
   std::ostringstream out;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (dir != 0) {
         out << ',';
      }
      out << geometry.isPeriodic(dir);
   }
   return out.str();
}

std::string refinement_ratios_string(
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   int finest_level
)
{
   std::ostringstream out;
   for (int lev = 0; lev < finest_level; ++lev) {
      if (lev != 0) {
         out << ';';
      }
      out << int_vect_string(refinement_ratios.at(lev));
   }
   return out.str();
}

void require_metadata(
   std::string const& checkpoint_name,
   CheckpointMetadata const& metadata,
   std::string const& key,
   std::string const& value
)
{
   const auto it = metadata.find(key);
   if (it == metadata.end()) {
      amrex::Abort("checkpoint metadata missing key '" + key + "' in " + std::string(checkpoint_name) + "/Header.meta");
   }
   if (it->second != value) {
      amrex::Abort(
         "checkpoint metadata mismatch for '" + key + "': checkpoint='" + it->second + "' current='" + value + "'"
      );
   }
}

int to_int(std::string const& text, std::string const& checkpoint_name, std::string const& key)
{
   try {
      return std::stoi(text);
   } catch (std::exception const&) {
      amrex::Abort("checkpoint metadata key '" + key + "' is not an int in " + std::string(checkpoint_name) + "/Header.meta");
   }
   return 0;
}

void require_metadata_int(
   std::string const& checkpoint_name,
   CheckpointMetadata const& metadata,
   std::string const& key,
   int value
)
{
   const auto it = metadata.find(key);
   if (it == metadata.end()) {
      amrex::Abort("checkpoint metadata missing key '" + key + "' in " + std::string(checkpoint_name) + "/Header.meta");
   }
   const int checked = to_int(it->second, checkpoint_name, key);
   if (checked != value) {
      amrex::Abort(
         "checkpoint metadata mismatch for '" + key + "': checkpoint='" + it->second
         + "' current='" + std::to_string(value) + "'"
      );
   }
}

bool checkpoint_metadata_is_complete(CheckpointMetadata const& metadata)
{
   constexpr std::array required = {
      "source",
      "checkpoint_schema_version",
      "checkpoint_format",
      "checkpoint_state_components",
      "checkpoint_mesh_coordinate",
      "checkpoint_mesh_axes",
      "checkpoint_amrex_spacedim",
      "checkpoint_domain",
      "checkpoint_prob_lo",
      "checkpoint_prob_hi",
      "checkpoint_periodicity",
      "checkpoint_refinement_ratios",
      "physics",
      "spacetime_evolution",
      "spacetime_matter_coupling",
      "spacetime_fixed_metric",
      "spacetime_acceleration1",
      "spacetime_acceleration2",
      "spacetime_acceleration3",
      "spacetime_origin1",
      "spacetime_origin2",
      "spacetime_origin3",
      "spacetime_mass",
      "eos",
      "checkpoint_ct",
      "checkpoint_passive_scalars",
   };
   for (auto const* key : required) {
      if (metadata.find(key) == metadata.end()) {
         return false;
      }
   }
   return true;
}

void write_checkpoint_metadata(
   std::string const& checkpoint_name,
   int step,
   amrex::Real time,
   int finest_level,
   RunConfig const& cfg,
   int state_components,
   amrex::Geometry const& base_geometry,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   std::string const& checkpoint_compression
)
{
   const std::filesystem::path metadata = std::filesystem::path(checkpoint_name) / "Header.meta";
   std::ofstream out(metadata, std::ios::out | std::ofstream::trunc);
   if (!out.good()) {
      amrex::FileOpenFailed(metadata.string());
   }
   const std::string compression = resolved_plotfile_compression(checkpoint_compression);
   append_output_metadata(
      out,
      "checkpoint",
      cfg,
      step,
      time,
      "state",
      finest_level,
      compression,
      {
         {"checkpoint_schema_version", std::to_string(CheckpointSchemaVersion)},
         {"checkpoint_format", "hdf5"},
         {"checkpoint_compression", compression},
      }
   );
   out << "# type=checkpoint\n";
   out << "# checkpoint_state_components=" << state_components << "\n";
   out << "# checkpoint_amrex_spacedim=" << AMREX_SPACEDIM << "\n";
   out << "# checkpoint_mesh_coordinate=" << cfg.mesh_coordinate << "\n";
   out << "# checkpoint_mesh_axes="
       << mesh_axes_or_default(mesh_coordinate_code(cfg.mesh_coordinate), cfg.mesh_axes) << "\n";
   out << "# checkpoint_domain=" << geometry_domain_string(base_geometry) << "\n";
   out << "# checkpoint_prob_lo=" << geometry_real_vector_string(base_geometry, false) << "\n";
   out << "# checkpoint_prob_hi=" << geometry_real_vector_string(base_geometry, true) << "\n";
   out << "# checkpoint_periodicity=" << geometry_periodicity_string(base_geometry) << "\n";
   out << "# checkpoint_refinement_ratios="
       << refinement_ratios_string(refinement_ratios, finest_level) << "\n";
   out << "# checkpoint_gamma=" << output_metadata_real_string(cfg.gamma) << "\n";
   out << "# checkpoint_polytropic_k=" << output_metadata_real_string(cfg.polytropic_k) << "\n";
   out << "# checkpoint_polytropic_gamma=" << output_metadata_real_string(cfg.polytropic_gamma) << "\n";
   out << "# checkpoint_polytropic_density_breaks="
       << output_metadata_real_vector_string(cfg.polytropic_density_breaks) << "\n";
   out << "# checkpoint_polytropic_segment_gammas="
       << output_metadata_real_vector_string(cfg.polytropic_segment_gammas) << "\n";
   out << "# checkpoint_thermal_gamma=" << output_metadata_real_string(cfg.thermal_gamma) << "\n";
   out << "# checkpoint_timestep=" << cfg.timestep << "\n";
   out << "# checkpoint_recon=" << cfg.recon << "\n";
   out << "# checkpoint_recon_fine=" << (cfg.recon_fine.empty() ? "none" : cfg.recon_fine) << "\n";
   out << "# checkpoint_recon_fine_start_level=" << cfg.recon_fine_start_level << "\n";
   out << "# checkpoint_ct=" << cfg.ct << "\n";
   out << "# checkpoint_passive_scalars=" << cfg.passive_scalars << "\n";
   out << "# checkpoint_subcycling=" << cfg.subcycling << "\n";
   out << "# checkpoint_diffusion=" << cfg.diffusion << "\n";
   out << "# checkpoint_gravity=" << cfg.gravity << "\n";
}

std::string checkpoint_field_prefix(std::string const& checkpoint_name, std::string const& field_name)
{
   return (std::filesystem::path(checkpoint_name) / field_name).string();
}

void write_checkpoint_field(
   amrex::Vector<amrex::MultiFab const*> const& fields,
   amrex::Vector<amrex::Geometry> const& geometry,
   amrex::Vector<int> const& level_steps,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   int finest_level,
   amrex::Real time,
   std::string const& checkpoint_name,
   std::string const& field_name,
   std::string const& compression,
   std::string const& format_label
)
{
   const int num_components = fields.at(0)->nComp();
   amrex::Vector<std::string> component_names(num_components);
   for (int comp = 0; comp < num_components; ++comp) {
      component_names[comp] = field_name + '_' + std::to_string(comp);
   }
   amrex::WriteMultiLevelPlotfileHDF5(
      checkpoint_field_prefix(checkpoint_name, field_name),
      finest_level + 1,
      fields,
      component_names,
      geometry,
      time,
      level_steps,
      refinement_ratios,
      compression,
      format_label
   );
}

CheckpointFieldInfo checkpoint_field_info(
   std::string const& name,
   amrex::MultiFab const& field
)
{
   return {name, field.nComp(), field.ixType()};
}

void validate_checkpoint_field_levels(
   CheckpointFieldInfo const& info,
   amrex::Vector<amrex::MultiFab const*> const& fields
)
{
   for (amrex::MultiFab const* field : fields) {
      if (field == nullptr || field->nComp() != info.num_components
          || field->ixType() != info.index_type) {
         amrex::Abort(
            "checkpoint field set '" + info.name
            + "' has inconsistent component or centering metadata"
         );
      }
   }
}

[[noreturn]] void checkpoint_hdf5_error(std::string const& message, std::string const& path)
{
   amrex::Abort(message + ": " + path);
   std::abort();
}

void read_checkpoint_field_hdf5(amrex::MultiFab& field, std::string const& path, int level)
{
   const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   if (file < 0) {
      checkpoint_hdf5_error("cannot open checkpoint HDF5 field", path);
   }

   const std::string level_name = "level_" + std::to_string(level);
   const hid_t group = H5Gopen2(file, level_name.c_str(), H5P_DEFAULT);
   if (group < 0) {
      checkpoint_hdf5_error("checkpoint HDF5 field is missing " + level_name, path);
   }

   const hid_t boxes_dataset = H5Dopen2(group, "boxes", H5P_DEFAULT);
   const hid_t offsets_dataset = H5Dopen2(group, "data:offsets=0", H5P_DEFAULT);
   const hid_t data_dataset = H5Dopen2(group, "data:datatype=0", H5P_DEFAULT);
   if (boxes_dataset < 0 || offsets_dataset < 0 || data_dataset < 0) {
      checkpoint_hdf5_error("checkpoint HDF5 field has an incomplete level layout", path);
   }

   const hid_t boxes_space = H5Dget_space(boxes_dataset);
   hsize_t num_boxes_hdf5 = 0;
   if (H5Sget_simple_extent_ndims(boxes_space) != 1 ||
       H5Sget_simple_extent_dims(boxes_space, &num_boxes_hdf5, nullptr) < 0) {
      checkpoint_hdf5_error("checkpoint HDF5 boxes dataset has an invalid shape", path);
   }
   H5Sclose(boxes_space);

   const std::size_t num_boxes = static_cast<std::size_t>(num_boxes_hdf5);
   const hid_t boxes_type = H5Dget_type(boxes_dataset);
   if (H5Tget_size(boxes_type) != 2 * AMREX_SPACEDIM * sizeof(int)) {
      checkpoint_hdf5_error("checkpoint HDF5 boxes dataset has an incompatible type", path);
   }

   std::vector<int> box_bounds(num_boxes * 2 * AMREX_SPACEDIM);
   std::vector<unsigned long long> offsets(num_boxes + 1);
   if (H5Dread(boxes_dataset, boxes_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, box_bounds.data()) < 0 ||
       H5Dread(offsets_dataset, H5T_NATIVE_ULLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, offsets.data()) < 0) {
      checkpoint_hdf5_error("cannot read checkpoint HDF5 box metadata", path);
   }
   H5Tclose(boxes_type);

   field.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
   const hid_t data_space = H5Dget_space(data_dataset);
   const hid_t transfer = H5Pcreate(H5P_DATASET_XFER);
#ifdef AMREX_USE_MPI
   H5Pset_dxpl_mpio(transfer, H5FD_MPIO_INDEPENDENT);
#endif
   const hid_t real_type = sizeof(amrex::Real) == sizeof(double) ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;

   for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
      const amrex::Box& target_box = mfi.validbox();
      std::size_t stored_index = num_boxes;
      for (std::size_t box_index = 0; box_index < num_boxes; ++box_index) {
         bool matches = true;
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const int lo = box_bounds[box_index * 2 * AMREX_SPACEDIM + dir];
            const int hi = box_bounds[box_index * 2 * AMREX_SPACEDIM + AMREX_SPACEDIM + dir];
            matches = matches && lo == target_box.smallEnd(dir) && hi == target_box.bigEnd(dir);
         }
         if (matches) {
            stored_index = box_index;
            break;
         }
      }
      if (stored_index == num_boxes) {
         checkpoint_hdf5_error("checkpoint HDF5 field does not contain a required box", path);
      }

      const hsize_t data_offset = offsets[stored_index];
      const hsize_t data_count = offsets[stored_index + 1] - offsets[stored_index];
      const hsize_t expected_count = static_cast<hsize_t>(target_box.numPts()) * field.nComp();
      if (data_count != expected_count) {
         checkpoint_hdf5_error("checkpoint HDF5 field box has an incompatible component count", path);
      }

      const hid_t memory_space = H5Screate_simple(1, &data_count, nullptr);
      H5Sselect_hyperslab(data_space, H5S_SELECT_SET, &data_offset, nullptr, &data_count, nullptr);
      amrex::Vector<amrex::Real> buffer(data_count);
      if (H5Dread(data_dataset, real_type, memory_space, data_space, transfer, buffer.data()) < 0) {
         checkpoint_hdf5_error("cannot read checkpoint HDF5 field data", path);
      }
      H5Sclose(memory_space);

      amrex::FArrayBox unpacked(target_box, field.nComp());
      std::copy(buffer.begin(), buffer.end(), unpacked.dataPtr());
      field[mfi].copy(unpacked, target_box, 0, target_box, 0, field.nComp());
   }

   H5Pclose(transfer);
   H5Sclose(data_space);
   H5Dclose(boxes_dataset);
   H5Dclose(offsets_dataset);
   H5Dclose(data_dataset);
   H5Gclose(group);
   H5Fclose(file);
}

} // namespace

void write_multilevel_hdf5_field(
   amrex::Vector<amrex::MultiFab const*> const& fields,
   amrex::Vector<amrex::Geometry> const& geometry,
   amrex::Vector<int> const& level_steps,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   int finest_level,
   amrex::Real time,
   std::string const& root,
   std::string const& field_name,
   std::string const& compression,
   std::string const& format_label
)
{
   write_checkpoint_field(
      fields,
      geometry,
      level_steps,
      refinement_ratios,
      finest_level,
      time,
      root,
      field_name,
      compression,
      format_label
   );
}

bool multilevel_hdf5_field_exists(
   std::string const& root,
   std::string const& field_name
)
{
   return std::filesystem::is_regular_file(
      checkpoint_field_path(root, field_name)
   );
}

void read_multilevel_hdf5_field(
   amrex::MultiFab& field,
   std::string const& root,
   int level,
   std::string const& field_name
)
{
   read_checkpoint_field_hdf5(
      field, checkpoint_field_path(root, field_name), level
   );
}

MultilevelHdf5FieldLayout read_multilevel_hdf5_field_layout(
   std::string const& root,
   std::string const& field_name
)
{
   const std::string path = checkpoint_field_path(root, field_name);
   const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   if (file < 0) {
      checkpoint_hdf5_error("cannot open multilevel HDF5 field", path);
   }
   auto read_root_int = [&](char const* name) {
      const hid_t attribute = H5Aopen(file, name, H5P_DEFAULT);
      int value = 0;
      if (attribute < 0
          || H5Aread(attribute, H5T_NATIVE_INT, &value) < 0) {
         checkpoint_hdf5_error(
            std::string("multilevel HDF5 field is missing attribute ") + name,
            path
         );
      }
      H5Aclose(attribute);
      return value;
   };
   const int num_levels = read_root_int("num_levels");
   MultilevelHdf5FieldLayout layout;
   layout.num_components = read_root_int("num_components");
   if (num_levels <= 0 || layout.num_components <= 0) {
      checkpoint_hdf5_error("multilevel HDF5 field has invalid root metadata", path);
   }
   layout.boxes.resize(num_levels);
   layout.domains.resize(num_levels);
   layout.prob_lo.resize(num_levels);
   layout.prob_hi.resize(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      const std::string level_name = "level_" + std::to_string(level);
      const hid_t group = H5Gopen2(file, level_name.c_str(), H5P_DEFAULT);
      if (group < 0) {
         checkpoint_hdf5_error(
            "multilevel HDF5 field is missing " + level_name, path
         );
      }
      const hid_t boxes_dataset = H5Dopen2(group, "boxes", H5P_DEFAULT);
      if (boxes_dataset < 0) {
         checkpoint_hdf5_error("multilevel HDF5 field is missing boxes", path);
      }
      const hid_t boxes_space = H5Dget_space(boxes_dataset);
      hsize_t num_boxes_hdf5 = 0;
      if (H5Sget_simple_extent_ndims(boxes_space) != 1
          || H5Sget_simple_extent_dims(
                boxes_space, &num_boxes_hdf5, nullptr
             ) < 0) {
         checkpoint_hdf5_error("multilevel HDF5 boxes have invalid shape", path);
      }
      H5Sclose(boxes_space);
      const hid_t boxes_type = H5Dget_type(boxes_dataset);
      if (H5Tget_size(boxes_type) != 2 * AMREX_SPACEDIM * sizeof(int)) {
         checkpoint_hdf5_error("multilevel HDF5 boxes have incompatible type", path);
      }
      std::vector<int> bounds(
         static_cast<std::size_t>(num_boxes_hdf5)
         * 2 * AMREX_SPACEDIM
      );
      if (H5Dread(
             boxes_dataset,
             boxes_type,
             H5S_ALL,
             H5S_ALL,
             H5P_DEFAULT,
             bounds.data()
          ) < 0) {
         checkpoint_hdf5_error("cannot read multilevel HDF5 boxes", path);
      }
      amrex::BoxList box_list;
      for (std::size_t box = 0;
           box < static_cast<std::size_t>(num_boxes_hdf5); ++box) {
         amrex::IntVect lo;
         amrex::IntVect hi;
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            lo[direction] = bounds[
               box * 2 * AMREX_SPACEDIM + direction
            ];
            hi[direction] = bounds[
               box * 2 * AMREX_SPACEDIM + AMREX_SPACEDIM + direction
            ];
         }
         box_list.push_back(amrex::Box(lo, hi));
      }
      layout.boxes[level] = amrex::BoxArray(box_list);
      H5Tclose(boxes_type);
      H5Dclose(boxes_dataset);

      const hid_t domain_attribute = H5Aopen(group, "prob_domain", H5P_DEFAULT);
      if (domain_attribute < 0) {
         checkpoint_hdf5_error("multilevel HDF5 field is missing prob_domain", path);
      }
      const hid_t domain_type = H5Aget_type(domain_attribute);
      std::array<int, 2 * AMREX_SPACEDIM> domain_bounds{};
      if (H5Tget_size(domain_type) != domain_bounds.size() * sizeof(int)
          || H5Aread(
                domain_attribute, domain_type, domain_bounds.data()
             ) < 0) {
         checkpoint_hdf5_error("cannot read multilevel HDF5 prob_domain", path);
      }
      amrex::IntVect domain_lo;
      amrex::IntVect domain_hi;
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         domain_lo[direction] = domain_bounds[direction];
         domain_hi[direction] = domain_bounds[AMREX_SPACEDIM + direction];
      }
      layout.domains[level] = amrex::Box(domain_lo, domain_hi);
      H5Tclose(domain_type);
      H5Aclose(domain_attribute);

      auto read_real_vector = [&](char const* name, auto& output) {
         const hid_t attribute = H5Aopen(group, name, H5P_DEFAULT);
         const hid_t real_type = sizeof(amrex::Real) == sizeof(double)
            ? H5T_NATIVE_DOUBLE
            : H5T_NATIVE_FLOAT;
         if (attribute < 0
             || H5Aread(attribute, real_type, output.data()) < 0) {
            checkpoint_hdf5_error(
               std::string("cannot read multilevel HDF5 ") + name, path
            );
         }
         H5Aclose(attribute);
      };
      read_real_vector("prob_lo", layout.prob_lo[level]);
      read_real_vector("prob_hi", layout.prob_hi[level]);
      H5Gclose(group);
   }
   H5Fclose(file);
   return layout;
}

RuntimeDimensionHdf5Field read_runtime_dimension_hdf5_field(
   std::string const& root,
   std::string const& field_name
)
{
   const std::string path = checkpoint_field_path(root, field_name);
   const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   if (file < 0) {
      checkpoint_hdf5_error("cannot open runtime-dimension HDF5 field", path);
   }
   auto read_root_int = [&](char const* name) {
      const hid_t attribute = H5Aopen(file, name, H5P_DEFAULT);
      int value = 0;
      if (attribute < 0
          || H5Aread(attribute, H5T_NATIVE_INT, &value) < 0) {
         checkpoint_hdf5_error(
            std::string("runtime-dimension HDF5 field is missing attribute ")
               + name,
            path
         );
      }
      H5Aclose(attribute);
      return value;
   };

   RuntimeDimensionHdf5Field field;
   field.dimension = read_root_int("dim");
   field.num_components = read_root_int("num_components");
   const int num_levels = read_root_int("num_levels");
   if (field.dimension < 1 || field.dimension > 3
       || field.num_components <= 0 || num_levels <= 0) {
      checkpoint_hdf5_error(
         "runtime-dimension HDF5 field has invalid root metadata", path
      );
   }
   field.levels.resize(num_levels);
   const hid_t real_type = sizeof(amrex::Real) == sizeof(double)
      ? H5T_NATIVE_DOUBLE
      : H5T_NATIVE_FLOAT;

   for (int level = 0; level < num_levels; ++level) {
      RuntimeDimensionHdf5Level& output = field.levels[level];
      const std::string level_name = "level_" + std::to_string(level);
      const hid_t group = H5Gopen2(file, level_name.c_str(), H5P_DEFAULT);
      if (group < 0) {
         checkpoint_hdf5_error(
            "runtime-dimension HDF5 field is missing " + level_name, path
         );
      }

      auto read_real_vector = [&](char const* name, auto& values) {
         const hid_t attribute = H5Aopen(group, name, H5P_DEFAULT);
         if (attribute < 0
             || H5Aread(attribute, real_type, values.data()) < 0) {
            checkpoint_hdf5_error(
               std::string("cannot read runtime-dimension HDF5 ") + name,
               path
            );
         }
         H5Aclose(attribute);
      };
      read_real_vector("prob_lo", output.prob_lo);
      read_real_vector("prob_hi", output.prob_hi);

      const hid_t domain_attribute = H5Aopen(
         group, "prob_domain", H5P_DEFAULT
      );
      const hid_t domain_type = domain_attribute >= 0
         ? H5Aget_type(domain_attribute)
         : -1;
      std::array<int, 6> domain_bounds{};
      if (domain_attribute < 0 || domain_type < 0
          || H5Tget_size(domain_type)
             != static_cast<std::size_t>(2 * field.dimension) * sizeof(int)
          || H5Aread(
                domain_attribute, domain_type, domain_bounds.data()
             ) < 0) {
         checkpoint_hdf5_error(
            "cannot read runtime-dimension HDF5 prob_domain", path
         );
      }
      for (int direction = 0; direction < field.dimension; ++direction) {
         output.domain_lower[direction] = domain_bounds[direction];
         output.domain_upper[direction] =
            domain_bounds[field.dimension + direction];
      }
      H5Tclose(domain_type);
      H5Aclose(domain_attribute);

      const hid_t boxes_dataset = H5Dopen2(
         group, "boxes", H5P_DEFAULT
      );
      const hid_t offsets_dataset = H5Dopen2(
         group, "data:offsets=0", H5P_DEFAULT
      );
      const hid_t data_dataset = H5Dopen2(
         group, "data:datatype=0", H5P_DEFAULT
      );
      if (boxes_dataset < 0 || offsets_dataset < 0 || data_dataset < 0) {
         checkpoint_hdf5_error(
            "runtime-dimension HDF5 field has an incomplete level layout",
            path
         );
      }

      const hid_t boxes_space = H5Dget_space(boxes_dataset);
      hsize_t num_boxes_hdf5 = 0;
      if (H5Sget_simple_extent_ndims(boxes_space) != 1
          || H5Sget_simple_extent_dims(
                boxes_space, &num_boxes_hdf5, nullptr
             ) < 0) {
         checkpoint_hdf5_error(
            "runtime-dimension HDF5 boxes have invalid shape", path
         );
      }
      H5Sclose(boxes_space);
      const std::size_t num_boxes = static_cast<std::size_t>(num_boxes_hdf5);
      const hid_t boxes_type = H5Dget_type(boxes_dataset);
      if (H5Tget_size(boxes_type)
          != static_cast<std::size_t>(2 * field.dimension) * sizeof(int)) {
         checkpoint_hdf5_error(
            "runtime-dimension HDF5 boxes have incompatible type", path
         );
      }
      std::vector<int> box_bounds(
         num_boxes * static_cast<std::size_t>(2 * field.dimension)
      );
      std::vector<unsigned long long> offsets(num_boxes + 1);
      if (H5Dread(
             boxes_dataset,
             boxes_type,
             H5S_ALL,
             H5S_ALL,
             H5P_DEFAULT,
             box_bounds.data()
          ) < 0
          || H5Dread(
                offsets_dataset,
                H5T_NATIVE_ULLONG,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                offsets.data()
             ) < 0) {
         checkpoint_hdf5_error(
            "cannot read runtime-dimension HDF5 box metadata", path
         );
      }
      H5Tclose(boxes_type);

      const hid_t data_space = H5Dget_space(data_dataset);
      hsize_t data_size_hdf5 = 0;
      if (H5Sget_simple_extent_ndims(data_space) != 1
          || H5Sget_simple_extent_dims(
                data_space, &data_size_hdf5, nullptr
             ) < 0) {
         checkpoint_hdf5_error(
            "runtime-dimension HDF5 data have invalid shape", path
         );
      }
      H5Sclose(data_space);
      std::vector<amrex::Real> all_data(
         static_cast<std::size_t>(data_size_hdf5)
      );
      if (H5Dread(
             data_dataset,
             real_type,
             H5S_ALL,
             H5S_ALL,
             H5P_DEFAULT,
             all_data.data()
          ) < 0) {
         checkpoint_hdf5_error(
            "cannot read runtime-dimension HDF5 data", path
         );
      }

      if (offsets.empty() || offsets.front() != 0
          || offsets.back() != all_data.size()) {
         checkpoint_hdf5_error(
            "runtime-dimension HDF5 offsets are inconsistent", path
         );
      }
      output.boxes.resize(num_boxes);
      for (std::size_t box_index = 0; box_index < num_boxes; ++box_index) {
         RuntimeDimensionHdf5Box& box = output.boxes[box_index];
         std::size_t points = 1;
         for (int direction = 0; direction < field.dimension; ++direction) {
            box.lower[direction] = box_bounds[
               box_index * 2 * field.dimension + direction
            ];
            box.upper[direction] = box_bounds[
               box_index * 2 * field.dimension
                  + field.dimension + direction
            ];
            if (box.upper[direction] < box.lower[direction]) {
               checkpoint_hdf5_error(
                  "runtime-dimension HDF5 box has invalid bounds", path
               );
            }
            points *= static_cast<std::size_t>(
               box.upper[direction] - box.lower[direction] + 1
            );
         }
         const std::size_t begin = static_cast<std::size_t>(
            offsets[box_index]
         );
         const std::size_t end = static_cast<std::size_t>(
            offsets[box_index + 1]
         );
         if (end < begin
             || end - begin
                != points * static_cast<std::size_t>(field.num_components)) {
            checkpoint_hdf5_error(
               "runtime-dimension HDF5 box has incompatible data size", path
            );
         }
         box.data.assign(all_data.begin() + begin, all_data.begin() + end);
      }

      H5Dclose(boxes_dataset);
      H5Dclose(offsets_dataset);
      H5Dclose(data_dataset);
      H5Gclose(group);
   }
   H5Fclose(file);
   return field;
}

std::string checkpoint_field_path(std::string const& checkpoint_name, std::string const& field_name)
{
   return checkpoint_field_prefix(checkpoint_name, field_name) + ".h5";
}

bool checkpoint_field_exists(std::string const& checkpoint_name, std::string const& field_name)
{
   return multilevel_hdf5_field_exists(checkpoint_name, field_name);
}

void read_checkpoint_field(
   amrex::MultiFab& field,
   std::string const& checkpoint_name,
   int level,
   std::string const& field_name
)
{
   read_multilevel_hdf5_field(field, checkpoint_name, level, field_name);
}

void write_checkpoint(
   std::vector<std::unique_ptr<amrex::MultiFab>> const& state,
   amrex::Vector<int> const& level_steps,
   amrex::Vector<amrex::Real> const& level_times,
   amrex::Vector<amrex::Real> const& level_dts,
   amrex::Vector<amrex::Geometry> const& geometry,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   int finest_level,
   std::string const& checkpoint_dir,
   int step,
   RunConfig const& cfg,
   std::vector<FaceMagneticField> const* face_b,
   std::vector<CheckpointFieldSet> const& extra_fields,
   amrex::Real reference_mass,
   amrex::Real reference_energy,
   amrex::Real reference_max_rho,
   amrex::Long aligned_regrid_ticks,
   amrex::Long aligned_base_regrid_tick,
   std::string const& checkpoint_compression
)
{
   const std::filesystem::path root = checkpoint_dir.empty()
      ? std::filesystem::path("checkpoints")
      : std::filesystem::path(checkpoint_dir);
   const std::string checkpoint_name = (root / amrex::Concatenate("chk", step, 5)).string();
   const int nlevels = finest_level + 1;
   if (state.size() < nlevels || level_steps.size() < nlevels || level_times.size() < nlevels ||
       level_dts.size() < nlevels ||
       geometry.size() < nlevels || refinement_ratios.size() < finest_level) {
      amrex::Abort("checkpoint metadata does not cover every active AMR level");
   }
   if (face_b != nullptr && face_b->size() < nlevels) {
      amrex::Abort("checkpoint face-centered magnetic state does not cover every active AMR level");
   }
   for (CheckpointFieldSet const& field_set : extra_fields) {
      if (field_set.name.empty() || field_set.levels == nullptr
          || field_set.levels->size() < nlevels) {
         amrex::Abort("checkpoint extra field set does not cover every active AMR level");
      }
   }

   const amrex::Real base_time = level_times[0];
   const amrex::Real sync_tol =
      64.0 * std::numeric_limits<amrex::Real>::epsilon() * std::max<amrex::Real>(1.0, std::abs(base_time));
   for (int lev = 1; lev <= finest_level; ++lev) {
      if (std::abs(level_times[lev] - base_time) > sync_tol) {
         amrex::Abort("checkpoint requires synchronized AMR level times");
      }
   }
   const int state_components = state.empty() ? state_component_count(cfg) : state[0]->nComp();

   std::vector<CheckpointFieldInfo> checkpoint_fields;
   checkpoint_fields.push_back(
      checkpoint_field_info(checkpoint_state_field_name(), *state[0])
   );
   if (face_b != nullptr) {
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         checkpoint_fields.push_back(checkpoint_field_info(
            checkpoint_face_b_field_name(dir), *face_b->at(0).b[dir]
         ));
      }
   }
   for (CheckpointFieldSet const& field_set : extra_fields) {
      checkpoint_fields.push_back(
         checkpoint_field_info(field_set.name, *field_set.levels->at(0))
      );
   }
   for (std::size_t left = 0; left < checkpoint_fields.size(); ++left) {
      for (std::size_t right = left + 1; right < checkpoint_fields.size(); ++right) {
         if (checkpoint_fields[left].name == checkpoint_fields[right].name) {
            amrex::Abort(
               "checkpoint field name is not unique: " + checkpoint_fields[left].name
            );
         }
      }
   }

   amrex::PreBuildDirectorHierarchy(checkpoint_name, std::string(CheckpointLevelPrefix), nlevels, true);

   if (amrex::ParallelDescriptor::IOProcessor()) {
      std::filesystem::create_directories(root);
      const std::string header_name = checkpoint_name + "/" + std::string(CheckpointHeaderName);
      amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
      std::ofstream header;
      header.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
      header.open(header_name.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      if (!header.good()) {
         amrex::FileOpenFailed(header_name);
      }
      header.precision(17);
      header << CheckpointHeaderLabel << "\n";
      header << finest_level << "\n";
      header << state_components << "\n";
      for (int lev = 0; lev <= finest_level; ++lev) {
         header << level_steps[lev] << " ";
      }
      header << "\n";
      for (int lev = 0; lev <= finest_level; ++lev) {
         header << level_dts[lev] << " ";
      }
      header << "\n";
      for (int lev = 0; lev <= finest_level; ++lev) {
         header << level_times[lev] << " ";
      }
      header << "\n";
      for (int lev = 0; lev <= finest_level; ++lev) {
         state[lev]->boxArray().writeOn(header);
         header << "\n";
      }
      header << checkpoint_fields.size() << "\n";
      for (CheckpointFieldInfo const& field : checkpoint_fields) {
         header << field.name << " " << field.num_components;
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            header << " " << field.index_type.nodeCentered(dir);
         }
         header << "\n";
      }
      header << reference_mass << " " << reference_energy << " "
             << reference_max_rho << "\n";
      header << aligned_regrid_ticks << "\n";
      header << aligned_base_regrid_tick << "\n";
      write_checkpoint_metadata(
         checkpoint_name,
         step,
         base_time,
         finest_level,
         cfg,
         state_components,
         geometry[0],
         refinement_ratios,
         checkpoint_compression
      );
   }

   const std::string compression = resolved_plotfile_compression(checkpoint_compression);
   amrex::Vector<amrex::MultiFab const*> state_fields(nlevels);
   for (int lev = 0; lev < nlevels; ++lev) {
      state_fields[lev] = state[lev].get();
   }
   validate_checkpoint_field_levels(checkpoint_fields[0], state_fields);
   write_checkpoint_field(
      state_fields,
      geometry,
      level_steps,
      refinement_ratios,
      finest_level,
      base_time,
      checkpoint_name,
      checkpoint_state_field_name(),
      compression,
      "PyFluidCheckpointHDF5-v1"
   );

   if (face_b != nullptr) {
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         amrex::Vector<amrex::MultiFab const*> face_fields(nlevels);
         for (int lev = 0; lev < nlevels; ++lev) {
            face_fields[lev] = (*face_b)[lev].b[dir].get();
         }
         validate_checkpoint_field_levels(checkpoint_fields[1 + dir], face_fields);
         write_checkpoint_field(
            face_fields,
            geometry,
            level_steps,
            refinement_ratios,
            finest_level,
            base_time,
            checkpoint_name,
            checkpoint_face_b_field_name(dir),
            compression,
            "PyFluidCheckpointHDF5-v1"
         );
      }
   }
   for (std::size_t extra_index = 0; extra_index < extra_fields.size(); ++extra_index) {
      CheckpointFieldSet const& field_set = extra_fields[extra_index];
      amrex::Vector<amrex::MultiFab const*> fields(nlevels);
      for (int lev = 0; lev < nlevels; ++lev) {
         fields[lev] = field_set.levels->at(lev).get();
      }
      const std::size_t field_index = 1 + (face_b == nullptr ? 0 : AMREX_SPACEDIM)
         + extra_index;
      validate_checkpoint_field_levels(checkpoint_fields[field_index], fields);
      write_checkpoint_field(
         fields,
         geometry,
         level_steps,
         refinement_ratios,
         finest_level,
         base_time,
         checkpoint_name,
         field_set.name,
         compression,
         "PyFluidCheckpointHDF5-v1"
      );
   }
}

CheckpointHeader read_checkpoint_header(std::string const& checkpoint_name)
{
   const std::string header_name = (std::filesystem::path(checkpoint_name) / std::string(CheckpointHeaderName)).string();
   std::ifstream header(header_name, std::ifstream::in | std::ifstream::binary);
   if (!header.good()) {
      amrex::FileOpenFailed(header_name);
   }

   std::string label;
   std::getline(header, label);
   if (label != CheckpointHeaderLabel) {
      amrex::Abort("unsupported checkpoint format: " + header_name);
   }

   CheckpointHeader chk;
   header >> chk.finest_level;
   header >> chk.state_components;
   if (chk.finest_level < 0 || chk.state_components <= 0) {
      amrex::Abort("checkpoint Header has invalid hierarchy or state-component metadata: " + header_name);
   }
   const int nlevels = chk.finest_level + 1;
   chk.level_steps.resize(nlevels);
   chk.dt.resize(nlevels);
   chk.time.resize(nlevels);
   chk.box_arrays.resize(nlevels);
   for (int lev = 0; lev < nlevels; ++lev) {
      header >> chk.level_steps[lev];
   }
   for (int lev = 0; lev < nlevels; ++lev) {
      header >> chk.dt[lev];
   }
   for (int lev = 0; lev < nlevels; ++lev) {
      header >> chk.time[lev];
   }
   for (int lev = 0; lev < nlevels; ++lev) {
      chk.box_arrays[lev].readFrom(header);
   }
   int num_fields = 0;
   header >> num_fields;
   if (num_fields <= 0) {
      amrex::Abort("checkpoint Header has no evolved fields: " + header_name);
   }
   chk.fields.reserve(num_fields);
   for (int field_index = 0; field_index < num_fields; ++field_index) {
      CheckpointFieldInfo field;
      amrex::IntVect nodal(AMREX_D_DECL(0, 0, 0));
      header >> field.name >> field.num_components;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         header >> nodal[dir];
      }
      if (field.name.empty() || field.num_components <= 0) {
         amrex::Abort("checkpoint Header has invalid field metadata: " + header_name);
      }
      field.index_type = amrex::IndexType(nodal);
      chk.fields.push_back(std::move(field));
   }
   header >> chk.reference_mass >> chk.reference_energy >> chk.reference_max_rho;
   header >> chk.aligned_regrid_ticks;
   header >> chk.aligned_base_regrid_tick;
   if (!header.good()) {
      amrex::Abort("checkpoint Header is incomplete: " + header_name);
   }
   return chk;
}

void validate_checkpoint_compatibility(
   std::string const& checkpoint_name,
   RunConfig const& cfg,
   int state_components,
   amrex::Geometry const& base_geometry,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   int finest_level
)
{
   const CheckpointMetadata metadata = parse_checkpoint_metadata(checkpoint_name);
   if (!checkpoint_metadata_is_complete(metadata)) {
      amrex::Abort(
         "checkpoint metadata is incomplete in " + checkpoint_name
         + "/Header.meta; restart requires the current checkpoint schema"
      );
   }

   const std::string mesh_axes = mesh_axes_or_default(mesh_coordinate_code(cfg.mesh_coordinate), cfg.mesh_axes);

   require_metadata(checkpoint_name, metadata, "source", "checkpoint");
   require_metadata_int(checkpoint_name, metadata, "checkpoint_schema_version", CheckpointSchemaVersion);
   require_metadata(checkpoint_name, metadata, "checkpoint_format", "hdf5");
   require_metadata_int(checkpoint_name, metadata, "checkpoint_state_components", state_components);
   require_metadata(checkpoint_name, metadata, "checkpoint_amrex_spacedim", std::to_string(AMREX_SPACEDIM));
   require_metadata(checkpoint_name, metadata, "checkpoint_mesh_coordinate", cfg.mesh_coordinate);
   require_metadata(checkpoint_name, metadata, "checkpoint_mesh_axes", mesh_axes);
   require_metadata(checkpoint_name, metadata, "checkpoint_domain", geometry_domain_string(base_geometry));
   require_metadata(checkpoint_name, metadata, "checkpoint_prob_lo", geometry_real_vector_string(base_geometry, false));
   require_metadata(checkpoint_name, metadata, "checkpoint_prob_hi", geometry_real_vector_string(base_geometry, true));
   require_metadata(checkpoint_name, metadata, "checkpoint_periodicity", geometry_periodicity_string(base_geometry));
   require_metadata(
      checkpoint_name,
      metadata,
      "checkpoint_refinement_ratios",
      refinement_ratios_string(refinement_ratios, finest_level)
   );
   require_metadata(checkpoint_name, metadata, "physics", cfg.physics);
   require_metadata(
      checkpoint_name,
      metadata,
      "spacetime_evolution",
      cfg.spacetime_evolution
   );
   require_metadata(
      checkpoint_name,
      metadata,
      "spacetime_matter_coupling",
      cfg.spacetime_matter_coupling
   );
   require_metadata(
      checkpoint_name, metadata, "spacetime_fixed_metric", cfg.spacetime_fixed_metric
   );
   require_metadata(
      checkpoint_name, metadata, "spacetime_acceleration1",
      output_metadata_real_string(cfg.spacetime_acceleration1)
   );
   require_metadata(
      checkpoint_name, metadata, "spacetime_acceleration2",
      output_metadata_real_string(cfg.spacetime_acceleration2)
   );
   require_metadata(
      checkpoint_name, metadata, "spacetime_acceleration3",
      output_metadata_real_string(cfg.spacetime_acceleration3)
   );
   require_metadata(
      checkpoint_name, metadata, "spacetime_origin1",
      output_metadata_real_string(cfg.spacetime_origin1)
   );
   require_metadata(
      checkpoint_name, metadata, "spacetime_origin2",
      output_metadata_real_string(cfg.spacetime_origin2)
   );
   require_metadata(
      checkpoint_name, metadata, "spacetime_origin3",
      output_metadata_real_string(cfg.spacetime_origin3)
   );
   require_metadata(
      checkpoint_name, metadata, "spacetime_mass",
      output_metadata_real_string(cfg.spacetime_mass)
   );
   require_metadata(checkpoint_name, metadata, "eos", cfg.eos);
   require_metadata(checkpoint_name, metadata, "checkpoint_ct", cfg.ct);
   require_metadata_int(checkpoint_name, metadata, "checkpoint_passive_scalars", cfg.passive_scalars);

   if (cfg.eos == "ideal_gas") {
      require_metadata(checkpoint_name, metadata, "checkpoint_gamma", output_metadata_real_string(cfg.gamma));
   } else if (cfg.eos == "polytrope") {
      require_metadata(
         checkpoint_name,
         metadata,
         "checkpoint_polytropic_k",
         output_metadata_real_string(cfg.polytropic_k)
      );
      require_metadata(
         checkpoint_name,
         metadata,
         "checkpoint_polytropic_gamma",
         output_metadata_real_string(cfg.polytropic_gamma)
      );
      require_metadata(
         checkpoint_name,
         metadata,
         "checkpoint_polytropic_density_breaks",
         output_metadata_real_vector_string(cfg.polytropic_density_breaks)
      );
      require_metadata(
         checkpoint_name,
         metadata,
         "checkpoint_polytropic_segment_gammas",
         output_metadata_real_vector_string(cfg.polytropic_segment_gammas)
      );
   } else if (cfg.eos == "hybrid_polytrope") {
      require_metadata(
         checkpoint_name,
         metadata,
         "checkpoint_polytropic_k",
         output_metadata_real_string(cfg.polytropic_k)
      );
      require_metadata(
         checkpoint_name,
         metadata,
         "checkpoint_polytropic_gamma",
         output_metadata_real_string(cfg.polytropic_gamma)
      );
      require_metadata(
         checkpoint_name,
         metadata,
         "checkpoint_polytropic_density_breaks",
         output_metadata_real_vector_string(cfg.polytropic_density_breaks)
      );
      require_metadata(
         checkpoint_name,
         metadata,
         "checkpoint_polytropic_segment_gammas",
         output_metadata_real_vector_string(cfg.polytropic_segment_gammas)
      );
      require_metadata(
         checkpoint_name,
         metadata,
         "checkpoint_thermal_gamma",
         output_metadata_real_string(cfg.thermal_gamma)
      );
   }
}

} // namespace pyfluid_amrex
