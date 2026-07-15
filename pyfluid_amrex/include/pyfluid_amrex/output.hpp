#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H>

#ifdef AMREX_USE_HDF5
#include <AMReX_PlotFileUtilHDF5.H>
#endif

#include <fstream>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/derived_aux.hpp"
#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/solid_fields.hpp"
#include "pyfluid_amrex/gravity.hpp"
#include "pyfluid_amrex/output_metadata.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/config.hpp"

namespace pyfluid_amrex {

enum class TemporalCadenceUnit {
   Steps,
   Disabled,
};

struct TemporalCadence {
   TemporalCadenceUnit unit = TemporalCadenceUnit::Disabled;
   int interval = -1;

   constexpr bool should_record(int step, bool reached_final) const noexcept
   {
      return unit == TemporalCadenceUnit::Steps
         && interval > 0
         && step > 0
         && (step % interval == 0 || reached_final);
   }

   constexpr bool should_record_initial() const noexcept
   {
      return unit == TemporalCadenceUnit::Steps && interval == 0;
   }
};

inline void append_plotfile_metadata_on_io_rank(
   std::string const& plotfile_stem,
   RunConfig const& cfg,
   int step,
   amrex::Real time,
   std::string_view const kind
)
{
   const std::string compression = cfg.plotfile_format == "hdf5"
      ? resolved_plotfile_compression(cfg.plotfile_compression)
      : "None@0";
   if (cfg.plotfile_format == "hdf5") {
#ifdef AMREX_USE_HDF5
      const bool has_h5_suffix =
         plotfile_stem.size() >= 3 && plotfile_stem.compare(plotfile_stem.size() - 3, 3, ".h5") == 0;
      const std::string hdf5_path = has_h5_suffix ? plotfile_stem : (plotfile_stem + ".h5");
      if (write_output_metadata_hdf5_attributes(
         hdf5_path,
         "plotfile",
         cfg,
         step,
         time,
         kind,
         -1,
         compression,
         {{"plotfile_format", cfg.plotfile_format}, {"plotfile_compression", compression}}
      )) {
         return;
      }
#else
      amrex::Abort("plotfile_format = hdf5 requires AMReX built with HDF5 support");
#endif

      const std::filesystem::path meta_file = std::filesystem::path(plotfile_stem + ".meta");
      std::ofstream out(meta_file, std::ios::out | std::ios::trunc);
      if (!out.good()) {
         return;
      }
      append_output_metadata(
         out,
         "plotfile",
         cfg,
         step,
         time,
         kind,
         -1,
         compression,
         {{"plotfile_format", cfg.plotfile_format}, {"plotfile_compression", compression}}
      );
      return;
   }

   std::filesystem::path header = std::filesystem::path(plotfile_stem) / "Header";
   std::ofstream out(header, std::ios::out | std::ios::app);
   if (!out.good()) {
      return;
   }
   append_output_metadata(
      out,
      "plotfile",
      cfg,
      step,
      time,
      kind,
      -1,
      compression,
      {{"plotfile_format", cfg.plotfile_format}, {"plotfile_compression", compression}}
   );
}

inline void append_plotfile_metadata(
   std::string const& plotfile_stem,
   RunConfig const& cfg,
   int step,
   amrex::Real time,
   std::string_view const kind
)
{
   amrex::ParallelDescriptor::Barrier();
   if (amrex::ParallelDescriptor::IOProcessor()) {
      append_plotfile_metadata_on_io_rank(plotfile_stem, cfg, step, time, kind);
   }
   amrex::ParallelDescriptor::Barrier();
}

inline void write_single_level_plotfile_impl(
   amrex::MultiFab const& mf,
   std::string const& plotfile_stem,
   amrex::Geometry const& geom,
   amrex::Vector<std::string> const& names,
   amrex::Real time,
   int step,
   RunConfig const& cfg,
   std::string_view const kind = "state"
)
{
   const std::string plotfile = amrex::Concatenate(plotfile_stem, step, 5);
   if (cfg.plotfile_format == "hdf5") {
#ifdef AMREX_USE_HDF5
      const std::string compression = resolved_plotfile_compression(cfg.plotfile_compression);
      amrex::WriteSingleLevelPlotfileHDF5(plotfile, mf, names, geom, time, step, compression);
#else
      amrex::Abort("plotfile_format = hdf5 requires AMReX built with HDF5 support");
#endif
   } else {
      amrex::WriteSingleLevelPlotfile(plotfile, mf, names, geom, time, step);
   }
   append_plotfile_metadata(plotfile, cfg, step, time, kind);
}

inline void write_multilevel_plotfile_impl(
   int finest_level,
   std::vector<std::unique_ptr<amrex::MultiFab>> const& levels,
   std::string const& plotfile_stem,
   amrex::Vector<amrex::Geometry> const& geom,
   amrex::Vector<int> const& level_steps,
   amrex::Vector<amrex::IntVect> const& ref_ratio,
   amrex::Vector<std::string> const& names,
   amrex::Real time,
   int step,
   RunConfig const& cfg,
   std::string_view const kind = "state"
)
{
   amrex::Vector<amrex::MultiFab const*> output_levels(finest_level + 1);
   for (int lev = 0; lev <= finest_level; ++lev) {
      output_levels[lev] = levels[lev].get();
   }
   const std::string plotfile = amrex::Concatenate(plotfile_stem, step, 5);
   if (cfg.plotfile_format == "hdf5") {
#ifdef AMREX_USE_HDF5
      const std::string compression = resolved_plotfile_compression(cfg.plotfile_compression);
      amrex::WriteMultiLevelPlotfileHDF5(
         plotfile,
         finest_level + 1,
         output_levels,
         names,
         geom,
         time,
         level_steps,
         ref_ratio,
         compression
      );
#else
      amrex::Abort("plotfile_format = hdf5 requires AMReX built with HDF5 support");
#endif
   } else {
      amrex::WriteMultiLevelPlotfile(
         plotfile,
         finest_level + 1,
         output_levels,
         names,
         geom,
         time,
         level_steps,
         ref_ratio
      );
   }
   append_plotfile_metadata(plotfile, cfg, step, time, kind);
}

inline void write_output_gf(
   amrex::MultiFab const& mf,
   std::string const& plotfile_stem,
   amrex::Geometry const& geom,
   amrex::Vector<std::string> const& names,
   amrex::Real time,
   int step,
   RunConfig const& cfg,
   std::string_view const kind = "state"
)
{
   write_single_level_plotfile_impl(mf, plotfile_stem, geom, names, time, step, cfg, kind);
}

inline void write_output_gf(
   int finest_level,
   std::vector<std::unique_ptr<amrex::MultiFab>> const& levels,
   amrex::Vector<amrex::Geometry> const& geom,
   amrex::Vector<int> const& level_steps,
   amrex::Vector<amrex::IntVect> const& ref_ratio,
   std::string const& plotfile_stem,
   amrex::Vector<std::string> const& names,
   amrex::Real time,
   int step,
   RunConfig const& cfg,
   std::string_view const kind = "state"
)
{
   if (finest_level <= 0) {
      write_output_gf(*levels[0], plotfile_stem, geom[0], names, time, step, cfg, kind);
      return;
   }
   write_multilevel_plotfile_impl(
      finest_level,
      levels,
      plotfile_stem,
      geom,
      level_steps,
      ref_ratio,
      names,
      time,
      step,
      cfg,
      kind
   );
}

inline void write_plotfile(
   amrex::MultiFab const& state,
   amrex::Geometry const& geom,
   std::string const& output_dir,
   int step,
   amrex::Real time,
   bool mhd = false,
   int passive_scalars = 0,
   RunConfig const& cfg = RunConfig{}
)
{
   const std::filesystem::path plot_dir = std::filesystem::path(output_dir) / "stateGF";
   std::filesystem::create_directories(plot_dir);
   write_output_gf(
      state,
      (plot_dir / "stateGF").string(),
      geom,
      state_grid_field_names(physical_system(cfg), passive_scalars),
      time,
      step,
      cfg,
      "state"
   );
}

inline std::array<GridFieldCentering, 7> aux_output_centerings()
{
   return {
      GridFieldCentering::Cell,
      GridFieldCentering::Face1,
      GridFieldCentering::Face2,
      GridFieldCentering::Face3,
      GridFieldCentering::Edge1,
      GridFieldCentering::Edge2,
      GridFieldCentering::Edge3,
   };
}

inline amrex::IndexType grid_field_index_type(GridFieldCentering centering)
{
   if (centering == GridFieldCentering::Cell) {
      return amrex::IndexType(amrex::IntVect(AMREX_D_DECL(0, 0, 0)));
   }
   if (centering == GridFieldCentering::Face1) {
      return amrex::IndexType(amrex::IntVect::TheDimensionVector(0));
   }
#if (AMREX_SPACEDIM >= 2)
   if (centering == GridFieldCentering::Face2) {
      return amrex::IndexType(amrex::IntVect::TheDimensionVector(1));
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (centering == GridFieldCentering::Face3) {
      return amrex::IndexType(amrex::IntVect::TheDimensionVector(2));
   }
#endif
   if (centering == GridFieldCentering::Edge1) {
      return edge_index_type(0);
   }
   if (centering == GridFieldCentering::Edge2) {
      return edge_index_type(1);
   }
   if (centering == GridFieldCentering::Edge3) {
      return edge_index_type(2);
   }
   amrex::Abort("unsupported grid-field centering for output");
   return amrex::IndexType(amrex::IntVect(AMREX_D_DECL(0, 0, 0)));
}

inline std::string aux_plotfile_stem_suffix(GridFieldCentering centering)
{
   switch (centering) {
   case GridFieldCentering::Cell:
      return "";
   case GridFieldCentering::Face1:
      return "_face1";
   case GridFieldCentering::Face2:
      return "_face2";
   case GridFieldCentering::Face3:
      return "_face3";
   case GridFieldCentering::Edge1:
      return "_edge1";
   case GridFieldCentering::Edge2:
      return "_edge2";
   case GridFieldCentering::Edge3:
      return "_edge3";
   }
   return "_unknown";
}

inline bool aux_field_has_centering(AuxGridFieldInfo const& info, GridFieldCentering centering)
{
   for (int component = 0; component < info.components; ++component) {
      if (info.centerings[component] == centering) {
         return true;
      }
   }
   return false;
}

inline amrex::Vector<std::string> aux_output_names(RunConfig const& cfg, GridFieldCentering centering)
{
   amrex::Vector<std::string> output_names;
   for (std::string const& name : aux_gf_names(cfg)) {
      const AuxGridField field = aux_grid_field_from_name(name);
      const AuxGridFieldInfo info = aux_grid_field_info(field);
      const auto component_names = aux_grid_field_output_names(field);
      for (int component = 0; component < info.components; ++component) {
         if (info.centerings[component] == centering) {
            output_names.push_back(component_names[component]);
         }
      }
   }
   return output_names;
}

inline int aux_output_component_count(RunConfig const& cfg, GridFieldCentering centering)
{
   int ncomp = 0;
   for (std::string const& name : aux_gf_names(cfg)) {
      const AuxGridFieldInfo info = aux_grid_field_info(aux_grid_field_from_name(name));
      for (int component = 0; component < info.components; ++component) {
         if (info.centerings[component] == centering) {
            ++ncomp;
         }
      }
   }
   return ncomp;
}

inline void copy_matching_aux_components(
   amrex::MultiFab& output,
   amrex::MultiFab const& source,
   AuxGridFieldInfo const& info,
   GridFieldCentering centering,
   int& dst
)
{
   for (int component = 0; component < info.components; ++component) {
      if (info.centerings[component] == centering) {
         amrex::MultiFab::Copy(output, source, component, dst, 1, 0);
         ++dst;
      }
   }
}

inline void fill_aux_gf_output(
   amrex::MultiFab& output,
   amrex::MultiFab const& state,
   FaceMagneticField const* face_b,
   EdgeElectricField const* edge_current,
   EdgeElectricField const* edge_electric,
   amrex::MultiFab const* gravity_phi,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& gravity_gradient,
   amrex::iMultiFab const* solid_mask,
   amrex::MultiFab const* solid_volume_fraction,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& solid_aperture,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   amrex::Real time,
   GridFieldCentering centering
)
{
   const std::vector<std::string> aux_names = aux_gf_names(cfg);
   std::unique_ptr<EdgeElectricField> edge_current_work;
   std::unique_ptr<EdgeElectricField> edge_electric_work;

   auto edge_current_for_output = [&]() -> EdgeElectricField const& {
      if (edge_current != nullptr) {
         return *edge_current;
      }
      if (!edge_current_work) {
         edge_current_work = std::make_unique<EdgeElectricField>();
         allocate_edge_electric_field(*edge_current_work, state.boxArray(), state.DistributionMap());
         if (face_b != nullptr) {
            fill_edge_current_density_from_faces(*edge_current_work, *face_b, geom, cfg);
         } else {
            fill_edge_current_density_from_cells(*edge_current_work, state, geom);
         }
         enforce_edge_electric_periodicity(*edge_current_work, geom);
      }
      return *edge_current_work;
   };

   auto edge_electric_for_output = [&]() -> EdgeElectricField const& {
      if (edge_electric != nullptr) {
         return *edge_electric;
      }
      if (!edge_electric_work) {
         edge_electric_work = std::make_unique<EdgeElectricField>();
         allocate_edge_electric_field(*edge_electric_work, state.boxArray(), state.DistributionMap());
         EdgeElectricField const* j = nullptr;
         if (cfg.resistivity != 0.0) {
            j = &edge_current_for_output();
         }
         fill_edge_electric_field_for_output(*edge_electric_work, state, face_b, j, geom, cfg.resistivity, cfg);
      }
      return *edge_electric_work;
   };

   int dst = 0;
   for (std::string const& name : aux_names) {
      const AuxGridField field = aux_grid_field_from_name(name);
      const AuxGridFieldInfo info = aux_grid_field_info(field);
      if (!aux_field_has_centering(info, centering)) {
         continue;
      }
      switch (field) {
      case AuxGridField::CurrentDensity: {
         const EdgeElectricField& edge_j = edge_current_for_output();
         for (int component = 0; component < info.components; ++component) {
            if (info.centerings[component] == centering) {
               if (!edge_j.e[component]) {
                  amrex::Abort("requested edge current-density output component is unavailable");
               }
               amrex::MultiFab::Copy(output, *edge_j.e[component], 0, dst, 1, 0);
               ++dst;
            }
         }
         break;
      }
      case AuxGridField::ElectricField: {
         const EdgeElectricField& edge_e = edge_electric_for_output();
         for (int component = 0; component < info.components; ++component) {
            if (info.centerings[component] == centering) {
               if (!edge_e.e[component]) {
                  amrex::Abort("requested edge electric-field output component is unavailable");
               }
               amrex::MultiFab::Copy(output, *edge_e.e[component], 0, dst, 1, 0);
               ++dst;
            }
         }
         break;
      }
      case AuxGridField::DivB: {
         amrex::MultiFab divb(state.boxArray(), state.DistributionMap(), 1, 0);
         if (face_b != nullptr) {
            fill_face_divergence(
               divb,
               *face_b,
               geom,
               mesh_coordinate_code(cfg.mesh_coordinate),
               mesh_axes_layout(cfg),
               false
            );
         } else {
            fill_cell_divergence(divb, state, geom, false);
         }
         copy_matching_aux_components(output, divb, info, centering, dst);
         break;
      }
      case AuxGridField::MagneticEnergy: {
         amrex::MultiFab magnetic_energy(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_magnetic_energy_density(magnetic_energy, state);
         copy_matching_aux_components(output, magnetic_energy, info, centering, dst);
         break;
      }
      case AuxGridField::Vorticity: {
         amrex::MultiFab vorticity(state.boxArray(), state.DistributionMap(), 3, 0);
         fill_vorticity_output(vorticity, state, geom);
         copy_matching_aux_components(output, vorticity, info, centering, dst);
         break;
      }
      case AuxGridField::MachNumber: {
         amrex::MultiFab mach(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_mach_number_output(mach, state, geom, time, cfg);
         copy_matching_aux_components(output, mach, info, centering, dst);
         break;
      }
      case AuxGridField::PlasmaBeta: {
         amrex::MultiFab plasma_beta(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_plasma_beta_output(plasma_beta, state, geom, time, cfg);
         copy_matching_aux_components(output, plasma_beta, info, centering, dst);
         break;
      }
      case AuxGridField::EntropyProxy: {
         amrex::MultiFab entropy(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_entropy_proxy_output(entropy, state, geom, time, cfg);
         copy_matching_aux_components(output, entropy, info, centering, dst);
         break;
      }
      case AuxGridField::AmrTagMetric: {
         amrex::MultiFab tag_metric(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_amr_tag_metric_output(tag_metric, state, geom, time, cfg);
         copy_matching_aux_components(output, tag_metric, info, centering, dst);
         break;
      }
      case AuxGridField::GravityPotential:
         if (gravity_phi == nullptr) {
            amrex::Abort("phi output requested before gravity potential storage exists");
         }
         copy_matching_aux_components(output, *gravity_phi, info, centering, dst);
         break;
      case AuxGridField::GravityGradient: {
         amrex::MultiFab cell_gradient(state.boxArray(), state.DistributionMap(), 3, 0);
         fill_cell_centered_gravity_gradient(cell_gradient, gravity_gradient);
         copy_matching_aux_components(output, cell_gradient, info, centering, dst);
         break;
      }
      case AuxGridField::SolidMask: {
         amrex::MultiFab mask_output(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_solid_mask_output(mask_output, solid_mask);
         copy_matching_aux_components(output, mask_output, info, centering, dst);
         break;
      }
      case AuxGridField::SolidVolumeFraction: {
         amrex::MultiFab volume_fraction(state.boxArray(), state.DistributionMap(), 1, 0);
         if (solid_volume_fraction != nullptr) {
            amrex::MultiFab::Copy(volume_fraction, *solid_volume_fraction, 0, 0, 1, 0);
         } else {
            fill_solid_volume_fraction_output(volume_fraction, solid_mask, geom, cfg, time);
         }
         copy_matching_aux_components(output, volume_fraction, info, centering, dst);
         break;
      }
      case AuxGridField::SolidSweptVolume: {
         amrex::MultiFab swept_volume(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_solid_swept_volume_output(swept_volume, geom, cfg, cfg.solid_swept_reference_time, time);
         copy_matching_aux_components(output, swept_volume, info, centering, dst);
         break;
      }
      case AuxGridField::SolidAperture1:
         if (solid_aperture[0] == nullptr) {
            amrex::Abort("solid_aperture1 output requested before the discrete AMR aperture field exists");
         }
         copy_matching_aux_components(output, *solid_aperture[0], info, centering, dst);
         break;
#if (AMREX_SPACEDIM >= 2)
      case AuxGridField::SolidAperture2:
         if (solid_aperture[1] == nullptr) {
            amrex::Abort("solid_aperture2 output requested before the discrete AMR aperture field exists");
         }
         copy_matching_aux_components(output, *solid_aperture[1], info, centering, dst);
         break;
#endif
#if (AMREX_SPACEDIM == 3)
      case AuxGridField::SolidAperture3:
         if (solid_aperture[2] == nullptr) {
            amrex::Abort("solid_aperture3 output requested before the discrete AMR aperture field exists");
         }
         copy_matching_aux_components(output, *solid_aperture[2], info, centering, dst);
         break;
#endif
      case AuxGridField::SolidPhi: {
         amrex::MultiFab phi_output(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_solid_phi_output(phi_output, geom, cfg, time);
         copy_matching_aux_components(output, phi_output, info, centering, dst);
         break;
      }
      default:
         amrex::Abort("unsupported auxiliary grid function in output: " + name);
      }
   }
}

inline void write_aux_gf_plotfile(
   amrex::MultiFab const& state,
   FaceMagneticField const* face_b,
   EdgeElectricField const* edge_current,
   EdgeElectricField const* edge_electric,
   amrex::MultiFab const* gravity_phi,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& gravity_gradient,
   amrex::iMultiFab const* solid_mask,
   amrex::MultiFab const* solid_volume_fraction,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& solid_aperture,
   amrex::Geometry const& geom,
   RunConfig const& cfg,
   int step,
   amrex::Real time,
   RunConfig const& output_cfg = RunConfig{}
)
{
   if (aux_gf_names(cfg).empty()) {
      return;
   }

   const std::filesystem::path plot_dir = std::filesystem::path(cfg.output_dir) / "auxGF";
   std::filesystem::create_directories(plot_dir);
   for (GridFieldCentering centering : aux_output_centerings()) {
      const int ncomp = aux_output_component_count(cfg, centering);
      if (ncomp == 0) {
         continue;
      }
      const amrex::BoxArray output_ba = amrex::convert(state.boxArray(), grid_field_index_type(centering));
      amrex::MultiFab output(output_ba, state.DistributionMap(), ncomp, 0);
      fill_aux_gf_output(
         output,
         state,
         face_b,
         edge_current,
         edge_electric,
         gravity_phi,
         gravity_gradient,
         solid_mask,
         solid_volume_fraction,
         solid_aperture,
         geom,
         cfg,
         time,
         centering
      );

      const std::string stem = (plot_dir / ("auxGF" + aux_plotfile_stem_suffix(centering))).string();
      write_output_gf(
         output,
         stem,
         geom,
         aux_output_names(cfg, centering),
         time,
         step,
         output_cfg,
         "aux"
      );
   }
}

inline void write_multilevel_aux_gf_plotfile(
   std::vector<std::unique_ptr<amrex::MultiFab>> const& state,
   std::vector<FaceMagneticField> const& face_b,
   std::vector<EdgeElectricField> const& edge_current,
   std::vector<EdgeElectricField> const& edge_electric,
   std::vector<std::unique_ptr<amrex::MultiFab>> const& gravity_phi,
   std::vector<GravityGradient> const& gravity_gradient,
   std::vector<std::unique_ptr<amrex::iMultiFab>> const& solid_mask,
   std::vector<std::unique_ptr<amrex::MultiFab>> const& solid_volume_fraction,
   std::array<std::vector<std::unique_ptr<amrex::MultiFab>>, AMREX_SPACEDIM> const& solid_aperture,
   amrex::Vector<amrex::Geometry> const& geom,
   amrex::Vector<int> const& level_steps,
   amrex::Vector<amrex::IntVect> const& ref_ratio,
   int finest_level,
   RunConfig const& cfg,
   int step,
   amrex::Real time,
   RunConfig const& output_cfg = RunConfig{}
)
{
   if (aux_gf_names(cfg).empty()) {
      return;
   }

   const std::filesystem::path plot_dir = std::filesystem::path(cfg.output_dir) / "auxGF";
   std::filesystem::create_directories(plot_dir);
   for (GridFieldCentering centering : aux_output_centerings()) {
      const int ncomp = aux_output_component_count(cfg, centering);
      if (ncomp == 0) {
         continue;
      }
      std::vector<std::unique_ptr<amrex::MultiFab>> output(finest_level + 1);
      amrex::Vector<amrex::MultiFab const*> levels(finest_level + 1);
      for (int lev = 0; lev <= finest_level; ++lev) {
         const amrex::BoxArray output_ba = amrex::convert(state[lev]->boxArray(), grid_field_index_type(centering));
         output[lev] = std::make_unique<amrex::MultiFab>(output_ba, state[lev]->DistributionMap(), ncomp, 0);
         FaceMagneticField const* face_b_ptr = ct_enabled(cfg) ? &face_b[lev] : nullptr;
         EdgeElectricField const* edge_current_ptr = edge_current[lev].e[0] ? &edge_current[lev] : nullptr;
         EdgeElectricField const* edge_electric_ptr = edge_electric[lev].e[0] ? &edge_electric[lev] : nullptr;
         amrex::MultiFab const* gravity_phi_ptr = gravity_phi[lev] ? gravity_phi[lev].get() : nullptr;
         const auto gravity_gradient_ptrs_for_level = gravity_gradient_const_ptrs(gravity_gradient[lev]);
         amrex::iMultiFab const* solid_ptr = solid_mask[lev] ? solid_mask[lev].get() : nullptr;
         amrex::MultiFab const* volume_fraction_ptr =
            solid_volume_fraction[lev] ? solid_volume_fraction[lev].get() : nullptr;
         std::array<amrex::MultiFab const*, AMREX_SPACEDIM> aperture_ptr{};
         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            aperture_ptr[dir] = solid_aperture[dir][lev] ? solid_aperture[dir][lev].get() : nullptr;
         }
         fill_aux_gf_output(
            *output[lev],
            *state[lev],
            face_b_ptr,
            edge_current_ptr,
            edge_electric_ptr,
            gravity_phi_ptr,
            gravity_gradient_ptrs_for_level,
            solid_ptr,
            volume_fraction_ptr,
            aperture_ptr,
            geom[lev],
            cfg,
            time,
            centering
         );
         levels[lev] = output[lev].get();
      }

   const std::string stem = (plot_dir / ("auxGF" + aux_plotfile_stem_suffix(centering))).string();
   write_output_gf(
      finest_level,
      output,
      geom,
      level_steps,
      ref_ratio,
      stem,
      aux_output_names(cfg, centering),
      time,
      step,
      output_cfg,
      "aux"
      );
   }
}

inline void write_multilevel_plotfile(
   std::vector<std::unique_ptr<amrex::MultiFab>> const& state,
   amrex::Vector<amrex::Geometry> const& geom,
   amrex::Vector<int> const& level_steps,
   amrex::Vector<amrex::IntVect> const& ref_ratio,
   int finest_level,
   std::string const& output_dir,
   int step,
   amrex::Real time,
   bool mhd = false,
   int passive_scalars = 0,
   RunConfig const& cfg = RunConfig{}
)
{
   std::filesystem::path const plot_dir = std::filesystem::path(output_dir) / "stateGF";
   std::filesystem::create_directories(plot_dir);
   write_output_gf(
      finest_level,
      state,
      geom,
      level_steps,
      ref_ratio,
      (plot_dir / "stateGF").string(),
      state_grid_field_names(physical_system(cfg), passive_scalars),
      time,
      step,
      cfg,
      "state"
   );
}

} // namespace pyfluid_amrex
