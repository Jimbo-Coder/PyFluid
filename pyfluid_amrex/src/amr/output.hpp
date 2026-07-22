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
#include <utility>
#include <vector>

#include "pyfluid_amrex/ct.hpp"
#include "amr/derived_aux.hpp"
#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/solid_fields.hpp"
#include "pyfluid_amrex/gravity.hpp"
#include "pyfluid_amrex/output_metadata.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/temporal_cadence.hpp"

namespace pyfluid_amrex {

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

struct AuxOutputEntry {
   AuxGridField field = AuxGridField::Unknown;
   std::vector<int> source_components{};
};

struct AuxOutputLayout {
   GridFieldCentering centering = GridFieldCentering::Cell;
   amrex::Vector<std::string> component_names{};
   std::vector<AuxOutputEntry> entries{};
};

inline std::vector<AuxOutputLayout> aux_output_layouts(RunConfig const& cfg)
{
   const std::vector<std::string> selected_names = aux_gf_names(cfg);
   std::vector<AuxOutputLayout> layouts;
   for (GridFieldCentering centering : grid_field_centerings()) {
      AuxOutputLayout layout;
      layout.centering = centering;
      for (std::string const& name : selected_names) {
         const AuxGridField field = aux_grid_field_from_name(name);
         const AuxGridFieldInfo info = aux_grid_field_info(field);
         if (info.field == AuxGridField::Unknown) {
            amrex::Abort("cannot build output layout for unknown auxiliary grid field " + name);
         }
         AuxOutputEntry entry;
         entry.field = field;
         for (int component = 0; component < static_cast<int>(info.components.size()); ++component) {
            if (info.components[component].centering == centering) {
               entry.source_components.push_back(component);
               layout.component_names.emplace_back(info.components[component].name);
            }
         }
         if (!entry.source_components.empty()) {
            layout.entries.push_back(std::move(entry));
         }
      }
      if (!layout.component_names.empty()) {
         layouts.push_back(std::move(layout));
      }
   }
   return layouts;
}

inline void copy_selected_aux_components(
   amrex::MultiFab& output,
   amrex::MultiFab const& source,
   AuxOutputEntry const& entry,
   int& dst
)
{
   for (int component : entry.source_components) {
      amrex::MultiFab::Copy(output, source, component, dst, 1, 0);
      ++dst;
   }
}

inline void fill_aux_gf_output(
   amrex::MultiFab& output,
   amrex::MultiFab const& state,
   amrex::MultiFab const* spacetime_state,
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
   AuxOutputLayout const& layout
)
{
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
   for (AuxOutputEntry const& entry : layout.entries) {
      const AuxGridField field = entry.field;
      const AuxGridFieldInfo info = aux_grid_field_info(field);
      switch (field) {
      case AuxGridField::CurrentDensity: {
         const EdgeElectricField& edge_j = edge_current_for_output();
         for (int component : entry.source_components) {
            if (!edge_j.e[component]) {
               amrex::Abort("requested edge current-density output component is unavailable");
            }
            amrex::MultiFab::Copy(output, *edge_j.e[component], 0, dst, 1, 0);
            ++dst;
         }
         break;
      }
      case AuxGridField::ElectricField: {
         const EdgeElectricField& edge_e = edge_electric_for_output();
         for (int component : entry.source_components) {
            if (!edge_e.e[component]) {
               amrex::Abort("requested edge electric-field output component is unavailable");
            }
            amrex::MultiFab::Copy(output, *edge_e.e[component], 0, dst, 1, 0);
            ++dst;
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
         copy_selected_aux_components(output, divb, entry, dst);
         break;
      }
      case AuxGridField::MagneticEnergy: {
         amrex::MultiFab magnetic_energy(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_magnetic_energy_density(magnetic_energy, state);
         copy_selected_aux_components(output, magnetic_energy, entry, dst);
         break;
      }
      case AuxGridField::Vorticity: {
         amrex::MultiFab vorticity(state.boxArray(), state.DistributionMap(), 3, 0);
         fill_vorticity_output(vorticity, state, geom);
         copy_selected_aux_components(output, vorticity, entry, dst);
         break;
      }
      case AuxGridField::MachNumber: {
         amrex::MultiFab mach(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_mach_number_output(mach, state, geom, time, cfg);
         copy_selected_aux_components(output, mach, entry, dst);
         break;
      }
      case AuxGridField::PlasmaBeta: {
         amrex::MultiFab plasma_beta(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_plasma_beta_output(plasma_beta, state, geom, time, cfg);
         copy_selected_aux_components(output, plasma_beta, entry, dst);
         break;
      }
      case AuxGridField::ThermalEntropyProxy: {
         amrex::MultiFab thermal_entropy_proxy(
            state.boxArray(), state.DistributionMap(), 1, 0
         );
         fill_thermal_entropy_proxy_output(
            thermal_entropy_proxy, state, geom, time, cfg
         );
         copy_selected_aux_components(
            output, thermal_entropy_proxy, entry, dst
         );
         break;
      }
      case AuxGridField::AmrTagMetric: {
         amrex::MultiFab tag_metric(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_amr_tag_metric_output(tag_metric, state, spacetime_state, geom, time, cfg);
         copy_selected_aux_components(output, tag_metric, entry, dst);
         break;
      }
      case AuxGridField::GravityPotential:
         if (gravity_phi == nullptr) {
            amrex::Abort("phi output requested before gravity potential storage exists");
         }
         copy_selected_aux_components(output, *gravity_phi, entry, dst);
         break;
      case AuxGridField::GravityGradient: {
         amrex::MultiFab cell_gradient(state.boxArray(), state.DistributionMap(), 3, 0);
         fill_cell_centered_gravity_gradient(cell_gradient, gravity_gradient);
         copy_selected_aux_components(output, cell_gradient, entry, dst);
         break;
      }
      case AuxGridField::SolidMask: {
         amrex::MultiFab mask_output(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_solid_mask_output(mask_output, solid_mask);
         copy_selected_aux_components(output, mask_output, entry, dst);
         break;
      }
      case AuxGridField::SolidVolumeFraction: {
         amrex::MultiFab volume_fraction(state.boxArray(), state.DistributionMap(), 1, 0);
         if (solid_volume_fraction != nullptr) {
            amrex::MultiFab::Copy(volume_fraction, *solid_volume_fraction, 0, 0, 1, 0);
         } else {
            fill_solid_volume_fraction_output(volume_fraction, solid_mask, geom, cfg, time);
         }
         copy_selected_aux_components(output, volume_fraction, entry, dst);
         break;
      }
      case AuxGridField::SolidSweptVolume: {
         amrex::MultiFab swept_volume(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_solid_swept_volume_output(swept_volume, geom, cfg, cfg.solid_swept_reference_time, time);
         copy_selected_aux_components(output, swept_volume, entry, dst);
         break;
      }
      case AuxGridField::SolidAperture1:
         if (solid_aperture[0] == nullptr) {
            amrex::Abort("solid_aperture1 output requested before the discrete AMR aperture field exists");
         }
         copy_selected_aux_components(output, *solid_aperture[0], entry, dst);
         break;
#if (AMREX_SPACEDIM >= 2)
      case AuxGridField::SolidAperture2:
         if (solid_aperture[1] == nullptr) {
            amrex::Abort("solid_aperture2 output requested before the discrete AMR aperture field exists");
         }
         copy_selected_aux_components(output, *solid_aperture[1], entry, dst);
         break;
#endif
#if (AMREX_SPACEDIM == 3)
      case AuxGridField::SolidAperture3:
         if (solid_aperture[2] == nullptr) {
            amrex::Abort("solid_aperture3 output requested before the discrete AMR aperture field exists");
         }
         copy_selected_aux_components(output, *solid_aperture[2], entry, dst);
         break;
#endif
      case AuxGridField::SolidPhi: {
         amrex::MultiFab phi_output(state.boxArray(), state.DistributionMap(), 1, 0);
         fill_solid_phi_output(phi_output, geom, cfg, time);
         copy_selected_aux_components(output, phi_output, entry, dst);
         break;
      }
      default:
         amrex::Abort("unsupported auxiliary grid function in output: " + info.name);
      }
   }
   AMREX_ALWAYS_ASSERT(dst == static_cast<int>(layout.component_names.size()));
}

inline void write_aux_gf_plotfile(
   amrex::MultiFab const& state,
   amrex::MultiFab const* spacetime_state,
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
   const auto layouts = aux_output_layouts(cfg);
   if (layouts.empty()) {
      return;
   }

   const std::filesystem::path plot_dir = std::filesystem::path(cfg.output_dir) / "auxGF";
   std::filesystem::create_directories(plot_dir);
   for (AuxOutputLayout const& layout : layouts) {
      const int ncomp = static_cast<int>(layout.component_names.size());
      const amrex::BoxArray output_ba = amrex::convert(
         state.boxArray(), grid_field_index_type(layout.centering)
      );
      amrex::MultiFab output(output_ba, state.DistributionMap(), ncomp, 0);
      fill_aux_gf_output(
         output,
         state,
         spacetime_state,
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
         layout
      );

      const std::string stem = (
         plot_dir / ("auxGF" + grid_field_stem_suffix(layout.centering))
      ).string();
      write_output_gf(
         output,
         stem,
         geom,
         layout.component_names,
         time,
         step,
         output_cfg,
         "aux"
      );
   }
}

inline void write_multilevel_aux_gf_plotfile(
   std::vector<std::unique_ptr<amrex::MultiFab>> const& state,
   std::vector<std::unique_ptr<amrex::MultiFab>> const& spacetime_state,
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
   const auto layouts = aux_output_layouts(cfg);
   if (layouts.empty()) {
      return;
   }

   const std::filesystem::path plot_dir = std::filesystem::path(cfg.output_dir) / "auxGF";
   std::filesystem::create_directories(plot_dir);
   for (AuxOutputLayout const& layout : layouts) {
      const int ncomp = static_cast<int>(layout.component_names.size());
      std::vector<std::unique_ptr<amrex::MultiFab>> output(finest_level + 1);
      amrex::Vector<amrex::MultiFab const*> levels(finest_level + 1);
      for (int lev = 0; lev <= finest_level; ++lev) {
         const amrex::BoxArray output_ba = amrex::convert(
            state[lev]->boxArray(), grid_field_index_type(layout.centering)
         );
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
            spacetime_state.empty() ? nullptr : spacetime_state[lev].get(),
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
            layout
         );
         levels[lev] = output[lev].get();
      }

      const std::string stem = (
         plot_dir / ("auxGF" + grid_field_stem_suffix(layout.centering))
      ).string();
      write_output_gf(
         finest_level,
         output,
         geom,
         level_steps,
         ref_ratio,
         stem,
         layout.component_names,
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
