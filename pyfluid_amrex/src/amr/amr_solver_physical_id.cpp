#include "amr/amr_solver_base.hpp"

#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/initial_data.hpp"
#include "pyfluid_amrex/initial_data/atmosphere.hpp"
#include "pyfluid_amrex/initial_data/elliptic.hpp"
#include "pyfluid_amrex/initial_data/equilibrium_torus.hpp"
#include "pyfluid_amrex/initial_data/fluid.hpp"
#include "pyfluid_amrex/initial_data/materialization.hpp"
#include "pyfluid_amrex/initial_data/perturbation.hpp"
#include "pyfluid_amrex/initial_data/physical_id.hpp"
#include "pyfluid_amrex/initial_data/provider.hpp"
#include "pyfluid_amrex/initial_data/rotating_star.hpp"
#include "pyfluid_amrex/z4c_runtime.hpp"

#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_GpuAsyncArray.H>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

namespace pyfluid_amrex {

using namespace amrex::literals;
namespace {

std::string sequence_path_component(std::string value)
{
   for (char& character : value) {
      const unsigned char byte = static_cast<unsigned char>(character);
      if (!std::isalnum(byte) && character != '-' && character != '_'
          && character != '.') {
         character = '_';
      }
   }
   return value;
}

std::string sequence_real_label(amrex::Real value)
{
   std::ostringstream label;
   label << std::scientific << std::setprecision(6) << value;
   return label.str();
}

std::string sequence_eos_label(RunConfig const& cfg)
{
   std::string label = "eos-" + cfg.eos;
   if (cfg.eos_preset != "custom") {
      label += "-" + cfg.eos_preset;
   } else {
      label += "-custom";
   }
   return sequence_path_component(std::move(label));
}

std::string sequence_rotation_label(RunConfig const& cfg)
{
   std::string label = "rotation-" + cfg.initial_data_rotation_law;
   if (cfg.initial_data_rotation_law != "uniform") {
      label += "-A-"
         + sequence_real_label(
            cfg.initial_data_differential_rotation_length
         );
   }
   if (cfg.initial_data_rotation_law == "uryu_plus_13"
       || cfg.initial_data_rotation_law == "uryu_j_envelope_p1") {
      label += "-B-" + sequence_real_label(
         cfg.initial_data_differential_rotation_length_b
      );
   }
   if (cfg.initial_data_rotation_law == "q_power") {
      label += "-q-" + sequence_real_label(cfg.initial_data_rotation_power_q);
   }
   if (cfg.initial_data_disk == "equilibrium_torus"
       || cfg.initial_data_disk == "tilted_torus") {
      label += "-disk-" + cfg.initial_data_disk_rotation_law
         + "-gravity-" + cfg.initial_data_disk_gravity;
      if (cfg.initial_data_disk_rotation_law == "constant_l") {
         label += "-ell-" + sequence_real_label(
            cfg.initial_data_disk_specific_angular_momentum
         );
      } else {
         label += "-omega-" + sequence_real_label(
            cfg.initial_data_disk_angular_velocity
         );
      }
      if (cfg.initial_data_disk_rotation_law == "j_constant"
          || cfg.initial_data_disk_rotation_law == "q_power") {
         label += "-A-" + sequence_real_label(
            cfg.initial_data_disk_differential_rotation_length
         );
      }
      if (cfg.initial_data_disk_rotation_law == "q_power") {
         label += "-q-" + sequence_real_label(
            cfg.initial_data_disk_rotation_power_q
         );
      }
      label += "-rin-" + sequence_real_label(
         cfg.initial_data_disk_inner_radius
      );
   }
   return sequence_path_component(std::move(label));
}

void set_rotating_star_metadata(
   PhysicalInitialDataMetadata& metadata,
   RotatingStarUpdateSummary const& rotating_star
)
{
   metadata.maximum_density = rotating_star.maximum_density;
   metadata.maximum_velocity = rotating_star.maximum_velocity;
   metadata.injection_energy = rotating_star.injection_energy;
   metadata.equatorial_radius = rotating_star.equatorial_radius;
   metadata.polar_radius = rotating_star.polar_radius;
   metadata.equatorial_log_enthalpy_gradient =
      rotating_star.equatorial_log_enthalpy_gradient;
   metadata.polar_log_enthalpy_gradient =
      rotating_star.polar_log_enthalpy_gradient;
   metadata.mass_shedding_indicator = rotating_star.mass_shedding_indicator;
   metadata.adm_mass = rotating_star.adm_mass;
   metadata.baryonic_mass = rotating_star.baryonic_mass;
   metadata.angular_momentum = rotating_star.angular_momentum;
   metadata.dimensionless_spin = rotating_star.dimensionless_spin;
}

void set_equilibrium_torus_metadata(
   PhysicalInitialDataMetadata& metadata,
   EquilibriumTorusUpdateSummary const& equilibrium_torus
)
{
   metadata.disk_maximum_density = equilibrium_torus.maximum_density;
   metadata.disk_maximum_velocity = equilibrium_torus.maximum_velocity;
   metadata.disk_resolved_inner_radius = equilibrium_torus.inner_radius;
   metadata.disk_outer_radius = equilibrium_torus.outer_radius;
   metadata.disk_density_maximum_radius =
      equilibrium_torus.density_maximum_radius;
   metadata.disk_maximum_log_specific_enthalpy =
      equilibrium_torus.maximum_log_specific_enthalpy;
   metadata.disk_baryonic_mass = equilibrium_torus.baryonic_mass;
   metadata.disk_angular_momentum = equilibrium_torus.angular_momentum;
}

EquilibriumTorusUpdateSummary equilibrium_torus_summary(
   PhysicalInitialDataMetadata const& metadata
)
{
   EquilibriumTorusUpdateSummary result;
   result.maximum_density = metadata.disk_maximum_density;
   result.maximum_velocity = metadata.disk_maximum_velocity;
   result.inner_radius = metadata.disk_resolved_inner_radius;
   result.outer_radius = metadata.disk_outer_radius;
   result.density_maximum_radius = metadata.disk_density_maximum_radius;
   result.maximum_log_specific_enthalpy =
      metadata.disk_maximum_log_specific_enthalpy;
   result.baryonic_mass = metadata.disk_baryonic_mass;
   result.angular_momentum = metadata.disk_angular_momentum;
   return result;
}

RotatingStarUpdateSummary rotating_star_summary(
   PhysicalInitialDataMetadata const& metadata
)
{
   RotatingStarUpdateSummary result;
   result.maximum_density = metadata.maximum_density;
   result.maximum_velocity = metadata.maximum_velocity;
   result.injection_energy = metadata.injection_energy;
   result.equatorial_radius = metadata.equatorial_radius;
   result.polar_radius = metadata.polar_radius;
   result.equatorial_log_enthalpy_gradient =
      metadata.equatorial_log_enthalpy_gradient;
   result.polar_log_enthalpy_gradient =
      metadata.polar_log_enthalpy_gradient;
   result.mass_shedding_indicator = metadata.mass_shedding_indicator;
   result.adm_mass = metadata.adm_mass;
   result.baryonic_mass = metadata.baryonic_mass;
   result.angular_momentum = metadata.angular_momentum;
   result.dimensionless_spin = metadata.dimensionless_spin;
   result.matter_outer_radius = metadata.matter_outer_radius;
   return result;
}

amrex::GpuArray<amrex::Real, 3> normalized_component_axis(
   amrex::GpuArray<amrex::Real, 3> axis,
   bool zero_is_unoriented
)
{
   amrex::Real norm_squared = 0.0_rt;
   for (amrex::Real value : axis) {
      if (!std::isfinite(value)) {
         amrex::Abort("physical-ID component axis is nonfinite");
      }
      norm_squared += value * value;
   }
   if (!(norm_squared > 0.0_rt)) {
      if (!zero_is_unoriented) {
         amrex::Abort("physical-ID component axis is zero");
      }
      return {{0.0_rt, 0.0_rt, 1.0_rt}};
   }
   const amrex::Real inverse_norm = 1.0_rt / std::sqrt(norm_squared);
   for (amrex::Real& value : axis) {
      value *= inverse_norm;
   }
   return axis;
}

std::vector<PhysicalIdComponentMetadata> configured_physical_id_components(
   RunConfig const& cfg
)
{
   std::vector<PhysicalIdComponentMetadata> components;
   if ((cfg.spacetime_initial_data == "kerr_schild"
        || cfg.spacetime_initial_data == "static_trumpet")
       || cfg.ic == "black_hole") {
      PhysicalIdComponentMetadata black_hole;
      black_hole.role = "central_object";
      black_hole.kind = "black_hole";
      black_hole.gravity =
         cfg.spacetime_initial_data == "kerr_schild"
            || cfg.spacetime_initial_data == "static_trumpet"
         ? "analytic" : "exterior_shell";
      black_hole.origin = {{
         cfg.spacetime_origin1,
         cfg.spacetime_origin2,
         cfg.spacetime_origin3,
      }};
      black_hole.axis = normalized_component_axis({{
         cfg.spacetime_dimensionless_spin1,
         cfg.spacetime_dimensionless_spin2,
         cfg.spacetime_dimensionless_spin3,
      }}, true);
      components.push_back(std::move(black_hole));
   } else if (cfg.ic == "tov" || cfg.ic == "rotating_star") {
      PhysicalIdComponentMetadata star;
      star.role = "central_object";
      star.kind = cfg.ic;
      star.gravity = cfg.ic == "tov"
         && cfg.spacetime_initial_data != "ctt"
         && cfg.spacetime_initial_data != "cttk"
         && cfg.spacetime_initial_data != "xcts"
         ? "analytic" : "coupled";
      star.origin = {{
         cfg.initial_data_rotation_center1,
         cfg.initial_data_rotation_center2,
         cfg.initial_data_rotation_center3,
      }};
      star.axis = normalized_component_axis({{
         cfg.initial_data_rotation_axis1,
         cfg.initial_data_rotation_axis2,
         cfg.initial_data_rotation_axis3,
      }}, false);
      components.push_back(std::move(star));
   }
   if (cfg.initial_data_disk == "equilibrium_torus"
       || cfg.initial_data_disk == "tilted_torus") {
      PhysicalIdComponentMetadata disk;
      disk.role = "matter";
      disk.kind = cfg.initial_data_disk;
      disk.gravity = cfg.initial_data_disk_gravity;
      disk.origin = {{
         cfg.initial_data_rotation_center1,
         cfg.initial_data_rotation_center2,
         cfg.initial_data_rotation_center3,
      }};
      disk.axis = normalized_component_axis({{
         cfg.initial_data_disk_axis1,
         cfg.initial_data_disk_axis2,
         cfg.initial_data_disk_axis3,
      }}, false);
      components.push_back(std::move(disk));
   }
   return components;
}

PhysicalInitialDataMetadata physical_id_artifact_metadata(
   RunConfig const& cfg,
   PhysicalInitialDataMetadata const* loaded,
   amrex::Real black_hole_throat_area,
   amrex::Real black_hole_irreducible_mass,
   amrex::Real black_hole_throat_equatorial_circumference,
   amrex::Real black_hole_throat_polar_circumference,
   amrex::Real black_hole_throat_kerr_mass,
   amrex::Real black_hole_throat_kerr_spin,
   amrex::Real black_hole_throat_angular_momentum,
   amrex::Real black_hole_throat_christodoulou_mass,
   amrex::Real black_hole_throat_christodoulou_spin,
   RotatingStarUpdateSummary const* rotating_star,
   EquilibriumTorusUpdateSummary const* equilibrium_torus
)
{
   PhysicalInitialDataMetadata metadata;
   metadata.components = loaded == nullptr
      ? std::vector<PhysicalIdComponentMetadata>{}
      : loaded->components;
   for (PhysicalIdComponentMetadata const& configured
        : configured_physical_id_components(cfg)) {
      // Loading preserves the accepted central-object provenance. The only
      // currently supported composition operation is adding a matter disk;
      // future central-object assembly belongs to the global N-object owner.
      if (loaded != nullptr && configured.role != "matter") {
         continue;
      }
      const bool already_present = std::any_of(
         metadata.components.begin(), metadata.components.end(),
         [&configured](PhysicalIdComponentMetadata const& existing) {
            return existing.role == configured.role
               && existing.kind == configured.kind;
         }
      );
      if (!already_present) {
         metadata.components.push_back(configured);
      }
   }
   metadata.matter_outer_radius = loaded == nullptr
      ? -1.0_rt : loaded->matter_outer_radius;
   if (black_hole_throat_area > 0.0_rt) {
      metadata.black_hole_throat_area = black_hole_throat_area;
      metadata.black_hole_irreducible_mass = black_hole_irreducible_mass;
      metadata.black_hole_throat_equatorial_circumference =
         black_hole_throat_equatorial_circumference;
      metadata.black_hole_throat_polar_circumference =
         black_hole_throat_polar_circumference;
      metadata.black_hole_throat_kerr_mass = black_hole_throat_kerr_mass;
      metadata.black_hole_throat_kerr_spin = black_hole_throat_kerr_spin;
      metadata.black_hole_throat_angular_momentum =
         black_hole_throat_angular_momentum;
      metadata.black_hole_throat_christodoulou_mass =
         black_hole_throat_christodoulou_mass;
      metadata.black_hole_throat_christodoulou_spin =
         black_hole_throat_christodoulou_spin;
   } else if (loaded != nullptr) {
      metadata.black_hole_throat_area = loaded->black_hole_throat_area;
      metadata.black_hole_irreducible_mass =
         loaded->black_hole_irreducible_mass;
      metadata.black_hole_throat_equatorial_circumference =
         loaded->black_hole_throat_equatorial_circumference;
      metadata.black_hole_throat_polar_circumference =
         loaded->black_hole_throat_polar_circumference;
      metadata.black_hole_throat_kerr_mass =
         loaded->black_hole_throat_kerr_mass;
      metadata.black_hole_throat_kerr_spin =
         loaded->black_hole_throat_kerr_spin;
      metadata.black_hole_throat_angular_momentum =
         loaded->black_hole_throat_angular_momentum;
      metadata.black_hole_throat_christodoulou_mass =
         loaded->black_hole_throat_christodoulou_mass;
      metadata.black_hole_throat_christodoulou_spin =
         loaded->black_hole_throat_christodoulou_spin;
   }
   if (rotating_star != nullptr) {
      set_rotating_star_metadata(metadata, *rotating_star);
      metadata.matter_outer_radius = std::max(
         metadata.matter_outer_radius, rotating_star->matter_outer_radius
      );
   } else if (loaded != nullptr && loaded->source_ic == "rotating_star") {
      const RotatingStarUpdateSummary summary = rotating_star_summary(*loaded);
      set_rotating_star_metadata(metadata, summary);
   }
   if (equilibrium_torus != nullptr) {
      set_equilibrium_torus_metadata(metadata, *equilibrium_torus);
      metadata.matter_outer_radius = std::max(
         metadata.matter_outer_radius, equilibrium_torus->outer_radius
      );
   } else if (loaded != nullptr
              && (loaded->disk == "equilibrium_torus"
                  || loaded->disk == "tilted_torus")) {
      const EquilibriumTorusUpdateSummary summary =
         equilibrium_torus_summary(*loaded);
      set_equilibrium_torus_metadata(metadata, summary);
   }
   return metadata;
}

} // namespace

void AmrSolverBase::materialize_configured_initial_data()
{
   if (cfg_.initial_data_input_file.empty()) {
      const bool materialized = materialize_constraint_solved_initial_data();
      if (!materialized && spacetime_state_.active()
          && !uses_analytic_fixed_spacetime_kernels()
          && physical_system_is_relativistic(physics_system_)) {
         // Non-analytic stored spacetime is authoritative only after nodal
         // initialization and centering. Recover passive concentrations,
         // resample the configured physical primitives, then construct the
         // conserved state against that final cell metric. Analytic Minkowski
         // initialization already uses its final metric and keeps its direct
         // zero-overhead path unchanged.
         for (int level = 0; level <= finestLevel(); ++level) {
            fill_level_primitive_cache(level, 0.0_rt);
            initialize_relativistic_primitive_cache_from_initial_data(
               *primitive_[level],
               Geom(level),
               cfg_,
               initial_data_provider_->view()
            );
         }
         finalize_constraint_solved_initial_data_state();
      }
      if (!materialized && cfg_.ic == "equilibrium_torus") {
         summarize_configured_equilibrium_torus();
      }
      return;
   }
   if (load_physical_initial_data_artifact()) {
      materialize_constraint_solved_initial_data(true);
   }
}

void AmrSolverBase::apply_configured_initial_data_perturbation()
{
   if (cfg_.initial_data_perturbation == "none") {
      return;
   }
   if (!physical_system_is_relativistic(physics_system_)
       || !spacetime_state_.active()) {
      amrex::Abort(
         "equilibrium initial-data perturbations require relativistic fluid "
         "and stored spacetime state"
      );
   }
   if (mixed_patch_system_enabled()) {
      amrex::Abort(
         "equilibrium initial-data perturbations currently require a single "
         "mapped patch"
      );
   }
   InitialDataPerturbation kind = InitialDataPerturbation::None;
   bool found = false;
   for (InitialDataPerturbationInfo const& perturbation
        : InitialDataPerturbationCatalog) {
      if (perturbation.name == cfg_.initial_data_perturbation) {
         kind = perturbation.id;
         found = true;
         break;
      }
   }
   if (!found) {
      amrex::Abort("unknown generated initial-data perturbation");
   }
   amrex::Real reference_radius = cfg_.initial_data_perturbation_radius;
   const bool density_shape_perturbation =
      kind == InitialDataPerturbation::DensityAzimuthalEspino2019
      || kind == InitialDataPerturbation::DensityRadialFont2002;
   if (density_shape_perturbation
       && !(reference_radius > 0.0_rt)) {
      if (initial_data_rotating_star_summary_
          && initial_data_rotating_star_summary_->equatorial_radius > 0.0_rt) {
         reference_radius =
            initial_data_rotating_star_summary_->equatorial_radius;
      } else if (loaded_physical_id_metadata_
                 && loaded_physical_id_metadata_->equatorial_radius > 0.0_rt) {
         reference_radius = loaded_physical_id_metadata_->equatorial_radius;
      } else if (initial_data_provider_->view().tov.surface_radius > 0.0_rt) {
         reference_radius = initial_data_provider_->view().tov.surface_radius;
      }
   }
   if (density_shape_perturbation
       && !(reference_radius > 0.0_rt)) {
      amrex::Abort(
         "density-shape initial-data perturbation could not resolve a "
         "positive stellar reference radius"
      );
   }
   const amrex::GpuArray<amrex::Real, 3> origin{{
      cfg_.initial_data_rotation_center1,
      cfg_.initial_data_rotation_center2,
      cfg_.initial_data_rotation_center3,
   }};
   const bool disk_equilibrium = cfg_.ic == "equilibrium_torus";
   const amrex::GpuArray<amrex::Real, 3> axis = disk_equilibrium
      ? amrex::GpuArray<amrex::Real, 3>{{
           cfg_.initial_data_disk_axis1,
           cfg_.initial_data_disk_axis2,
           cfg_.initial_data_disk_axis3,
        }}
      : amrex::GpuArray<amrex::Real, 3>{{
           cfg_.initial_data_rotation_axis1,
           cfg_.initial_data_rotation_axis2,
           cfg_.initial_data_rotation_axis3,
        }};
   const InitialDataPerturbationParameters parameters{
      kind,
      cfg_.initial_data_perturbation_pressure_scale,
      cfg_.initial_data_perturbation_amplitude,
      cfg_.initial_data_perturbation_mode,
      cfg_.initial_data_perturbation_phase,
      reference_radius,
      make_azimuthal_frame(origin, axis),
      eos_params(cfg_),
   };
   if (!cold_barotrope_supported(parameters.eos)) {
      amrex::Abort(
         "equilibrium initial-data perturbation requires a cold barotrope"
      );
   }

   if (kind == InitialDataPerturbation::DensityAzimuthalEspino2019) {
      int symmetry_direction = -1;
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         if (!symmetry_axis_active(direction)) {
            continue;
         }
         if (symmetry_direction >= 0) {
            amrex::Abort(
               "nonaxisymmetric initial-data perturbations support only one "
               "equatorial reflection plane"
            );
         }
         symmetry_direction = direction;
      }
      if (symmetry_direction >= 0) {
         const bool low_symmetry = side_boundary_kind(
            cfg_, symmetry_direction, false
         ) == BoundaryKind::Symmetry;
         const bool high_symmetry = side_boundary_kind(
            cfg_, symmetry_direction, true
         ) == BoundaryKind::Symmetry;
         const amrex::Real plane = low_symmetry
            ? Geom(0).ProbLo(symmetry_direction)
            : Geom(0).ProbHi(symmetry_direction);
         const amrex::Real tolerance = 256.0_rt
            * std::numeric_limits<amrex::Real>::epsilon()
            * std::max<amrex::Real>(1.0_rt, std::abs(plane));
         if (low_symmetry == high_symmetry
             || std::abs(
                std::abs(parameters.frame.axis[symmetry_direction]) - 1.0_rt
             ) > tolerance
             || std::abs(origin[symmetry_direction] - plane) > tolerance) {
            amrex::Abort(
               "nonaxisymmetric initial-data perturbation is incompatible "
               "with the configured reflection symmetry"
            );
         }
      }
   }

   amrex::Real maximum_fractional_change = 0.0_rt;
   amrex::Real invalid_points = 0.0_rt;
   const InitialAtmosphereParameters atmosphere =
      initial_atmosphere_parameters(cfg_, initial_data_provider_->view());
   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      fill_level_primitive_cache(lev, 0.0_rt);
      amrex::MultiFab reduction(
         primitive_[lev]->boxArray(),
         primitive_[lev]->DistributionMap(),
         2,
         0
      );
      reduction.setVal(0.0_rt);
      const auto problo = Geom(lev).ProbLoArray();
      const auto dx = Geom(lev).CellSizeArray();
      const amrex::Dim3 domain_lo = amrex::lbound(Geom(lev).Domain());
      std::unique_ptr<amrex::Gpu::AsyncArray<amrex::Real>>
         mapped_radial_center_storage;
      amrex::Real const* mapped_radial_center = nullptr;
      if (coordinate == MeshSpherical
          && lev < static_cast<int>(initial_data_coordinate_faces_.size())
          && !initial_data_coordinate_faces_[lev][0].empty()) {
         const auto& radial_faces = initial_data_coordinate_faces_[lev][0];
         amrex::Vector<amrex::Real> radial_centers(radial_faces.size() - 1);
         for (int radial_index = 0;
              radial_index < static_cast<int>(radial_centers.size());
              ++radial_index) {
            radial_centers[radial_index] = 0.5_rt * (
               radial_faces[radial_index] + radial_faces[radial_index + 1]
            );
         }
         mapped_radial_center_storage =
            std::make_unique<amrex::Gpu::AsyncArray<amrex::Real>>(
               radial_centers.data(), radial_centers.size()
            );
         mapped_radial_center = mapped_radial_center_storage->data();
      }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(*primitive_[lev], amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto primitive = primitive_[lev]->array(mfi);
         const auto output = reduction.array(mfi);
         amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               auto mesh_point = mesh_coordinate_triplet_at_location(
                  coordinate,
                  axes,
                  problo,
                  dx,
                  i,
                  j,
                  k,
                  false,
                  false,
                  false
               );
               if (mapped_radial_center != nullptr) {
                  mesh_point[0] = mapped_radial_center[i - domain_lo.x];
               }
               const auto cartesian = mesh_coordinate_triplet_to_cartesian(
                  coordinate, mesh_point
               );
               const PatchCoordinate physical{
                  cartesian[0], cartesian[1], cartesian[2]
               };
               const InitialDataSamplePoint sample{physical, physical};
               const EOSThermodynamicState atmosphere_state =
                  initial_atmosphere_state<EOSRuntime>(atmosphere, sample);
               const amrex::Real rho = primitive(i, j, k, RHO);
               if (!initial_matter_exceeds_atmosphere(
                      rho, atmosphere_state.rho
                   )) {
                  return;
               }
               const InitialDataPerturbationPoint perturbed =
                  initial_data_perturbation_point(
                     parameters, physical, rho
                  );
               if (!perturbed.valid) {
                  output(i, j, k, 1) = 1.0_rt;
                  return;
               }
               primitive(i, j, k, RHO) = perturbed.matter.rho;
               primitive(i, j, k, ENERGY) = perturbed.matter.pressure;
               output(i, j, k, 0) = std::abs(
                  perturbed.fractional_density_change
               );
            }
         );
      }
      maximum_fractional_change = std::max(
         maximum_fractional_change, reduction.max(0, 0, false)
      );
      invalid_points += reduction.sum(1, false);
   }
   if (invalid_points != 0.0_rt) {
      amrex::Abort(
         "initial-data perturbation produced an invalid cold-EoS state"
      );
   }
   finalize_constraint_solved_initial_data_state();
   const bool polish_constraints =
      cfg_.spacetime_initial_data == "ctt"
      || cfg_.spacetime_initial_data == "cttk"
      || cfg_.spacetime_initial_data == "xcts";
   const bool reseed_static_tov_spacetime =
      cfg_.initial_data_perturbation == "density_radial_font2002"
      && cfg_.ic == "rotating_star"
      && cfg_.initial_data_angular_velocity == 0.0_rt
      && cfg_.initial_data_disk == "none"
      && !initial_data_coordinate_faces_.empty();
   if (polish_constraints
       && !materialize_constraint_solved_initial_data(
          true, reseed_static_tov_spacetime,
          reseed_static_tov_spacetime
       )) {
      amrex::Abort(
         "configured initial-data perturbation constraint polish was not "
         "materialized"
      );
   }
   amrex::Print()
      << "applied initial-data perturbation: kind="
      << cfg_.initial_data_perturbation
      << " maximum_fractional_density_change="
      << maximum_fractional_change
      << " constraint_polish="
      << (polish_constraints ? cfg_.spacetime_initial_data : "none") << '\n';
}

void AmrSolverBase::summarize_configured_equilibrium_torus()
{
#if (AMREX_SPACEDIM != 3)
   amrex::Abort("direct equilibrium-torus diagnostics require 3D");
#else
   if (cfg_.mesh_coordinate != "cartesian"
       || cfg_.initial_data_disk != "equilibrium_torus") {
      amrex::Abort("invalid direct equilibrium-torus diagnostic request");
   }
   const RelativisticInitialDataView provider = initial_data_provider_->view();
   const FluidInitialDataParameters fluid_parameters =
      fluid_initial_data_parameters(cfg_, provider);
   const EquilibriumTorusParameters torus =
      fluid_parameters.equilibrium_torus;
   const FixedSpacetimeParameters spacetime = fixed_spacetime_parameters(cfg_);
   constexpr int density_component = 0;
   constexpr int velocity_squared_component = 1;
   constexpr int log_enthalpy_component = 2;
   constexpr int inner_radius_component = 3;
   constexpr int outer_radius_component = 4;
   constexpr int boundary_excess_component = 5;
   constexpr int density_maximum_radius_component = 6;
   constexpr int baryonic_mass_component = 7;
   constexpr int angular_momentum1_component = 8;
   constexpr int angular_momentum2_component = 9;
   constexpr int angular_momentum3_component = 10;
   constexpr int num_components = 11;
   const amrex::Real invalid_min = std::numeric_limits<amrex::Real>::max();
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> diagnostics;
   diagnostics.reserve(finestLevel() + 1);
   EquilibriumTorusUpdateSummary summary;
   amrex::Real baryonic_mass = 0.0_rt;
   amrex::GpuArray<amrex::Real, 3> angular_momentum{{0.0_rt, 0.0_rt, 0.0_rt}};
   for (int level = 0; level <= finestLevel(); ++level) {
      fill_level_primitive_cache(level, 0.0_rt);
      amrex::iMultiFab valid_mask = valid_cell_mask(level);
      diagnostics.push_back(std::make_unique<amrex::MultiFab>(
         primitive_[level]->boxArray(),
         primitive_[level]->DistributionMap(),
         num_components,
         0
      ));
      amrex::MultiFab& diagnostic = *diagnostics.back();
      diagnostic.setVal(0.0_rt);
      diagnostic.setVal(invalid_min, inner_radius_component, 1, 0);
      const auto problo = Geom(level).ProbLoArray();
      const auto dx = Geom(level).CellSizeArray();
      const amrex::Box domain = Geom(level).Domain();
      const amrex::Dim3 domain_lo = amrex::lbound(domain);
      const amrex::Dim3 domain_hi = amrex::ubound(domain);
      amrex::GpuArray<int, 3> non_reflecting_lo{{0, 0, 0}};
      amrex::GpuArray<int, 3> non_reflecting_hi{{0, 0, 0}};
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         non_reflecting_lo[direction] = reflective_boundary(
            side_boundary_kind(cfg_, direction, false)
         ) ? 0 : 1;
         non_reflecting_hi[direction] = reflective_boundary(
            side_boundary_kind(cfg_, direction, true)
         ) ? 0 : 1;
      }
      const amrex::Real half_diagonal = 0.5_rt * std::sqrt(
         dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]
      );
      const amrex::Real cell_volume = dx[0] * dx[1] * dx[2];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(diagnostic, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto primitive = primitive_[level]->const_array(mfi);
         const auto conserved = fluid_state_.state()[level]->const_array(mfi);
         const auto valid = valid_mask.const_array(mfi);
         const auto output = diagnostic.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            if (valid(i, j, k, 0) == 0) {
               return;
            }
            const PatchCoordinate physical{
               problo[0] + (static_cast<amrex::Real>(i) + 0.5_rt) * dx[0],
               problo[1] + (static_cast<amrex::Real>(j) + 0.5_rt) * dx[1],
               problo[2] + (static_cast<amrex::Real>(k) + 0.5_rt) * dx[2],
            };
            const InitialDataSamplePoint sample{physical, physical};
            const EOSThermodynamicState atmosphere =
               initial_atmosphere_state<EOSRuntime>(
                  fluid_parameters.atmosphere, sample
               );
            const amrex::Real rho = primitive(i, j, k, RHO);
            if (!initial_matter_exceeds_atmosphere(rho, atmosphere.rho)) {
               return;
            }
            const amrex::Real displacement1 = physical.q1 - torus.center1;
            const amrex::Real displacement2 = physical.q2 - torus.center2;
            const amrex::Real displacement3 = physical.q3 - torus.center3;
            const amrex::Real radius = std::sqrt(
               displacement1 * displacement1
               + displacement2 * displacement2
               + displacement3 * displacement3
            );
            const amrex::Real tangent1 =
               torus.axis2 * displacement3 - torus.axis3 * displacement2;
            const amrex::Real tangent2 =
               torus.axis3 * displacement1 - torus.axis1 * displacement3;
            const amrex::Real tangent3 =
               torus.axis1 * displacement2 - torus.axis2 * displacement1;
            const amrex::Real cylindrical_radius = std::sqrt(
               tangent1 * tangent1 + tangent2 * tangent2
               + tangent3 * tangent3
            );
            const FixedSpacetimeGeometry local_geometry =
               initial_data_spacetime_geometry(
                  spacetime,
                  physical.q1,
                  physical.q2,
                  physical.q3,
                  0.0_rt
               );
            const EquilibriumTorusPoint target = equilibrium_torus_point(
               torus,
               fluid_parameters.equilibrium_torus_inner_normalization,
               physical,
               local_geometry.metric
            );
            const amrex::Real velocity_squared = spatial_vector_norm_squared(
               primitive(i, j, k, MOM1),
               primitive(i, j, k, MOM2),
               primitive(i, j, k, MOM3),
               local_geometry.metric
            );
            output(i, j, k, density_component) = rho;
            output(i, j, k, velocity_squared_component) = velocity_squared;
            output(i, j, k, log_enthalpy_component) =
               target.log_specific_enthalpy;
            output(i, j, k, inner_radius_component) = radius;
            output(i, j, k, outer_radius_component) =
               radius + half_diagonal;
            output(i, j, k, density_maximum_radius_component) =
               cylindrical_radius;
            const bool boundary_cell =
               (i == domain_lo.x && non_reflecting_lo[0] != 0)
               || (i == domain_hi.x && non_reflecting_hi[0] != 0)
               || (j == domain_lo.y && non_reflecting_lo[1] != 0)
               || (j == domain_hi.y && non_reflecting_hi[1] != 0)
               || (k == domain_lo.z && non_reflecting_lo[2] != 0)
               || (k == domain_hi.z && non_reflecting_hi[2] != 0);
            output(i, j, k, boundary_excess_component) = boundary_cell
               ? rho - atmosphere.rho : 0.0_rt;
            output(i, j, k, baryonic_mass_component) =
               cell_volume * conserved(i, j, k, RHO);
            const amrex::Real momentum1 = conserved(i, j, k, MOM1);
            const amrex::Real momentum2 = conserved(i, j, k, MOM2);
            const amrex::Real momentum3 = conserved(i, j, k, MOM3);
            output(i, j, k, angular_momentum1_component) = cell_volume * (
               displacement2 * momentum3 - displacement3 * momentum2
            );
            output(i, j, k, angular_momentum2_component) = cell_volume * (
               displacement3 * momentum1 - displacement1 * momentum3
            );
            output(i, j, k, angular_momentum3_component) = cell_volume * (
               displacement1 * momentum2 - displacement2 * momentum1
            );
         });
      }
      summary.maximum_density = std::max(
         summary.maximum_density,
         diagnostic.max(density_component, 0, false)
      );
      summary.maximum_velocity = std::max(
         summary.maximum_velocity,
         std::sqrt(std::max(
            diagnostic.max(velocity_squared_component, 0, false),
            amrex::Real(0.0_rt)
         ))
      );
      summary.maximum_log_specific_enthalpy = std::max(
         summary.maximum_log_specific_enthalpy,
         diagnostic.max(log_enthalpy_component, 0, false)
      );
      summary.inner_radius = level == 0
         ? diagnostic.min(inner_radius_component, 0, false)
         : std::min(
              summary.inner_radius,
              diagnostic.min(inner_radius_component, 0, false)
           );
      summary.outer_radius = std::max(
         summary.outer_radius,
         diagnostic.max(outer_radius_component, 0, false)
      );
      summary.boundary_density_excess = std::max(
         summary.boundary_density_excess,
         diagnostic.max(boundary_excess_component, 0, false)
      );
      baryonic_mass += diagnostic.sum(baryonic_mass_component, false);
      angular_momentum[0] += diagnostic.sum(
         angular_momentum1_component, false
      );
      angular_momentum[1] += diagnostic.sum(
         angular_momentum2_component, false
      );
      angular_momentum[2] += diagnostic.sum(
         angular_momentum3_component, false
      );
   }
   if (!(summary.maximum_density > 0.0_rt)
       || !(summary.maximum_velocity >= 0.0_rt)
       || !(summary.maximum_velocity < 1.0_rt)
       || !(summary.outer_radius > summary.inner_radius)
       || !(summary.maximum_log_specific_enthalpy > 0.0_rt)
       || !std::isfinite(summary.inner_radius)) {
      amrex::Abort(
         "equilibrium-torus parameters do not produce resolved finite support"
      );
   }
   if (summary.boundary_density_excess > 0.0_rt) {
      amrex::Abort(
         "equilibrium-torus matter reaches the Cartesian boundary; enlarge "
         "the ID domain"
      );
   }
   const amrex::Real density_tolerance = 128.0_rt
      * std::numeric_limits<amrex::Real>::epsilon();
   for (auto& diagnostic_storage : diagnostics) {
      amrex::MultiFab& diagnostic = *diagnostic_storage;
      const amrex::Real maximum_density = summary.maximum_density;
      for (amrex::MFIter mfi(diagnostic, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto values = diagnostic.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            if (values(i, j, k, density_component)
                < maximum_density * (1.0_rt - density_tolerance)) {
               values(i, j, k, density_maximum_radius_component) = 0.0_rt;
            }
         });
      }
      summary.density_maximum_radius = std::max(
         summary.density_maximum_radius,
         diagnostic.max(density_maximum_radius_component, 0, false)
      );
   }
   const amrex::Real volume_factor = symmetry_volume_factor();
   summary.baryonic_mass = volume_factor * baryonic_mass;
   summary.angular_momentum =
      torus.axis1 * symmetry_angular_momentum_factor(0, volume_factor)
         * angular_momentum[0]
      + torus.axis2 * symmetry_angular_momentum_factor(1, volume_factor)
         * angular_momentum[1]
      + torus.axis3 * symmetry_angular_momentum_factor(2, volume_factor)
         * angular_momentum[2];
   if (!(summary.baryonic_mass > 0.0_rt)
       || !std::isfinite(summary.angular_momentum)) {
      amrex::Abort("equilibrium-torus global diagnostics are invalid");
   }
   initial_data_equilibrium_torus_summary_ =
      std::make_unique<EquilibriumTorusUpdateSummary>(summary);
   amrex::Print()
      << "equilibrium torus: gravity=" << cfg_.initial_data_disk_gravity
      << " rotation_law=" << cfg_.initial_data_disk_rotation_law
      << " specific_angular_momentum="
      << torus.specific_angular_momentum
      << " angular_velocity=" << torus.angular_velocity
      << " differential_rotation_length="
      << torus.differential_rotation_length
      << " rotation_power_q=" << torus.rotation_power_q
      << " requested_inner_radius=" << torus.inner_radius
      << " resolved_inner_radius=" << summary.inner_radius
      << " outer_radius=" << summary.outer_radius
      << " density_maximum_radius=" << summary.density_maximum_radius
      << " maximum_density=" << summary.maximum_density
      << " maximum_velocity=" << summary.maximum_velocity
      << " maximum_log_specific_enthalpy="
      << summary.maximum_log_specific_enthalpy
      << " baryonic_mass=" << summary.baryonic_mass
      << " angular_momentum=" << summary.angular_momentum << '\n';
#endif
}

bool AmrSolverBase::load_physical_initial_data_artifact()
{
   amrex::Vector<amrex::Geometry> geometry;
   amrex::Vector<amrex::IntVect> refinement_ratios;
   MutablePhysicalIdField primitive;
   MutablePhysicalIdField fluid;
   MutablePhysicalIdField spacetime;
   MutablePhysicalIdFaceFields face_magnetic;
#if (AMREX_SPACEDIM >= 2)
   if (cfg_.mesh_coordinate == "spherical"
       && cfg_.initial_data_radial_grid == "adaptive") {
      const RelativisticInitialDataView provider = initial_data_provider_->view();
      const bool black_hole_xcts = cfg_.ic == "black_hole"
         && cfg_.spacetime_initial_data == "xcts";
      if ((!provider.tov.active() && !black_hole_xcts)
          || finestLevel() != 0) {
         amrex::Abort(
            "adaptive reduced ID loading requires one level and a native "
            "TOV or black-hole XCTS source"
         );
      }
      initial_data_coordinate_faces_.resize(1);
      initial_data_coordinate_faces_[0][0] = black_hole_xcts
         ? stretched_radial_shell_faces(
              0.5_rt * cfg_.spacetime_mass,
              Geom(0).ProbHi(0),
              Geom(0).Domain().length(0),
              cfg_.initial_data_adaptive_shell_width_ratio
           )
         : stellar_adaptive_radial_faces(
              provider.tov.surface_radius,
              Geom(0).ProbHi(0),
              Geom(0).Domain().length(0),
              cfg_.initial_data_adaptive_source_radius_factor,
              cfg_.initial_data_adaptive_exterior_fraction
           );
   }
#endif
   for (int level = 0; level <= finestLevel(); ++level) {
      geometry.push_back(Geom(level));
      primitive.push_back(primitive_[level].get());
      fluid.push_back(fluid_state_.state()[level].get());
      if (spacetime_state_.active()) {
         spacetime.push_back(spacetime_state_.state()[level].get());
      }
      if (ct_enabled(cfg_)) {
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            face_magnetic[direction].push_back(face_b_[level].b[direction].get());
         }
      }
      if (level < finestLevel()) {
         refinement_ratios.push_back(refRatio(level));
      }
   }
   const PhysicalInitialDataMetadata metadata = read_physical_initial_data(
      cfg_.initial_data_input_file,
      cfg_,
      physics_system_,
      initial_atmosphere_parameters(cfg_, initial_data_provider_->view()),
      geometry,
      refinement_ratios,
      primitive,
      fluid,
      spacetime,
      face_magnetic,
      &initial_data_coordinate_faces_
   );
   loaded_physical_id_metadata_ =
      std::make_unique<PhysicalInitialDataMetadata>(metadata);
   // A non-Minkowski analytic fixed metric is the runtime metric provider.
   // A portable artifact may supply its fluid on a different hierarchy, but
   // interpolating the artifact's redundant metric would make horizon and
   // constraint diagnostics disagree with the exact geometry used by the
   // Valencia evolution. Re-evaluate that metric on the target hierarchy
   // before rematerializing conserved fluid variables. Fixed Minkowski plus a
   // loaded XCTS/CTT state intentionally remains the stored-metric contract.
   const bool resample_analytic_fixed_spacetime =
      spacetime_state_.active()
      && !spacetime_state_.evolves()
      && cfg_.spacetime_fixed_metric != "minkowski";
   if (resample_analytic_fixed_spacetime) {
      for (int level = 0; level <= finestLevel(); ++level) {
         initialize_spacetime_level(level, 0.0_rt);
      }
   }
   select_analytic_fixed_spacetime_kernels(
      metadata.source_ic, metadata.spacetime_initial_data
   );
   if (!metadata.used_exact_cache) {
      ConstPhysicalIdField cell_spacetime;
      if (spacetime_state_.active()) {
         for (int level = 0; level <= finestLevel(); ++level) {
            spacetime_coupling_.invalidate_cell_spacetime(level);
         }
         fill_all_spacetime_boundaries(spacetime_state_.state(), 0.0_rt);
         if (initial_data_coordinate_faces_.empty()) {
            for (int level = 0; level <= finestLevel(); ++level) {
               complete_spacetime_connections(
                  *spacetime_state_.state()[level], Geom(level), cfg_
               );
            }
         }
         fill_all_spacetime_boundaries(spacetime_state_.state(), 0.0_rt);
         prepare_spacetime_coupling_range(
            spacetime_state_.state(),
            0,
            finestLevel() + 1,
            0.0_rt,
            -1.0_rt,
            0
         );
         for (int level = 0; level <= finestLevel(); ++level) {
            cell_spacetime.push_back(
               spacetime_coupling_.cell_spacetime()[level].get()
            );
         }
      }
      materialize_physical_initial_data_fluid(
         cfg_,
         physics_system_,
         geometry,
         ConstPhysicalIdField(primitive.begin(), primitive.end()),
         cell_spacetime,
         fluid
      );
      if (ct_enabled(cfg_)) {
         project_materialized_initial_face_b();
      }
      amrex::Print()
         << "materialized physical initial data onto target hierarchy\n";
   }
   if (ct_enabled(cfg_)) {
      sync_all_cell_b_from_faces(fluid_state_.state(), false);
   }
   for (int level = 0; level <= finestLevel(); ++level) {
      if (spacetime_state_.active()) {
         spacetime_coupling_.invalidate_cell_spacetime(level);
      }
      copy_level_state_to_old(level);
   }
   amrex::Print()
      << "loaded physical initial data: " << cfg_.initial_data_input_file << '\n';
   const bool configured_rotation = metadata.source_ic == "rotating_star"
      && cfg_.ic == "rotating_star";
   if (configured_rotation
       && metadata.rotation_law != cfg_.initial_data_rotation_law) {
      amrex::Abort(
         "rotating-star physical ID continuation cannot change rotation law"
      );
   }
   if (configured_rotation
       && cfg_.initial_data_rotation_law != "uniform"
       && metadata.differential_rotation_length
          != cfg_.initial_data_differential_rotation_length) {
      amrex::Abort(
         "rotating-star physical ID continuation cannot change the "
         "differential rotation length"
      );
   }
   if (configured_rotation
       && (cfg_.initial_data_rotation_law == "uryu_plus_13"
           || cfg_.initial_data_rotation_law == "uryu_j_envelope_p1")
       && metadata.differential_rotation_length_b
          != cfg_.initial_data_differential_rotation_length_b) {
      amrex::Abort(
         "rotating-star physical ID continuation cannot change the Uryu "
         "differential rotation length B"
      );
   }
   if (configured_rotation
       && cfg_.initial_data_rotation_law == "q_power"
       && metadata.rotation_power_q != cfg_.initial_data_rotation_power_q) {
      amrex::Abort(
         "rotating-star physical ID continuation cannot change the q-power "
         "exponent"
      );
   }
   if (metadata.used_rigid_transform) {
      amrex::Print()
         << "rigidly transformed physical initial data: source_axis=("
         << metadata.rotation_axis[0] << ',' << metadata.rotation_axis[1]
         << ',' << metadata.rotation_axis[2] << ") target_axis=("
         << cfg_.initial_data_rotation_axis1 << ','
         << cfg_.initial_data_rotation_axis2 << ','
         << cfg_.initial_data_rotation_axis3 << ") source_center=("
         << metadata.rotation_center[0] << ','
         << metadata.rotation_center[1] << ','
         << metadata.rotation_center[2] << ") target_center=("
         << cfg_.initial_data_rotation_center1 << ','
         << cfg_.initial_data_rotation_center2 << ','
         << cfg_.initial_data_rotation_center3 << ")\n";
   }
   const bool continue_rotation = configured_rotation
      && metadata.angular_velocity != cfg_.initial_data_angular_velocity;
   const bool continue_disk =
      physical_id_disk_continuation_requested(metadata, cfg_);
   const bool add_torus = metadata.disk == "none"
      && (cfg_.initial_data_disk == "equilibrium_torus"
          || cfg_.initial_data_disk == "tilted_torus")
      && cfg_.spacetime_initial_data == "xcts";
   const bool constructed_magnetic_artifact =
      metadata.magnetic_seed != "none"
      || metadata.magnetic_equilibrium != "none";
   if (constructed_magnetic_artifact
       && (metadata.source_ic != cfg_.ic || continue_rotation
           || continue_disk || add_torus)) {
      amrex::Abort(
         "constructed stellar magnetic physical-ID artifacts cannot yet be "
         "continued or recomposed; a target-grid magnetic equilibrium owner "
         "is required"
      );
   }
   if constexpr (AMREX_SPACEDIM == 3) {
      const bool axisymmetric_disk = cfg_.initial_data_disk == "none"
         || (cfg_.initial_data_disk == "equilibrium_torus"
             && cfg_.initial_data_disk_axis1 == 0.0_rt
             && cfg_.initial_data_disk_axis2 == 0.0_rt
             && cfg_.initial_data_disk_axis3 > 0.0_rt);
      const bool spherical_axisymmetric_target =
         cfg_.mesh_coordinate == "spherical"
         && cfg_.mesh_axes == "r theta phi"
         && cfg_.initial_data_rotation_center1 == 0.0_rt
         && cfg_.initial_data_rotation_center2 == 0.0_rt
         && cfg_.initial_data_rotation_center3 == 0.0_rt
         && cfg_.initial_data_rotation_axis1 == 0.0_rt
         && cfg_.initial_data_rotation_axis2 == 0.0_rt
         && cfg_.initial_data_rotation_axis3 > 0.0_rt
         && axisymmetric_disk;
      if (cfg_.mesh_coordinate != "cartesian"
          && !spherical_axisymmetric_target
          && (continue_rotation || continue_disk || add_torus)) {
         amrex::Abort(
            "3D curvilinear physical-ID targets may materialize a completed "
            "model, but continuation or component addition requires the "
            "native 3D Cartesian coupled-XCTS owner"
         );
      }
   }
   if (continue_rotation) {
      if (cfg_.spacetime_initial_data != "xcts") {
         amrex::Abort(
            "saved rotating-star continuation currently requires XCTS"
         );
      }
      if (metadata.angular_velocity * cfg_.initial_data_angular_velocity < 0.0_rt
          || std::abs(metadata.angular_velocity)
             >= std::abs(cfg_.initial_data_angular_velocity)) {
         amrex::Abort(
            "rotating-star physical ID continuation requires a larger target "
            "angular velocity with the saved rotation sign"
         );
      }
      amrex::Real previous_magnitude = std::abs(metadata.angular_velocity);
      for (amrex::Real target :
           cfg_.initial_data_rotation_continuation_targets) {
         if (!(std::abs(target) > previous_magnitude)) {
            amrex::Abort(
               "saved rotating-star continuation targets must all exceed the "
               "saved angular-velocity magnitude"
            );
         }
         previous_magnitude = std::abs(target);
      }
      amrex::Print()
         << "continuing rotating physical initial data: source_omega="
         << metadata.angular_velocity
         << " target_omega=" << cfg_.initial_data_angular_velocity << '\n';
   }
   if (continue_disk) {
      if (metadata.disk_angular_velocity
             * cfg_.initial_data_disk_angular_velocity < 0.0_rt
          || std::abs(metadata.disk_angular_velocity)
             >= std::abs(cfg_.initial_data_disk_angular_velocity)) {
         amrex::Abort(
            "equilibrium-torus physical ID continuation requires a larger "
            "target angular-velocity magnitude with the saved rotation sign"
         );
      }
      amrex::Print()
         << "continuing equilibrium-torus physical initial data: "
         << "source_omega=" << metadata.disk_angular_velocity
         << " target_omega="
         << cfg_.initial_data_disk_angular_velocity << '\n';
   }
   if (add_torus) {
      amrex::Print()
         << "continuing physical initial data with a "
         << cfg_.initial_data_disk_gravity << ' '
         << cfg_.initial_data_disk << " matter component\n";
   }
   bool resolve_changed_reduced_rns = false;
   bool resolve_spherical_black_hole = false;
#if (AMREX_SPACEDIM >= 2)
   resolve_changed_reduced_rns = configured_rotation
      && cfg_.initial_data_solve_only
      && cfg_.spacetime_initial_data == "xcts"
      && cfg_.mesh_coordinate == "spherical"
      && (cfg_.mesh_axes == "r theta"
          || cfg_.mesh_axes == "r theta phi")
      // A tilted disk makes the full-storage source genuinely phi-dependent.
      // The native spherical continuation owner is axisymmetric, so this
      // route remains a portable rematerialization rather than a false
      // axisymmetric constraint re-solve.
      && cfg_.initial_data_disk != "tilted_torus"
      && !metadata.exact_layout;
#endif
#if (AMREX_SPACEDIM >= 2)
   resolve_spherical_black_hole = metadata.source_ic == "black_hole"
      && cfg_.ic == "black_hole"
      && cfg_.initial_data_solve_only
      && cfg_.spacetime_initial_data == "xcts"
      && cfg_.mesh_coordinate == "spherical"
      && (cfg_.mesh_axes == "r theta"
          || cfg_.mesh_axes == "r theta phi");
#endif
   if (resolve_changed_reduced_rns && !continue_rotation) {
      amrex::Print()
         << "re-solving rotating physical initial data on the changed "
            "r-theta grid at omega="
         << cfg_.initial_data_angular_velocity << '\n';
   }
   if (resolve_spherical_black_hole) {
      amrex::Print()
         << "re-solving black-hole physical initial data on the configured "
         << cfg_.mesh_axes << " shell\n";
   }
   if ((continue_rotation || resolve_changed_reduced_rns)
       && !(metadata.injection_energy > 0.0_rt)) {
      amrex::Abort(
         "saved rotating-star continuation requires persisted injection "
         "energy; regenerate this legacy physical-ID artifact"
      );
   }
   return continue_rotation || continue_disk || resolve_changed_reduced_rns
      || resolve_spherical_black_hole || add_torus;
}

void AmrSolverBase::project_materialized_initial_face_b()
{
#if (AMREX_SPACEDIM > 1)
   const int num_levels = finestLevel() + 1;
   amrex::Vector<amrex::Geometry> geometry(num_levels);
   amrex::Vector<amrex::BoxArray> grids(num_levels);
   amrex::Vector<amrex::DistributionMapping> distribution(num_levels);
   amrex::Vector<FaceMagneticField> logical_face_b(num_levels);
   amrex::Vector<FaceMagneticField> gradient(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> potential_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> rhs_storage(num_levels);
   amrex::Vector<amrex::MultiFab*> potential(num_levels);
   amrex::Vector<amrex::MultiFab const*> rhs(num_levels);
   amrex::Real maximum_divergence = 0.0_rt;
   for (int level = 0; level < num_levels; ++level) {
      geometry[level] = Geom(level);
      grids[level] = boxArray(level);
      distribution[level] = DistributionMap(level);
      allocate_face_magnetic_field_like(logical_face_b[level], face_b_[level]);
      allocate_face_magnetic_field_like(gradient[level], face_b_[level]);
      fill_face_fillpatch_from_face_b(
         logical_face_b[level], face_b_[level], Geom(level), cfg_
      );
      rhs_storage[level] = std::make_unique<amrex::MultiFab>(
         grids[level], distribution[level], 1, 0
      );
      fill_face_divergence(
         *rhs_storage[level], logical_face_b[level], Geom(level), false
      );
      maximum_divergence = std::max(
         maximum_divergence, rhs_storage[level]->norm0(0, 0, true)
      );
      potential_storage[level] = std::make_unique<amrex::MultiFab>(
         grids[level], distribution[level], 1, 1
      );
      potential_storage[level]->setVal(0.0_rt);
      potential[level] = potential_storage[level].get();
      rhs[level] = rhs_storage[level].get();
   }
   amrex::ParallelDescriptor::ReduceRealMax(maximum_divergence);
   if (maximum_divergence <= 64.0_rt * std::numeric_limits<amrex::Real>::epsilon()) {
      return;
   }
   amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lower_boundary{};
   amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> upper_boundary{};
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      const amrex::LinOpBCType type = Geom(0).isPeriodic(direction)
         ? amrex::LinOpBCType::Periodic
         : amrex::LinOpBCType::Dirichlet;
      lower_boundary[direction] = type;
      upper_boundary[direction] = type;
   }
   amrex::LPInfo info;
   amrex::MLPoisson poisson(geometry, grids, distribution, info);
   poisson.setMaxOrder(2);
   poisson.setDomainBC(lower_boundary, upper_boundary);
   for (int level = 0; level < num_levels; ++level) {
      poisson.setLevelBC(level, nullptr);
   }
   amrex::MLMG mlmg(poisson);
   mlmg.setMaxIter(200);
   mlmg.setMaxFmgIter(1);
   mlmg.setVerbose(0);
   const amrex::Real residual = mlmg.solve(potential, rhs, 1.0e-12_rt, 0.0_rt);
   if (!std::isfinite(residual)) {
      amrex::Abort("physical-ID magnetic projection produced a nonfinite residual");
   }
   amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> gradients(
      num_levels
   );
   for (int level = 0; level < num_levels; ++level) {
      gradients[level] = face_b_array(gradient[level]);
   }
   mlmg.getGradSolution(gradients);
   for (int level = 0; level < num_levels; ++level) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         amrex::MultiFab::Add(
            *logical_face_b[level].b[direction],
            *gradient[level].b[direction],
            0,
            0,
            1,
            0
         );
      }
      fill_face_b_from_face_fillpatch(
         face_b_[level], logical_face_b[level], Geom(level), cfg_
      );
   }
   amrex::Print()
      << "projected materialized physical-ID magnetic field: initial_divb_linf="
      << maximum_divergence << " residual=" << residual << '\n';
#endif
}

void AmrSolverBase::write_physical_initial_data_artifact()
{
   // An exact-layout artifact load restores both the portable primitive core
   // and evolution caches.  Reconstructing that already-authoritative
   // primitive core before a solve-only reserialization introduces a tiny C2P
   // round-trip change and defeats exact continuation.
   const bool preserve_exact_primitive = loaded_physical_id_metadata_
      && loaded_physical_id_metadata_->used_exact_cache
      && !initial_data_rotating_star_summary_
      && !initial_data_equilibrium_torus_summary_;
   if (preserve_exact_primitive) {
      MutablePhysicalIdField primitive;
      for (int level = 0; level <= finestLevel(); ++level) {
         primitive.push_back(primitive_[level].get());
      }
      restore_physical_initial_data_primitive(
         cfg_.initial_data_input_file, primitive
      );
   }
   write_physical_initial_data_artifact(
      cfg_.initial_data_output_file,
      cfg_.initial_data_angular_velocity,
      cfg_.initial_data_disk_angular_velocity,
      !preserve_exact_primitive
   );
}

void AmrSolverBase::write_physical_initial_data_artifact(
   std::string const& path,
   amrex::Real stellar_angular_velocity,
   amrex::Real disk_angular_velocity,
   bool refresh_primitive,
   RotatingStarUpdateSummary const* rotating_star,
   EquilibriumTorusUpdateSummary const* equilibrium_torus,
   ConformalBlackHoleSummary const* black_hole
)
{
   amrex::Vector<amrex::Geometry> geometry;
   amrex::Vector<amrex::IntVect> refinement_ratios;
   ConstPhysicalIdField primitive;
   ConstPhysicalIdField fluid;
   ConstPhysicalIdField spacetime;
   ConstPhysicalIdFaceFields face_magnetic;
   for (int level = 0; level <= finestLevel(); ++level) {
      if (refresh_primitive) {
         fill_level_primitive_cache(level, 0.0_rt);
      }
      geometry.push_back(Geom(level));
      primitive.push_back(primitive_[level].get());
      fluid.push_back(fluid_state_.state()[level].get());
      if (spacetime_state_.active()) {
         spacetime.push_back(spacetime_state_.state()[level].get());
      }
      if (ct_enabled(cfg_)) {
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            face_magnetic[direction].push_back(
               face_b_[level].b[direction].get()
            );
         }
      }
      if (level < finestLevel()) {
         refinement_ratios.push_back(refRatio(level));
      }
   }
   RunConfig artifact_cfg = cfg_;
   artifact_cfg.initial_data_angular_velocity = stellar_angular_velocity;
   artifact_cfg.initial_data_disk_angular_velocity = disk_angular_velocity;
   if (loaded_physical_id_metadata_) {
      // Magnetic construction and force-balance identity belong to the
      // physical artifact, not to the target run's construction request.
      // An exact-layout solve-only reload therefore reserializes the saved
      // identity even though no seed or equilibrium solve is requested again.
      artifact_cfg.initial_data_magnetic_seed =
         loaded_physical_id_metadata_->magnetic_seed;
      artifact_cfg.initial_data_magnetic_seed_pressure_cutoff =
         loaded_physical_id_metadata_->magnetic_seed_pressure_cutoff;
      artifact_cfg.initial_data_magnetic_seed_exponent =
         loaded_physical_id_metadata_->magnetic_seed_exponent;
      artifact_cfg.initial_data_magnetic_seed_maximum_comoving_field =
         loaded_physical_id_metadata_->magnetic_seed_maximum_comoving_field;
      artifact_cfg.initial_data_magnetic_equilibrium =
         loaded_physical_id_metadata_->magnetic_equilibrium;
      artifact_cfg.initial_data_magnetic_equilibrium_poloidal_magnetization =
         loaded_physical_id_metadata_
            ->magnetic_equilibrium_poloidal_magnetization;
      artifact_cfg.initial_data_magnetic_equilibrium_nonlinear_poloidal =
         loaded_physical_id_metadata_
            ->magnetic_equilibrium_nonlinear_poloidal;
      artifact_cfg.initial_data_magnetic_equilibrium_relaxation =
         loaded_physical_id_metadata_->magnetic_equilibrium_relaxation;
   }
   RotatingStarUpdateSummary const* selected_rotating_star = rotating_star;
   if (selected_rotating_star == nullptr
       && stellar_angular_velocity == cfg_.initial_data_angular_velocity
       && initial_data_rotating_star_summary_) {
      selected_rotating_star = initial_data_rotating_star_summary_.get();
   }
   EquilibriumTorusUpdateSummary const* selected_equilibrium_torus =
      equilibrium_torus;
   if (selected_equilibrium_torus == nullptr
       && disk_angular_velocity == cfg_.initial_data_disk_angular_velocity
       && initial_data_equilibrium_torus_summary_) {
      selected_equilibrium_torus = initial_data_equilibrium_torus_summary_.get();
   }
   const amrex::Real black_hole_throat_area = black_hole == nullptr
      ? initial_data_black_hole_throat_area_ : black_hole->throat_area;
   const amrex::Real black_hole_irreducible_mass = black_hole == nullptr
      ? initial_data_black_hole_irreducible_mass_
      : black_hole->irreducible_mass;
   const amrex::Real black_hole_throat_equatorial_circumference =
      black_hole == nullptr
         ? initial_data_black_hole_throat_equatorial_circumference_
         : black_hole->equatorial_circumference;
   const amrex::Real black_hole_throat_polar_circumference =
      black_hole == nullptr
         ? initial_data_black_hole_throat_polar_circumference_
         : black_hole->polar_circumference;
   const amrex::Real black_hole_throat_kerr_mass = black_hole == nullptr
      ? initial_data_black_hole_throat_kerr_mass_
      : black_hole->kerr_horizon_mass;
   const amrex::Real black_hole_throat_kerr_spin = black_hole == nullptr
      ? initial_data_black_hole_throat_kerr_spin_
      : black_hole->kerr_dimensionless_spin;
   const amrex::Real black_hole_throat_angular_momentum =
      black_hole == nullptr
         ? initial_data_black_hole_throat_angular_momentum_
         : black_hole->angular_momentum;
   const amrex::Real black_hole_throat_christodoulou_mass =
      black_hole == nullptr
         ? initial_data_black_hole_throat_christodoulou_mass_
         : black_hole->christodoulou_mass;
   const amrex::Real black_hole_throat_christodoulou_spin =
      black_hole == nullptr
         ? initial_data_black_hole_throat_christodoulou_spin_
         : black_hole->christodoulou_dimensionless_spin;
   const PhysicalInitialDataMetadata artifact_metadata =
      physical_id_artifact_metadata(
         artifact_cfg,
         loaded_physical_id_metadata_.get(),
         black_hole_throat_area,
         black_hole_irreducible_mass,
         black_hole_throat_equatorial_circumference,
         black_hole_throat_polar_circumference,
         black_hole_throat_kerr_mass,
         black_hole_throat_kerr_spin,
         black_hole_throat_angular_momentum,
         black_hole_throat_christodoulou_mass,
         black_hole_throat_christodoulou_spin,
         selected_rotating_star,
         selected_equilibrium_torus
      );
   write_physical_initial_data(
      path,
      artifact_cfg,
      physics_system_,
      geometry,
      refinement_ratios,
      primitive,
      fluid,
      spacetime,
      face_magnetic,
      initial_data_coordinate_faces_,
      artifact_metadata
   );
   amrex::Print() << "wrote physical initial data: " << path << '\n';
}

void AmrSolverBase::initialize_physical_initial_data_sequence_output()
{
   if (!physical_initial_data_sequence_output_enabled()) {
      return;
   }
   const std::filesystem::path sequence_dir =
      physical_initial_data_sequence_output_dir();
   if (amrex::ParallelDescriptor::IOProcessor()) {
      if (std::filesystem::exists(sequence_dir)) {
         amrex::Abort(
            "physical initial-data sequence output already exists: "
            + sequence_dir.string()
         );
      }
      std::filesystem::create_directories(sequence_dir);
      std::ofstream index(sequence_dir / "sequence.tsv");
      index
         << "index\ttargets\tangular_velocity\tdisk_angular_velocity"
         << "\tmaximum_density"
         << "\tmaximum_velocity\tinjection_energy"
         << "\tequatorial_radius\tpolar_radius"
         << "\tequatorial_log_enthalpy_gradient"
         << "\tpolar_log_enthalpy_gradient\tmass_shedding_indicator"
         << "\tadm_mass\tbaryonic_mass\tangular_momentum\tdimensionless_spin"
         << "\tdisk_maximum_density\tdisk_inner_radius\tdisk_outer_radius"
         << "\tdisk_density_maximum_radius\tdisk_baryonic_mass"
         << "\tdisk_angular_momentum"
         << "\tartifact\n";
      std::ofstream metadata(sequence_dir / "sequence.meta");
      metadata << "# type=physical_initial_data_sequence\n"
               << "# schema_version=6\n"
               << "# eos=" << cfg_.eos << '\n'
               << "# eos_preset=" << cfg_.eos_preset << '\n'
               << "# thermal_gamma=" << std::setprecision(17)
               << cfg_.thermal_gamma << '\n'
               << "# central_density=" << std::setprecision(17)
               << cfg_.tov_central_density << '\n'
               << "# rotation_law=" << cfg_.initial_data_rotation_law << '\n'
               << "# differential_rotation_length=" << std::setprecision(17)
               << cfg_.initial_data_differential_rotation_length << '\n'
               << "# differential_rotation_length_b="
               << std::setprecision(17)
               << cfg_.initial_data_differential_rotation_length_b << '\n'
               << "# rotation_power_q=" << std::setprecision(17)
               << cfg_.initial_data_rotation_power_q << '\n'
               << "# rotation_axis=" << std::setprecision(17)
               << cfg_.initial_data_rotation_axis1 << ','
               << cfg_.initial_data_rotation_axis2 << ','
               << cfg_.initial_data_rotation_axis3 << '\n'
               << "# rotation_center=" << std::setprecision(17)
               << cfg_.initial_data_rotation_center1 << ','
               << cfg_.initial_data_rotation_center2 << ','
               << cfg_.initial_data_rotation_center3 << '\n'
               << "# disk=" << cfg_.initial_data_disk << '\n'
               << "# disk_gravity=" << cfg_.initial_data_disk_gravity << '\n'
               << "# disk_rotation_law="
               << cfg_.initial_data_disk_rotation_law << '\n'
               << "# disk_specific_angular_momentum="
               << std::setprecision(17)
               << cfg_.initial_data_disk_specific_angular_momentum << '\n'
               << "# disk_angular_velocity=" << std::setprecision(17)
               << cfg_.initial_data_disk_angular_velocity << '\n'
               << "# disk_differential_rotation_length="
               << std::setprecision(17)
               << cfg_.initial_data_disk_differential_rotation_length << '\n'
               << "# disk_rotation_power_q=" << std::setprecision(17)
               << cfg_.initial_data_disk_rotation_power_q << '\n'
               << "# disk_inner_radius=" << std::setprecision(17)
               << cfg_.initial_data_disk_inner_radius << '\n'
               << "# disk_axis=" << std::setprecision(17)
               << cfg_.initial_data_disk_axis1 << ','
               << cfg_.initial_data_disk_axis2 << ','
               << cfg_.initial_data_disk_axis3 << '\n'
               << "# artifact_contract=PyFluid physical initial data v3\n";
      if (cfg_.eos_preset == "custom") {
         metadata << "# polytropic_k=" << std::setprecision(17)
                  << cfg_.polytropic_k << '\n'
                  << "# polytropic_gamma=" << std::setprecision(17)
                  << cfg_.polytropic_gamma << '\n'
                  << "# polytropic_density_breaks=";
         for (std::size_t index = 0;
              index < cfg_.polytropic_density_breaks.size(); ++index) {
            metadata << (index == 0 ? "" : ",") << std::setprecision(17)
                     << cfg_.polytropic_density_breaks[index];
         }
         metadata << '\n' << "# polytropic_segment_gammas=";
         for (std::size_t index = 0;
              index < cfg_.polytropic_segment_gammas.size(); ++index) {
            metadata << (index == 0 ? "" : ",") << std::setprecision(17)
                     << cfg_.polytropic_segment_gammas[index];
         }
         metadata << '\n';
      }
   }
   amrex::ParallelDescriptor::Barrier();
}

bool AmrSolverBase::physical_initial_data_sequence_output_enabled() const
{
   const bool multiple_targets =
      cfg_.initial_data_rotation_continuation_steps > 1
      || !cfg_.initial_data_rotation_continuation_targets.empty();
   const bool continues_saved_model = !cfg_.initial_data_input_file.empty();
   return cfg_.initial_data_save_sequence
      && (multiple_targets || continues_saved_model
          || !cfg_.initial_data_sequence_output_dir.empty());
}

std::string AmrSolverBase::physical_initial_data_sequence_output_dir() const
{
   if (!cfg_.initial_data_sequence_output_dir.empty()) {
      return cfg_.initial_data_sequence_output_dir;
   }
   return (std::filesystem::path(cfg_.output_dir)
           / "initial_data_sequences"
           / sequence_eos_label(cfg_)
           / sequence_rotation_label(cfg_)
           / ("rho_c-" + sequence_real_label(cfg_.tov_central_density)))
      .string();
}

void AmrSolverBase::write_physical_initial_data_sequence_model(
   int continuation_step,
   int continuation_steps,
   amrex::Real stellar_angular_velocity,
   amrex::Real disk_angular_velocity,
   ConformalBlackHoleSummary const& black_hole,
   RotatingStarUpdateSummary const& rotating_star,
   EquilibriumTorusUpdateSummary const& equilibrium_torus
)
{
   std::ostringstream model_name;
   model_name << "model_" << std::setw(4) << std::setfill('0')
              << continuation_step
              << "_omega-" << sequence_real_label(stellar_angular_velocity);
   if (cfg_.initial_data_disk == "equilibrium_torus"
       || cfg_.initial_data_disk == "tilted_torus") {
      model_name
         << "_diskomega-" << sequence_real_label(disk_angular_velocity);
   }
   model_name
              << "_baseN-";
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      if (direction != 0) {
         model_name << 'x';
      }
      model_name << Geom(0).Domain().length(direction);
   }
   model_name << "_levels-" << finestLevel() + 1;
   const std::filesystem::path relative_artifact = model_name.str();
   const std::filesystem::path artifact =
      std::filesystem::path(physical_initial_data_sequence_output_dir())
      / relative_artifact;
   write_physical_initial_data_artifact(
      artifact.string(),
      stellar_angular_velocity,
      disk_angular_velocity,
      false,
      &rotating_star,
      (cfg_.initial_data_disk == "equilibrium_torus"
       || cfg_.initial_data_disk == "tilted_torus")
         ? &equilibrium_torus : nullptr,
      &black_hole
   );
   if (amrex::ParallelDescriptor::IOProcessor()) {
      std::ofstream index(
         std::filesystem::path(physical_initial_data_sequence_output_dir())
            / "sequence.tsv",
         std::ios::app
      );
      index << continuation_step << '\t' << continuation_steps << '\t'
            << std::setprecision(17) << stellar_angular_velocity << '\t'
            << disk_angular_velocity << '\t'
            << rotating_star.maximum_density << '\t'
            << rotating_star.maximum_velocity << '\t'
            << rotating_star.injection_energy << '\t'
            << rotating_star.equatorial_radius << '\t'
            << rotating_star.polar_radius << '\t'
            << rotating_star.equatorial_log_enthalpy_gradient << '\t'
            << rotating_star.polar_log_enthalpy_gradient << '\t'
            << rotating_star.mass_shedding_indicator << '\t'
            << rotating_star.adm_mass << '\t'
            << rotating_star.baryonic_mass << '\t'
            << rotating_star.angular_momentum << '\t'
            << rotating_star.dimensionless_spin << '\t'
            << equilibrium_torus.maximum_density << '\t'
            << equilibrium_torus.inner_radius << '\t'
            << equilibrium_torus.outer_radius << '\t'
            << equilibrium_torus.density_maximum_radius << '\t'
            << equilibrium_torus.baryonic_mass << '\t'
            << equilibrium_torus.angular_momentum << '\t'
            << relative_artifact.string() << '\n';
   }
   amrex::ParallelDescriptor::Barrier();
}

} // namespace pyfluid_amrex
