#include "amr/amr_solver_base.hpp"

#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/initial_data.hpp"
#include "pyfluid_amrex/initial_data/atmosphere.hpp"
#include "pyfluid_amrex/initial_data/elliptic.hpp"
#include "pyfluid_amrex/initial_data/equilibrium_torus.hpp"
#include "pyfluid_amrex/initial_data/fluid.hpp"
#include "pyfluid_amrex/initial_data/perturbation.hpp"
#include "pyfluid_amrex/initial_data/physical_id.hpp"
#include "pyfluid_amrex/initial_data/provider.hpp"
#include "pyfluid_amrex/initial_data/rotating_star.hpp"
#include "pyfluid_amrex/z4c_runtime.hpp"

#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>

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

namespace pyfluid_amrex {
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

void append_rotating_star_metadata(
   std::filesystem::path const& artifact,
   RotatingStarUpdateSummary const& rotating_star
)
{
   std::ofstream metadata(artifact / "Header.meta", std::ios::app);
   metadata
      << "# maximum_density=" << std::setprecision(17)
      << rotating_star.maximum_density << '\n'
      << "# maximum_velocity=" << std::setprecision(17)
      << rotating_star.maximum_velocity << '\n'
      << "# injection_energy=" << std::setprecision(17)
      << rotating_star.injection_energy << '\n'
      << "# equatorial_radius=" << std::setprecision(17)
      << rotating_star.equatorial_radius << '\n'
      << "# polar_radius=" << std::setprecision(17)
      << rotating_star.polar_radius << '\n'
      << "# equatorial_log_enthalpy_gradient=" << std::setprecision(17)
      << rotating_star.equatorial_log_enthalpy_gradient << '\n'
      << "# polar_log_enthalpy_gradient=" << std::setprecision(17)
      << rotating_star.polar_log_enthalpy_gradient << '\n'
      << "# mass_shedding_indicator=" << std::setprecision(17)
      << rotating_star.mass_shedding_indicator << '\n'
      << "# adm_mass=" << std::setprecision(17)
      << rotating_star.adm_mass << '\n'
      << "# baryonic_mass=" << std::setprecision(17)
      << rotating_star.baryonic_mass << '\n'
      << "# angular_momentum=" << std::setprecision(17)
      << rotating_star.angular_momentum << '\n'
      << "# dimensionless_spin=" << std::setprecision(17)
      << rotating_star.dimensionless_spin << '\n';
}

void append_equilibrium_torus_metadata(
   std::filesystem::path const& artifact,
   EquilibriumTorusUpdateSummary const& equilibrium_torus
)
{
   std::ofstream metadata(artifact / "Header.meta", std::ios::app);
   metadata
      << "# disk_maximum_density=" << std::setprecision(17)
      << equilibrium_torus.maximum_density << '\n'
      << "# disk_maximum_velocity=" << std::setprecision(17)
      << equilibrium_torus.maximum_velocity << '\n'
      << "# disk_resolved_inner_radius=" << std::setprecision(17)
      << equilibrium_torus.inner_radius << '\n'
      << "# disk_outer_radius=" << std::setprecision(17)
      << equilibrium_torus.outer_radius << '\n'
      << "# disk_density_maximum_radius=" << std::setprecision(17)
      << equilibrium_torus.density_maximum_radius << '\n'
      << "# disk_maximum_log_specific_enthalpy=" << std::setprecision(17)
      << equilibrium_torus.maximum_log_specific_enthalpy << '\n'
      << "# disk_baryonic_mass=" << std::setprecision(17)
      << equilibrium_torus.baryonic_mass << '\n'
      << "# disk_angular_momentum=" << std::setprecision(17)
      << equilibrium_torus.angular_momentum << '\n';
}

void append_matter_outer_radius(
   std::filesystem::path const& artifact,
   amrex::Real matter_outer_radius
)
{
   std::ofstream metadata(artifact / "Header.meta", std::ios::app);
   metadata << "# matter_outer_radius=" << std::setprecision(17)
            << matter_outer_radius << '\n';
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
            fill_level_primitive_cache(level, 0.0);
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
   if (kind == InitialDataPerturbation::DensityAzimuthalEspino2019
       && !(reference_radius > 0.0)) {
      if (initial_data_rotating_star_summary_) {
         reference_radius =
            initial_data_rotating_star_summary_->equatorial_radius;
      } else if (loaded_physical_id_metadata_
                 && loaded_physical_id_metadata_->equatorial_radius > 0.0) {
         reference_radius = loaded_physical_id_metadata_->equatorial_radius;
      } else {
         reference_radius = initial_data_provider_->view().tov.surface_radius;
      }
   }
   if (kind == InitialDataPerturbation::DensityAzimuthalEspino2019
       && !(reference_radius > 0.0)) {
      amrex::Abort(
         "density_azimuthal_espino2019 could not resolve a positive "
         "equatorial reference radius"
      );
   }
   const amrex::GpuArray<amrex::Real, 3> origin{{
      cfg_.initial_data_rotation_center1,
      cfg_.initial_data_rotation_center2,
      cfg_.initial_data_rotation_center3,
   }};
   const amrex::GpuArray<amrex::Real, 3> axis{{
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
         const amrex::Real tolerance = 256.0
            * std::numeric_limits<amrex::Real>::epsilon()
            * std::max<amrex::Real>(1.0, std::abs(plane));
         if (low_symmetry == high_symmetry
             || std::abs(
                std::abs(parameters.frame.axis[symmetry_direction]) - 1.0
             ) > tolerance
             || std::abs(origin[symmetry_direction] - plane) > tolerance) {
            amrex::Abort(
               "nonaxisymmetric initial-data perturbation is incompatible "
               "with the configured reflection symmetry"
            );
         }
      }
   }

   amrex::Real maximum_fractional_change = 0.0;
   amrex::Real invalid_points = 0.0;
   const InitialAtmosphereParameters atmosphere =
      initial_atmosphere_parameters(cfg_, initial_data_provider_->view());
   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      fill_level_primitive_cache(lev, 0.0);
      amrex::MultiFab reduction(
         primitive_[lev]->boxArray(),
         primitive_[lev]->DistributionMap(),
         2,
         0
      );
      reduction.setVal(0.0);
      const auto problo = Geom(lev).ProbLoArray();
      const auto dx = Geom(lev).CellSizeArray();
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
               const auto mesh_point = mesh_coordinate_triplet_at_location(
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
                  output(i, j, k, 1) = 1.0;
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
   if (invalid_points != 0.0) {
      amrex::Abort(
         "initial-data perturbation produced an invalid cold-EoS state"
      );
   }
   finalize_constraint_solved_initial_data_state();
   amrex::Print()
      << "applied initial-data perturbation: kind="
      << cfg_.initial_data_perturbation
      << " maximum_fractional_density_change="
      << maximum_fractional_change
      << " constraint_polish=none\n";
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
   amrex::Real baryonic_mass = 0.0;
   amrex::GpuArray<amrex::Real, 3> angular_momentum{{0.0, 0.0, 0.0}};
   for (int level = 0; level <= finestLevel(); ++level) {
      fill_level_primitive_cache(level, 0.0);
      amrex::iMultiFab valid_mask = valid_cell_mask(level);
      diagnostics.push_back(std::make_unique<amrex::MultiFab>(
         primitive_[level]->boxArray(),
         primitive_[level]->DistributionMap(),
         num_components,
         0
      ));
      amrex::MultiFab& diagnostic = *diagnostics.back();
      diagnostic.setVal(0.0);
      diagnostic.setVal(invalid_min, inner_radius_component, 1, 0);
      const auto problo = Geom(level).ProbLoArray();
      const auto dx = Geom(level).CellSizeArray();
      const amrex::Box domain = Geom(level).Domain();
      const amrex::Dim3 domain_lo = amrex::lbound(domain);
      const amrex::Dim3 domain_hi = amrex::ubound(domain);
      const amrex::Real half_diagonal = 0.5 * std::sqrt(
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
               problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0],
               problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1],
               problo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2],
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
                  0.0
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
            const bool boundary_cell = i == domain_lo.x || i == domain_hi.x
               || j == domain_lo.y || j == domain_hi.y
               || k == domain_lo.z || k == domain_hi.z;
            output(i, j, k, boundary_excess_component) = boundary_cell
               ? rho - atmosphere.rho : 0.0;
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
            amrex::Real(0.0)
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
   if (!(summary.maximum_density > 0.0)
       || !(summary.maximum_velocity >= 0.0)
       || !(summary.maximum_velocity < 1.0)
       || !(summary.outer_radius > summary.inner_radius)
       || !(summary.maximum_log_specific_enthalpy > 0.0)
       || !std::isfinite(summary.inner_radius)) {
      amrex::Abort(
         "equilibrium-torus parameters do not produce resolved finite support"
      );
   }
   if (summary.boundary_density_excess > 0.0) {
      amrex::Abort(
         "equilibrium-torus matter reaches the Cartesian boundary; enlarge "
         "the ID domain"
      );
   }
   const amrex::Real density_tolerance = 128.0
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
                < maximum_density * (1.0 - density_tolerance)) {
               values(i, j, k, density_maximum_radius_component) = 0.0;
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
   if (!(summary.baryonic_mass > 0.0)
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
              0.5 * cfg_.spacetime_mass,
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
   analytic_fixed_spacetime_kernels_ =
      !spacetime_state_.evolves()
      && cfg_.spacetime_initial_data == "minkowski"
      && metadata.spacetime_initial_data == "minkowski";
   if (!metadata.used_exact_cache) {
      ConstPhysicalIdField cell_spacetime;
      if (spacetime_state_.active()) {
         for (int level = 0; level <= finestLevel(); ++level) {
            spacetime_coupling_.invalidate_cell_spacetime(level);
         }
         fill_all_spacetime_boundaries(spacetime_state_.state(), 0.0);
         if (initial_data_coordinate_faces_.empty()) {
            for (int level = 0; level <= finestLevel(); ++level) {
               complete_z4c_contracted_connections(
                  *spacetime_state_.state()[level], Geom(level)
               );
            }
         }
         fill_all_spacetime_boundaries(spacetime_state_.state(), 0.0);
         prepare_spacetime_coupling_range(
            spacetime_state_.state(),
            0,
            finestLevel() + 1,
            0.0,
            -1.0,
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
   const bool add_torus = metadata.disk == "none"
      && (cfg_.initial_data_disk == "equilibrium_torus"
          || cfg_.initial_data_disk == "tilted_torus")
      && cfg_.spacetime_initial_data == "xcts";
   if constexpr (AMREX_SPACEDIM == 3) {
      const bool spherical_axisymmetric_target =
         cfg_.mesh_coordinate == "spherical"
         && cfg_.mesh_axes == "r theta phi"
         && cfg_.initial_data_rotation_center1 == 0.0
         && cfg_.initial_data_rotation_center2 == 0.0
         && cfg_.initial_data_rotation_center3 == 0.0
         && cfg_.initial_data_rotation_axis1 == 0.0
         && cfg_.initial_data_rotation_axis2 == 0.0
         && cfg_.initial_data_rotation_axis3 > 0.0
         && (!add_torus
             || (cfg_.initial_data_disk == "equilibrium_torus"
                 && cfg_.initial_data_disk_axis1 == 0.0
                 && cfg_.initial_data_disk_axis2 == 0.0
                 && cfg_.initial_data_disk_axis3 > 0.0));
      if (cfg_.mesh_coordinate != "cartesian"
          && !spherical_axisymmetric_target
          && (continue_rotation || add_torus)) {
         amrex::Abort(
            "3D curvilinear XCTS continuation requires a centered, +z "
            "axisymmetric r-theta-phi target and an aligned equilibrium torus"
         );
      }
   }
   if (continue_rotation) {
      if (cfg_.spacetime_initial_data != "xcts") {
         amrex::Abort(
            "saved rotating-star continuation currently requires XCTS"
         );
      }
      if (metadata.angular_velocity * cfg_.initial_data_angular_velocity < 0.0
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
   if (add_torus) {
      amrex::Print()
         << "continuing physical initial data with a "
         << cfg_.initial_data_disk_gravity << ' '
         << cfg_.initial_data_disk << " matter component\n";
   }
   bool resolve_changed_reduced_rns = false;
   bool resolve_changed_spherical_black_hole = false;
#if (AMREX_SPACEDIM >= 2)
   resolve_changed_reduced_rns = configured_rotation
      && cfg_.initial_data_solve_only
      && cfg_.spacetime_initial_data == "xcts"
      && cfg_.mesh_coordinate == "spherical"
      && (cfg_.mesh_axes == "r theta"
          || cfg_.mesh_axes == "r theta phi")
      && !metadata.exact_layout;
#endif
#if (AMREX_SPACEDIM >= 2)
   resolve_changed_spherical_black_hole = metadata.source_ic == "black_hole"
      && cfg_.ic == "black_hole"
      && cfg_.initial_data_solve_only
      && cfg_.spacetime_initial_data == "xcts"
      && cfg_.mesh_coordinate == "spherical"
      && (cfg_.mesh_axes == "r theta"
          || cfg_.mesh_axes == "r theta phi")
      && !metadata.exact_layout;
#endif
   if (resolve_changed_reduced_rns && !continue_rotation) {
      amrex::Print()
         << "re-solving rotating physical initial data on the changed "
            "r-theta grid at omega="
         << cfg_.initial_data_angular_velocity << '\n';
   }
   if (resolve_changed_spherical_black_hole) {
      amrex::Print()
         << "re-solving black-hole physical initial data on the changed "
         << cfg_.mesh_axes << " shell\n";
   }
   if ((continue_rotation || resolve_changed_reduced_rns)
       && !(metadata.injection_energy > 0.0)) {
      amrex::Abort(
         "saved rotating-star continuation requires persisted injection "
         "energy; regenerate this legacy physical-ID artifact"
      );
   }
   return continue_rotation || resolve_changed_reduced_rns
      || resolve_changed_spherical_black_hole || add_torus;
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
   amrex::Real maximum_divergence = 0.0;
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
      potential_storage[level]->setVal(0.0);
      potential[level] = potential_storage[level].get();
      rhs[level] = rhs_storage[level].get();
   }
   amrex::ParallelDescriptor::ReduceRealMax(maximum_divergence);
   if (maximum_divergence <= 64.0 * std::numeric_limits<amrex::Real>::epsilon()) {
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
   const amrex::Real residual = mlmg.solve(potential, rhs, 1.0e-12, 0.0);
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
   write_physical_initial_data_artifact(
      cfg_.initial_data_output_file,
      cfg_.initial_data_angular_velocity,
      true
   );
}

void AmrSolverBase::write_physical_initial_data_artifact(
   std::string const& path,
   amrex::Real angular_velocity,
   bool refresh_primitive
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
         fill_level_primitive_cache(level, 0.0);
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
   artifact_cfg.initial_data_angular_velocity = angular_velocity;
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
      initial_data_coordinate_faces_
   );
   if (angular_velocity == cfg_.initial_data_angular_velocity
       && amrex::ParallelDescriptor::IOProcessor()) {
      amrex::Real matter_outer_radius = loaded_physical_id_metadata_
         ? loaded_physical_id_metadata_->matter_outer_radius : 0.0;
      if (initial_data_rotating_star_summary_) {
         append_rotating_star_metadata(
            path, *initial_data_rotating_star_summary_
         );
         matter_outer_radius = std::max(
            matter_outer_radius,
            initial_data_rotating_star_summary_->matter_outer_radius
         );
      } else if (loaded_physical_id_metadata_
                 && loaded_physical_id_metadata_->source_ic
                    == "rotating_star") {
         append_rotating_star_metadata(
            path, rotating_star_summary(*loaded_physical_id_metadata_)
         );
      }
      if (initial_data_equilibrium_torus_summary_) {
         append_equilibrium_torus_metadata(
            path, *initial_data_equilibrium_torus_summary_
         );
         matter_outer_radius = std::max(
            matter_outer_radius,
            initial_data_equilibrium_torus_summary_->outer_radius
         );
      } else if (loaded_physical_id_metadata_
                 && (loaded_physical_id_metadata_->disk
                        == "equilibrium_torus"
                     || loaded_physical_id_metadata_->disk
                        == "tilted_torus")) {
         append_equilibrium_torus_metadata(
            path, equilibrium_torus_summary(*loaded_physical_id_metadata_)
         );
      }
      if (matter_outer_radius > 0.0) {
         append_matter_outer_radius(path, matter_outer_radius);
      }
   }
   amrex::ParallelDescriptor::Barrier();
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
         << "index\ttargets\tangular_velocity\tmaximum_density"
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
               << "# schema_version=5\n"
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
   amrex::Real angular_velocity,
   RotatingStarUpdateSummary const& rotating_star,
   EquilibriumTorusUpdateSummary const& equilibrium_torus
)
{
   std::ostringstream model_name;
   model_name << "model_" << std::setw(4) << std::setfill('0')
              << continuation_step
              << "_omega-" << sequence_real_label(angular_velocity)
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
      artifact.string(), angular_velocity, false
   );
   if (amrex::ParallelDescriptor::IOProcessor()) {
      append_rotating_star_metadata(artifact, rotating_star);
      if (cfg_.initial_data_disk == "equilibrium_torus"
          || cfg_.initial_data_disk == "tilted_torus") {
         append_equilibrium_torus_metadata(artifact, equilibrium_torus);
      }
      append_matter_outer_radius(
         artifact,
         std::max(
            rotating_star.matter_outer_radius,
            equilibrium_torus.outer_radius
         )
      );
      std::ofstream index(
         std::filesystem::path(physical_initial_data_sequence_output_dir())
            / "sequence.tsv",
         std::ios::app
      );
      index << continuation_step << '\t' << continuation_steps << '\t'
            << std::setprecision(17) << angular_velocity << '\t'
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
