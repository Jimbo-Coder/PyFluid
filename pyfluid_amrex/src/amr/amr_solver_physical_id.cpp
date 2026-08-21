#include "amr/amr_solver_base.hpp"

#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/initial_data/atmosphere.hpp"
#include "pyfluid_amrex/initial_data/elliptic.hpp"
#include "pyfluid_amrex/initial_data/equilibrium_torus.hpp"
#include "pyfluid_amrex/initial_data/physical_id.hpp"
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
   if (cfg.initial_data_rotation_law == "j_constant"
       || cfg.initial_data_rotation_law == "q_power") {
      label += "-A-"
         + sequence_real_label(
            cfg.initial_data_differential_rotation_length
         );
   }
   if (cfg.initial_data_rotation_law == "q_power") {
      label += "-q-" + sequence_real_label(cfg.initial_data_rotation_power_q);
   }
   if (cfg.initial_data_disk == "equilibrium_torus") {
      label += "-disk-" + cfg.initial_data_disk_rotation_law
         + "-gravity-" + cfg.initial_data_disk_gravity
         + "-ell-"
         + sequence_real_label(
              cfg.initial_data_disk_specific_angular_momentum
           )
         + "-rin-" + sequence_real_label(cfg.initial_data_disk_inner_radius);
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

} // namespace

void AmrSolverBase::materialize_configured_initial_data()
{
   if (cfg_.initial_data_input_file.empty()) {
      materialize_constraint_solved_initial_data();
      return;
   }
   if (load_physical_initial_data_artifact()) {
      materialize_constraint_solved_initial_data(true);
   }
}

bool AmrSolverBase::load_physical_initial_data_artifact()
{
   amrex::Vector<amrex::Geometry> geometry;
   amrex::Vector<amrex::IntVect> refinement_ratios;
   MutablePhysicalIdField primitive;
   MutablePhysicalIdField fluid;
   MutablePhysicalIdField spacetime;
   MutablePhysicalIdFaceFields face_magnetic;
#if (AMREX_SPACEDIM == 2)
   if (cfg_.initial_data_radial_grid == "adaptive") {
      const RelativisticInitialDataView provider = initial_data_provider_.view();
      if (!provider.tov.active() || finestLevel() != 0) {
         amrex::Abort(
            "adaptive stellar ID loading requires one level and a TOV seed"
         );
      }
      initial_data_coordinate_faces_.resize(1);
      initial_data_coordinate_faces_[0][0] = stellar_adaptive_radial_faces(
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
      initial_atmosphere_parameters(cfg_, initial_data_provider_.view()),
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
       && cfg_.initial_data_rotation_law == "q_power"
       && metadata.rotation_power_q != cfg_.initial_data_rotation_power_q) {
      amrex::Abort(
         "rotating-star physical ID continuation cannot change the q-power "
         "exponent"
      );
   }
   if (configured_rotation
       && (metadata.rotation_center[0] != cfg_.initial_data_rotation_center1
           || metadata.rotation_center[1] != cfg_.initial_data_rotation_center2
           || metadata.rotation_center[2] != cfg_.initial_data_rotation_center3)) {
      amrex::Abort(
         "rotating-star physical ID continuation cannot change the rotation center"
      );
   }
   const bool continue_rotation = configured_rotation
      && metadata.angular_velocity != cfg_.initial_data_angular_velocity;
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
   bool resolve_changed_reduced_rns = false;
#if (AMREX_SPACEDIM == 2)
   resolve_changed_reduced_rns = configured_rotation
      && cfg_.initial_data_solve_only
      && cfg_.spacetime_initial_data == "xcts"
      && cfg_.mesh_coordinate == "spherical"
      && cfg_.mesh_axes == "r theta"
      && !metadata.exact_layout;
#endif
   if (resolve_changed_reduced_rns && !continue_rotation) {
      amrex::Print()
         << "re-solving rotating physical initial data on the changed "
            "r-theta grid at omega="
         << cfg_.initial_data_angular_velocity << '\n';
   }
   if ((continue_rotation || resolve_changed_reduced_rns)
       && !(metadata.injection_energy > 0.0)) {
      amrex::Abort(
         "saved rotating-star continuation requires persisted injection "
         "energy; regenerate this legacy physical-ID artifact"
      );
   }
   return continue_rotation || resolve_changed_reduced_rns;
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
                 && loaded_physical_id_metadata_->disk
                    == "equilibrium_torus") {
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
               << "# rotation_power_q=" << std::setprecision(17)
               << cfg_.initial_data_rotation_power_q << '\n'
               << "# disk=" << cfg_.initial_data_disk << '\n'
               << "# disk_gravity=" << cfg_.initial_data_disk_gravity << '\n'
               << "# disk_rotation_law="
               << cfg_.initial_data_disk_rotation_law << '\n'
               << "# disk_specific_angular_momentum="
               << std::setprecision(17)
               << cfg_.initial_data_disk_specific_angular_momentum << '\n'
               << "# disk_inner_radius=" << std::setprecision(17)
               << cfg_.initial_data_disk_inner_radius << '\n'
               << "# artifact_contract=PyFluid physical initial data v2\n";
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
      if (cfg_.initial_data_disk == "equilibrium_torus") {
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
