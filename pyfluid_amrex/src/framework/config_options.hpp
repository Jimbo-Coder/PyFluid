#pragma once

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos/presets.hpp"
#include "pyfluid_amrex/method_codes.hpp"

namespace pyfluid_amrex::config_options {

struct NamedOptionCode {
   std::string_view name;
   int code;
};

static constexpr std::array<NamedOptionCode, 5> RiemannOptions{{
   {"llf", RiemannLLF},
   {"hll", RiemannHLL},
   {"hlle", RiemannHLLE},
   {"hllc", RiemannHLLC},
   {"hlld", RiemannHLLD},
}};

inline constexpr auto const& ReconOptions = ReconstructionMethodCatalog;

static constexpr std::array<NamedOptionCode, 5> TimestepOptions{{
   {"euler", TimeEuler},
   {"corrector", TimeCorrector},
   {"midpoint", TimeMidpoint},
   {"ssprk3", TimeSSPRK3},
   {"rk4", TimeRK4},
}};

static constexpr std::array<NamedOptionCode, 3> DiffusionOptions{{
   {"none", DiffusionNone},
   {"explicit", DiffusionExplicit},
   {"implicit", DiffusionImplicit},
}};

static constexpr std::array<NamedOptionCode, 4> EOSOptions{{
   {"ideal_gas", EOSIdealGas},
   {"polytrope", EOSPolytrope},
   {"hybrid_polytrope", EOSHybridPolytrope},
   {"compose", EOSCompose},
}};

inline constexpr auto EOSPresetOptions = eos::PiecewisePolytropePresetNames;

static constexpr std::array<std::string_view, 4> PhysicsOptions{{"hydro", "mhd", "rhd", "rmhd"}};
static constexpr std::array<std::string_view, 3> SpacetimeEvolutionOptions{{
   "none", "fixed", "evolved"
}};
static constexpr std::array<std::string_view, 2> SpacetimeMatterCouplingOptions{{"coupled", "vacuum"}};
static constexpr std::array<std::string_view, 8> SpacetimeInitialDataOptions{{
   "minkowski", "gauge_wave", "linear_wave", "robust_stability",
   "kerr_schild", "ctt", "cttk", "xcts"
}};
static constexpr std::array<std::string_view, 3> BlackHoleInteriorOptions{{
   "exterior", "filled", "excision"
}};
static constexpr std::array<std::string_view, 3> MagneticInitialDataOptions{{
   "id", "zero", "none"
}};
static constexpr std::array<std::string_view, 3> SpacetimeGroupInitialDataOptions{{
   "id", "flat", "none"
}};
static constexpr std::array<std::string_view, 3> LapseInitialDataOptions{{
   "id", "one", "none"
}};
static constexpr std::array<std::string_view, 3> ShiftInitialDataOptions{{
   "id", "zero", "none"
}};
static constexpr std::array<std::string_view, 2> InitialAtmosphereOptions{{
   "flat", "power_law"
}};
static constexpr std::array<std::string_view, 2> InitialDiskOptions{{
   "none", "equilibrium_torus"
}};
static constexpr std::array<std::string_view, 2> InitialDiskGravityOptions{{
   "test_fluid", "coupled"
}};
static constexpr std::array<std::string_view, 4> Z4cLapseGaugeOptions{{
   "frozen", "harmonic", "one_plus_log", "slow_start"
}};
static constexpr std::array<std::string_view, 2> Z4cShiftGaugeOptions{{
   "frozen", "gamma_driver"
}};
static constexpr std::array<std::string_view, 4> FixedSpacetimeMetricOptions{{
   "minkowski", "rindler", "schwarzschild_isotropic", "kerr_schild",
}};
static constexpr std::array<std::string_view, 2> CTOptions{{"off", "on"}};
static constexpr std::array<std::string_view, 2> PlotfileFormatOptions{{"amrex", "hdf5"}};
static constexpr std::array<std::string_view, 5> PassiveProfileOptions{{"none", "uniform", "gaussian", "sin_wave", "step_x"}};
static constexpr std::array<std::string_view, 2> ProfileOptions{{"on", "off"}};
static constexpr std::array<std::string_view, 2> ProfileDetailOptions{{"basic", "flux"}};
static constexpr std::array<std::string_view, 8> BoundaryOptions{{
   "periodic",
   "outflow",
   "reflecting",
   "symmetry",
   "inflow",
   "diode",
   "origin",
   "analytic",
}};
static constexpr std::array<std::string_view, 2> BoundaryProfileOptions{{"none", "double_mach_reflection"}};
static constexpr std::array<std::string_view, 3> InflowProfileOptions{{"uniform", "gaussian_beam", "uniform_beam"}};
static constexpr std::array<std::string_view, 2> ScalarOutputModeOptions{{"stream", "final"}};
static constexpr std::array<std::string_view, 10> ValidationPolicyOptions{{
   "closed",
   "open",
   "open_mhd",
   "driven",
   "wall",
   "diffusive",
   "diffusive_wall",
   "smoke",
   "elliptic",
   "dynamic_spacetime",
}};
static constexpr std::array<std::string_view, 3> SourceOptions{{"none", "uniform_acceleration", "rotating_compact_heating"}};
static constexpr std::array<std::string_view, 3> GravityOptions{{"none", "poisson", "fixed"}};
static constexpr std::array<std::string_view, 2> GravityCouplingOptions{{"athena", "source"}};
static constexpr std::array<std::string_view, 2> GravityFixedPotentialOptions{{"uniform_acceleration", "harmonic"}};
static constexpr std::array<std::string_view, 2> SubcyclingOptions{{"off", "on"}};
static constexpr std::array<std::string_view, 2> SubcyclingStepOptions{{"spatial", "local_cfl"}};
static constexpr std::array<std::string_view, 4> SolidGeometryOptions{{"off", "boxes", "wedge", "circle"}};
static constexpr std::array<std::string_view, 4> SolidMotionOptions{{"static", "translate", "rotate", "pulse"}};
static constexpr std::array<std::string_view, 2> SolidWallConditionOptions{{"slip", "no_slip"}};
static constexpr std::array<std::string_view, 3> MeshPatchBoxOwnershipOptions{{"exact", "owner_first", "owner_last"}};

static constexpr std::array<std::string_view, 2> MeshPatchBoxDomainOptions{{"rectangular", "union"}};
static constexpr std::array<std::string_view, 2> AMRTagMetricOptions{{"none", "lohner"}};
static constexpr std::array<std::string_view, 2> AMRRegridModeOptions{{"synchronized", "aligned"}};
static constexpr std::array<std::string_view, 5> AMRStateInterpOptions{{
   "component-conservative",
   "coupled-conservative",
   "quadratic",
   "quartic",
   "quartic-conservative",
}};
static constexpr std::array<std::string_view, 2> MetricRegridOptions{{"auto", "always"}};
static constexpr std::array<std::string_view, 2> InitialRefineOptions{{"none", "box"}};
static constexpr std::array<std::string_view, 3> KHPerturbationOptions{{"normal", "stream", "vortex"}};
static constexpr std::array<std::string_view, 5> HydroWaveFamilyOptions{{
   "acoustic_plus",
   "acoustic_minus",
   "entropy",
   "shear_y",
   "shear_z",
}};
static constexpr std::array<std::string_view, 7> MHDWaveFamilyOptions{{
   "fast_plus",
   "fast_minus",
   "alfven_plus",
   "alfven_minus",
   "slow_plus",
   "slow_minus",
   "entropy",
}};
using PhysicalSystemSet = std::uint8_t;

constexpr PhysicalSystemSet physical_system_set(PhysicalSystem system) noexcept
{
   return PhysicalSystemSet{1U} << static_cast<std::uint8_t>(system);
}

constexpr PhysicalSystemSet operator|(PhysicalSystem lhs, PhysicalSystem rhs) noexcept
{
   return physical_system_set(lhs) | physical_system_set(rhs);
}

constexpr PhysicalSystemSet operator|(
   PhysicalSystemSet systems,
   PhysicalSystem system
) noexcept
{
   return systems | physical_system_set(system);
}

constexpr bool contains_physical_system(
   PhysicalSystemSet systems,
   PhysicalSystem system
) noexcept
{
   return (systems & physical_system_set(system)) != 0;
}

struct InitialDataOption {
   std::string_view name;
   PhysicalSystemSet physical_systems;
};

static constexpr PhysicalSystemSet HydroSystems = PhysicalSystem::HD | PhysicalSystem::RHD;
static constexpr PhysicalSystemSet MagneticSystems = PhysicalSystem::MHD | PhysicalSystem::RMHD;
static constexpr PhysicalSystemSet AllFluidSystems = HydroSystems | PhysicalSystem::MHD | PhysicalSystem::RMHD;

// This is the single source of truth for IC/system compatibility. A Newtonian
// profile admitted for RHD/RMHD is converted through primitive variables; that
// does not make its Newtonian analytic evolution an exact relativistic test.
static constexpr std::array<InitialDataOption, 33> InitialDataOptions{{
   {"sin_wave", AllFluidSystems},
   {"advection_medley", HydroSystems},
   {"characteristic_medley", HydroSystems},
   {"mhd_characteristic_medley", PhysicalSystem::MHD | PhysicalSystem::RMHD},
   {"uniform", AllFluidSystems},
   {"cartesian_advection", AllFluidSystems},
   {"linear_wave_x", AllFluidSystems},
   {"acoustic_wave_x", HydroSystems},
   {"acoustic_packet_x", HydroSystems},
   {"jeans_wave_x", physical_system_set(PhysicalSystem::HD)},
   {"shear_wave_x", HydroSystems},
   {"pressure_bump", AllFluidSystems},
   {"vortex_blast", AllFluidSystems},
   {"density_cloud", AllFluidSystems},
   {"hse_polytrope_x", physical_system_set(PhysicalSystem::HD)},
   {"fixed_metric_hse", physical_system_set(PhysicalSystem::RHD)},
   {"tov", physical_system_set(PhysicalSystem::RHD)},
   {"rotating_star", physical_system_set(PhysicalSystem::RHD)},
   {"hydro_beam", HydroSystems},
   {"sod_x", AllFluidSystems},
   {"sod_plane", AllFluidSystems},
   {"sod_periodic_plane", AllFluidSystems},
   {"double_mach_reflection", physical_system_set(PhysicalSystem::HD)},
   {"circular_alfven_x", MagneticSystems},
   {"resistive_wave_x", physical_system_set(PhysicalSystem::MHD)},
   {"magnetic_tube_z", MagneticSystems},
   {"mhd_rotor", MagneticSystems},
   {"kelvin_helmholtz", HydroSystems},
   {"kelvin_helmholtz_mhd", MagneticSystems},
   {"orszag_tang", MagneticSystems},
   {"magnetic_blast_noise", MagneticSystems},
   {"resistive_current_sheet", physical_system_set(PhysicalSystem::MHD)},
   {"reconnecting_shear_layer", physical_system_set(PhysicalSystem::MHD)},
}};

template <std::size_t N>
inline bool option_supported(std::string const& value, std::array<std::string_view, N> const& options)
{
   for (std::string_view option : options) {
      if (value == option) {
         return true;
      }
   }
   return false;
}

template <std::size_t N>
inline bool option_supported(std::string const& value, std::array<NamedOptionCode, N> const& options)
{
   for (NamedOptionCode const& option : options) {
      if (value == option.name) {
         return true;
      }
   }
   return false;
}

template <std::size_t N>
inline bool option_supported(
   std::string const& value,
   std::array<ReconstructionMethodInfo, N> const& options
)
{
   for (ReconstructionMethodInfo const& option : options) {
      if (value == option.name) {
         return true;
      }
   }
   return false;
}

template <std::size_t N>
inline bool option_supported(
   std::string const& value,
   std::array<InitialDataOption, N> const& options
)
{
   for (InitialDataOption const& option : options) {
      if (value == option.name) {
         return true;
      }
   }
   return false;
}

template <std::size_t N>
inline std::string supported_option_names(std::array<std::string_view, N> const& options)
{
   std::string names;
   for (std::size_t n = 0; n < N; ++n) {
      if (n > 0) {
         names += ", ";
      }
      names += options[n];
   }
   return names;
}

template <std::size_t N>
inline std::string supported_option_names(std::array<NamedOptionCode, N> const& options)
{
   std::string names;
   for (std::size_t n = 0; n < N; ++n) {
      if (n > 0) {
         names += ", ";
      }
      names += options[n].name;
   }
   return names;
}

template <std::size_t N>
inline std::string supported_option_names(
   std::array<ReconstructionMethodInfo, N> const& options
)
{
   std::string names;
   for (std::size_t n = 0; n < N; ++n) {
      if (n > 0) {
         names += ", ";
      }
      names += options[n].name;
   }
   return names;
}

template <std::size_t N>
inline std::string supported_option_names(
   std::array<InitialDataOption, N> const& options
)
{
   std::string names;
   for (std::size_t n = 0; n < N; ++n) {
      if (n > 0) {
         names += ", ";
      }
      names += options[n].name;
   }
   return names;
}

template <std::size_t N>
inline int option_code(std::string const& value, std::array<NamedOptionCode, N> const& options, char const* key)
{
   for (NamedOptionCode const& option : options) {
      if (value == option.name) {
         return option.code;
      }
   }
   amrex::Abort(std::string(key) + " supports " + supported_option_names(options));
   return -1;
}

template <std::size_t N>
inline int option_code(
   std::string const& value,
   std::array<ReconstructionMethodInfo, N> const& options,
   char const* key
)
{
   for (ReconstructionMethodInfo const& option : options) {
      if (value == option.name) {
         return option.code;
      }
   }
   amrex::Abort(std::string(key) + " supports " + supported_option_names(options));
   return -1;
}

template <std::size_t N>
inline void require_supported_option(std::string const& value, std::array<std::string_view, N> const& options, char const* key)
{
   if (!option_supported(value, options)) {
      amrex::Abort(std::string(key) + " supports " + supported_option_names(options));
   }
}

template <std::size_t N>
inline void require_supported_option(std::string const& value, std::array<NamedOptionCode, N> const& options, char const* key)
{
   if (!option_supported(value, options)) {
      amrex::Abort(std::string(key) + " supports " + supported_option_names(options));
   }
}

template <std::size_t N>
inline void require_supported_option(
   std::string const& value,
   std::array<ReconstructionMethodInfo, N> const& options,
   char const* key
)
{
   if (!option_supported(value, options)) {
      amrex::Abort(std::string(key) + " supports " + supported_option_names(options));
   }
}

template <std::size_t N>
inline void require_supported_option(
   std::string const& value,
   std::array<InitialDataOption, N> const& options,
   char const* key
)
{
   if (!option_supported(value, options)) {
      amrex::Abort(std::string(key) + " supports " + supported_option_names(options));
   }
}

inline InitialDataOption const& initial_data_option(std::string const& name)
{
   for (InitialDataOption const& option : InitialDataOptions) {
      if (name == option.name) {
         return option;
      }
   }
   amrex::Abort("pyfluid.ic supports " + supported_option_names(InitialDataOptions));
   return InitialDataOptions.front();
}

inline bool valid_recon_name(std::string const& name)
{
   return option_supported(name, ReconOptions);
}

inline bool valid_plotfile_format(std::string const& value)
{
   return option_supported(value, PlotfileFormatOptions);
}

inline bool valid_bc_name(std::string const& value)
{
   return option_supported(value, BoundaryOptions);
}

inline bool valid_hydro_wave_family(std::string const& value)
{
   return option_supported(value, HydroWaveFamilyOptions);
}

inline bool valid_mhd_wave_family(std::string const& value)
{
   return option_supported(value, MHDWaveFamilyOptions);
}

} // namespace pyfluid_amrex::config_options
