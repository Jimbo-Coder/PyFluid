cmake_minimum_required(VERSION 3.25)

project(pyfluid_amrex LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
option(PYFLUID_AMREX_ENABLE_IMPLICIT_GRAVITY "Compile implicit-gravity paths that use AMReX linear solvers." ON)

# Track the real installation in include/link commands even when a preset
# selects it through the stable managed link. Version switches then invalidate
# dependent objects normally, including switches to older file timestamps.
set(CMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS TRUE)
find_package(AMReX REQUIRED)
if(AMReX_CUDA_FOUND)
   enable_language(CUDA)
   include(AMReXTargetHelpers)
   set(CMAKE_CUDA_STANDARD 20)
   set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(PYFLUID_AMREX_PRECISION "${AMReX_PRECISION}" CACHE STRING
   "Floating-point precision; must match the selected AMReX install.")
set_property(CACHE PYFLUID_AMREX_PRECISION PROPERTY STRINGS SINGLE DOUBLE)
if(NOT PYFLUID_AMREX_PRECISION STREQUAL AMReX_PRECISION)
   message(FATAL_ERROR
      "PyFluid precision ${PYFLUID_AMREX_PRECISION} does not match "
      "AMReX precision ${AMReX_PRECISION}. Select the matching AMReX install "
      "and a separate build directory for each precision.")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
   "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/amrex.toml")
if(NOT AMReX_MPI_FOUND OR NOT AMReX_MPI)
   message(FATAL_ERROR
      "PyFluid AMReX requires an MPI-enabled AMReX install. "
      "Set AMReX_DIR to an AMReX build configured with -DAMReX_MPI=ON.")
endif()
if(NOT AMReX_HDF5_FOUND OR NOT AMReX_HDF5)
   message(FATAL_ERROR
      "PyFluid AMReX requires an AMReX install with parallel HDF5 output. "
      "Set AMReX_DIR to an MPI AMReX build configured with -DAMReX_HDF5=ON.")
endif()
if(PYFLUID_AMREX_ENABLE_IMPLICIT_GRAVITY)
   if(NOT AMReX_LSOLVERS_FOUND AND NOT AMReX_LSOLVERS)
      message(FATAL_ERROR
         "pyfluid_amrex is configured to compile implicit-gravity paths but this AMReX install "
         "does not include linear solvers (AMReX_LSOLVERS=OFF). "
         "Rebuild AMReX with -DAMReX_LSOLVERS=ON and set AMReX_DIR to that build, "
         "or configure PyFluid with -DPYFLUID_AMREX_ENABLE_IMPLICIT_GRAVITY=OFF.")
   endif()
endif()

option(PYFLUID_AMREX_NATIVE "Tune PyFluid AMReX binaries for the build host CPU." ON)
option(
   PYFLUID_AMREX_HORIZON_OPENMP
   "Use isolated OpenMP parallelism in the host apparent-horizon finder."
   ON
)
option(PYFLUID_AMREX_ENABLE_HYDRO "Compile hydro evolution branches." ON)
option(PYFLUID_AMREX_ENABLE_MHD "Compile MHD evolution branches." ON)
option(PYFLUID_AMREX_ENABLE_CT "Compile constrained-transport MHD branches." ON)
option(PYFLUID_AMREX_ENABLE_RECON_PWC "Compile piecewise-constant reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_PWL "Compile piecewise-linear reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_WENOZ "Compile component WENO-Z reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_WENOZ_CHAR "Compile characteristic WENO-Z reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_MP5 "Compile component MP5 reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_MP5_CHAR "Compile characteristic MP5 reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_TENO5 "Compile component TENO5 reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_TENO5_CHAR "Compile characteristic TENO5 reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_PARABOLIC "Compile component unlimited parabolic reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_PARABOLIC_CHAR "Compile characteristic unlimited parabolic reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_PPM4 "Compile component PPM4 reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_PPM4_CHAR "Compile characteristic PPM4 reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_PPMX "Compile component PPMX reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_PPMX_CHAR "Compile characteristic PPMX reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_PPM3 "Compile component PPM3 reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RECON_PPM3_CHAR "Compile characteristic PPM3 reconstruction." ON)
option(PYFLUID_AMREX_ENABLE_RIEMANN_LLF "Compile LLF Riemann solver." ON)
option(PYFLUID_AMREX_ENABLE_RIEMANN_HLL "Compile HLL Riemann solver." ON)
option(PYFLUID_AMREX_ENABLE_RIEMANN_HLLE "Compile HLLE Riemann solver." ON)
option(PYFLUID_AMREX_ENABLE_RIEMANN_HLLC "Compile HLLC Riemann solver." ON)
option(PYFLUID_AMREX_ENABLE_RIEMANN_HLLD "Compile HLLD Riemann solver." ON)
option(PYFLUID_AMREX_ENABLE_TIME_EULER "Compile Euler time integration." ON)
option(PYFLUID_AMREX_ENABLE_TIME_CORRECTOR "Compile corrector time integration." ON)
option(PYFLUID_AMREX_ENABLE_TIME_MIDPOINT "Compile midpoint time integration." ON)
option(PYFLUID_AMREX_ENABLE_TIME_SSPRK3 "Compile SSPRK3 time integration." ON)
option(PYFLUID_AMREX_ENABLE_TIME_RK4 "Compile RK4 time integration." ON)
option(PYFLUID_AMREX_ENABLE_DIFFUSION_NONE "Compile no-diffusion evolution branches." ON)
option(PYFLUID_AMREX_ENABLE_DIFFUSION_EXPLICIT "Compile explicit-diffusion evolution branches." ON)
option(PYFLUID_AMREX_ENABLE_DIFFUSION_IMPLICIT "Compile implicit-diffusion evolution branches." ON)
option(PYFLUID_AMREX_ENABLE_SUBCYCLING_OFF "Compile non-subcycled AMR evolution branches." ON)
option(PYFLUID_AMREX_ENABLE_SUBCYCLING_ON "Compile subcycled AMR evolution branches." ON)
if(PYFLUID_AMREX_NATIVE)
   check_cxx_compiler_flag("-mcpu=native" PYFLUID_AMREX_HAS_MCPU_NATIVE)
endif()

set(PYFLUID_AMREX_DRIVER_SOURCES
   src/driver/main.cpp
   src/amr/amr_solver_driver.cpp
   src/driver/evolution_kernel_bundle.cpp
)

set(PYFLUID_AMREX_KERNEL_BUNDLE_SOURCES
   src/kernel_bundles/hydro.cpp
)
if(PYFLUID_AMREX_ENABLE_HYDRO)
   list(APPEND PYFLUID_AMREX_KERNEL_BUNDLE_SOURCES
      src/kernel_bundles/hydro_passive.cpp
   )
endif()
if(PYFLUID_AMREX_ENABLE_MHD)
   list(APPEND PYFLUID_AMREX_KERNEL_BUNDLE_SOURCES
      src/kernel_bundles/mhd.cpp
      src/kernel_bundles/mhd_passive.cpp
   )
   if(PYFLUID_AMREX_ENABLE_CT)
      list(APPEND PYFLUID_AMREX_KERNEL_BUNDLE_SOURCES
         src/kernel_bundles/mhd_ct.cpp
         src/kernel_bundles/mhd_passive_ct.cpp
      )
   endif()
endif()

set(PYFLUID_AMREX_AMR_SOURCES
   src/amr/gravity_fields.cpp
   src/amr/amr_solver_average.cpp
   src/amr/amr_solver_auxiliary.cpp
   src/amr/amr_solver_conservation.cpp
   src/amr/amr_solver_debug.cpp
   src/amr/amr_solver_diagnostics.cpp
   src/amr/amr_solver_fillpatch.cpp
   src/amr/amr_solver_flux_registers.cpp
   src/amr/amr_solver_gravity_sync.cpp
   src/amr/amr_solver_gravity.cpp
   src/amr/amr_solver_horizon.cpp
   src/amr/amr_solver_solid.cpp
   src/amr/amr_solver_hierarchy.cpp
   src/amr/amr_solver_levels.cpp
   src/amr/amr_solver_metrics.cpp
   src/amr/amr_solver_multipatch.cpp
   src/amr/amr_solver_output.cpp
   src/amr/amr_solver_physical_id.cpp
   src/amr/amr_solver_radiation.cpp
   src/amr/amr_solver_spacetime.cpp
   src/amr/amr_solver_spacetime_evolution.cpp
   src/amr/amr_solver_valid_cells.cpp
   src/amr/state_interpolation.cpp
)

set(PYFLUID_AMREX_FRAMEWORK_SOURCES
   src/framework/checkpoint.cpp
   src/framework/config.cpp
   src/framework/config_runtime.cpp
   src/framework/evolution_sector_storage.cpp
   src/framework/fields.cpp
   src/framework/profiling.cpp
   src/framework/multipoles.cpp
   src/framework/surface_geometry.cpp
)

set(PYFLUID_AMREX_EOS_SOURCES
   src/eos/eos_config.cpp
   src/eos/presets.cpp
   src/eos/tabulated.cpp
)

set(PYFLUID_AMREX_NUMERICS_SOURCES
   src/numerics/origin_boundary.cpp
   src/numerics/ct_edge_flux_register.cpp
   src/numerics/flux_factory.cpp
   src/numerics/sources.cpp
   src/numerics/implicit_resistivity.cpp
   src/numerics/diagnostic_reductions.cpp
   src/numerics/nodal_interpolater.cpp
   src/numerics/hierarchy_interpolation.cpp
)

set(PYFLUID_AMREX_FLUX_KERNEL_SOURCES)
if(PYFLUID_AMREX_ENABLE_RECON_PWC)
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/pwc.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_PWL)
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/pwl.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_WENOZ)
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/wenoz.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_WENOZ_CHAR AND (PYFLUID_AMREX_ENABLE_HYDRO OR PYFLUID_AMREX_ENABLE_MHD))
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/wenoz_char.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_MP5)
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/mp5.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_MP5_CHAR AND (PYFLUID_AMREX_ENABLE_HYDRO OR PYFLUID_AMREX_ENABLE_MHD))
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/mp5_char.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_TENO5)
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/teno5.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_TENO5_CHAR AND (PYFLUID_AMREX_ENABLE_HYDRO OR PYFLUID_AMREX_ENABLE_MHD))
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/teno5_char.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_PARABOLIC)
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/parabolic.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_PARABOLIC_CHAR AND (PYFLUID_AMREX_ENABLE_HYDRO OR PYFLUID_AMREX_ENABLE_MHD))
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/parabolic_char.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_PPM4)
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/ppm4.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_PPM4_CHAR AND (PYFLUID_AMREX_ENABLE_HYDRO OR PYFLUID_AMREX_ENABLE_MHD))
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/ppm4_char.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_PPMX)
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/ppmx.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_PPMX_CHAR AND (PYFLUID_AMREX_ENABLE_HYDRO OR PYFLUID_AMREX_ENABLE_MHD))
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/ppmx_char.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_PPM3)
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/ppm3.cpp)
endif()
if(PYFLUID_AMREX_ENABLE_RECON_PPM3_CHAR AND (PYFLUID_AMREX_ENABLE_HYDRO OR PYFLUID_AMREX_ENABLE_MHD))
   list(APPEND PYFLUID_AMREX_FLUX_KERNEL_SOURCES src/flux_kernels/ppm3_char.cpp)
endif()

set(PYFLUID_AMREX_INITIAL_DATA_SOURCES
   src/initial_data/equilibrium_torus_initial_data.cpp
   src/initial_data/initial_data_centering.cpp
   src/initial_data/initial_data_ctt.cpp
   src/initial_data/initial_data_elliptic.cpp
   src/initial_data/initial_data_elliptic_spherical.cpp
   src/initial_data/initial_data_magnetic.cpp
   src/initial_data/initial_data_materialization.cpp
   src/initial_data/initial_data_provider.cpp
   src/initial_data/physical_id.cpp
   src/initial_data/physical_id_fluid_materialization.cpp
   src/initial_data/physical_id_interpolation.cpp
   src/initial_data/initial_data_rhd.cpp
   src/initial_data/initial_data_state.cpp
   src/initial_data/rotating_star_initial_data.cpp
   src/initial_data/tov_initial_data.cpp
)

set(PYFLUID_AMREX_RELATIVITY_SOURCES
   src/relativity/apparent_horizon_surface.cpp
   src/relativity/apparent_horizon_tracker.cpp
   src/relativity/black_hole_excision.cpp
   src/relativity/primitive_cache_factory.cpp
   src/relativity/reference_metric_valencia.cpp
   src/relativity/relativistic_exterior.cpp
   src/relativity/relativistic_flux_limiter.cpp
   src/relativity/relativistic_source_factory.cpp
   src/relativity/signal_speed_factory.cpp
   src/relativity/spacetime_coupling_storage.cpp
   src/relativity/z4c_runtime.cpp
   src/relativity/fccz4_runtime.cpp
   src/relativity/z4c_outer_boundary.cpp
)

# LLF/HLL are required correction fallbacks even when not advertised as
# primary methods. A hydro-only configuration owns no RMHD flux kernels.
if(PYFLUID_AMREX_ENABLE_MHD)
   list(APPEND PYFLUID_AMREX_RELATIVITY_SOURCES
      src/relativity/rmhd_correction_flux.cpp
      src/relativity/rmhd_correction_flux_llf.cpp
      src/relativity/rmhd_correction_flux_hll.cpp
   )
endif()
if(PYFLUID_AMREX_ENABLE_MHD AND PYFLUID_AMREX_ENABLE_RIEMANN_HLLC)
   list(APPEND PYFLUID_AMREX_RELATIVITY_SOURCES
      src/relativity/rmhd_correction_flux_hllc.cpp
   )
endif()
if(PYFLUID_AMREX_ENABLE_MHD AND PYFLUID_AMREX_ENABLE_RIEMANN_HLLD)
   list(APPEND PYFLUID_AMREX_RELATIVITY_SOURCES
      src/relativity/rmhd_correction_flux_hlld.cpp
   )
endif()

if(PYFLUID_AMREX_ENABLE_MHD AND
   (PYFLUID_AMREX_ENABLE_RIEMANN_HLLC OR
    PYFLUID_AMREX_ENABLE_RIEMANN_HLLD))
   list(APPEND PYFLUID_AMREX_RELATIVITY_SOURCES
      src/relativity/rmhd_advanced_flux.cpp
   )
endif()

set(PYFLUID_AMREX_SOURCES
   ${PYFLUID_AMREX_DRIVER_SOURCES}
   ${PYFLUID_AMREX_KERNEL_BUNDLE_SOURCES}
   ${PYFLUID_AMREX_AMR_SOURCES}
   ${PYFLUID_AMREX_FRAMEWORK_SOURCES}
   ${PYFLUID_AMREX_EOS_SOURCES}
   ${PYFLUID_AMREX_NUMERICS_SOURCES}
   ${PYFLUID_AMREX_FLUX_KERNEL_SOURCES}
   ${PYFLUID_AMREX_INITIAL_DATA_SOURCES}
   ${PYFLUID_AMREX_RELATIVITY_SOURCES}
)

# Guard the compiled production owners, including shared object-library uses.
# Integer indices/signs converting to Real are a separate range contract; this
# check prevents accidental floating-point widening or narrowing in FP32.
if(PYFLUID_AMREX_PRECISION STREQUAL "SINGLE" AND
   CMAKE_CXX_COMPILER_ID MATCHES "Clang")
   set_property(SOURCE
      ${PYFLUID_AMREX_SOURCES}
      src/relativity/apparent_horizon_finder.cpp
      APPEND PROPERTY COMPILE_OPTIONS
      -Werror=double-promotion
      -Werror=implicit-float-conversion
      -Wno-implicit-int-float-conversion
   )
endif()

set(PYFLUID_AMREX_CHECKPOINT_COMPARE_SOURCES
   src/framework/checkpoint.cpp
   tools/cpp/compare_checkpoint.cpp
   src/framework/config.cpp
   src/framework/config_runtime.cpp
   src/eos/eos_config.cpp
   src/eos/presets.cpp
   src/eos/tabulated.cpp
   src/framework/fields.cpp
)

set(PYFLUID_AMREX_CODEGEN_CHECK_SOURCES
   src/eos/presets.cpp
   tests/generated_flux_compile_check.cpp
   tests/relativistic_vacuum_check.cpp
)

set(PYFLUID_AMREX_C2P_BENCHMARK_SOURCES
   benchmarks/c2p_benchmark.cpp
)

set(PYFLUID_AMREX_TABULATED_EOS_CHECK_SOURCES
   src/eos/tabulated.cpp
   tests/tabulated_eos_check.cpp
)

set(PYFLUID_AMREX_RECONSTRUCTION_CHECK_SOURCES
   tests/reconstruction_check.cpp
)

set(PYFLUID_AMREX_STATE_UPDATE_CHECK_SOURCES
   tests/state_update_check.cpp
   src/numerics/diagnostic_reductions.cpp
)

set(PYFLUID_AMREX_RMHD_RIEMANN_CHECK_SOURCES
   tests/rmhd_riemann_check.cpp
)

set(PYFLUID_AMREX_STATE_INTERPOLATION_CHECK_SOURCES
   src/amr/state_interpolation.cpp
   tests/state_interpolation_check.cpp
)

set(PYFLUID_AMREX_PHYSICAL_ID_INTERPOLATION_CHECK_SOURCES
   src/framework/fields.cpp
   src/initial_data/physical_id_interpolation.cpp
   tests/physical_id_interpolation_check.cpp
)

set(PYFLUID_AMREX_MULTIPATCH_CHECK_SOURCES
   tests/multipatch_check.cpp
   tests/mapped_patch_flux_check.cpp
)

set(PYFLUID_AMREX_ORIGIN_BOUNDARY_CHECK_SOURCES
   src/framework/fields.cpp
   src/numerics/origin_boundary.cpp
   src/amr/state_interpolation.cpp
   tests/origin_boundary_check.cpp
)

set(PYFLUID_AMREX_BOUNDARY_TILING_CHECK_SOURCES
   src/numerics/origin_boundary.cpp
   src/framework/fields.cpp
   tests/boundary_tiling_check.cpp
)

set(PYFLUID_AMREX_SURFACE_CHECK_SOURCES
   src/framework/multipoles.cpp
   src/framework/surface_geometry.cpp
   src/numerics/hierarchy_interpolation.cpp
   tests/surface_interpolation_check.cpp
)

set(PYFLUID_AMREX_APPARENT_HORIZON_CHECK_SOURCES
   src/relativity/apparent_horizon_surface.cpp
   tests/apparent_horizon_finder_check.cpp
)

set(PYFLUID_AMREX_INITIAL_DATA_ELLIPTIC_CHECK_SOURCES
   tests/cartesian_id_convergence.cpp
   src/framework/config.cpp
   src/framework/config_runtime.cpp
   src/framework/fields.cpp
   src/eos/eos_config.cpp
   src/eos/presets.cpp
   src/eos/tabulated.cpp
   src/initial_data/initial_data_centering.cpp
   src/initial_data/initial_data_ctt.cpp
   src/initial_data/initial_data_elliptic.cpp
   src/initial_data/initial_data_elliptic_spherical.cpp
   tests/initial_data_elliptic_check.cpp
   src/numerics/nodal_interpolater.cpp
   src/initial_data/tov_initial_data.cpp
)

set(PYFLUID_AMREX_TOV_SEQUENCE_SOURCES
   src/framework/config.cpp
   src/framework/config_runtime.cpp
   src/framework/fields.cpp
   src/eos/eos_config.cpp
   src/eos/presets.cpp
   src/eos/tabulated.cpp
   src/initial_data/tov_initial_data.cpp
   tools/cpp/tov_sequence.cpp
)

find_program(PYFLUID_UV_EXECUTABLE uv)
if(PYFLUID_UV_EXECUTABLE)
   set(PYFLUID_AMREX_GENERATED_FLUX_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/ideal_mhd_flux_cartesian.hpp
   )
   set(PYFLUID_AMREX_GENERATED_EOS_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/eos.hpp
   )
   set(PYFLUID_AMREX_GENERATED_OPERATOR_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/scalar_ops.hpp
   )
   set(PYFLUID_AMREX_GENERATED_DIFFERENTIAL_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/differential_ops.hpp
   )
   set(PYFLUID_AMREX_GENERATED_DIAGNOSTIC_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/diagnostics.hpp
   )
   set(PYFLUID_AMREX_GENERATED_RIEMANN_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/riemann_ops.hpp
   )
   set(PYFLUID_AMREX_GENERATED_CHARACTERISTIC_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/characteristic_transforms.hpp
   )
   set(PYFLUID_AMREX_GENERATED_VECTOR_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/vector_ops.hpp
   )
   set(PYFLUID_AMREX_GENERATED_METRIC_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/coordinate_metrics.hpp
   )
   set(PYFLUID_AMREX_GENERATED_PATCH_MAP_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/patch_maps.hpp
   )
   set(PYFLUID_AMREX_GENERATED_RELATIVISTIC_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/relativistic_fluid.hpp
   )
   set(PYFLUID_AMREX_GENERATED_RELATIVISTIC_HEADERS
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/relativistic/core.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/relativistic/flux.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/relativistic/sources.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/relativistic/c2p_rhd.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/relativistic/c2p_rmhd.hpp
   )
   set(PYFLUID_AMREX_GENERATED_Z4C_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/z4c.hpp
   )
   set(PYFLUID_AMREX_GENERATED_Z4C_HEADERS
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/z4c/geometry.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/z4c/constraints.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/z4c/gauge.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/z4c/evolution.hpp
   )
   set(PYFLUID_AMREX_GENERATED_INITIAL_DATA_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/initial_data.hpp
   )
   set(PYFLUID_AMREX_GENERATED_FIELD_IDS_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/generated/field_ids.hpp
   )
   set(PYFLUID_AMREX_GENERATED_FIELD_CATALOG_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/generated/field_catalog.hpp
   )
   set(PYFLUID_AMREX_GENERATED_DIAGNOSTIC_CATALOG_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/generated/diagnostic_catalog.hpp
   )
   set(PYFLUID_AMREX_GENERATED_METHOD_CATALOG_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/generated/method_catalog.hpp
   )
   set(PYFLUID_AMREX_GENERATED_AMR_TAG_CATALOG_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/generated/amr_tag_catalog.hpp
   )
   set(PYFLUID_AMREX_GENERATED_INITIAL_DATA_METHOD_CATALOG_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/generated/initial_data_method_catalog.hpp
   )
   set(PYFLUID_AMREX_GENERATED_PHYSICAL_ID_CATALOG_HEADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/generated/physical_id_catalog.hpp
   )
   set(PYFLUID_AMREX_GENERATED_EQUATION_MARKDOWN
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/equations.md
   )
   set(PYFLUID_AMREX_GENERATED_FIELD_CATALOG_JSON
      ${CMAKE_CURRENT_SOURCE_DIR}/generated/field_catalog.json
   )
   set(PYFLUID_AMREX_GENERATED_PHYSICAL_ID_CATALOG_JSON
      ${CMAKE_CURRENT_SOURCE_DIR}/generated/physical_id_catalog.json
   )
   set(PYFLUID_AMREX_GENERATED_EQUATION_LATEX
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/equations.tex
   )
   set(PYFLUID_AMREX_CODEGEN_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/codegen/generate_kernels.py)
   # Generated files belong to the source tree, not a build configuration.
   # Check their shared provenance cheaply on every build before importing SymPy.
   function(pyfluid_amrex_add_codegen_check family)
      cmake_parse_arguments(CODEGEN "" "" "OUTPUTS;DEPENDS;ARGUMENTS" ${ARGN})
      set(freshness_file "${CMAKE_CURRENT_SOURCE_DIR}/.cache/codegen/${family}.json")
      set(dependency_arguments)
      foreach(dependency IN ITEMS
            ${PYFLUID_AMREX_CODEGEN_SCRIPT}
            ${CMAKE_CURRENT_SOURCE_DIR}/codegen/generation_cache.py
            ${CODEGEN_DEPENDS})
         list(APPEND dependency_arguments --dependency ${dependency})
      endforeach()
      set(output_arguments)
      foreach(output IN LISTS CODEGEN_OUTPUTS)
         list(APPEND output_arguments --expected-output ${output})
      endforeach()
      add_custom_target(pyfluid_amrex_codegen_${family}
         BYPRODUCTS ${CODEGEN_OUTPUTS}
         COMMAND ${CMAKE_COMMAND} -E env
            UV_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/uv-cache
            ${PYFLUID_UV_EXECUTABLE} run --locked python ${PYFLUID_AMREX_CODEGEN_SCRIPT}
            ${CODEGEN_ARGUMENTS}
            --freshness-file ${freshness_file}
            ${dependency_arguments} ${output_arguments}
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
         COMMENT "Checking shared PyFluid AMReX ${family} generation"
         VERBATIM
      )
   endfunction()
   pyfluid_amrex_add_codegen_check(catalog
      OUTPUTS
         ${PYFLUID_AMREX_GENERATED_FIELD_IDS_HEADER}
         ${PYFLUID_AMREX_GENERATED_FIELD_CATALOG_HEADER}
         ${PYFLUID_AMREX_GENERATED_DIAGNOSTIC_CATALOG_HEADER}
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/generated/diagnostic_types.hpp
         ${PYFLUID_AMREX_GENERATED_METHOD_CATALOG_HEADER}
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/generated/spatial_order_catalog.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/generated/time_integrator_catalog.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/generated/time_integrator_ids.hpp
         ${PYFLUID_AMREX_GENERATED_AMR_TAG_CATALOG_HEADER}
         ${PYFLUID_AMREX_GENERATED_INITIAL_DATA_METHOD_CATALOG_HEADER}
         ${PYFLUID_AMREX_GENERATED_PHYSICAL_ID_CATALOG_HEADER}
         ${PYFLUID_AMREX_GENERATED_FIELD_CATALOG_JSON}
         ${PYFLUID_AMREX_GENERATED_PHYSICAL_ID_CATALOG_JSON}
         ${PYFLUID_AMREX_GENERATED_EQUATION_MARKDOWN}
         ${PYFLUID_AMREX_GENERATED_EQUATION_LATEX}
      ARGUMENTS
         --family catalog
         --field-ids-header ${PYFLUID_AMREX_GENERATED_FIELD_IDS_HEADER}
         --field-catalog-header ${PYFLUID_AMREX_GENERATED_FIELD_CATALOG_HEADER}
         --diagnostic-catalog-header ${PYFLUID_AMREX_GENERATED_DIAGNOSTIC_CATALOG_HEADER}
         --method-catalog-header ${PYFLUID_AMREX_GENERATED_METHOD_CATALOG_HEADER}
         --amr-tag-catalog-header ${PYFLUID_AMREX_GENERATED_AMR_TAG_CATALOG_HEADER}
         --initial-data-method-catalog-header ${PYFLUID_AMREX_GENERATED_INITIAL_DATA_METHOD_CATALOG_HEADER}
         --physical-id-catalog-header ${PYFLUID_AMREX_GENERATED_PHYSICAL_ID_CATALOG_HEADER}
         --field-catalog-json ${PYFLUID_AMREX_GENERATED_FIELD_CATALOG_JSON}
         --physical-id-catalog-json ${PYFLUID_AMREX_GENERATED_PHYSICAL_ID_CATALOG_JSON}
         --equation-markdown ${PYFLUID_AMREX_GENERATED_EQUATION_MARKDOWN}
         --equation-latex ${PYFLUID_AMREX_GENERATED_EQUATION_LATEX}
      DEPENDS
         ${PYFLUID_AMREX_CODEGEN_SCRIPT}
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/__init__.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/diagnostic_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/equation_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/emit_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/emit_spatial_order.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/emit_time_integrators.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/time_integrator_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/schema/time_integrators.toml
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/field_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/method_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/numerical_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/physical_id_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/schema/diagnostics.toml
         ${CMAKE_CURRENT_SOURCE_DIR}/schema/fields.toml
         ${CMAKE_CURRENT_SOURCE_DIR}/schema/methods.toml
         ${CMAKE_CURRENT_SOURCE_DIR}/schema/numerics.toml
         ${CMAKE_CURRENT_SOURCE_DIR}/schema/physical_id.toml
   )
   pyfluid_amrex_add_codegen_check(physics
      OUTPUTS
         ${PYFLUID_AMREX_GENERATED_FLUX_HEADER}
         ${PYFLUID_AMREX_GENERATED_EOS_HEADER}
         ${PYFLUID_AMREX_GENERATED_OPERATOR_HEADER}
         ${PYFLUID_AMREX_GENERATED_DIFFERENTIAL_HEADER}
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/boundary_differential_ops.hpp
         ${PYFLUID_AMREX_GENERATED_RIEMANN_HEADER}
         ${PYFLUID_AMREX_GENERATED_CHARACTERISTIC_HEADER}
         ${PYFLUID_AMREX_GENERATED_METRIC_HEADER}
      ARGUMENTS
         --family eos
         --family flux
         --family operator
         --family differential
         --family riemann
         --family characteristic
         --family metric
         --flux-header ${PYFLUID_AMREX_GENERATED_FLUX_HEADER}
         --eos-header ${PYFLUID_AMREX_GENERATED_EOS_HEADER}
         --operator-header ${PYFLUID_AMREX_GENERATED_OPERATOR_HEADER}
         --differential-header ${PYFLUID_AMREX_GENERATED_DIFFERENTIAL_HEADER}
         --riemann-header ${PYFLUID_AMREX_GENERATED_RIEMANN_HEADER}
         --characteristic-header ${PYFLUID_AMREX_GENERATED_CHARACTERISTIC_HEADER}
         --metric-header ${PYFLUID_AMREX_GENERATED_METRIC_HEADER}
      DEPENDS
         ${PYFLUID_AMREX_CODEGEN_SCRIPT}
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/__init__.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/characteristic_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/differential_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/numerical_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/emit_spatial_order.py
         ${CMAKE_CURRENT_SOURCE_DIR}/schema/numerics.toml
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/discretization.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/emit_cpp.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/emit_boundary_differential.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/eos_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/flux_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/field_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/schema/fields.toml
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/kernel_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/metric_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/operator_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/physical_system.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/riemann_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/tensor_algebra.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/coordinates.py
   )
   # These expensive symbolic families have separate ownership. A diagnostic
   # formula edit must not regenerate evolution/C2P kernels, and vice versa.
   function(pyfluid_amrex_add_kernel_family family)
      cmake_parse_arguments(KERNEL "" "HEADER" "BYPRODUCTS;DEPENDS" ${ARGN})
      pyfluid_amrex_add_codegen_check(${family}
         OUTPUTS ${KERNEL_HEADER} ${KERNEL_BYPRODUCTS}
         ARGUMENTS --family ${family} --${family}-header ${KERNEL_HEADER}
         DEPENDS
            ${PYFLUID_AMREX_CODEGEN_SCRIPT}
            ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/__init__.py
            ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/emit_cpp.py
            ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/kernel_equations.py
            ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/tensor_algebra.py
            ${KERNEL_DEPENDS}
      )
      set(PYFLUID_AMREX_FAMILY_CODEGEN_TARGETS
         ${PYFLUID_AMREX_FAMILY_CODEGEN_TARGETS} pyfluid_amrex_codegen_${family} PARENT_SCOPE)
   endfunction()
   pyfluid_amrex_add_kernel_family(vector
      HEADER ${PYFLUID_AMREX_GENERATED_VECTOR_HEADER}
      BYPRODUCTS
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/vector/calculus.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/vector/reference_metric.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/vector/viscous.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/vector/emf.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/vector/magnetic.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/vector/ct.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/vector/finite_volume.hpp
      DEPENDS
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/vector_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/physical_system.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/eos_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/discretization.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/coordinates.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/reference_metric.py
   )
   pyfluid_amrex_add_kernel_family(diagnostic
      HEADER ${PYFLUID_AMREX_GENERATED_DIAGNOSTIC_HEADER}
      DEPENDS
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/diagnostic_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/z4c_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/metric_differential.py
   )
   pyfluid_amrex_add_kernel_family(z4c
      HEADER ${PYFLUID_AMREX_GENERATED_Z4C_HEADER}
      BYPRODUCTS ${PYFLUID_AMREX_GENERATED_Z4C_HEADERS}
      DEPENDS
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/z4c_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/metric_differential.py
   )
   pyfluid_amrex_add_kernel_family(spacetime
      HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/spacetime_geometry.hpp
      BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/reference_metric_evolution.hpp
      DEPENDS
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/reference_metric_evolution.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/tensor_jets.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/spacetime_geometry.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/metric_differential.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/coordinates.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/reference_metric.py
   )
   pyfluid_amrex_add_kernel_family(fccz4
      HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/fccz4.hpp
      BYPRODUCTS
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/fccz4/geometry.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/fccz4/rhs.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/fccz4/gauge.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/pyfluid_amrex/kernels_generated/fccz4/constraints.hpp
      DEPENDS
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/fccz4_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/fccz4_kernels.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/reference_metric_evolution.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/spacetime_geometry.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/tensor_jets.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/metric_differential.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/coordinates.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/reference_metric.py
   )
   pyfluid_amrex_add_kernel_family(relativistic
      HEADER ${PYFLUID_AMREX_GENERATED_RELATIVISTIC_HEADER}
      BYPRODUCTS ${PYFLUID_AMREX_GENERATED_RELATIVISTIC_HEADERS}
      DEPENDS
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/relativistic_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/eos_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/field_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/physical_system.py
         ${CMAKE_CURRENT_SOURCE_DIR}/schema/fields.toml
   )
   pyfluid_amrex_add_codegen_check(initial_data
      OUTPUTS ${PYFLUID_AMREX_GENERATED_INITIAL_DATA_HEADER}
      ARGUMENTS
         --family initial_data
         --initial-data-header ${PYFLUID_AMREX_GENERATED_INITIAL_DATA_HEADER}
      DEPENDS
         ${PYFLUID_AMREX_CODEGEN_SCRIPT}
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/emit_cpp.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/initial_data_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/kernel_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/tensor_algebra.py
   )
   pyfluid_amrex_add_codegen_check(patch_map
      OUTPUTS ${PYFLUID_AMREX_GENERATED_PATCH_MAP_HEADER}
      ARGUMENTS
         --family patch_map
         --patch-map-header ${PYFLUID_AMREX_GENERATED_PATCH_MAP_HEADER}
      DEPENDS
         ${PYFLUID_AMREX_CODEGEN_SCRIPT}
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/emit_cpp.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/patch_map_equations.py
   )
   add_custom_target(
      pyfluid_amrex_generate_catalog
      DEPENDS pyfluid_amrex_codegen_catalog
   )
   add_custom_target(
      pyfluid_amrex_generate_kernels
      DEPENDS
         pyfluid_amrex_codegen_physics
         pyfluid_amrex_codegen_initial_data
         pyfluid_amrex_codegen_patch_map
         ${PYFLUID_AMREX_FAMILY_CODEGEN_TARGETS}
   )
   add_dependencies(pyfluid_amrex_generate_kernels pyfluid_amrex_generate_catalog)
else()
   message(WARNING "uv not found; pyfluid_amrex_generate_kernels target will not be available")
endif()

set(PYFLUID_AMREX_BUILD_FEATURES
   PYFLUID_AMREX_ENABLE_HYDRO
   PYFLUID_AMREX_ENABLE_MHD
   PYFLUID_AMREX_ENABLE_CT
   PYFLUID_AMREX_ENABLE_RECON_PWC
   PYFLUID_AMREX_ENABLE_RECON_PWL
   PYFLUID_AMREX_ENABLE_RECON_WENOZ
   PYFLUID_AMREX_ENABLE_RECON_WENOZ_CHAR
   PYFLUID_AMREX_ENABLE_RECON_MP5
   PYFLUID_AMREX_ENABLE_RECON_MP5_CHAR
   PYFLUID_AMREX_ENABLE_RECON_TENO5
   PYFLUID_AMREX_ENABLE_RECON_TENO5_CHAR
   PYFLUID_AMREX_ENABLE_RECON_PARABOLIC
   PYFLUID_AMREX_ENABLE_RECON_PARABOLIC_CHAR
   PYFLUID_AMREX_ENABLE_RECON_PPM4
   PYFLUID_AMREX_ENABLE_RECON_PPM4_CHAR
   PYFLUID_AMREX_ENABLE_RECON_PPMX
   PYFLUID_AMREX_ENABLE_RECON_PPMX_CHAR
   PYFLUID_AMREX_ENABLE_RECON_PPM3
   PYFLUID_AMREX_ENABLE_RECON_PPM3_CHAR
   PYFLUID_AMREX_ENABLE_RIEMANN_LLF
   PYFLUID_AMREX_ENABLE_RIEMANN_HLL
   PYFLUID_AMREX_ENABLE_RIEMANN_HLLE
   PYFLUID_AMREX_ENABLE_RIEMANN_HLLC
   PYFLUID_AMREX_ENABLE_RIEMANN_HLLD
   PYFLUID_AMREX_ENABLE_TIME_EULER
   PYFLUID_AMREX_ENABLE_TIME_CORRECTOR
   PYFLUID_AMREX_ENABLE_TIME_MIDPOINT
   PYFLUID_AMREX_ENABLE_TIME_SSPRK3
   PYFLUID_AMREX_ENABLE_TIME_RK4
   PYFLUID_AMREX_ENABLE_DIFFUSION_NONE
   PYFLUID_AMREX_ENABLE_DIFFUSION_EXPLICIT
   PYFLUID_AMREX_ENABLE_DIFFUSION_IMPLICIT
   PYFLUID_AMREX_ENABLE_SUBCYCLING_OFF
   PYFLUID_AMREX_ENABLE_SUBCYCLING_ON
)

# Derive executable capabilities from the same switches that configure owners.
# This header is local to this build directory, never shared across profiles.
list(LENGTH PYFLUID_AMREX_BUILD_FEATURES PYFLUID_AMREX_BUILD_FEATURE_COUNT)
set(PYFLUID_AMREX_BUILD_FEATURE_ROWS "")
foreach(feature IN LISTS PYFLUID_AMREX_BUILD_FEATURES)
   if(${feature})
      set(enabled true)
   else()
      set(enabled false)
   endif()
   string(APPEND PYFLUID_AMREX_BUILD_FEATURE_ROWS
      "   {\"${feature}\", ${enabled}},\n")
endforeach()
configure_file(cmake/build_configuration.hpp.in
   generated/pyfluid_amrex/build_configuration.hpp @ONLY)
set_property(SOURCE src/framework/config.cpp APPEND PROPERTY COMPILE_DEFINITIONS
   "PYFLUID_AMREX_BUILD_CONFIGURATION_HEADER=\"${CMAKE_CURRENT_BINARY_DIR}/generated/pyfluid_amrex/build_configuration.hpp\"")

function(pyfluid_amrex_configure_compile_target target)
   target_include_directories(${target} PRIVATE include src)
   target_compile_features(${target} PRIVATE cxx_std_20)
   foreach(feature ${PYFLUID_AMREX_BUILD_FEATURES})
      if(${feature})
         target_compile_definitions(${target} PRIVATE ${feature}=1)
      else()
         target_compile_definitions(${target} PRIVATE ${feature}=0)
      endif()
   endforeach()
   if(PYFLUID_AMREX_HAS_MCPU_NATIVE)
      target_compile_options(${target} PRIVATE "$<$<CONFIG:Release>:-mcpu=native>")
   endif()
endfunction()

function(pyfluid_amrex_configure_target target dim)
   pyfluid_amrex_configure_compile_target(${target})
   target_link_libraries(${target} PRIVATE AMReX::amrex_${dim}d)
   set_property(TARGET ${target} PROPERTY PYFLUID_AMREX_DIM ${dim})
   if(AMReX_CUDA_FOUND)
      setup_target_for_cuda_compilation(${target})
   endif()
endfunction()

# Keep the host-only horizon solver out of the fluid template products and out
# of the 1D/2D executables.  OpenMP is deliberately private to this library:
# AMReX remains MPI-only, so enabling the finder does not change the evolution
# launch model or activate unrelated AMREX_USE_OMP loops.
add_library(
   pyfluid_amrex_apparent_horizon
   STATIC
   src/relativity/apparent_horizon_finder.cpp
)
pyfluid_amrex_configure_compile_target(pyfluid_amrex_apparent_horizon)
target_link_libraries(
   pyfluid_amrex_apparent_horizon
   PUBLIC
   AMReX::amrex_3d
)
if(AMReX_CUDA_FOUND)
   # The finder still executes on the host, but its shared AMReX value types
   # carry CUDA qualifiers and require the CUDA compiler.
   setup_target_for_cuda_compilation(pyfluid_amrex_apparent_horizon)
endif()
if(PYFLUID_AMREX_HORIZON_OPENMP)
   find_package(OpenMP QUIET COMPONENTS CXX)
   if(OpenMP_CXX_FOUND)
      target_link_libraries(
         pyfluid_amrex_apparent_horizon
         PRIVATE
         OpenMP::OpenMP_CXX
      )
      if(AMReX_CUDA_FOUND)
         separate_arguments(horizon_openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
         foreach(flag IN LISTS horizon_openmp_flags)
            target_compile_options(pyfluid_amrex_apparent_horizon PRIVATE
               "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>"
            )
         endforeach()
      endif()
   elseif(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
      # Homebrew's keg-only libomp is not discovered by FindOpenMP when the
      # MPI wrapper selects AppleClang.  Keep this fallback target-local.
      find_path(
         PYFLUID_AMREX_LIBOMP_INCLUDE_DIR
         omp.h
         HINTS
         /opt/homebrew/opt/libomp/include
         /usr/local/opt/libomp/include
      )
      find_library(
         PYFLUID_AMREX_LIBOMP_LIBRARY
         NAMES omp
         HINTS
         /opt/homebrew/opt/libomp/lib
         /usr/local/opt/libomp/lib
      )
      if(PYFLUID_AMREX_LIBOMP_INCLUDE_DIR AND PYFLUID_AMREX_LIBOMP_LIBRARY)
         target_compile_options(
            pyfluid_amrex_apparent_horizon
            PRIVATE
            -Xpreprocessor
            -fopenmp
         )
         target_include_directories(
            pyfluid_amrex_apparent_horizon
            PRIVATE
            ${PYFLUID_AMREX_LIBOMP_INCLUDE_DIR}
         )
         target_link_libraries(
            pyfluid_amrex_apparent_horizon
            PRIVATE
            ${PYFLUID_AMREX_LIBOMP_LIBRARY}
         )
         set(OpenMP_CXX_FOUND TRUE)
      endif()
   endif()
   if(OpenMP_CXX_FOUND)
      target_compile_definitions(
         pyfluid_amrex_apparent_horizon
         PRIVATE
         PYFLUID_AMREX_HORIZON_OPENMP=1
      )
   else()
      message(WARNING
         "OpenMP was not found; the apparent-horizon finder will be serial."
      )
   endif()
endif()

# This host bookkeeping owner has no AMReX, generated-kernel or dimension
# dependencies. Compile it once; implementation edits do not rebuild drivers.
add_library(pyfluid_amrex_regrid STATIC src/amr/regrid_lifecycle.cpp)

foreach(dim 1 2 3)
   if(TARGET AMReX::amrex_${dim}d)
      add_executable(pyfluid_amrex_${dim}d ${PYFLUID_AMREX_SOURCES})
      target_link_libraries(pyfluid_amrex_${dim}d PRIVATE pyfluid_amrex_regrid)
      list(APPEND PYFLUID_AMREX_PRODUCTION_TARGETS pyfluid_amrex_${dim}d)
      if(dim EQUAL 3)
         # The horizon library already supplies the 3D AMReX dependency.
         pyfluid_amrex_configure_compile_target(pyfluid_amrex_${dim}d)
         set_property(TARGET pyfluid_amrex_${dim}d PROPERTY PYFLUID_AMREX_DIM ${dim})
         if(AMReX_CUDA_FOUND)
            setup_target_for_cuda_compilation(pyfluid_amrex_${dim}d)
         endif()
         target_link_libraries(
            pyfluid_amrex_${dim}d
            PRIVATE
            pyfluid_amrex_apparent_horizon
         )
      else()
         pyfluid_amrex_configure_target(pyfluid_amrex_${dim}d ${dim})
      endif()
      if(TARGET pyfluid_amrex_generate_kernels)
         add_dependencies(pyfluid_amrex_${dim}d pyfluid_amrex_generate_kernels)
      endif()

      add_executable(pyfluid_amrex_compare_checkpoint_${dim}d EXCLUDE_FROM_ALL ${PYFLUID_AMREX_CHECKPOINT_COMPARE_SOURCES})
      pyfluid_amrex_configure_target(pyfluid_amrex_compare_checkpoint_${dim}d ${dim})
      if(TARGET pyfluid_amrex_generate_kernels)
         add_dependencies(pyfluid_amrex_compare_checkpoint_${dim}d pyfluid_amrex_generate_kernels)
      endif()

      add_executable(pyfluid_amrex_codegen_check_${dim}d EXCLUDE_FROM_ALL ${PYFLUID_AMREX_CODEGEN_CHECK_SOURCES})
      pyfluid_amrex_configure_target(pyfluid_amrex_codegen_check_${dim}d ${dim})
      if(TARGET pyfluid_amrex_generate_kernels)
         add_dependencies(pyfluid_amrex_codegen_check_${dim}d pyfluid_amrex_generate_kernels)
      endif()

      add_executable(
         pyfluid_amrex_initial_data_elliptic_check_${dim}d
         EXCLUDE_FROM_ALL
         ${PYFLUID_AMREX_INITIAL_DATA_ELLIPTIC_CHECK_SOURCES}
      )
      pyfluid_amrex_configure_target(pyfluid_amrex_initial_data_elliptic_check_${dim}d ${dim})
      if(TARGET pyfluid_amrex_generate_kernels)
         add_dependencies(
            pyfluid_amrex_initial_data_elliptic_check_${dim}d
            pyfluid_amrex_generate_kernels
         )
      endif()

      if(dim EQUAL 1)
         add_executable(
            pyfluid_amrex_state_update_check
            EXCLUDE_FROM_ALL
            ${PYFLUID_AMREX_STATE_UPDATE_CHECK_SOURCES}
         )
         pyfluid_amrex_configure_target(pyfluid_amrex_state_update_check ${dim})
         add_executable(
            pyfluid_amrex_reconstruction_check
            EXCLUDE_FROM_ALL
            ${PYFLUID_AMREX_RECONSTRUCTION_CHECK_SOURCES}
         )
         pyfluid_amrex_configure_target(pyfluid_amrex_reconstruction_check ${dim})
         if(TARGET pyfluid_amrex_generate_kernels)
            add_dependencies(
               pyfluid_amrex_reconstruction_check
               pyfluid_amrex_generate_kernels
            )
         endif()

         add_executable(
            pyfluid_amrex_rmhd_riemann_check
            EXCLUDE_FROM_ALL
            ${PYFLUID_AMREX_RMHD_RIEMANN_CHECK_SOURCES}
         )
         pyfluid_amrex_configure_target(pyfluid_amrex_rmhd_riemann_check ${dim})
         if(TARGET pyfluid_amrex_generate_kernels)
            add_dependencies(
               pyfluid_amrex_rmhd_riemann_check
               pyfluid_amrex_generate_kernels
            )
         endif()

         add_executable(
            pyfluid_amrex_state_interpolation_check
            EXCLUDE_FROM_ALL
            ${PYFLUID_AMREX_STATE_INTERPOLATION_CHECK_SOURCES}
         )
         pyfluid_amrex_configure_target(
            pyfluid_amrex_state_interpolation_check
            ${dim}
         )
         if(TARGET pyfluid_amrex_generate_kernels)
            add_dependencies(
               pyfluid_amrex_state_interpolation_check
               pyfluid_amrex_generate_kernels
            )
         endif()

         add_executable(pyfluid_amrex_c2p_benchmark EXCLUDE_FROM_ALL ${PYFLUID_AMREX_C2P_BENCHMARK_SOURCES})
         pyfluid_amrex_configure_target(pyfluid_amrex_c2p_benchmark ${dim})
         if(TARGET pyfluid_amrex_generate_kernels)
            add_dependencies(pyfluid_amrex_c2p_benchmark pyfluid_amrex_generate_kernels)
         endif()

         add_executable(
            pyfluid_amrex_tabulated_eos_check
            EXCLUDE_FROM_ALL
            ${PYFLUID_AMREX_TABULATED_EOS_CHECK_SOURCES}
         )
         pyfluid_amrex_configure_target(pyfluid_amrex_tabulated_eos_check ${dim})
         if(TARGET pyfluid_amrex_generate_kernels)
            add_dependencies(
               pyfluid_amrex_tabulated_eos_check
               pyfluid_amrex_generate_kernels
            )
         endif()

         add_executable(pyfluid_amrex_tov_sequence EXCLUDE_FROM_ALL ${PYFLUID_AMREX_TOV_SEQUENCE_SOURCES})
         pyfluid_amrex_configure_target(pyfluid_amrex_tov_sequence ${dim})
         if(TARGET pyfluid_amrex_generate_kernels)
            add_dependencies(pyfluid_amrex_tov_sequence pyfluid_amrex_generate_kernels)
         endif()
      endif()

      if(dim EQUAL 3)
         add_executable(
            pyfluid_amrex_excision_stencil_check
            EXCLUDE_FROM_ALL
            tests/excision_stencil_check.cpp
            src/numerics/nodal_interpolater.cpp
         )
         pyfluid_amrex_configure_target(pyfluid_amrex_excision_stencil_check ${dim})
         if(TARGET pyfluid_amrex_generate_kernels)
            add_dependencies(pyfluid_amrex_excision_stencil_check pyfluid_amrex_generate_kernels)
         endif()

         add_executable(
            pyfluid_amrex_apparent_horizon_check
            EXCLUDE_FROM_ALL
            ${PYFLUID_AMREX_APPARENT_HORIZON_CHECK_SOURCES}
         )
         pyfluid_amrex_configure_compile_target(
            pyfluid_amrex_apparent_horizon_check
         )
         set_property(TARGET pyfluid_amrex_apparent_horizon_check PROPERTY PYFLUID_AMREX_DIM ${dim})
         target_link_libraries(
            pyfluid_amrex_apparent_horizon_check
            PRIVATE
            pyfluid_amrex_apparent_horizon
         )

         add_executable(
            pyfluid_amrex_physical_id_interpolation_check
            EXCLUDE_FROM_ALL
            ${PYFLUID_AMREX_PHYSICAL_ID_INTERPOLATION_CHECK_SOURCES}
         )
         pyfluid_amrex_configure_target(
            pyfluid_amrex_physical_id_interpolation_check
            ${dim}
         )
         if(TARGET pyfluid_amrex_generate_kernels)
            add_dependencies(
               pyfluid_amrex_physical_id_interpolation_check
               pyfluid_amrex_generate_kernels
            )
         endif()

         add_executable(
            pyfluid_amrex_surface_check
            EXCLUDE_FROM_ALL
            ${PYFLUID_AMREX_SURFACE_CHECK_SOURCES}
         )
         pyfluid_amrex_configure_target(pyfluid_amrex_surface_check ${dim})
      endif()

      if(dim GREATER 1)
         add_executable(pyfluid_amrex_multipatch_check_${dim}d EXCLUDE_FROM_ALL ${PYFLUID_AMREX_MULTIPATCH_CHECK_SOURCES})
         pyfluid_amrex_configure_target(pyfluid_amrex_multipatch_check_${dim}d ${dim})
         if(TARGET pyfluid_amrex_generate_kernels)
            add_dependencies(pyfluid_amrex_multipatch_check_${dim}d pyfluid_amrex_generate_kernels)
         endif()

         add_executable(pyfluid_amrex_origin_boundary_check_${dim}d EXCLUDE_FROM_ALL
            ${PYFLUID_AMREX_ORIGIN_BOUNDARY_CHECK_SOURCES})
         pyfluid_amrex_configure_target(pyfluid_amrex_origin_boundary_check_${dim}d ${dim})
         if(TARGET pyfluid_amrex_generate_kernels)
            add_dependencies(pyfluid_amrex_origin_boundary_check_${dim}d pyfluid_amrex_generate_kernels)
         endif()

         add_executable(
            pyfluid_amrex_boundary_tiling_check_${dim}d
            EXCLUDE_FROM_ALL
            ${PYFLUID_AMREX_BOUNDARY_TILING_CHECK_SOURCES}
         )
         pyfluid_amrex_configure_target(pyfluid_amrex_boundary_tiling_check_${dim}d ${dim})
         if(TARGET pyfluid_amrex_generate_kernels)
            add_dependencies(
               pyfluid_amrex_boundary_tiling_check_${dim}d
               pyfluid_amrex_generate_kernels
            )
         endif()
      endif()
   endif()
endforeach()

include(cmake/PyFluidSharedObjects.cmake)
pyfluid_amrex_share_implementation_objects()

if(PYFLUID_AMREX_PRODUCTION_TARGETS)
   add_custom_target(pyfluid_amrex_production)
   add_dependencies(pyfluid_amrex_production ${PYFLUID_AMREX_PRODUCTION_TARGETS})
endif()
