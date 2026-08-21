cmake_minimum_required(VERSION 3.20)

project(pyfluid_amrex LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
option(PYFLUID_AMREX_ENABLE_IMPLICIT_GRAVITY "Compile implicit-gravity paths that use AMReX linear solvers." ON)

find_package(AMReX REQUIRED)
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
   src/amr/amr_solver_average.cpp
   src/amr/amr_solver_auxiliary.cpp
   src/amr/amr_solver_conservation.cpp
   src/amr/amr_solver_debug.cpp
   src/amr/amr_solver_diagnostics.cpp
   src/amr/amr_solver_fillpatch.cpp
   src/amr/amr_solver_flux_registers.cpp
   src/amr/amr_solver_gravity_sync.cpp
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
)

set(PYFLUID_AMREX_NUMERICS_SOURCES
   src/numerics/ct_edge_flux_register.cpp
   src/numerics/flux_factory.cpp
   src/numerics/implicit_resistivity.cpp
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
   src/initial_data/initial_data_materialization.cpp
   src/initial_data/initial_data_provider.cpp
   src/initial_data/physical_id.cpp
   src/initial_data/physical_id_interpolation.cpp
   src/initial_data/initial_data_rhd.cpp
   src/initial_data/initial_data_state.cpp
   src/initial_data/rotating_star_initial_data.cpp
   src/initial_data/tov_initial_data.cpp
)

set(PYFLUID_AMREX_RELATIVITY_SOURCES
   src/relativity/primitive_cache_factory.cpp
   src/relativity/relativistic_flux_limiter.cpp
   src/relativity/relativistic_source_factory.cpp
   src/relativity/signal_speed_factory.cpp
   src/relativity/spacetime_coupling_storage.cpp
   src/relativity/z4c_runtime.cpp
)

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

set(PYFLUID_AMREX_CHECKPOINT_COMPARE_SOURCES
   src/framework/checkpoint.cpp
   tools/cpp/compare_checkpoint.cpp
   src/framework/config.cpp
   src/framework/config_runtime.cpp
   src/eos/eos_config.cpp
   src/eos/presets.cpp
   src/framework/fields.cpp
)

set(PYFLUID_AMREX_CODEGEN_CHECK_SOURCES
   src/eos/presets.cpp
   tests/generated_flux_compile_check.cpp
)

set(PYFLUID_AMREX_C2P_BENCHMARK_SOURCES
   benchmarks/c2p_benchmark.cpp
)

set(PYFLUID_AMREX_RECONSTRUCTION_CHECK_SOURCES
   tests/reconstruction_check.cpp
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
)

set(PYFLUID_AMREX_BOUNDARY_TILING_CHECK_SOURCES
   src/framework/fields.cpp
   tests/boundary_tiling_check.cpp
)

set(PYFLUID_AMREX_SURFACE_CHECK_SOURCES
   src/framework/multipoles.cpp
   src/framework/surface_geometry.cpp
   src/numerics/hierarchy_interpolation.cpp
   tests/surface_interpolation_check.cpp
)

set(PYFLUID_AMREX_INITIAL_DATA_ELLIPTIC_CHECK_SOURCES
   src/framework/config.cpp
   src/framework/config_runtime.cpp
   src/framework/fields.cpp
   src/eos/eos_config.cpp
   src/eos/presets.cpp
   src/initial_data/initial_data_centering.cpp
   src/initial_data/initial_data_ctt.cpp
   src/initial_data/initial_data_elliptic.cpp
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
   set(PYFLUID_AMREX_GENERATED_EQUATION_MARKDOWN
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/equations.md
   )
   set(PYFLUID_AMREX_GENERATED_FIELD_CATALOG_JSON
      ${CMAKE_CURRENT_SOURCE_DIR}/generated/field_catalog.json
   )
   set(PYFLUID_AMREX_GENERATED_EQUATION_LATEX
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/equations.tex
   )
   set(PYFLUID_AMREX_CODEGEN_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/codegen/generate_kernels.py)
   set(PYFLUID_AMREX_CODEGEN_STAMP ${CMAKE_CURRENT_BINARY_DIR}/pyfluid_amrex_codegen.stamp)
   set(PYFLUID_AMREX_INITIAL_DATA_CODEGEN_STAMP
      ${CMAKE_CURRENT_BINARY_DIR}/pyfluid_amrex_initial_data_codegen.stamp
   )
   set(PYFLUID_AMREX_PATCH_MAP_CODEGEN_STAMP
      ${CMAKE_CURRENT_BINARY_DIR}/pyfluid_amrex_patch_map_codegen.stamp
   )
   add_custom_command(
      OUTPUT ${PYFLUID_AMREX_CODEGEN_STAMP}
      BYPRODUCTS
         ${PYFLUID_AMREX_GENERATED_FLUX_HEADER}
         ${PYFLUID_AMREX_GENERATED_EOS_HEADER}
         ${PYFLUID_AMREX_GENERATED_OPERATOR_HEADER}
         ${PYFLUID_AMREX_GENERATED_DIFFERENTIAL_HEADER}
         ${PYFLUID_AMREX_GENERATED_DIAGNOSTIC_HEADER}
         ${PYFLUID_AMREX_GENERATED_RIEMANN_HEADER}
         ${PYFLUID_AMREX_GENERATED_CHARACTERISTIC_HEADER}
         ${PYFLUID_AMREX_GENERATED_VECTOR_HEADER}
         ${PYFLUID_AMREX_GENERATED_METRIC_HEADER}
         ${PYFLUID_AMREX_GENERATED_RELATIVISTIC_HEADER}
         ${PYFLUID_AMREX_GENERATED_RELATIVISTIC_HEADERS}
         ${PYFLUID_AMREX_GENERATED_Z4C_HEADER}
         ${PYFLUID_AMREX_GENERATED_Z4C_HEADERS}
         ${PYFLUID_AMREX_GENERATED_FIELD_IDS_HEADER}
         ${PYFLUID_AMREX_GENERATED_FIELD_CATALOG_HEADER}
         ${PYFLUID_AMREX_GENERATED_DIAGNOSTIC_CATALOG_HEADER}
         ${PYFLUID_AMREX_GENERATED_METHOD_CATALOG_HEADER}
         ${PYFLUID_AMREX_GENERATED_FIELD_CATALOG_JSON}
         ${PYFLUID_AMREX_GENERATED_EQUATION_MARKDOWN}
         ${PYFLUID_AMREX_GENERATED_EQUATION_LATEX}
      COMMAND ${CMAKE_COMMAND} -E env
         UV_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/uv-cache
         ${PYFLUID_UV_EXECUTABLE} run python
         ${PYFLUID_AMREX_CODEGEN_SCRIPT}
         --family catalog
         --family eos
         --family flux
         --family operator
         --family differential
         --family diagnostic
         --family riemann
         --family characteristic
         --family vector
         --family metric
         --family relativistic
         --family z4c
         --flux-header ${PYFLUID_AMREX_GENERATED_FLUX_HEADER}
         --eos-header ${PYFLUID_AMREX_GENERATED_EOS_HEADER}
         --operator-header ${PYFLUID_AMREX_GENERATED_OPERATOR_HEADER}
         --differential-header ${PYFLUID_AMREX_GENERATED_DIFFERENTIAL_HEADER}
         --diagnostic-header ${PYFLUID_AMREX_GENERATED_DIAGNOSTIC_HEADER}
         --riemann-header ${PYFLUID_AMREX_GENERATED_RIEMANN_HEADER}
         --characteristic-header ${PYFLUID_AMREX_GENERATED_CHARACTERISTIC_HEADER}
         --vector-header ${PYFLUID_AMREX_GENERATED_VECTOR_HEADER}
         --metric-header ${PYFLUID_AMREX_GENERATED_METRIC_HEADER}
         --relativistic-header ${PYFLUID_AMREX_GENERATED_RELATIVISTIC_HEADER}
         --z4c-header ${PYFLUID_AMREX_GENERATED_Z4C_HEADER}
         --field-ids-header ${PYFLUID_AMREX_GENERATED_FIELD_IDS_HEADER}
         --field-catalog-header ${PYFLUID_AMREX_GENERATED_FIELD_CATALOG_HEADER}
         --diagnostic-catalog-header ${PYFLUID_AMREX_GENERATED_DIAGNOSTIC_CATALOG_HEADER}
         --method-catalog-header ${PYFLUID_AMREX_GENERATED_METHOD_CATALOG_HEADER}
         --field-catalog-json ${PYFLUID_AMREX_GENERATED_FIELD_CATALOG_JSON}
         --equation-markdown ${PYFLUID_AMREX_GENERATED_EQUATION_MARKDOWN}
         --equation-latex ${PYFLUID_AMREX_GENERATED_EQUATION_LATEX}
      COMMAND ${CMAKE_COMMAND} -E touch ${PYFLUID_AMREX_CODEGEN_STAMP}
      DEPENDS
         ${PYFLUID_AMREX_CODEGEN_SCRIPT}
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/__init__.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/characteristic_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/differential_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/diagnostic_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/discretization.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/emit_cpp.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/eos_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/diagnostic_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/equation_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/emit_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/flux_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/field_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/schema/fields.toml
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/method_catalog.py
         ${CMAKE_CURRENT_SOURCE_DIR}/schema/methods.toml
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/kernel_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/metric_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/operator_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/physical_system.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/riemann_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/relativistic_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/tensor_algebra.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/vector_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/z4c_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/coordinates.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
      COMMENT "Generating PyFluid AMReX physics kernels"
      VERBATIM
   )
   add_custom_command(
      OUTPUT ${PYFLUID_AMREX_INITIAL_DATA_CODEGEN_STAMP}
      BYPRODUCTS ${PYFLUID_AMREX_GENERATED_INITIAL_DATA_HEADER}
      COMMAND ${CMAKE_COMMAND} -E env
         UV_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/uv-cache
         ${PYFLUID_UV_EXECUTABLE} run python
         ${PYFLUID_AMREX_CODEGEN_SCRIPT}
         --family initial_data
         --initial-data-header ${PYFLUID_AMREX_GENERATED_INITIAL_DATA_HEADER}
      COMMAND ${CMAKE_COMMAND} -E touch ${PYFLUID_AMREX_INITIAL_DATA_CODEGEN_STAMP}
      DEPENDS
         ${PYFLUID_AMREX_CODEGEN_SCRIPT}
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/emit_cpp.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/initial_data_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/kernel_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/tensor_algebra.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
      COMMENT "Generating PyFluid AMReX initial-data kernels"
      VERBATIM
   )
   add_custom_command(
      OUTPUT ${PYFLUID_AMREX_PATCH_MAP_CODEGEN_STAMP}
      BYPRODUCTS ${PYFLUID_AMREX_GENERATED_PATCH_MAP_HEADER}
      COMMAND ${CMAKE_COMMAND} -E env
         UV_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/uv-cache
         ${PYFLUID_UV_EXECUTABLE} run python
         ${PYFLUID_AMREX_CODEGEN_SCRIPT}
         --family patch_map
         --patch-map-header ${PYFLUID_AMREX_GENERATED_PATCH_MAP_HEADER}
      COMMAND ${CMAKE_COMMAND} -E touch ${PYFLUID_AMREX_PATCH_MAP_CODEGEN_STAMP}
      DEPENDS
         ${PYFLUID_AMREX_CODEGEN_SCRIPT}
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/emit_cpp.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/patch_map_equations.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
      COMMENT "Generating PyFluid AMReX patch-map kernels"
      VERBATIM
   )
   add_custom_target(
      pyfluid_amrex_generate_kernels
      DEPENDS
         ${PYFLUID_AMREX_CODEGEN_STAMP}
         ${PYFLUID_AMREX_INITIAL_DATA_CODEGEN_STAMP}
         ${PYFLUID_AMREX_PATCH_MAP_CODEGEN_STAMP}
   )
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

function(pyfluid_amrex_configure_target target dim)
   target_include_directories(${target} PRIVATE include src)
   target_link_libraries(${target} PRIVATE AMReX::amrex_${dim}d)
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

foreach(dim 1 2 3)
   if(TARGET AMReX::amrex_${dim}d)
      add_executable(pyfluid_amrex_${dim}d ${PYFLUID_AMREX_SOURCES})
      list(APPEND PYFLUID_AMREX_PRODUCTION_TARGETS pyfluid_amrex_${dim}d)
      pyfluid_amrex_configure_target(pyfluid_amrex_${dim}d ${dim})
      if(TARGET pyfluid_amrex_generate_kernels)
         add_dependencies(pyfluid_amrex_${dim}d pyfluid_amrex_generate_kernels)
      endif()

      add_executable(pyfluid_amrex_compare_checkpoint_${dim}d ${PYFLUID_AMREX_CHECKPOINT_COMPARE_SOURCES})
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

         add_executable(pyfluid_amrex_tov_sequence EXCLUDE_FROM_ALL ${PYFLUID_AMREX_TOV_SEQUENCE_SOURCES})
         pyfluid_amrex_configure_target(pyfluid_amrex_tov_sequence ${dim})
         if(TARGET pyfluid_amrex_generate_kernels)
            add_dependencies(pyfluid_amrex_tov_sequence pyfluid_amrex_generate_kernels)
         endif()
      endif()

      if(dim EQUAL 3)
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

if(PYFLUID_AMREX_PRODUCTION_TARGETS)
   add_custom_target(pyfluid_amrex_production)
   add_dependencies(pyfluid_amrex_production ${PYFLUID_AMREX_PRODUCTION_TARGETS})
endif()
