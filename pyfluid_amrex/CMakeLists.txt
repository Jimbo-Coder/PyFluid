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

set(PYFLUID_AMREX_SOURCES
   src/main.cpp
   src/config.cpp
   src/amr_solver_average.cpp
   src/amr_solver_auxiliary.cpp
   src/amr_solver_debug.cpp
   src/amr_solver_diagnostics.cpp
   src/amr_solver_hierarchy.cpp
   src/amr_solver_levels.cpp
   src/amr_solver_metrics.cpp
   src/amr_solver_multipatch.cpp
   src/amr_solver_output.cpp
   src/amr_solver_valid_cells.cpp
   src/checkpoint.cpp
   src/ct_edge_flux_register.cpp
   src/fields.cpp
   src/implicit_resistivity.cpp
   src/profiling.cpp
   src/solver.cpp
)

set(PYFLUID_AMREX_CHECKPOINT_COMPARE_SOURCES
   src/checkpoint.cpp
   src/compare_checkpoint.cpp
   src/config.cpp
   src/fields.cpp
)

set(PYFLUID_AMREX_CODEGEN_CHECK_SOURCES
   src/generated_flux_compile_check.cpp
)

set(PYFLUID_AMREX_MULTIPATCH_CHECK_SOURCES
   src/multipatch_check.cpp
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
   set(PYFLUID_AMREX_CODEGEN_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/codegen/generate_kernels.py)
   set(PYFLUID_AMREX_CODEGEN_STAMP ${CMAKE_CURRENT_BINARY_DIR}/pyfluid_amrex_codegen.stamp)
   set(PYFLUID_AMREX_PATCH_MAP_CODEGEN_STAMP
      ${CMAKE_CURRENT_BINARY_DIR}/pyfluid_amrex_patch_map_codegen.stamp
   )
   add_custom_command(
      OUTPUT ${PYFLUID_AMREX_CODEGEN_STAMP}
      BYPRODUCTS
         ${PYFLUID_AMREX_GENERATED_FLUX_HEADER}
         ${PYFLUID_AMREX_GENERATED_EOS_HEADER}
         ${PYFLUID_AMREX_GENERATED_OPERATOR_HEADER}
         ${PYFLUID_AMREX_GENERATED_RIEMANN_HEADER}
         ${PYFLUID_AMREX_GENERATED_CHARACTERISTIC_HEADER}
         ${PYFLUID_AMREX_GENERATED_VECTOR_HEADER}
         ${PYFLUID_AMREX_GENERATED_METRIC_HEADER}
      COMMAND ${CMAKE_COMMAND} -E env
         UV_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/uv-cache
         ${PYFLUID_UV_EXECUTABLE} run python
         ${PYFLUID_AMREX_CODEGEN_SCRIPT}
         --family eos
         --family flux
         --family operator
         --family riemann
         --family characteristic
         --family vector
         --family metric
         --flux-header ${PYFLUID_AMREX_GENERATED_FLUX_HEADER}
         --eos-header ${PYFLUID_AMREX_GENERATED_EOS_HEADER}
         --operator-header ${PYFLUID_AMREX_GENERATED_OPERATOR_HEADER}
         --riemann-header ${PYFLUID_AMREX_GENERATED_RIEMANN_HEADER}
         --characteristic-header ${PYFLUID_AMREX_GENERATED_CHARACTERISTIC_HEADER}
         --vector-header ${PYFLUID_AMREX_GENERATED_VECTOR_HEADER}
         --metric-header ${PYFLUID_AMREX_GENERATED_METRIC_HEADER}
      COMMAND ${CMAKE_COMMAND} -E touch ${PYFLUID_AMREX_CODEGEN_STAMP}
      DEPENDS
         ${PYFLUID_AMREX_CODEGEN_SCRIPT}
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/__init__.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/characteristic_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/discretization.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/emit_cpp.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/eos_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/flux_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/kernel_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/metric_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/operator_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/physical_system.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/riemann_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/tensor_algebra.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/vector_equations.py
         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/coordinates.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
      COMMENT "Generating PyFluid AMReX physics kernels"
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
      DEPENDS ${PYFLUID_AMREX_CODEGEN_STAMP} ${PYFLUID_AMREX_PATCH_MAP_CODEGEN_STAMP}
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
   target_include_directories(${target} PRIVATE include)
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

      if(dim GREATER 1)
         add_executable(pyfluid_amrex_multipatch_check_${dim}d EXCLUDE_FROM_ALL ${PYFLUID_AMREX_MULTIPATCH_CHECK_SOURCES})
         pyfluid_amrex_configure_target(pyfluid_amrex_multipatch_check_${dim}d ${dim})
         if(TARGET pyfluid_amrex_generate_kernels)
            add_dependencies(pyfluid_amrex_multipatch_check_${dim}d pyfluid_amrex_generate_kernels)
         endif()
      endif()
   endif()
endforeach()
