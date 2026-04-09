# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)

# Get processor count for parallel builds
include(ProcessorCount)
ProcessorCount(NPROC)
if(NPROC EQUAL 0)
    set(NPROC 4)
endif()

# GPU target architecture (passed from command line or default to gfx942)
if(NOT DEFINED GPU_TARGETS OR GPU_TARGETS STREQUAL "")
    set(GPU_TARGETS "gfx942" CACHE STRING "GPU architecture target")
endif()
# Extract first target if multiple are provided (we only support single target builds)
string(REPLACE ";" " " GPU_TARGETS_SPACE "${GPU_TARGETS}")
string(REPLACE " " ";" GPU_TARGETS_LIST "${GPU_TARGETS_SPACE}")
list(GET GPU_TARGETS_LIST 0 GPU_TARGET)
message(STATUS "Building for GPU target: ${GPU_TARGET}")

# NOTE: Per-kernel compilation is now automatic via declarative examples
# Each example generates only its declared kernels (from DECL_KERNEL_SET)

# Link to dispatcher library
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build)

# =============================================================================
# Kernel Output Directory
# =============================================================================

set(KERNEL_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels")
file(MAKE_DIRECTORY ${KERNEL_OUTPUT_DIR})

# =============================================================================
# Kernel Generation Targets (run during 'make', not 'cmake')
# =============================================================================

# Sentinel files to track generation
set(GEMM_SENTINEL "${KERNEL_OUTPUT_DIR}/.gemm_generated")

# Generate GEMM kernels (standard + preshuffle + multi_d) - runs with internal parallelism
# Note: 4-char layout "rcrr" means A=row, B=col, C=row, D=row (for multi-d)
add_custom_command(
    OUTPUT ${GEMM_SENTINEL}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_gemm_codegen.py 
            --datatype fp16 --layout rcrr --variants standard preshuffle multi_d
            --output ${KERNEL_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${GEMM_SENTINEL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
    COMMENT "Generating GEMM kernels (fp16, rcrr, standard + preshuffle + multi_d) with internal parallelism..."
    VERBATIM
)

add_custom_target(generate_gemm_kernels
    DEPENDS ${GEMM_SENTINEL}
    COMMENT "GEMM kernel generation target"
)

# Alias for generate_all_kernels (GEMM only now)
add_custom_target(generate_all_kernels 
    DEPENDS generate_gemm_kernels
)

# =============================================================================
# Per-Kernel Compilation (Maximum Parallelism)
# =============================================================================
# Enable with: cmake -DPER_KERNEL_COMPILATION=ON
#
# This creates ONE translation unit per kernel, enabling:
#   1. Maximum parallelism with make -j$(nproc)
#   2. Per-kernel build progress: "[1/128] Building kernel: gemm_fp16_128x128"
#   3. Incremental rebuilds (only changed kernels recompile)
#   4. Fine-grained build time analysis
#
# Build process:
#   1. Generate kernel headers (.hpp)
#   2. Generate wrapper files (.cpp) - one per kernel
#   3. Compile each wrapper in parallel
#   4. Link all objects into libdispatcher_kernels.so
#
# Example output:
#   [  1/128] Building kernel: gemm_fp16_rcr_128x128x32
#   [  2/128] Building kernel: gemm_fp16_rcr_256x256x64
#   ...
#   [128/128] Linking: libdispatcher_kernels.so
# =============================================================================

set(WRAPPER_DIR "${CMAKE_BINARY_DIR}/kernel_wrappers")
set(WRAPPER_SENTINEL "${WRAPPER_DIR}/.wrappers_generated")

# Target: Generate wrapper .cpp files (one per kernel)
add_custom_command(
    OUTPUT ${WRAPPER_SENTINEL}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/generate_kernel_wrappers.py
            --kernel-dir ${KERNEL_OUTPUT_DIR}
            --output-dir ${WRAPPER_DIR}
            --generate-makefile
            --generate-cmake
    COMMAND ${CMAKE_COMMAND} -E touch ${WRAPPER_SENTINEL}
    DEPENDS ${GEMM_SENTINEL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
    COMMENT "Generating per-kernel wrapper .cpp files..."
    VERBATIM
)

add_custom_target(generate_kernel_wrappers
    DEPENDS ${WRAPPER_SENTINEL}
    COMMENT "Kernel wrapper generation target"
)

# Target: Build kernels using generated Makefile (true per-kernel progress)
add_custom_target(build_kernels_parallel
    COMMAND ${CMAKE_COMMAND} -E echo "Building kernels with per-kernel progress..."
    COMMAND make -C ${WRAPPER_DIR} -j${NPROC} 2>&1 | grep -E "^\\[|Built|Linking|Error"
    DEPENDS generate_kernel_wrappers
    WORKING_DIRECTORY ${WRAPPER_DIR}
    COMMENT "Compiling kernels in parallel (one translation unit per kernel)..."
    VERBATIM
)

# Global kernel build (optional - prefer per-example builds for minimal compilation)
# This builds ALL kernels into a shared library - use for Python bindings or full library
# For C++ examples, use declarative approach which builds only needed kernels
add_custom_target(dispatcher_kernels
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/parallel_kernel_builder.py
            --kernel-dir ${KERNEL_OUTPUT_DIR}
            --output-dir ${CMAKE_BINARY_DIR}
            --include-dirs "${CMAKE_CURRENT_SOURCE_DIR}/../../include,${CMAKE_CURRENT_SOURCE_DIR}/../include"
            --jobs ${NPROC}
    DEPENDS generate_all_kernels
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../scripts
    COMMENT "Building ALL kernels in parallel (prefer per-example builds for minimal compilation)..."
    VERBATIM
)

# =============================================================================
# Force regeneration targets (useful when you want to regenerate)
# =============================================================================

add_custom_target(regenerate_gemm_kernels
    COMMAND ${CMAKE_COMMAND} -E remove -f ${GEMM_SENTINEL}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_gemm_codegen.py 
            --datatype fp16 --layout rcr --variants standard preshuffle multi_d
            --output ${KERNEL_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${GEMM_SENTINEL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
    COMMENT "Force regenerating GEMM kernels (standard + preshuffle + multi_d)..."
    VERBATIM
)

add_custom_target(regenerate_all_kernels
    DEPENDS regenerate_gemm_kernels
)

# Clean all per-example kernel directories
add_custom_target(clean_example_kernels
    COMMAND ${CMAKE_COMMAND} -E echo "Removing per-example kernel directories..."
    COMMAND find ${CMAKE_BINARY_DIR} -maxdepth 1 -type d -name "*_kernels" -exec rm -rf {} +
    COMMENT "Cleaning all per-example kernel directories..."
    VERBATIM
)

# =============================================================================
# Helper function to add a GPU example with force-included kernel
# =============================================================================

# Helper for GPU examples that use the dispatcher registry
# KERNEL_HEADER can be:
#   - A registration header (register_all_kernels.hpp) - included directly in source
#   - A specific kernel header - force-included via compiler flag
function(add_gpu_example NAME SOURCE KERNEL_HEADER)
    add_executable(${NAME} ${SOURCE})
    
    target_link_libraries(${NAME} PRIVATE ck_tile_dispatcher)
    
    target_include_directories(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include              # CK root include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include                 # Dispatcher include
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels # Generated kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels/dispatcher_wrappers # Wrapper headers
    )
    
    # Check if using registration header (no force-include needed)
    get_filename_component(HEADER_NAME ${KERNEL_HEADER} NAME)
    if(HEADER_NAME STREQUAL "register_all_kernels.hpp")
        # Registration header - examples include it directly
        target_compile_options(${NAME} PRIVATE
            -mllvm -enable-noalias-to-md-conversion=0
            -Wno-undefined-func-template
            -Wno-float-equal
            --offload-compress
        )
    else()
        # Specific kernel header - force-include it
        target_compile_options(${NAME} PRIVATE
            -include ${KERNEL_HEADER}
            -mllvm -enable-noalias-to-md-conversion=0
            -Wno-undefined-func-template
            -Wno-float-equal
            --offload-compress
        )
    endif()
    
    if(hip_FOUND)
        target_link_libraries(${NAME} PRIVATE hip::device hip::host)
    endif()
endfunction()

# Helper for standalone GPU examples (instantiate kernel directly, no pre-generated header)
function(add_standalone_gpu_example NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    
    target_link_libraries(${NAME} PRIVATE ck_tile_dispatcher)
    
    target_include_directories(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include              # CK root include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include                 # Dispatcher include
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels # Generated kernels (optional)
    )
    
    target_compile_options(${NAME} PRIVATE
        -mllvm -enable-noalias-to-md-conversion=0
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
    )
    
    if(hip_FOUND)
        target_link_libraries(${NAME} PRIVATE hip::device hip::host)
    endif()
endfunction()

# Helper for declarative examples (configuration demo, still needs HIP compiler for CK headers)
function(add_declarative_example NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    
    target_include_directories(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
    
    target_compile_options(${NAME} PRIVATE 
        -Wno-float-equal
        -Wno-unused-variable
        -Wno-undefined-func-template
        -mllvm -enable-noalias-to-md-conversion=0
    )
    
    target_link_libraries(${NAME} PRIVATE ck_tile_dispatcher)
    
    if(hip_FOUND)
        target_link_libraries(${NAME} PRIVATE hip::device hip::host)
    endif()
endfunction()

# =============================================================================
# GEMM Examples
# =============================================================================

# Per-example kernel directories are created from DECL_KERNEL_SET declarations
# Each example gets its own: build/<name>_kernels/
# This prevents clashes during parallel compilation of multiple examples.

# Helper function to add example with declarative kernel support
# Parses DECL_KERNEL_SET from source and generates ONLY the declared kernels
# This enables minimal builds: only kernels needed by this example are generated
#
# Key features:
# - Per-example kernel directories: build/<name>_kernels/ (no clashes)
# - Automatic header inclusion: No hardcoded #include needed in source
# - Minimal builds: Only declared kernels are generated
# - Auto-regeneration: Kernels regenerated if directory missing
# - Parallel compilation: Each kernel is a separate translation unit
function(add_declarative_gpu_example NAME SOURCE)
    set(EXAMPLE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}")
    get_filename_component(EXAMPLE_STEM ${SOURCE} NAME_WE)
    
    # Per-example kernel directories
    set(EXAMPLE_KERNEL_DIR "${CMAKE_BINARY_DIR}/${NAME}_kernels")
    set(EXAMPLE_HEADER "${EXAMPLE_KERNEL_DIR}/${EXAMPLE_STEM}_kernels.hpp")
    set(EXAMPLE_LIB "${EXAMPLE_KERNEL_DIR}/lib${NAME}_kernels.a")
    set(EXAMPLE_SENTINEL "${EXAMPLE_KERNEL_DIR}/.generated")
    
    # Generate AND compile kernels in parallel at make time
    # This avoids slow cmake and gets per-kernel progress
    add_custom_command(
        OUTPUT ${EXAMPLE_SENTINEL} ${EXAMPLE_LIB}
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/example_kernel_builder.py
                ${EXAMPLE_SOURCE}
                --output-dir ${EXAMPLE_KERNEL_DIR}
                --include-dirs "${CMAKE_CURRENT_SOURCE_DIR}/../../include,${CMAKE_CURRENT_SOURCE_DIR}/../include"
                --gpu-target ${GPU_TARGET}
                --jobs ${NPROC}
                --target-name ${NAME}
        COMMAND ${CMAKE_COMMAND} -E touch ${EXAMPLE_SENTINEL}
        DEPENDS ${EXAMPLE_SOURCE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../scripts
        COMMENT "[${NAME}] Generating and compiling kernels from DECL_KERNEL_SET..."
        VERBATIM
    )
    
    add_custom_target(generate_${NAME}_kernels DEPENDS ${EXAMPLE_SENTINEL})
    
    # Add the executable
    add_executable(${NAME} ${SOURCE})
    
    target_link_libraries(${NAME} PRIVATE ck_tile_dispatcher)
    
    # Link against the per-example kernel library
    target_link_libraries(${NAME} PRIVATE ${EXAMPLE_LIB})
    
    target_include_directories(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${EXAMPLE_KERNEL_DIR}
        ${EXAMPLE_KERNEL_DIR}/dispatcher_wrappers
    )
    
    # Force-include the generated registration header
    target_compile_options(${NAME} PRIVATE
        -include ${EXAMPLE_HEADER}
        -mllvm -enable-noalias-to-md-conversion=0
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
    )
    
    if(hip_FOUND)
        target_link_libraries(${NAME} PRIVATE hip::device hip::host)
    endif()
    
    # Only depends on generating THIS example's kernels
    add_dependencies(${NAME} generate_${NAME}_kernels)
endfunction()

# GEMM C++ examples with declarative kernel support
# Each example's C++ code contains DECL_KERNEL_SET which declares needed kernels
add_declarative_gpu_example(gemm_01_basic                gemm/cpp/01_basic_gemm.cpp)
add_declarative_gpu_example(gemm_02_multi_size           gemm/cpp/02_multi_size.cpp)
add_declarative_gpu_example(gemm_03_benchmark_validation gemm/cpp/03_benchmark_validation.cpp)
add_declarative_gpu_example(gemm_04_heuristics           gemm/cpp/04_heuristics.cpp)
add_declarative_gpu_example(gemm_05_json_export          gemm/cpp/05_json_export.cpp)
add_declarative_gpu_example(gemm_06_multi_registry       gemm/cpp/06_multi_registry.cpp)
add_declarative_gpu_example(gemm_07_gfx950_minimal       gemm/cpp/07_gfx950_minimal.cpp)

# ML Heuristic example -- requires LightGBM shared library
# Derive site-packages from active Python interpreter (respects virtualenvs)
find_package(Python3 COMPONENTS Interpreter)

set(LIGHTGBM_SEARCH_PATHS)
if(Python3_FOUND AND Python3_EXECUTABLE)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYTHON_SITE_PACKAGES)
        list(APPEND LIGHTGBM_SEARCH_PATHS "${PYTHON_SITE_PACKAGES}/lightgbm/lib")
    endif()
endif()

# Fallback to common Python 3.x site-packages if auto-detection failed
if(NOT PYTHON_SITE_PACKAGES)
    list(APPEND LIGHTGBM_SEARCH_PATHS
        "$ENV{HOME}/.local/lib/python3.12/site-packages/lightgbm/lib"
    )
endif()

find_library(LIGHTGBM_LIB NAMES LightGBM lib_lightgbm _lightgbm
    HINTS ${CMAKE_PREFIX_PATH}
    PATHS ${LIGHTGBM_SEARCH_PATHS}
    NO_DEFAULT_PATH
    DOC "LightGBM shared library for ML heuristics"
)

# Fallback: search default paths (respects LightGBM_DIR if set by user)
if(NOT LIGHTGBM_LIB)
    find_library(LIGHTGBM_LIB NAMES LightGBM lib_lightgbm)
endif()

if(LIGHTGBM_LIB)
    add_declarative_gpu_example(gemm_09_ml_heuristic gemm/cpp/09_ml_heuristic.cpp)
    target_link_libraries(gemm_09_ml_heuristic PRIVATE ${LIGHTGBM_LIB})
    message(STATUS "LightGBM found: ${LIGHTGBM_LIB} -- building gemm_09_ml_heuristic")
else()
    message(STATUS "LightGBM not found -- skipping gemm_09_ml_heuristic")
    message(STATUS "  To enable ML heuristic example:")
    message(STATUS "    1. Activate virtualenv: source .venv/bin/activate")
    message(STATUS "    2. Install: pip install -r ../requirements-ml.txt")
    message(STATUS "    3. Reconfigure: cmake ..")
    message(STATUS "  Or set CMAKE_PREFIX_PATH or LightGBM_DIR to LightGBM location")
endif()

# =============================================================================
# GEMM Python Library - Single Fallback Kernel
# =============================================================================

# Generate a single fallback kernel for the Python library (fp16, rcr, compv4)
set(GEMM_FALLBACK_KERNEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/gemm_python_fallback")
set(GEMM_FALLBACK_KERNEL "${GEMM_FALLBACK_KERNEL_DIR}/gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_128x128x32_2x2x1_32x32x16.hpp")

# Tile config JSON for single kernel generation
set(GEMM_FALLBACK_TILE_CONFIG "{\"tile_m\":[128],\"tile_n\":[128],\"tile_k\":[32],\"warp_m\":[2],\"warp_n\":[2],\"warp_k\":[1],\"warp_tile_m\":[32],\"warp_tile_n\":[32],\"warp_tile_k\":[16],\"pipeline\":[\"compv4\"],\"scheduler\":[\"intrawave\"],\"epilogue\":[\"cshuffle\"]}")

# Generate single fallback kernel (not all 6000+ kernels)
add_custom_command(
    OUTPUT ${GEMM_FALLBACK_KERNEL}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GEMM_FALLBACK_KERNEL_DIR}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_gemm_codegen.py 
            --datatype fp16 --layout rcr --variants standard
            --gpu-target ${GPU_TARGET}
            --output-dir ${GEMM_FALLBACK_KERNEL_DIR}
            --tile-config-json "${GEMM_FALLBACK_TILE_CONFIG}"
    COMMENT "Generating single fallback GEMM kernel for Python library"
    VERBATIM
)

add_custom_target(generate_gemm_fallback_kernel DEPENDS ${GEMM_FALLBACK_KERNEL})

# GEMM dynamic library for Python
add_library(dispatcher_gemm_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/ctypes/gemm_ctypes_lib.cpp)
target_link_libraries(dispatcher_gemm_lib PRIVATE ck_tile_dispatcher)
target_include_directories(dispatcher_gemm_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${GEMM_FALLBACK_KERNEL_DIR}
)
target_compile_options(dispatcher_gemm_lib PRIVATE
    -DCK_TILE_SINGLE_KERNEL_INCLUDE
    -include ${GEMM_FALLBACK_KERNEL}
    -DGFX_ARCH="${GPU_TARGET}"
    -mllvm -enable-noalias-to-md-conversion=0
    -Wno-undefined-func-template
    -Wno-float-equal
    --offload-compress
)
if(hip_FOUND)
    target_link_libraries(dispatcher_gemm_lib PRIVATE hip::device hip::host)
endif()
add_dependencies(dispatcher_gemm_lib generate_gemm_fallback_kernel)

# =============================================================================
# Grouped Convolution C++ Examples
# =============================================================================

add_declarative_gpu_example(grouped_conv_01_basic         grouped_conv/cpp/01_basic_grouped_conv.cpp)
add_declarative_gpu_example(grouped_conv_02_all_dirs      grouped_conv/cpp/02_all_directions.cpp)
add_declarative_gpu_example(grouped_conv_03_bench_val     grouped_conv/cpp/03_benchmark_validation.cpp)
add_declarative_gpu_example(grouped_conv_04_registry_json grouped_conv/cpp/04_registry_json.cpp)
add_declarative_gpu_example(grouped_conv_05_bwd_data      grouped_conv/cpp/05_bwd_data.cpp)
add_declarative_gpu_example(grouped_conv_06_bwd_weight    grouped_conv/cpp/06_bwd_weight.cpp)
add_declarative_gpu_example(grouped_conv_07_benchmark     grouped_conv/cpp/07_multi_tile_benchmark.cpp)

# =============================================================================
# Grouped Convolution Python Library - Multi-Kernel (fwd/bwd_data/bwd_weight x 2D/3D)
# =============================================================================

# Kernel output directory for the Python conv library
set(CONV_FALLBACK_KERNEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/conv_python_fallback")
set(CONV_DISPATCH_HEADER "${CONV_FALLBACK_KERNEL_DIR}/conv_python_dispatch.hpp")

# Generate ALL conv kernels (fwd/bwd_data/bwd_weight x 2D/3D x multiple tile configs)
# then create the dispatch header with 2D/3D aliases
add_custom_command(
    OUTPUT ${CONV_DISPATCH_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CONV_FALLBACK_KERNEL_DIR}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_grouped_conv_codegen.py
            --variant forward bwd_data bwd_weight --ndim 2 3
            --datatype fp16 --arch ${GPU_TARGET}
            --output ${CONV_FALLBACK_KERNEL_DIR}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_conv_dispatch_header.py
            --kernel-dir ${CONV_FALLBACK_KERNEL_DIR}
            --output ${CONV_DISPATCH_HEADER}
    COMMENT "Generating conv kernels (fwd/bwd_data/bwd_weight x 2D/3D) for Python library..."
    VERBATIM
)

add_custom_target(generate_conv_fallback_kernels DEPENDS ${CONV_DISPATCH_HEADER})

# Conv dynamic library for Python (all 6 kernel variants)
add_library(dispatcher_conv_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/ctypes/conv_ctypes_lib.cpp)
target_link_libraries(dispatcher_conv_lib PRIVATE ck_tile_dispatcher)
target_include_directories(dispatcher_conv_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CONV_FALLBACK_KERNEL_DIR}
)
target_compile_options(dispatcher_conv_lib PRIVATE
    -include ${CONV_DISPATCH_HEADER}
    -DGFX_ARCH="${GPU_TARGET}"
    -mllvm -enable-noalias-to-md-conversion=0
    -Wno-undefined-func-template
    -Wno-float-equal
    --offload-compress
)
if(hip_FOUND)
    target_link_libraries(dispatcher_conv_lib PRIVATE hip::device hip::host)
endif()
add_dependencies(dispatcher_conv_lib generate_conv_fallback_kernels)

message(STATUS "GEMM examples configured - kernels will be generated during 'make'")
message(STATUS "Grouped Conv examples configured - kernels will be generated during 'make'")

# Convenience target to build all Python ctypes libraries
add_custom_target(python_libs
    DEPENDS dispatcher_gemm_lib dispatcher_conv_lib
    COMMENT "Building Python ctypes libraries (GEMM + Conv)"
)

# =============================================================================
# Per-Architecture Kernel Generation Targets
# =============================================================================

set(SUPPORTED_GPU_ARCHS gfx942 gfx950 gfx90a gfx1100 gfx1030)

foreach(ARCH ${SUPPORTED_GPU_ARCHS})
    # GEMM kernels for this arch
    add_custom_target(generate_gemm_kernels_${ARCH}
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_gemm_codegen.py 
                --datatype fp16 --layout rcr --gpu-target ${ARCH}
                --output ${KERNEL_OUTPUT_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
        COMMENT "Generating GEMM kernels for ${ARCH}..."
        VERBATIM
    )
    
    # Alias for kernels (GEMM only now)
    add_custom_target(generate_kernels_${ARCH}
        DEPENDS generate_gemm_kernels_${ARCH}
        COMMENT "Generating all kernels for ${ARCH}..."
    )
endforeach()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== Dispatcher Examples Configuration ===")
message(STATUS "")
message(STATUS "Kernels will be generated automatically during 'make'")
message(STATUS "  Generated to: ${KERNEL_OUTPUT_DIR}")
message(STATUS "")
message(STATUS "Build targets:")
message(STATUS "  make                        - Build all examples (generates kernels first)")
message(STATUS "  make python_libs            - Build Python ctypes libraries")
message(STATUS "  make generate_all_kernels   - Generate all kernels only")
message(STATUS "  make regenerate_all_kernels - Force regenerate all kernels")
message(STATUS "")
message(STATUS "Per-architecture targets:")
message(STATUS "  make generate_kernels_<arch>  - Generate for specific arch")
message(STATUS "  Supported archs: ${SUPPORTED_GPU_ARCHS}")
message(STATUS "")
