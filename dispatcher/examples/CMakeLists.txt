# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
set(CONV_FWD_SENTINEL "${KERNEL_OUTPUT_DIR}/.conv_fwd_generated")
set(CONV_BWD_SENTINEL "${KERNEL_OUTPUT_DIR}/.conv_bwd_generated")
set(ALL_KERNELS_SENTINEL "${KERNEL_OUTPUT_DIR}/.all_generated")

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

# Generate Conv forward kernels (2D and 3D) - runs with internal parallelism
add_custom_command(
    OUTPUT ${CONV_FWD_SENTINEL}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_conv_codegen.py 
            --datatype fp16 --variant forward --ndim 2 3
            --output ${KERNEL_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${CONV_FWD_SENTINEL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
    COMMENT "Generating Conv forward kernels (fp16, 2D+3D) with internal parallelism..."
    VERBATIM
)

add_custom_target(generate_conv_fwd_kernels
    DEPENDS ${CONV_FWD_SENTINEL}
    COMMENT "Conv forward kernel generation target"
)

# Generate Conv backward kernels (bwd_data and bwd_weight, 2D) - runs with internal parallelism
add_custom_command(
    OUTPUT ${CONV_BWD_SENTINEL}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_conv_codegen.py 
            --datatype fp16 --variant bwd_data bwd_weight --ndim 2
            --output ${KERNEL_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${CONV_BWD_SENTINEL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
    COMMENT "Generating Conv backward kernels (fp16, 2D) with internal parallelism..."
    VERBATIM
)

add_custom_target(generate_conv_bwd_kernels
    DEPENDS ${CONV_BWD_SENTINEL}
    COMMENT "Conv backward kernel generation target"
)

# Combined kernel generation targets (these can run in parallel with make -j)
add_custom_target(generate_conv_kernels 
    DEPENDS generate_conv_fwd_kernels generate_conv_bwd_kernels
)

add_custom_target(generate_all_kernels 
    DEPENDS generate_gemm_kernels generate_conv_kernels
)

# Parallel kernel generation - generates all kernels in a single parallel job
# This is faster than running separate targets sequentially
# Uses separate commands that can run in parallel via make -j
add_custom_command(
    OUTPUT ${ALL_KERNELS_SENTINEL}
    COMMAND python3 unified_gemm_codegen.py --datatype fp16 --layout rcrr --variants standard preshuffle multi_d --output ${KERNEL_OUTPUT_DIR}
    COMMAND python3 unified_conv_codegen.py --datatype fp16 --variant forward --ndim 2 3 --output ${KERNEL_OUTPUT_DIR}
    COMMAND python3 unified_conv_codegen.py --datatype fp16 --variant bwd_data bwd_weight --ndim 2 --output ${KERNEL_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${GEMM_SENTINEL} ${CONV_FWD_SENTINEL} ${CONV_BWD_SENTINEL} ${ALL_KERNELS_SENTINEL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
    COMMENT "Generating ALL kernels (GEMM + Conv forward + Conv backward)..."
)

add_custom_target(generate_all_kernels_parallel
    DEPENDS ${ALL_KERNELS_SENTINEL}
    COMMENT "Parallel kernel generation target (fastest for full builds)"
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
#   [  3/128] Building kernel: conv_fwd_fp16_2d_128x128
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
    DEPENDS ${GEMM_SENTINEL} ${CONV_FWD_SENTINEL} ${CONV_BWD_SENTINEL}
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

add_custom_target(regenerate_conv_kernels
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CONV_FWD_SENTINEL} ${CONV_BWD_SENTINEL}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_conv_codegen.py 
            --datatype fp16 --variant forward bwd_data bwd_weight --ndim 2 3
            --output ${KERNEL_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${CONV_FWD_SENTINEL} ${CONV_BWD_SENTINEL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
    COMMENT "Force regenerating Conv kernels..."
    VERBATIM
)

add_custom_target(regenerate_all_kernels
    DEPENDS regenerate_gemm_kernels regenerate_conv_kernels
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
            -DCONV_KERNEL_AVAILABLE=1
            -DGEMM_KERNEL_AVAILABLE=1
            -mllvm -enable-noalias-to-md-conversion=0
            -Wno-undefined-func-template
            -Wno-float-equal
            --offload-compress
        )
    else()
        # Specific kernel header - force-include it
        target_compile_options(${NAME} PRIVATE
            -include ${KERNEL_HEADER}
            -DCONV_KERNEL_AVAILABLE=1
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

# Helper for conv GPU examples - uses include_all_conv_kernels.hpp 
# This header is generated during make and includes all conv kernel headers
function(add_conv_gpu_example NAME SOURCE KERNEL_TYPE)
    add_executable(${NAME} ${SOURCE})
    
    target_link_libraries(${NAME} PRIVATE ck_tile_dispatcher)
    
    target_include_directories(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include              # CK root include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include                 # Dispatcher include
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels # Generated kernels
    )
    
    # Force include the appropriate conv registration header
    set(CONV_REG_HEADER "${KERNEL_OUTPUT_DIR}/include_all_${KERNEL_TYPE}_kernels.hpp")
    
    target_compile_options(${NAME} PRIVATE
        -include ${CONV_REG_HEADER}
        -DCONV_KERNEL_AVAILABLE=1
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
        ${EXAMPLE_KERNEL_DIR}
        ${EXAMPLE_KERNEL_DIR}/dispatcher_wrappers
    )
    
    # Force-include the generated registration header
    target_compile_options(${NAME} PRIVATE
        -include ${EXAMPLE_HEADER}
        -DCONV_KERNEL_AVAILABLE=1
        -DGEMM_KERNEL_AVAILABLE=1
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
add_declarative_gpu_example(gemm_01_basic           gemm/cpp/01_basic_gemm.cpp)
add_declarative_gpu_example(gemm_02_multi_size      gemm/cpp/02_multi_size.cpp)
add_declarative_gpu_example(gemm_03_benchmark       gemm/cpp/03_benchmark.cpp)
add_declarative_gpu_example(gemm_04_validation      gemm/cpp/04_validation.cpp)
add_declarative_gpu_example(gemm_05_heuristics      gemm/cpp/05_heuristics.cpp)
add_declarative_gpu_example(gemm_06_json_export     gemm/cpp/06_json_export.cpp)
add_declarative_gpu_example(gemm_07_preshuffle      gemm/cpp/07_preshuffle.cpp)
add_declarative_gpu_example(gemm_08_multi_d         gemm/cpp/08_multi_d.cpp)
add_declarative_gpu_example(gemm_09_multi_registry  gemm/cpp/09_multi_registry.cpp)

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
            --gpu-target ${GPU_TARGETS}
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
    -DGFX_ARCH="${GPU_TARGETS}"
    -mllvm -enable-noalias-to-md-conversion=0
    -Wno-undefined-func-template
    -Wno-float-equal
    --offload-compress
)
if(hip_FOUND)
    target_link_libraries(dispatcher_gemm_lib PRIVATE hip::device hip::host)
endif()
add_dependencies(dispatcher_gemm_lib generate_gemm_fallback_kernel)

message(STATUS "GEMM examples configured - kernels will be generated during 'make'")

# =============================================================================
# Convolution Examples - Declarative kernel generation
# =============================================================================

# Conv examples use DECL_CONV_KERNEL_SET - declarative kernel specification
# No hardcoded paths - kernels are generated and compiled at make time

add_declarative_gpu_example(conv_01_forward         conv/cpp/01_conv_forward.cpp)
add_declarative_gpu_example(conv_02_validation      conv/cpp/02_conv_validation.cpp)
add_declarative_gpu_example(conv_03_multi_size      conv/cpp/03_multi_size.cpp)
add_declarative_gpu_example(conv_04_benchmark       conv/cpp/04_benchmark.cpp)
add_declarative_gpu_example(conv_05_heuristics      conv/cpp/05_heuristics.cpp)
add_declarative_gpu_example(conv_06_json_export     conv/cpp/06_json_export.cpp)
add_declarative_gpu_example(conv_07_multi_registry  conv/cpp/07_multi_registry.cpp)
add_declarative_gpu_example(conv_08_conv3d_forward  conv/cpp/08_conv3d_forward.cpp)
add_declarative_gpu_example(conv_09_bwd_data        conv/cpp/09_bwd_data.cpp)
add_declarative_gpu_example(conv_10_bwd_weight      conv/cpp/10_bwd_weight.cpp)
add_declarative_gpu_example(conv_11_advanced_benchmark conv/cpp/11_advanced_benchmark.cpp)

message(STATUS "Conv examples configured - kernels will be generated during 'make'")

# =============================================================================
# Python helper libraries for conv (from bindings) - Single Fallback Kernels
# =============================================================================

# Generate single fallback kernels for Python library (one each for fwd, bwdd, bwdw)
set(CONV_FALLBACK_DIR "${CMAKE_CURRENT_BINARY_DIR}/conv_python_fallback")
set(CONV_FWD_FALLBACK "${CONV_FALLBACK_DIR}/include_all_conv_fwd_kernels.hpp")
set(CONV_BWDD_FALLBACK "${CONV_FALLBACK_DIR}/include_all_conv_bwdd_kernels.hpp")
set(CONV_BWDW_FALLBACK "${CONV_FALLBACK_DIR}/include_all_conv_bwdw_kernels.hpp")
set(CONV_FALLBACK_SENTINEL "${CONV_FALLBACK_DIR}/.fallback_generated")

# Generate all conv fallback kernels in one command (uses --tile-m/n to produce 1 kernel per variant)
# Note: backward ops only support compv3 pipeline (compv4/compv5 have ck_tile template issues)
add_custom_command(
    OUTPUT ${CONV_FALLBACK_SENTINEL}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CONV_FALLBACK_DIR}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_conv_codegen.py 
            --datatype fp16 --variant forward --ndim 2
            --tile-m 128 --tile-n 128 --warp-m 2 --warp-n 2
            --output ${CONV_FALLBACK_DIR}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_conv_codegen.py 
            --datatype fp16 --variant bwd_data --ndim 2
            --tile-m 128 --tile-n 128 --warp-m 2 --warp-n 2
            --pipeline compv3
            --output ${CONV_FALLBACK_DIR}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_conv_codegen.py 
            --datatype fp16 --variant bwd_weight --ndim 2
            --tile-m 128 --tile-n 32 --warp-m 4 --warp-n 1
            --warp-tile-m 32 --warp-tile-n 32 --warp-tile-k 16
            --pipeline mem --scheduler interwave
            --output ${CONV_FALLBACK_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${CONV_FALLBACK_SENTINEL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
    COMMENT "Generating single fallback Conv kernels for Python library (3 kernels)"
    VERBATIM
)

add_custom_target(generate_conv_fallback_kernels DEPENDS ${CONV_FALLBACK_SENTINEL})

# Forward + backward data library (single fallback kernel each)
add_library(dispatcher_conv_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/ctypes/conv_ctypes_lib.cpp)
target_link_libraries(dispatcher_conv_lib PRIVATE ck_tile_dispatcher)
target_include_directories(dispatcher_conv_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CONV_FALLBACK_DIR}
)
target_compile_options(dispatcher_conv_lib PRIVATE
    "SHELL:-include ${CONV_FWD_FALLBACK}"
    # Note: Backward data disabled due to ck_tile compv3 template issues
    #"SHELL:-include ${CONV_BWDD_FALLBACK}"
    -DCONV_KERNEL_AVAILABLE=1
    #-DCONV_BWD_DATA_AVAILABLE=1
    -mllvm -enable-noalias-to-md-conversion=0
    -Wno-undefined-func-template
    -Wno-float-equal
    --offload-compress
)
if(hip_FOUND)
    target_link_libraries(dispatcher_conv_lib PRIVATE hip::device hip::host)
endif()
add_dependencies(dispatcher_conv_lib generate_conv_fallback_kernels)

# Backward weight library - DISABLED due to ck_tile transpose_tile2d template issues
# The generated bwdw kernels fail to compile with all pipeline types (mem, compv3, compv4)
# This is a fundamental ck_tile limitation, not a dispatcher issue
# 
# add_library(dispatcher_conv_bwdw_lib SHARED 
#     ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/ctypes/conv_bwdw_ctypes_lib.cpp)
# target_link_libraries(dispatcher_conv_bwdw_lib PRIVATE ck_tile_dispatcher)
# target_include_directories(dispatcher_conv_bwdw_lib PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../include
#     ${CMAKE_CURRENT_SOURCE_DIR}/../include
#     ${CONV_FALLBACK_DIR}
# )
# target_compile_options(dispatcher_conv_bwdw_lib PRIVATE
#     "SHELL:-include ${CONV_BWDW_FALLBACK}"
#     -DCONV_KERNEL_AVAILABLE=1
#     -DCONV_BWD_WEIGHT_AVAILABLE=1
#     -mllvm -enable-noalias-to-md-conversion=0
#     -Wno-undefined-func-template
#     -Wno-float-equal
#     --offload-compress
# )
# if(hip_FOUND)
#     target_link_libraries(dispatcher_conv_bwdw_lib PRIVATE hip::device hip::host)
# endif()
# add_dependencies(dispatcher_conv_bwdw_lib generate_conv_fallback_kernels)

# Convenience target to build all Python ctypes libraries
add_custom_target(python_libs
    DEPENDS dispatcher_gemm_lib dispatcher_conv_lib
    # Note: dispatcher_conv_bwdw_lib disabled due to ck_tile transpose_tile2d template issues
    COMMENT "Building all Python ctypes libraries"
)

# =============================================================================
# Per-Architecture Kernel Generation Targets
# =============================================================================

set(SUPPORTED_GPU_ARCHS gfx942 gfx90a gfx1100 gfx1030)

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
    
    # Conv kernels for this arch
    add_custom_target(generate_conv_kernels_${ARCH}
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_conv_codegen.py 
                --datatype fp16 --variant forward bwd_data bwd_weight --ndim 2 3 --arch ${ARCH}
                --output ${KERNEL_OUTPUT_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
        COMMENT "Generating Conv kernels for ${ARCH}..."
        VERBATIM
    )
    
    # All kernels for this arch
    add_custom_target(generate_kernels_${ARCH}
        DEPENDS generate_gemm_kernels_${ARCH} generate_conv_kernels_${ARCH}
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
