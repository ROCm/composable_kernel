# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.16)

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

# Generate GEMM kernels
add_custom_command(
    OUTPUT ${GEMM_SENTINEL}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_gemm_codegen.py 
            --datatype fp16 --layout rcr
            --output ${KERNEL_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${GEMM_SENTINEL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
    COMMENT "Generating GEMM kernels (fp16, rcr)..."
    VERBATIM
)

add_custom_target(generate_gemm_kernels
    DEPENDS ${GEMM_SENTINEL}
    COMMENT "GEMM kernel generation target"
)

# Generate Conv forward kernels (2D and 3D)
add_custom_command(
    OUTPUT ${CONV_FWD_SENTINEL}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_conv_codegen.py 
            --datatype fp16 --variant forward --ndim 2 3
            --output ${KERNEL_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${CONV_FWD_SENTINEL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
    COMMENT "Generating Conv forward kernels (fp16, 2D+3D)..."
    VERBATIM
)

add_custom_target(generate_conv_fwd_kernels
    DEPENDS ${CONV_FWD_SENTINEL}
    COMMENT "Conv forward kernel generation target"
)

# Generate Conv backward kernels (bwd_data and bwd_weight, 2D)
add_custom_command(
    OUTPUT ${CONV_BWD_SENTINEL}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_conv_codegen.py 
            --datatype fp16 --variant bwd_data bwd_weight --ndim 2
            --output ${KERNEL_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${CONV_BWD_SENTINEL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
    COMMENT "Generating Conv backward kernels (fp16, 2D)..."
    VERBATIM
)

add_custom_target(generate_conv_bwd_kernels
    DEPENDS ${CONV_BWD_SENTINEL}
    COMMENT "Conv backward kernel generation target"
)

# Combined kernel generation targets
add_custom_target(generate_conv_kernels 
    DEPENDS generate_conv_fwd_kernels generate_conv_bwd_kernels
)

add_custom_target(generate_all_kernels 
    DEPENDS generate_gemm_kernels generate_conv_kernels
)

# =============================================================================
# Force regeneration targets (useful when you want to regenerate)
# =============================================================================

add_custom_target(regenerate_gemm_kernels
    COMMAND ${CMAKE_COMMAND} -E remove -f ${GEMM_SENTINEL}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_gemm_codegen.py 
            --datatype fp16 --layout rcr
            --output ${KERNEL_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${GEMM_SENTINEL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
    COMMENT "Force regenerating GEMM kernels..."
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

# =============================================================================
# Helper function to add a GPU example with force-included kernel
# =============================================================================

function(add_gpu_example NAME SOURCE KERNEL_HEADER)
    add_executable(${NAME} ${SOURCE})
    
    target_link_libraries(${NAME} PRIVATE ck_tile_dispatcher)
    
    target_include_directories(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include              # CK root include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include                 # Dispatcher include
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels # Generated kernels
    )
    
    target_compile_options(${NAME} PRIVATE
        -include ${KERNEL_HEADER}
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

# Set default kernel header path (will be found after generation)
# Naming convention: gemm_<dtype>_<layout>_<pipeline>_<epilogue>_<scheduler>_<pad_m>_<pad_n>_<pad_k>_<preshuffle>_<tile>_<wave>_<warp>.hpp
set(GEMM_KERNEL_HEADER "${KERNEL_OUTPUT_DIR}/gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_128x128x32_2x2x1_32x32x16.hpp")

# GEMM C++ examples - these depend on generate_gemm_kernels
add_gpu_example(gemm_01_basic           gemm/cpp/01_basic_gemm.cpp       ${GEMM_KERNEL_HEADER})
add_gpu_example(gemm_02_multi_size      gemm/cpp/02_multi_size.cpp       ${GEMM_KERNEL_HEADER})
add_gpu_example(gemm_03_benchmark       gemm/cpp/03_benchmark.cpp        ${GEMM_KERNEL_HEADER})
add_gpu_example(gemm_04_validation      gemm/cpp/04_validation.cpp       ${GEMM_KERNEL_HEADER})
add_gpu_example(gemm_05_heuristics      gemm/cpp/05_heuristics.cpp       ${GEMM_KERNEL_HEADER})
add_gpu_example(gemm_06_json_export     gemm/cpp/06_json_export.cpp      ${GEMM_KERNEL_HEADER})
add_gpu_example(gemm_07_preshuffle      gemm/cpp/07_preshuffle.cpp       ${GEMM_KERNEL_HEADER})
add_gpu_example(gemm_08_multi_d         gemm/cpp/08_multi_d.cpp          ${GEMM_KERNEL_HEADER})
add_gpu_example(gemm_09_multi_registry  gemm/cpp/09_multi_registry.cpp   ${GEMM_KERNEL_HEADER})

# Make GEMM examples depend on kernel generation
add_dependencies(gemm_01_basic          generate_gemm_kernels)
add_dependencies(gemm_02_multi_size     generate_gemm_kernels)
add_dependencies(gemm_03_benchmark      generate_gemm_kernels)
add_dependencies(gemm_04_validation     generate_gemm_kernels)
add_dependencies(gemm_05_heuristics     generate_gemm_kernels)
add_dependencies(gemm_06_json_export    generate_gemm_kernels)
add_dependencies(gemm_07_preshuffle     generate_gemm_kernels)
add_dependencies(gemm_08_multi_d        generate_gemm_kernels)
add_dependencies(gemm_09_multi_registry generate_gemm_kernels)

# GEMM dynamic library for Python (from bindings)
add_library(dispatcher_gemm_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/ctypes/gemm_ctypes_lib.cpp)
target_link_libraries(dispatcher_gemm_lib PRIVATE ck_tile_dispatcher)
target_include_directories(dispatcher_gemm_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels
)
target_compile_options(dispatcher_gemm_lib PRIVATE
    -include ${GEMM_KERNEL_HEADER}
    -mllvm -enable-noalias-to-md-conversion=0
    -Wno-undefined-func-template
    -Wno-float-equal
    --offload-compress
)
if(hip_FOUND)
    target_link_libraries(dispatcher_gemm_lib PRIVATE hip::device hip::host)
endif()
add_dependencies(dispatcher_gemm_lib generate_gemm_kernels)

message(STATUS "GEMM examples configured - kernels will be generated during 'make'")

# =============================================================================
# Convolution Examples
# =============================================================================

# Set default kernel header paths (will be found after generation)
# These use compv3 with tile_k=32 which are the first configs generated
set(CONV_KERNEL_HEADER "${KERNEL_OUTPUT_DIR}/conv_fwd_fp16_2d_compv3_cshuffle_intrawave_128x128x32_2x2x1_32x32x16.hpp")
set(CONV_3D_KERNEL_HEADER "${KERNEL_OUTPUT_DIR}/conv_fwd_fp16_3d_compv3_cshuffle_intrawave_128x128x32_2x2x1_32x32x16.hpp")
# Backward data: ConvConfigComputeV3 validated config M=16, N=64, K=64
set(CONV_BWDD_KERNEL_HEADER "${KERNEL_OUTPUT_DIR}/conv_bwdd_fp16_2d_compv3_cshuffle_intrawave_16x64x64_1x4x1_16x16x32.hpp")
# Backward weight: ConvConfigComputeV3 validated config M=16, N=64, K=64
set(CONV_BWDW_KERNEL_HEADER "${KERNEL_OUTPUT_DIR}/conv_bwdw_fp16_2d_compv3_cshuffle_intrawave_16x64x64_1x4x1_16x16x32.hpp")

# 2D forward examples
add_gpu_example(conv_01_forward         conv/cpp/01_conv_forward.cpp     ${CONV_KERNEL_HEADER})
add_gpu_example(conv_02_validation      conv/cpp/02_conv_validation.cpp  ${CONV_KERNEL_HEADER})
add_gpu_example(conv_03_multi_size      conv/cpp/03_multi_size.cpp       ${CONV_KERNEL_HEADER})
add_gpu_example(conv_04_benchmark       conv/cpp/04_benchmark.cpp        ${CONV_KERNEL_HEADER})
add_gpu_example(conv_05_heuristics      conv/cpp/05_heuristics.cpp       ${CONV_KERNEL_HEADER})
add_gpu_example(conv_06_json_export     conv/cpp/06_json_export.cpp      ${CONV_KERNEL_HEADER})
add_gpu_example(conv_07_multi_registry  conv/cpp/07_multi_registry.cpp   ${CONV_KERNEL_HEADER})

# 3D forward example
add_gpu_example(conv_08_conv3d_forward  conv/cpp/08_conv3d_forward.cpp   ${CONV_3D_KERNEL_HEADER})

# Backward data example
add_gpu_example(conv_09_bwd_data        conv/cpp/09_bwd_data.cpp         ${CONV_BWDD_KERNEL_HEADER})

# Backward weight example
add_gpu_example(conv_10_bwd_weight      conv/cpp/10_bwd_weight.cpp       ${CONV_BWDW_KERNEL_HEADER})

# Make Conv examples depend on kernel generation
add_dependencies(conv_01_forward        generate_conv_fwd_kernels)
add_dependencies(conv_02_validation     generate_conv_fwd_kernels)
add_dependencies(conv_03_multi_size     generate_conv_fwd_kernels)
add_dependencies(conv_04_benchmark      generate_conv_fwd_kernels)
add_dependencies(conv_05_heuristics     generate_conv_fwd_kernels)
add_dependencies(conv_06_json_export    generate_conv_fwd_kernels)
add_dependencies(conv_07_multi_registry generate_conv_fwd_kernels)
add_dependencies(conv_08_conv3d_forward generate_conv_fwd_kernels)
add_dependencies(conv_09_bwd_data       generate_conv_bwd_kernels)
add_dependencies(conv_10_bwd_weight     generate_conv_bwd_kernels)

message(STATUS "Conv examples configured - kernels will be generated during 'make'")

# =============================================================================
# Python helper libraries for conv (from bindings)
# =============================================================================

# Forward + backward data library
add_library(dispatcher_conv_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/ctypes/conv_ctypes_lib.cpp)
target_link_libraries(dispatcher_conv_lib PRIVATE ck_tile_dispatcher)
target_include_directories(dispatcher_conv_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels
)
target_compile_options(dispatcher_conv_lib PRIVATE
    -include ${CONV_KERNEL_HEADER}
    "SHELL:-include ${CONV_BWDD_KERNEL_HEADER}"
    -DCONV_KERNEL_AVAILABLE=1
    -DCONV_BWD_DATA_AVAILABLE=1
    -mllvm -enable-noalias-to-md-conversion=0
    -Wno-undefined-func-template
    -Wno-float-equal
    --offload-compress
)
if(hip_FOUND)
    target_link_libraries(dispatcher_conv_lib PRIVATE hip::device hip::host)
endif()
add_dependencies(dispatcher_conv_lib generate_conv_kernels)

# Backward weight library (separate to avoid template conflicts)
add_library(dispatcher_conv_bwdw_lib SHARED 
    ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/ctypes/conv_bwdw_ctypes_lib.cpp)
target_link_libraries(dispatcher_conv_bwdw_lib PRIVATE ck_tile_dispatcher)
target_include_directories(dispatcher_conv_bwdw_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels
)
target_compile_options(dispatcher_conv_bwdw_lib PRIVATE
    -include ${CONV_BWDW_KERNEL_HEADER}
    -DCONV_KERNEL_AVAILABLE=1
    -DCONV_BWD_WEIGHT_AVAILABLE=1
    -mllvm -enable-noalias-to-md-conversion=0
    -Wno-undefined-func-template
    -Wno-float-equal
    --offload-compress
)
if(hip_FOUND)
    target_link_libraries(dispatcher_conv_bwdw_lib PRIVATE hip::device hip::host)
endif()
add_dependencies(dispatcher_conv_bwdw_lib generate_conv_bwd_kernels)

# Convenience target to build all Python ctypes libraries
add_custom_target(python_libs
    DEPENDS dispatcher_gemm_lib dispatcher_conv_lib dispatcher_conv_bwdw_lib
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
