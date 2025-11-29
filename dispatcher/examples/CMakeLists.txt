# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.16)

# Link to dispatcher library
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build)

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
# Auto-generate kernels if they don't exist
# =============================================================================

set(KERNEL_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels")
file(MAKE_DIRECTORY ${KERNEL_OUTPUT_DIR})

# Check if GEMM kernels exist, generate if not
file(GLOB EXISTING_GEMM_KERNELS "${KERNEL_OUTPUT_DIR}/gemm_fp16_rcr*.hpp")
if(NOT EXISTING_GEMM_KERNELS)
    message(STATUS "GEMM kernels not found - generating automatically...")
    execute_process(
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_gemm_codegen.py 
                --datatype fp16 --layout rcr
                --output ${KERNEL_OUTPUT_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
        RESULT_VARIABLE GEMM_CODEGEN_RESULT
    )
    if(NOT GEMM_CODEGEN_RESULT EQUAL 0)
        message(WARNING "GEMM kernel generation failed")
    endif()
endif()

# Check if Conv kernels exist, generate if not
file(GLOB EXISTING_CONV_KERNELS "${KERNEL_OUTPUT_DIR}/conv_fwd_fp16_2d*.hpp")
if(NOT EXISTING_CONV_KERNELS)
    message(STATUS "Conv forward kernels not found - generating automatically...")
    execute_process(
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_conv_codegen.py 
                --datatype fp16 --variant forward --ndim 2 3
                --output ${KERNEL_OUTPUT_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
        RESULT_VARIABLE CONV_FWD_CODEGEN_RESULT
    )
    if(NOT CONV_FWD_CODEGEN_RESULT EQUAL 0)
        message(WARNING "Conv forward kernel generation failed")
    endif()
endif()

# Check if Conv backward kernels exist, generate if not
file(GLOB EXISTING_CONV_BWD_KERNELS "${KERNEL_OUTPUT_DIR}/conv_bwd*.hpp")
if(NOT EXISTING_CONV_BWD_KERNELS)
    message(STATUS "Conv backward kernels not found - generating automatically...")
    execute_process(
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_conv_codegen.py 
                --datatype fp16 --variant bwd_data bwd_weight --ndim 2
                --output ${KERNEL_OUTPUT_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
        RESULT_VARIABLE CONV_BWD_CODEGEN_RESULT
    )
    if(NOT CONV_BWD_CODEGEN_RESULT EQUAL 0)
        message(WARNING "Conv backward kernel generation failed")
    endif()
endif()

# =============================================================================
# Manual generation targets (for regeneration)
# =============================================================================

# Generate GEMM kernels
add_custom_target(generate_gemm_kernels
    COMMAND ${CMAKE_COMMAND} -E make_directory ${KERNEL_OUTPUT_DIR}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_gemm_codegen.py 
            --datatype fp16 --layout rcr
            --output ${KERNEL_OUTPUT_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
    COMMENT "Generating GEMM kernels..."
)

# Generate Conv kernels
add_custom_target(generate_conv_kernels
    COMMAND ${CMAKE_COMMAND} -E make_directory ${KERNEL_OUTPUT_DIR}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_conv_codegen.py 
            --datatype fp16 --variant forward bwd_data bwd_weight --ndim 2 3
            --output ${KERNEL_OUTPUT_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
    COMMENT "Generating Conv kernels..."
)

# Combined target
add_custom_target(generate_all_kernels 
    DEPENDS generate_gemm_kernels generate_conv_kernels
)

# =============================================================================
# GEMM Examples
# =============================================================================

# Find generated GEMM kernel header
file(GLOB GEMM_KERNEL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels/gemm_fp16_rcr_compv4*128x128x32*.hpp")
if(GEMM_KERNEL_HEADERS)
    list(GET GEMM_KERNEL_HEADERS 0 GEMM_KERNEL_HEADER)
else()
    set(GEMM_KERNEL_HEADER "")
endif()

if(GEMM_KERNEL_HEADER AND EXISTS "${GEMM_KERNEL_HEADER}")
    message(STATUS "Building GEMM examples with kernel: ${GEMM_KERNEL_HEADER}")
    
    # GEMM C++ examples
    add_gpu_example(gemm_01_basic           gemm/cpp/01_basic_gemm.cpp       ${GEMM_KERNEL_HEADER})
    add_gpu_example(gemm_02_multi_size      gemm/cpp/02_multi_size.cpp       ${GEMM_KERNEL_HEADER})
    add_gpu_example(gemm_03_benchmark       gemm/cpp/03_benchmark.cpp        ${GEMM_KERNEL_HEADER})
    add_gpu_example(gemm_04_validation      gemm/cpp/04_validation.cpp       ${GEMM_KERNEL_HEADER})
    add_gpu_example(gemm_05_heuristics      gemm/cpp/05_heuristics.cpp       ${GEMM_KERNEL_HEADER})
    add_gpu_example(gemm_06_json_export     gemm/cpp/06_json_export.cpp      ${GEMM_KERNEL_HEADER})
    add_gpu_example(gemm_07_preshuffle      gemm/cpp/07_preshuffle.cpp       ${GEMM_KERNEL_HEADER})
    add_gpu_example(gemm_08_multi_d         gemm/cpp/08_multi_d.cpp          ${GEMM_KERNEL_HEADER})
    add_gpu_example(gemm_09_multi_registry  gemm/cpp/09_multi_registry.cpp   ${GEMM_KERNEL_HEADER})
    
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
    
    message(STATUS "  Built: gemm_01 through gemm_09, dispatcher_gemm_lib.so")
else()
    message(STATUS "GEMM kernels not found - skipping GPU GEMM examples")
    message(STATUS "  Generate with: make generate_gemm_kernels")
    message(STATUS "  Or: python3 codegen/unified_gemm_codegen.py --datatype fp16 --layout rcr")
endif()

# =============================================================================
# Convolution Examples
# =============================================================================

# Find generated Conv kernel header (use single kernel to avoid redefinition issues)
file(GLOB CONV_KERNEL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels/conv_fwd_fp16_2d_compv3_cshuffle_intrawave_128x128x32*.hpp")
if(CONV_KERNEL_HEADERS)
    list(GET CONV_KERNEL_HEADERS 0 CONV_KERNEL_HEADER)
else()
    set(CONV_KERNEL_HEADER "")
endif()

# ALL conv examples require generated kernels for GPU execution
if(CONV_KERNEL_HEADER AND EXISTS "${CONV_KERNEL_HEADER}")
    message(STATUS "Building ALL Conv examples with GPU kernels: ${CONV_KERNEL_HEADER}")
    
    # 2D forward examples
    add_gpu_example(conv_01_basic           conv/cpp/01_basic_conv.cpp       ${CONV_KERNEL_HEADER})
    add_gpu_example(conv_02_forward         conv/cpp/02_conv_forward.cpp     ${CONV_KERNEL_HEADER})
    add_gpu_example(conv_03_validation      conv/cpp/03_conv_validation.cpp  ${CONV_KERNEL_HEADER})
    add_gpu_example(conv_04_multi_size      conv/cpp/04_multi_size.cpp       ${CONV_KERNEL_HEADER})
    add_gpu_example(conv_05_benchmark       conv/cpp/05_benchmark.cpp        ${CONV_KERNEL_HEADER})
    add_gpu_example(conv_06_heuristics      conv/cpp/06_heuristics.cpp       ${CONV_KERNEL_HEADER})
    add_gpu_example(conv_07_json_export     conv/cpp/07_json_export.cpp      ${CONV_KERNEL_HEADER})
    add_gpu_example(conv_08_multi_registry  conv/cpp/08_multi_registry.cpp   ${CONV_KERNEL_HEADER})
    
    # 3D forward example
    file(GLOB CONV_3D_KERNEL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels/conv_fwd_fp16_3d_compv3*.hpp")
    if(CONV_3D_KERNEL_HEADERS)
        list(GET CONV_3D_KERNEL_HEADERS 0 CONV_3D_KERNEL_HEADER)
        add_gpu_example(conv_09_conv3d_forward  conv/cpp/09_conv3d_forward.cpp  ${CONV_3D_KERNEL_HEADER})
        message(STATUS "  Built: conv_09 (3D forward)")
    endif()
    
    # Backward data example
    file(GLOB CONV_BWDD_KERNEL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels/conv_bwdd_fp16_2d_compv3*.hpp")
    if(CONV_BWDD_KERNEL_HEADERS)
        list(GET CONV_BWDD_KERNEL_HEADERS 0 CONV_BWDD_KERNEL_HEADER)
        add_gpu_example(conv_10_bwd_data  conv/cpp/10_bwd_data.cpp  ${CONV_BWDD_KERNEL_HEADER})
        message(STATUS "  Built: conv_10 (backward data)")
    endif()
    
    # Backward weight example
    file(GLOB CONV_BWDW_KERNEL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels/conv_bwdw_fp16_2d_compv3*.hpp")
    if(CONV_BWDW_KERNEL_HEADERS)
        list(GET CONV_BWDW_KERNEL_HEADERS 0 CONV_BWDW_KERNEL_HEADER)
        add_gpu_example(conv_11_bwd_weight  conv/cpp/11_bwd_weight.cpp  ${CONV_BWDW_KERNEL_HEADER})
        message(STATUS "  Built: conv_11 (backward weight)")
    endif()
    
    message(STATUS "  Built: conv_01 through conv_08 (2D forward with GPU execution)")
else()
    message(STATUS "Conv kernels not found - skipping ALL Conv examples")
    message(STATUS "  Generate with: python3 codegen/unified_conv_codegen.py --datatype fp16 --variant forward bwd_data bwd_weight --ndim 2 3 -o build/generated_kernels")
endif()

# =============================================================================
# Python helper library for conv (from bindings)
# =============================================================================

if(CONV_KERNEL_HEADER AND EXISTS "${CONV_KERNEL_HEADER}")
    add_library(dispatcher_conv_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/ctypes/conv_ctypes_lib.cpp)
    target_link_libraries(dispatcher_conv_lib PRIVATE ck_tile_dispatcher)
    target_include_directories(dispatcher_conv_lib PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels
    )
    
    # Start with forward kernel
    set(CONV_LIB_COMPILE_OPTIONS
        -include ${CONV_KERNEL_HEADER}
        -DCONV_KERNEL_AVAILABLE=1
        -mllvm -enable-noalias-to-md-conversion=0
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
    )
    
    # Backward data kernel (optional)
    if(CONV_BWDD_KERNEL_HEADER AND EXISTS "${CONV_BWDD_KERNEL_HEADER}")
        list(APPEND CONV_LIB_COMPILE_OPTIONS
            "SHELL:-include ${CONV_BWDD_KERNEL_HEADER}"
            -DCONV_BWD_DATA_AVAILABLE=1
        )
        message(STATUS "  Conv lib: backward data kernel included")
    endif()
    
    target_compile_options(dispatcher_conv_lib PRIVATE ${CONV_LIB_COMPILE_OPTIONS})
    
    if(hip_FOUND)
        target_link_libraries(dispatcher_conv_lib PRIVATE hip::device hip::host)
    endif()
    message(STATUS "  Built: dispatcher_conv_lib.so (forward + bwd_data)")
endif()

# =============================================================================
# Separate backward weight library (avoids template conflicts)
# =============================================================================

if(CONV_BWDW_KERNEL_HEADER AND EXISTS "${CONV_BWDW_KERNEL_HEADER}")
    add_library(dispatcher_conv_bwdw_lib SHARED 
        ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/ctypes/conv_bwdw_ctypes_lib.cpp)
    target_link_libraries(dispatcher_conv_bwdw_lib PRIVATE ck_tile_dispatcher)
    target_include_directories(dispatcher_conv_bwdw_lib PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels
    )
    
    # Use same flags as C++ example (which compiles successfully)
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
    message(STATUS "  Built: dispatcher_conv_bwdw_lib.so (backward weight only)")
endif()

# Convenience target to build all Python ctypes libraries
add_custom_target(python_libs
    DEPENDS dispatcher_gemm_lib dispatcher_conv_lib dispatcher_conv_bwdw_lib
    COMMENT "Building all Python ctypes libraries"
)

message(STATUS "Examples configuration complete")
message(STATUS "  Use 'make python_libs' to build only the shared libraries for Python")
