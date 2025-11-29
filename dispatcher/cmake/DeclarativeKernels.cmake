# SPDX-License-Identifier: MIT
# Declarative Kernel Build Support for CMake

#[=============================================================================[
  DeclarativeKernels.cmake
  
  This module enables the declarative kernel workflow:
  1. C++ code declares kernels with DECLARE_GEMM_KERNEL()
  2. CMake extracts declarations and generates .cpp files
  3. Kernels compile in parallel
  4. Application links to all declared kernels
  
  Usage in CMakeLists.txt:
  
    include(DeclarativeKernels)
    
    # Add your application with declarative kernel support
    add_declarative_gemm_app(
        NAME my_app
        SOURCES main.cpp utils.cpp
        GPU_ARCH gfx942
    )
#]=============================================================================]

# Extract kernel declarations from source files
function(extract_kernel_declarations SOURCES OUTPUT_FILE)
    set(ALL_DECLS "")
    
    foreach(SRC ${SOURCES})
        # Read source file
        file(READ ${SRC} CONTENT)
        
        # Find all DECLARE_GEMM_KERNEL calls
        string(REGEX MATCHALL "DECLARE_GEMM_KERNEL\\([^)]+\\)" DECLS "${CONTENT}")
        
        foreach(DECL ${DECLS})
            # Extract arguments: dtype, layout, tile_m, tile_n, tile_k
            string(REGEX REPLACE "DECLARE_GEMM_KERNEL\\(([^)]+)\\)" "\\1" ARGS "${DECL}")
            string(REPLACE " " "" ARGS "${ARGS}")  # Remove spaces
            list(APPEND ALL_DECLS "${ARGS}")
        endforeach()
    endforeach()
    
    # Remove duplicates
    list(REMOVE_DUPLICATES ALL_DECLS)
    
    # Write to file
    file(WRITE ${OUTPUT_FILE} "")
    foreach(DECL ${ALL_DECLS})
        file(APPEND ${OUTPUT_FILE} "${DECL}\n")
    endforeach()
    
    # Return count
    list(LENGTH ALL_DECLS NUM_DECLS)
    set(NUM_KERNEL_DECLARATIONS ${NUM_DECLS} PARENT_SCOPE)
endfunction()

# Generate kernel instantiation .cpp file
function(generate_kernel_source DTYPE LAYOUT TILE_M TILE_N TILE_K OUTPUT_DIR)
    set(KERNEL_NAME "${DTYPE}_${LAYOUT}_${TILE_M}x${TILE_N}x${TILE_K}")
    set(OUTPUT_FILE "${OUTPUT_DIR}/kernel_${KERNEL_NAME}.cpp")
    
    # Determine wave/warp config
    if(${TILE_M} GREATER_EQUAL 256 AND ${TILE_N} GREATER_EQUAL 256)
        set(WAVE_M 4) set(WAVE_N 4) set(WAVE_K 1)
        set(WARP_M 32) set(WARP_N 32) set(WARP_K 16)
    elseif(${TILE_M} GREATER_EQUAL 128 AND ${TILE_N} GREATER_EQUAL 128)
        set(WAVE_M 2) set(WAVE_N 2) set(WAVE_K 1)
        set(WARP_M 32) set(WARP_N 32) set(WARP_K 16)
    else()
        set(WAVE_M 2) set(WAVE_N 2) set(WAVE_K 1)
        set(WARP_M 16) set(WARP_N 16) set(WARP_K 16)
    endif()
    
    # Map dtype to C++ type
    if(DTYPE STREQUAL "fp16")
        set(CPP_TYPE "fp16_t")
    elseif(DTYPE STREQUAL "bf16")
        set(CPP_TYPE "bf16_t")
    elseif(DTYPE STREQUAL "fp32")
        set(CPP_TYPE "float")
    else()
        set(CPP_TYPE "fp16_t")
    endif()
    
    # Map layout
    if(LAYOUT STREQUAL "rcr")
        set(LAY_A "RowMajor") set(LAY_B "ColMajor") set(LAY_C "RowMajor")
    elseif(LAYOUT STREQUAL "rrr")
        set(LAY_A "RowMajor") set(LAY_B "RowMajor") set(LAY_C "RowMajor")
    else()
        set(LAY_A "RowMajor") set(LAY_B "ColMajor") set(LAY_C "RowMajor")
    endif()
    
    # Generate source
    file(WRITE ${OUTPUT_FILE} "// Auto-generated kernel: ${KERNEL_NAME}
#include \"ck_tile/dispatcher/kernel_impl.hpp\"

namespace ck_tile {
namespace dispatcher {

using Kernel_${KERNEL_NAME} = GemmKernel<
    ${CPP_TYPE}, ${CPP_TYPE}, ${CPP_TYPE}, float,
    ${LAY_A}, ${LAY_B}, ${LAY_C},
    ${TILE_M}, ${TILE_N}, ${TILE_K},
    ${WAVE_M}, ${WAVE_N}, ${WAVE_K},
    ${WARP_M}, ${WARP_N}, ${WARP_K},
    true, true, true
>;

CK_TILE_INSTANTIATE_KERNEL(Kernel_${KERNEL_NAME});

} // namespace dispatcher
} // namespace ck_tile
")
    
    set(GENERATED_KERNEL_SOURCE ${OUTPUT_FILE} PARENT_SCOPE)
endfunction()

# Main function: add application with declarative kernel support
function(add_declarative_gemm_app)
    cmake_parse_arguments(ARG "" "NAME;GPU_ARCH" "SOURCES" ${ARGN})
    
    if(NOT ARG_NAME)
        message(FATAL_ERROR "add_declarative_gemm_app: NAME required")
    endif()
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "add_declarative_gemm_app: SOURCES required")
    endif()
    if(NOT ARG_GPU_ARCH)
        set(ARG_GPU_ARCH "gfx942")
    endif()
    
    set(KERNEL_DIR "${CMAKE_BINARY_DIR}/generated_kernels/${ARG_NAME}")
    file(MAKE_DIRECTORY ${KERNEL_DIR})
    
    # Phase 1: Extract declarations
    message(STATUS "[${ARG_NAME}] Scanning for kernel declarations...")
    set(DECL_FILE "${CMAKE_BINARY_DIR}/${ARG_NAME}_declarations.txt")
    extract_kernel_declarations("${ARG_SOURCES}" ${DECL_FILE})
    message(STATUS "[${ARG_NAME}] Found ${NUM_KERNEL_DECLARATIONS} declarations")
    
    # Phase 2: Generate kernel sources
    set(KERNEL_SOURCES "")
    file(STRINGS ${DECL_FILE} DECLARATIONS)
    
    foreach(DECL ${DECLARATIONS})
        string(REPLACE "," ";" ARGS "${DECL}")
        list(GET ARGS 0 DTYPE)
        list(GET ARGS 1 LAYOUT)
        list(GET ARGS 2 TILE_M)
        list(GET ARGS 3 TILE_N)
        list(GET ARGS 4 TILE_K)
        
        generate_kernel_source(${DTYPE} ${LAYOUT} ${TILE_M} ${TILE_N} ${TILE_K} ${KERNEL_DIR})
        list(APPEND KERNEL_SOURCES ${GENERATED_KERNEL_SOURCE})
        message(STATUS "[${ARG_NAME}]   Generated: kernel_${DTYPE}_${LAYOUT}_${TILE_M}x${TILE_N}x${TILE_K}.cpp")
    endforeach()
    
    # Phase 3: Add executable with all sources
    add_executable(${ARG_NAME} ${ARG_SOURCES} ${KERNEL_SOURCES})
    
    target_include_directories(${ARG_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/include
    )
    
    target_compile_options(${ARG_NAME} PRIVATE
        -std=c++17
        --offload-arch=${ARG_GPU_ARCH}
        -O3
    )
    
    target_link_libraries(${ARG_NAME} PRIVATE ck_tile_dispatcher)
    
    message(STATUS "[${ARG_NAME}] Configured with ${NUM_KERNEL_DECLARATIONS} kernels")
endfunction()

