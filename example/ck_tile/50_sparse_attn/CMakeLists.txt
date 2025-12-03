# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
# CMakeLists.txt for sparse attention (Jenga and VSA)

# Use SUPPORTED_GPU_TARGETS directly
set(INST_TARGETS ${SUPPORTED_GPU_TARGETS})
set(GPU_TARGETS ${SUPPORTED_GPU_TARGETS})

message(STATUS "VSA Sparse Attention: SUPPORTED_GPU_TARGETS=${SUPPORTED_GPU_TARGETS}, INST_TARGETS=${INST_TARGETS}")

list(FILTER INST_TARGETS INCLUDE REGEX "gfx9|gfx12")
if(NOT INST_TARGETS)
  message(WARNING "Skipping Tile Engine Sparse Attention: No supported GPU targets found")
  return()
endif()

message(STATUS "Building VSA Sparse Attention for targets: ${INST_TARGETS}")

# Code generation scripts
file(GLOB_RECURSE CODE_GEN_SCRIPTS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  ${CMAKE_CURRENT_LIST_DIR}/codegen/*.py
)
set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS "${CODE_GEN_SCRIPTS}")

# Code generation for VSA (receipt 600 for aiter integration)
set(SPARSE_ATTN_VSA_CODE_GEN_ARGS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api fwd_vsa
  --receipt 600
)

# Generate list of VSA kernels (at configure time, only list, not generate)
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${SPARSE_ATTN_VSA_CODE_GEN_ARGS}
  --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/vsa_blob_list.txt
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message(FATAL_ERROR "Failed to generate VSA kernel list")
endif()

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/vsa_blob_list.txt SPARSE_ATTN_VSA_GEN_BLOBS)

# Generate the kernel source files at build time (not configure time)
add_custom_command(
  OUTPUT ${SPARSE_ATTN_VSA_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${SPARSE_ATTN_VSA_CODE_GEN_ARGS}
  --output_dir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CODE_GEN_SCRIPTS}
  COMMENT "Generate CK Tile VSA Sparse Attention kernels"
)

message(STATUS "VSA kernel files to be generated: ${SPARSE_ATTN_VSA_GEN_BLOBS}")

# VSA Instances
set(SPARSE_ATTN_VSA_INSTANCES "tile_sparse_attn_vsa_instances")

add_library(${SPARSE_ATTN_VSA_INSTANCES} OBJECT EXCLUDE_FROM_ALL
  ${SPARSE_ATTN_VSA_GEN_BLOBS}
  ${CMAKE_CURRENT_LIST_DIR}/vsa_sparse_attention.cu
)
target_include_directories(${SPARSE_ATTN_VSA_INSTANCES} PRIVATE 
  ${CMAKE_CURRENT_LIST_DIR}
  ${PROJECT_SOURCE_DIR}/include/ck_tile/ops/sparse_attn
)
set_source_files_properties(${SPARSE_ATTN_VSA_GEN_BLOBS} PROPERTIES LANGUAGE HIP)
set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/vsa_sparse_attention.cu PROPERTIES LANGUAGE HIP)
set_property(TARGET ${SPARSE_ATTN_VSA_INSTANCES} PROPERTY HIP_ARCHITECTURES ${INST_TARGETS})

# Compile options
target_compile_options(${SPARSE_ATTN_VSA_INSTANCES} PRIVATE
  -DCK_TILE_USE_BUFFER_ADDRESSING_BUILTIN
  -DCK_TILE_FMHA_FWD_FAST_EXP2
  -Wno-undefined-func-template
  -Wno-float-equal
)

# Test executable
set(TEST_VSA_SPARSE_ATTN "tile_test_vsa_sparse_attn")
add_executable(${TEST_VSA_SPARSE_ATTN} EXCLUDE_FROM_ALL test_vsa_sparse_attn.cpp)
target_link_libraries(${TEST_VSA_SPARSE_ATTN} ${SPARSE_ATTN_VSA_INSTANCES})
target_include_directories(${TEST_VSA_SPARSE_ATTN} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(${TEST_VSA_SPARSE_ATTN} PRIVATE
  -Wno-undefined-func-template
  -Wno-float-equal
)

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
