# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(INST_TARGETS ${SUPPORTED_GPU_TARGETS})
# Currently only gfx9 arch is supported
list(FILTER INST_TARGETS INCLUDE REGEX "gfx9")
if(NOT INST_TARGETS)
  message(WARNING "Skipping SageAttention compilation: No supported GPU targets (gfx9) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
  return()
endif()

# ====================================================================
# SageAttention codegen - only FWD API, minimal instances
# ====================================================================
file(GLOB_RECURSE CODE_GEN_SCRIPTS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  ${CMAKE_CURRENT_LIST_DIR}/codegen/*.py
)
set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS "${CODE_GEN_SCRIPTS}")

list(JOIN INST_TARGETS , SAGEATTN_TARGETS_ARG)

# Only generate FWD API, only supported head dimension (128)
# Note: Only d=128, d_v=128 has kernel tile definitions in sageattn_fwd.py
set(SAGEATTN_FWD_CODE_GEN_COMMON_ARGS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --targets ${SAGEATTN_TARGETS_ARG}
  --api fwd
  --optdim 128
)

# Generate list of kernels to build
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${SAGEATTN_FWD_CODE_GEN_COMMON_ARGS}
  --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/sageattn_fwd_blob_list.txt
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message(FATAL_ERROR "SageAttention FAILED to generate kernel list via Python.")
endif()

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/sageattn_fwd_blob_list.txt SAGEATTN_FWD_GEN_BLOBS)

# Generate the kernel instance files
add_custom_command(
  OUTPUT ${SAGEATTN_FWD_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${SAGEATTN_FWD_CODE_GEN_COMMON_ARGS}
  --output_dir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CODE_GEN_SCRIPTS}
  COMMENT "Generate SageAttention FWD kernels"
  VERBATIM
)

# Build the kernel instances library
add_library(tile_sageattn_fwd_instances OBJECT EXCLUDE_FROM_ALL ${SAGEATTN_FWD_GEN_BLOBS})
target_include_directories(tile_sageattn_fwd_instances PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# Compile options for kernel instances
set(SAGEATTN_FWD_INSTANCE_COMPILE_OPTIONS)
list(APPEND SAGEATTN_FWD_INSTANCE_COMPILE_OPTIONS -Wno-undefined-func-template)
list(APPEND SAGEATTN_FWD_INSTANCE_COMPILE_OPTIONS -Wno-float-equal)
list(APPEND SAGEATTN_FWD_INSTANCE_COMPILE_OPTIONS -fgpu-flush-denormals-to-zero)

if(CK_USE_OCP_FP8)
  list(APPEND SAGEATTN_FWD_INSTANCE_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

target_compile_options(tile_sageattn_fwd_instances PRIVATE ${SAGEATTN_FWD_INSTANCE_COMPILE_OPTIONS})
set_property(TARGET tile_sageattn_fwd_instances PROPERTY HIP_ARCHITECTURES ${INST_TARGETS})
set_property(TARGET tile_sageattn_fwd_instances PROPERTY POSITION_INDEPENDENT_CODE ON)

# ====================================================================
# SageAttention FWD Example
# ====================================================================
set(EXAMPLE_SAGEATTN_FWD "tile_example_sageattn_fwd")

message(DEBUG "adding example ${EXAMPLE_SAGEATTN_FWD}")

add_executable(${EXAMPLE_SAGEATTN_FWD} EXCLUDE_FROM_ALL example_sageattn_fwd.cpp)
target_include_directories(${EXAMPLE_SAGEATTN_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# Link with our own minimal instances library (INDEPENDENT from FMHA!)
target_link_libraries(${EXAMPLE_SAGEATTN_FWD} tile_sageattn_fwd_instances)

set(SAGEATTN_FWD_COMPILE_OPTIONS)
list(APPEND SAGEATTN_FWD_COMPILE_OPTIONS -Wno-undefined-func-template)
list(APPEND SAGEATTN_FWD_COMPILE_OPTIONS -Wno-float-equal)
list(APPEND SAGEATTN_FWD_COMPILE_OPTIONS -fgpu-flush-denormals-to-zero)

if(CK_USE_OCP_FP8)
  list(APPEND SAGEATTN_FWD_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

target_compile_options(${EXAMPLE_SAGEATTN_FWD} PRIVATE ${SAGEATTN_FWD_COMPILE_OPTIONS})
set_property(TARGET ${EXAMPLE_SAGEATTN_FWD} PROPERTY HIP_ARCHITECTURES ${INST_TARGETS})
