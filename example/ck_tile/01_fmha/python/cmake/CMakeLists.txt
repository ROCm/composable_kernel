# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
project(ck_tile_fmha_kernels LANGUAGES CXX HIP)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
include(CheckCXXCompilerFlag)

if(NOT CK_ROOT OR NOT EXISTS "${CK_ROOT}/include/ck_tile/core.hpp")
  message(FATAL_ERROR "CK_ROOT must point to a Composable Kernel source tree")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(SUPPORTED_GPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
set(GPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
set(target_ids)
foreach(target IN LISTS GPU_TARGETS)
  if(NOT target MATCHES "^gfx([0-9a-f]+)$")
    message(FATAL_ERROR "Unsupported GPU target spelling: ${target}")
  endif()
  list(APPEND target_ids "0x${CMAKE_MATCH_1}")
endforeach()
list(JOIN target_ids , target_ids_arg)
add_compile_definitions(CK_CMAKE_GPU_TARGET_IDS=${target_ids_arg})
if(GPU_TARGETS MATCHES "gfx1250")
  # Match CK's top-level CMakeLists: the unified MMA framework is not yet
  # supported on gfx1250, including in the host compilation pass.
  add_compile_definitions(USE_NEW_UNIFIED_FRAMEWORK=0)
endif()
set(BUILD_TESTING OFF)
set(FMHA_FWD_ENABLE_APIS "fwd;fwd_splitkv;fwd_appendkv")
set(FMHA_FWD_CODEGEN_RECEIPT 5 CACHE STRING "Codegen receipt for Python FMHA forward instances")
set(FMHA_BWD_CODEGEN_RECEIPT 5 CACHE STRING "Codegen receipt for Python FMHA backward instances")

foreach(dtype FP16 BF16 FP32 FP64 FP8 BF8 INT8)
  set(CK_ENABLE_${dtype} ON)
endforeach()
set(CK_USE_XDL ON)
if(GPU_TARGETS MATCHES "gfx11|gfx12")
  set(CK_USE_WMMA ON)
  add_compile_definitions(CK_TILE_USE_WMMA=1)
endif()
if(GPU_TARGETS MATCHES "gfx12|gfx950")
  set(CK_USE_OCP_FP8 ON)
  add_compile_definitions(CK_TILE_USE_OCP_FP8)
endif()
configure_file("${CK_ROOT}/include/ck/config.h.in" "${CMAKE_BINARY_DIR}/include/ck/config.h")
include_directories("${CK_ROOT}/include" "${CK_ROOT}/library/include" "${CMAKE_BINARY_DIR}/include")
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
add_compile_options(-fbracket-depth=1024)

# Match CK's toolchain options; the shared FMHA targets supply per-direction
# BF16 conversion and per-instance gfx1250 scheduling flags.
foreach(flag IN ITEMS "-fno-offload-uniform-block" "-mllvm --lsr-drop-solution=1"
    "-mllvm -enable-post-misched=0" "-mllvm -amdgpu-early-inline-all=true"
    "-mllvm -amdgpu-function-calls=false")
  string(MAKE_C_IDENTIFIER "${flag}" flag_id)
  check_cxx_compiler_flag("${flag}" "HAS_${flag_id}")
  if(HAS_${flag_id})
    add_compile_options("SHELL:${flag}")
  endif()
endforeach()

add_subdirectory("${CK_ROOT}/example/ck_tile/01_fmha" "fmha")

# Match PyTorch and backward's BF16 rounding without changing the C++ example defaults.
get_target_property(fwd_sources tile_fmha_fwd_instances SOURCES)
foreach(source IN LISTS fwd_sources)
  if(source MATCHES "_gfx9[^/]*\\.cpp$")
    set_property(SOURCE "${source}" TARGET_DIRECTORY tile_fmha_fwd_instances
      APPEND PROPERTY COMPILE_DEFINITIONS CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3)
  endif()
endforeach()

add_library(ck_tile_fmha_kernels STATIC
  $<TARGET_OBJECTS:tile_fmha_fwd_instances>
  $<TARGET_OBJECTS:tile_fmha_bwd_instances>)
set_target_properties(ck_tile_fmha_kernels PROPERTIES
  LINKER_LANGUAGE HIP ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
