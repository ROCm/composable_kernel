# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Correctness check and timed benchmark of MX FP8 GEMM on the comp_tdm_v3 pipeline (gfx125x only:
# TDM and the scaled WMMA exist only there).
if(GPU_TARGETS MATCHES "gfx125")
  set(EXAMPLE_COMP_TDM_V3_MX_FP8 tile_example_gemm_comp_tdm_v3_mx_fp8_bench)
  set_source_files_properties(gemm_comp_tdm_v3_mx_fp8_bench.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${EXAMPLE_COMP_TDM_V3_MX_FP8} gemm_comp_tdm_v3_mx_fp8_bench.cpp)
  set(_comp_tdm_v3_mx_archs ${SUPPORTED_GPU_TARGETS})
  list(FILTER _comp_tdm_v3_mx_archs INCLUDE REGEX "^gfx125")
  set_property(TARGET ${EXAMPLE_COMP_TDM_V3_MX_FP8} PROPERTY HIP_ARCHITECTURES ${_comp_tdm_v3_mx_archs})
  # As for the comp_tdm_v3 gtests and example 54: re-enable the post-RA machine scheduler.
  target_compile_options(${EXAMPLE_COMP_TDM_V3_MX_FP8} PRIVATE
      -mllvm -enable-noalias-to-md-conversion=0 "SHELL:-mllvm -enable-post-misched=1")
  if(CK_USE_OCP_FP8)
    target_compile_options(${EXAMPLE_COMP_TDM_V3_MX_FP8} PRIVATE -DCK_TILE_USE_OCP_FP8)
  endif()
endif()
