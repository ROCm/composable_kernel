# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(ck_tile_fmha_fwd_get_source_compile_options fwd_blob_name out_var)
  set(compile_options)
  # Keep the compiler scheduling mode used by the validated qr_tdm_sched, qr_tdm_fp8_sched and
  # qr_tdm_pingpong instances. The qr_tdm_pingpong instances that store the LSE
  # keep the default: built in this mode, which leaves the ALU dependency checks
  # to the compiler, they produce intermittently wrong O and LSE values.
  # Clang before 24 omits a required wait after the MX async scale loads in this mode.
  if(fwd_blob_name MATCHES "^fmha_fwd_d(64|128|192)_(fp16|bf16)_.*_qr_tdm_(sched|pingpong_.*_nlse)(_.*)?_gfx125\\.cpp$"
     OR (fwd_blob_name MATCHES "^fmha_fwd_d[0-9]+_[a-z0-9]+_.*_qr_tdm_fp8_sched_.*_gfx125\\.cpp$"
         AND NOT (fwd_blob_name MATCHES "_mx_" AND DEFINED CMAKE_HIP_COMPILER_VERSION
                  AND CMAKE_HIP_COMPILER_VERSION VERSION_LESS 24)))
    list(APPEND compile_options -mllvm -amdgpu-expert-scheduling-mode)
  endif()
  # Break large PHIs so blockscale and non-MX D64 keep the loop-carried K group in dwords.
  if(fwd_blob_name MATCHES "^fmha_fwd_d[0-9]+_[a-z0-9]+_.*_qr_tdm_fp8_sched_.*_blockscale_ntrload_.*_gfx125\\.cpp$"
     OR fwd_blob_name MATCHES "^fmha_fwd_d64_[a-z0-9]+_.*_qr_tdm_fp8_sched_.*_(nqscale|pertensor|perhead)_ntrload_.*_gfx125\\.cpp$")
    list(APPEND compile_options -mllvm -amdgpu-codegenprepare-force-break-large-phis)
  endif()
  # D128/D256 MX instances keep the scale copy's uniform branches free of StructurizeCFG flow flags.
  if(fwd_blob_name MATCHES "^fmha_fwd_d(128|256)_mxfp8[a-z0-9]*_.*_qr_tdm_fp8_sched_.*_gfx125\\.cpp$")
    list(APPEND compile_options -mllvm -structurizecfg-skip-uniform-regions)
  endif()
  # D256 non-MX instances allocate better with the dummy register allocation priority advisor.
  if(fwd_blob_name MATCHES "^fmha_fwd_d256_fp8[a-z0-9]*_.*_qr_tdm_fp8_sched_.*_gfx125\\.cpp$")
    list(APPEND compile_options -mllvm -regalloc-enable-priority-advisor=dummy)
  endif()
  set(${out_var} "${compile_options}" PARENT_SCOPE)
endfunction()

# Apply the per-source options above when the targets include any gfx125 family arch.
function(ck_tile_fmha_fwd_apply_source_compile_options inst_targets)
  list(FILTER inst_targets INCLUDE REGEX "^gfx125")
  if(NOT inst_targets)
    return()
  endif()
  foreach(fwd_blob IN LISTS ARGN)
    get_filename_component(fwd_blob_name "${fwd_blob}" NAME)
    ck_tile_fmha_fwd_get_source_compile_options("${fwd_blob_name}" fwd_blob_compile_options)
    if(fwd_blob_compile_options)
      set_property(SOURCE "${fwd_blob}" APPEND PROPERTY COMPILE_OPTIONS ${fwd_blob_compile_options})
    endif()
  endforeach()
endfunction()
