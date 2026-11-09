# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(ck_tile_fmha_fwd_get_source_compile_options fwd_blob_name out_var)
  set(compile_options)
  # Keep the compiler scheduling mode used by the validated qr_tdm_sched instances,
  # and use it for every qr_tdm_pingpong instance (GDPA's d128 instances among them: they
  # keep qr_tdm_pingpong's name and run BlockFmhaPipelineQRKSVSTdmGdpaQLoop).
  if(fwd_blob_name MATCHES "^fmha_fwd_d(64|128|192)_(fp16|bf16)_.*_qr_tdm_sched(_.*)?_gfx125\\.cpp$" OR
     fwd_blob_name MATCHES "^fmha_fwd_d[0-9]+_(fp16|bf16)_.*_qr_tdm_pingpong(_.*)?_gfx125\\.cpp$")
    list(APPEND compile_options -mllvm -amdgpu-expert-scheduling-mode)
  endif()
  set(${out_var} "${compile_options}" PARENT_SCOPE)
endfunction()
