if(DL_KERNELS)
  add_custom_target(example_gemm_dl)

  add_example_executable(example_gemm_dl_fp32 gemm_dl_fp32.cpp)
  add_dependencies(example_gemm_dl example_gemm_dl_fp32)
  if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    add_example_executable(example_gemm_dl_fp16 gemm_dl_fp16.cpp)
    add_dependencies(example_gemm_dl example_gemm_dl_fp16)
    add_example_executable(example_gemm_dl_dpp8_fp16 gemm_dl_dpp8_fp16.cpp)
    add_dependencies(example_gemm_dl example_gemm_dl_dpp8_fp16)
  endif()
  if(DTYPES MATCHES "int8" OR NOT DEFINED DTYPES)
    add_example_executable(example_gemm_dl_int8 gemm_dl_int8.cpp)
    add_dependencies(example_gemm_dl example_gemm_dl_int8)
  endif()

  if(USE_BITINT_EXTENSION_INT4)
    add_example_executable(example_gemm_dl_int4 gemm_dl_int4.cpp)
    add_dependencies(example_gemm_dl example_gemm_dl_int4)
  endif(USE_BITINT_EXTENSION_INT4)
endif()

add_custom_target(example_gemm_xdl)
if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
  add_example_executable(example_gemm_xdl_fp16 gemm_xdl_fp16.cpp)
  add_example_executable(example_gemm_xdl_wavelet_fp16 gemm_xdl_wavelet_fp16.cpp)
  add_dependencies(example_gemm_xdl example_gemm_xdl_fp16)
  add_dependencies(example_gemm_xdl example_gemm_xdl_wavelet_fp16)
  add_example_executable(example_gemm_xdl_skip_b_lds_fp16 gemm_xdl_skip_b_lds_fp16.cpp)
  add_dependencies(example_gemm_xdl example_gemm_xdl_skip_b_lds_fp16)

  if(GPU_TARGETS MATCHES "gfx1100" OR GPU_TARGETS MATCHES "gfx1101" OR GPU_TARGETS MATCHES "gfx1102")
    add_custom_target(example_gemm_wmma)
    add_example_executable(example_gemm_wmma_fp16 gemm_wmma_fp16.cpp)
    add_dependencies(example_gemm_wmma example_gemm_wmma_fp16)
  endif()

endif()

if(DTYPES MATCHES "bf16" OR NOT DEFINED DTYPES)
  add_example_executable(example_gemm_xdl_bf16 gemm_xdl_bf16.cpp)
  add_dependencies(example_gemm_xdl example_gemm_xdl_bf16)

  add_example_executable(example_gemm_xdl_bf16_rtn gemm_xdl_bf16_rtn.cpp)
  add_dependencies(example_gemm_xdl example_gemm_xdl_bf16_rtn)
endif()

if(DTYPES MATCHES "int8" OR NOT DEFINED DTYPES)
  add_example_executable(example_gemm_xdl_int8 gemm_xdl_int8.cpp)
  add_dependencies(example_gemm_xdl example_gemm_xdl_int8)
endif()

if(USE_BITINT_EXTENSION_INT4)
  add_example_executable(example_gemm_xdl_int4 gemm_xdl_int4.cpp)
  add_dependencies(example_gemm_xdl example_gemm_xdl_int4)
endif(USE_BITINT_EXTENSION_INT4)

if(DTYPES MATCHES "fp64" OR NOT DEFINED DTYPES)
  # FIXME: re-enable this exampe as test when SWDEV-335738 is fixed
  add_example_executable_no_testing(example_gemm_xdl_fp64 gemm_xdl_fp64.cpp)
  add_dependencies(example_gemm_xdl example_gemm_xdl_fp64)
endif()

add_example_executable(example_gemm_xdl_streamk gemm_xdl_streamk.cpp)

if(DTYPES MATCHES "fp8" OR NOT DEFINED DTYPES)
  if(GPU_TARGETS MATCHES "gfx940" OR GPU_TARGETS MATCHES "gfx941" OR GPU_TARGETS MATCHES "gfx942")
    add_example_executable(example_gemm_xdl_f8 gemm_xdl_f8.cpp)
    add_dependencies(example_gemm_xdl example_gemm_xdl_f8)
  endif()
endif()

add_example_executable(example_gemm_xdl_fp16_f8 gemm_xdl_fp16_f8.cpp)
add_dependencies(example_gemm_xdl example_gemm_xdl_fp16_f8)
