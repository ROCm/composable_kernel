set(GROUPED_GEMM_FIXED_NK_INSTANCES)

if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
  list(APPEND GROUPED_GEMM_FIXED_NK_INSTANCES device_grouped_gemm_xdl_fixed_nk_f16_f16_f16_mk_kn_mn_instance.cpp)
  list(APPEND GROUPED_GEMM_FIXED_NK_INSTANCES device_grouped_gemm_xdl_fixed_nk_f16_f16_f16_mk_nk_mn_instance.cpp)
endif()

if((DTYPES MATCHES "fp8" AND DTYPES MATCHES "fp16") OR NOT DEFINED DTYPES)
  list(APPEND GROUPED_GEMM_FIXED_NK_INSTANCES device_grouped_gemm_xdl_fixed_nk_f16_fp8_f16_mk_kn_mn_instance.cpp)
  list(APPEND GROUPED_GEMM_FIXED_NK_INSTANCES device_grouped_gemm_xdl_fixed_nk_f16_fp8_f16_mk_nk_mn_instance.cpp)
endif()

if((DTYPES MATCHES "int8" AND DTYPES MATCHES "fp16") OR NOT DEFINED DTYPES)
  list(APPEND GROUPED_GEMM_FIXED_NK_INSTANCES device_grouped_gemm_xdl_fixed_nk_f16_i8_f16_mk_kn_mn_instance.cpp)
  list(APPEND GROUPED_GEMM_FIXED_NK_INSTANCES device_grouped_gemm_xdl_fixed_nk_f16_i8_f16_mk_nk_mn_instance.cpp)
endif()

add_instance_library(device_grouped_gemm_fixed_nk_instance ${GROUPED_GEMM_FIXED_NK_INSTANCES})