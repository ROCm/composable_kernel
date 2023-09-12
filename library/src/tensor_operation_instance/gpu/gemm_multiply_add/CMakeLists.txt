set(GEMM_MULTIPLY_ADD_INSTANCES)

if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
  list(APPEND GEMM_MULTIPLY_ADD_INSTANCES device_gemm_multiply_add_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instance.cpp)
  list(APPEND GEMM_MULTIPLY_ADD_INSTANCES device_gemm_multiply_add_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instance.cpp)
endif()

if((DTYPES MATCHES "fp16" AND DTYPES MATCHES "fp8") OR NOT DEFINED DTYPES)
  list(APPEND GEMM_MULTIPLY_ADD_INSTANCES device_gemm_multiply_add_xdl_c_shuffle_f16_f8_f32_f32_f16_mk_kn_mn_mn_mn_instance.cpp)
  list(APPEND GEMM_MULTIPLY_ADD_INSTANCES device_gemm_multiply_add_xdl_c_shuffle_f16_f8_f32_f32_f16_mk_nk_mn_mn_mn_instance.cpp)
endif()

add_instance_library(device_gemm_multiply_add_instance ${GEMM_MULTIPLY_ADD_INSTANCES})
