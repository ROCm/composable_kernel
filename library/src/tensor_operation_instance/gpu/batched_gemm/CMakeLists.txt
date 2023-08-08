set(BATCHED_GEMM_INSTANCES)
if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
   list(APPEND BATCHED_GEMM_INSTANCES device_batched_gemm_xdl_f16_f16_f16_gmk_gkn_gmn_instance.cpp
                                      device_batched_gemm_xdl_f16_f16_f16_gmk_gnk_gmn_instance.cpp
                                      device_batched_gemm_xdl_f16_f16_f16_gkm_gkn_gmn_instance.cpp
                                      device_batched_gemm_xdl_f16_f16_f16_gkm_gnk_gmn_instance.cpp)
endif()
if(DTYPES MATCHES "bf16" OR NOT DEFINED DTYPES)
   list(APPEND BATCHED_GEMM_INSTANCES device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gkn_gmn_instance.cpp
                                      device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gnk_gmn_instance.cpp
                                      device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gkn_gmn_instance.cpp
                                      device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gnk_gmn_instance.cpp)
endif()
if(DTYPES MATCHES "fp32" OR NOT DEFINED DTYPES)
   list(APPEND BATCHED_GEMM_INSTANCES device_batched_gemm_xdl_f32_f32_f32_gmk_gkn_gmn_instance.cpp
                                      device_batched_gemm_xdl_f32_f32_f32_gmk_gnk_gmn_instance.cpp
                                      device_batched_gemm_xdl_f32_f32_f32_gkm_gkn_gmn_instance.cpp
                                      device_batched_gemm_xdl_f32_f32_f32_gkm_gnk_gmn_instance.cpp)
endif()
if(DTYPES MATCHES "int8" OR NOT DEFINED DTYPES)
   list(APPEND BATCHED_GEMM_INSTANCES device_batched_gemm_xdl_int8_int8_int8_gmk_gkn_gmn_instance.cpp
                                      device_batched_gemm_xdl_int8_int8_int8_gmk_gnk_gmn_instance.cpp
                                      device_batched_gemm_xdl_int8_int8_int8_gkm_gkn_gmn_instance.cpp
                                      device_batched_gemm_xdl_int8_int8_int8_gkm_gnk_gmn_instance.cpp)
endif()
add_instance_library(device_batched_gemm_instance ${BATCHED_GEMM_INSTANCES})
