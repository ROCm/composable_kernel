if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
add_instance_library(device_batched_gemm_gemm_instance
    device_batched_gemm_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp
    device_batched_gemm_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gon_gmo_instance.cpp
)
endif()
