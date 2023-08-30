if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
add_instance_library(device_grouped_gemm_splitk_instance
   device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_instance.cpp
   device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_irregular_instance.cpp
)
endif()
