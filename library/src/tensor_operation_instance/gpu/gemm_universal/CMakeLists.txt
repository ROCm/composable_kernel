set(GEMM_UNIVERSAL_INSTANCES)

list(APPEND GEMM_UNIVERSAL_INSTANCES 
        device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_instance.cpp
        device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_instance.cpp)

add_instance_library(device_gemm_universal_instance ${GEMM_UNIVERSAL_INSTANCES})
