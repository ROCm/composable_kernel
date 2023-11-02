set(GEMM_SPLITK_INSTANCES)

list(APPEND GEMM_SPLITK_INSTANCES device_gemm_xdl_splitk_f32_f32_f32_mk_kn_mn_instance.cpp
                                  device_gemm_xdl_splitk_f32_f32_f32_mk_nk_mn_instance.cpp
                                  device_gemm_xdl_splitk_f32_f32_f32_km_kn_mn_instance.cpp
                                  device_gemm_xdl_splitk_f32_f32_f32_km_nk_mn_instance.cpp
                                  device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_instance.cpp
                                  device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_instance.cpp
                                  device_gemm_xdl_splitk_f16_f16_f16_km_kn_mn_instance.cpp
                                  device_gemm_xdl_splitk_f16_f16_f16_km_nk_mn_instance.cpp
                                  device_gemm_xdl_splitk_fp8_f16_f16_mk_kn_mn_instance.cpp
                                  device_gemm_xdl_splitk_fp8_f16_f16_mk_nk_mn_instance.cpp
                                  device_gemm_xdl_splitk_fp8_f16_f16_km_kn_mn_instance.cpp
                                  device_gemm_xdl_splitk_fp8_f16_f16_km_nk_mn_instance.cpp
                                  device_gemm_xdl_splitk_f16_fp8_f16_mk_kn_mn_instance.cpp
                                  device_gemm_xdl_splitk_f16_fp8_f16_mk_nk_mn_instance.cpp
                                  device_gemm_xdl_splitk_f16_fp8_f16_km_kn_mn_instance.cpp
                                  device_gemm_xdl_splitk_f16_fp8_f16_km_nk_mn_instance.cpp
                                  device_gemm_xdl_splitk_f16_f16_f16_comp_fp8_mk_kn_mn_instance.cpp
                                  device_gemm_xdl_splitk_f16_f16_f16_comp_fp8_mk_nk_mn_instance.cpp
                                  device_gemm_xdl_splitk_f16_f16_f16_comp_fp8_km_kn_mn_instance.cpp
                                  device_gemm_xdl_splitk_f16_f16_f16_comp_fp8_km_nk_mn_instance.cpp)

add_instance_library(device_gemm_splitk_instance ${GEMM_SPLITK_INSTANCES})
