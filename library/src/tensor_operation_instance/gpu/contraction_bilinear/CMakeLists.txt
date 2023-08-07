set(DEVICE_CONTRACTION_BILINEAR_INSTANCES)
if(DTYPES MATCHES "fp32" OR NOT DEFINED DTYPES)
    #float
    list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance.cpp
                                                      device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_knnn_instance.cpp
                                                      device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_mknn_instance.cpp
                                                      device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_mnnn_instance.cpp)
endif()
if(DTYPES MATCHES "fp64" OR NOT DEFINED DTYPES)
    #double
    list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_kknn_instance.cpp
                                                      device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_knnn_instance.cpp
                                                      device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_mknn_instance.cpp
                                                      device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_mnnn_instance.cpp)
endif()
add_instance_library(device_contraction_bilinear_instance ${DEVICE_CONTRACTION_BILINEAR_INSTANCES})

