set(DEVICE_CONTRACTION_BILINEAR_INSTANCES)
#float
list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance.cpp
                                                  device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_knnn_instance.cpp
                                                  device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_mknn_instance.cpp
                                                  device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_mnnn_instance.cpp)

#double
list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_kknn_instance.cpp
                                                  device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_knnn_instance.cpp
                                                  device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_mknn_instance.cpp
                                                  device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_mnnn_instance.cpp)

add_instance_library(device_contraction_bilinear_instance ${DEVICE_CONTRACTION_BILINEAR_INSTANCES})
