set(DEVICE_CONTRACTION_SCALE_INSTANCES)
if(DTYPES MATCHES "fp32" OR NOT DEFINED DTYPES)
    #float
    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_kkn_instance.cpp
                                                   device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_knn_instance.cpp
                                                   device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mkn_instance.cpp
                                                   device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mnn_instance.cpp)
endif()
if(DTYPES MATCHES "fp64" OR NOT DEFINED DTYPES)
    #double
    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_kkn_instance.cpp
                                                   device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_knn_instance.cpp
                                                   device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_mkn_instance.cpp
                                                   device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_mnn_instance.cpp)
endif()
add_instance_library(device_contraction_scale_instance ${DEVICE_CONTRACTION_SCALE_INSTANCES})

