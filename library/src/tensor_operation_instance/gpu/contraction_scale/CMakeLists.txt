set(DEVICE_CONTRACTION_SCALE_INSTANCES)

# FP32
list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_kkn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_knn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mkn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mnn_instance.cpp)

list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_f16_kkn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_f16_knn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_f16_mkn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_f16_mnn_instance.cpp)

list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_bf16_kkn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_bf16_knn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_bf16_mkn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_bf16_mnn_instance.cpp)

# FP64
list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_kkn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_knn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_mkn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_mnn_instance.cpp)

list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_compute_f32_kkn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_compute_f32_knn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_compute_f32_mkn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_compute_f32_mnn_instance.cpp)

# FP16
list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_compute_f32_kkn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_compute_f32_knn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_compute_f32_mkn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_compute_f32_mnn_instance.cpp)

# BF16
list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_kkn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_knn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mkn_instance.cpp
                                                device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mnn_instance.cpp)

add_instance_library(device_contraction_scale_instance ${DEVICE_CONTRACTION_SCALE_INSTANCES})

