set(DEVICE_CONTRACTION_SCALE_INSTANCES)

foreach(idx RANGE 2 5)
    set(DIM ${idx}D) # 2D, 3D, 4D, 5D
    set(DIMS_MNK m${idx}_n${idx}_k${idx}) # m2_n2_k2, m3_n3_k3, m4_n4_k4, m5_n5_k5
    # FP32
    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f32_f32_f32_kkn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f32_f32_f32_knn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f32_f32_f32_mkn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f32_f32_f32_mnn_instance.cpp)

    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f32_f32_f32_compute_f16_kkn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f32_f32_f32_compute_f16_knn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f32_f32_f32_compute_f16_mkn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f32_f32_f32_compute_f16_mnn_instance.cpp)

    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f32_f32_f32_compute_bf16_kkn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f32_f32_f32_compute_bf16_knn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f32_f32_f32_compute_bf16_mkn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f32_f32_f32_compute_bf16_mnn_instance.cpp)

    # FP64
    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f64_f64_f64_kkn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f64_f64_f64_knn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f64_f64_f64_mkn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f64_f64_f64_mnn_instance.cpp)

    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f64_f64_f64_compute_f32_kkn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f64_f64_f64_compute_f32_knn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f64_f64_f64_compute_f32_mkn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f64_f64_f64_compute_f32_mnn_instance.cpp)

    # FP16
    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f16_f16_f16_compute_f32_kkn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f16_f16_f16_compute_f32_knn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f16_f16_f16_compute_f32_mkn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_f16_f16_f16_compute_f32_mnn_instance.cpp)

    # BF16
    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_kkn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_knn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mkn_instance.cpp
                                                    ${DIM}/device_contraction_scale_${DIMS_MNK}_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mnn_instance.cpp)
endforeach()

add_instance_library(device_contraction_scale_instance ${DEVICE_CONTRACTION_SCALE_INSTANCES})

