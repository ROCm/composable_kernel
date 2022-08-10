include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/
)

# ck_profiler
set(PROFILER_SOURCE
    src/profiler.cpp
    src/profile_gemm.cpp
    src/profile_gemm_splitk.cpp
    src/profile_gemm_bilinear.cpp
    src/profile_gemm_bias_add_reduce.cpp
    src/profile_gemm_add_add_fastgelu.cpp
    src/profile_gemm_reduce.cpp
    src/profile_batched_gemm.cpp
    src/profile_batched_gemm_reduce.cpp
    src/profile_grouped_gemm.cpp
    src/profile_conv_fwd.cpp
    src/profile_conv_fwd_bias_relu.cpp
    src/profile_conv_fwd_bias_relu_add.cpp
    src/profile_conv_bwd_data.cpp
    src/profile_conv_bwd_weight.cpp
    src/profile_grouped_conv_fwd.cpp
    src/profile_reduce.cpp
    src/profile_normalization.cpp
)

add_executable(ckProfiler ${PROFILER_SOURCE})

target_link_libraries(ckProfiler PRIVATE utility)
target_link_libraries(ckProfiler PRIVATE device_gemm_instance)
target_link_libraries(ckProfiler PRIVATE device_gemm_splitk_instance)
target_link_libraries(ckProfiler PRIVATE device_gemm_bilinear_instance)
target_link_libraries(ckProfiler PRIVATE device_gemm_add_add_fastgelu_instance)
target_link_libraries(ckProfiler PRIVATE device_gemm_reduce_instance)
target_link_libraries(ckProfiler PRIVATE device_gemm_bias_add_reduce_instance)
target_link_libraries(ckProfiler PRIVATE device_batched_gemm_instance)
target_link_libraries(ckProfiler PRIVATE device_batched_gemm_reduce_instance)
target_link_libraries(ckProfiler PRIVATE device_grouped_gemm_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_instance)
target_link_libraries(ckProfiler PRIVATE device_grouped_conv1d_fwd_instance)
target_link_libraries(ckProfiler PRIVATE device_grouped_conv2d_fwd_instance)
target_link_libraries(ckProfiler PRIVATE device_grouped_conv3d_fwd_instance)
target_link_libraries(ckProfiler PRIVATE device_conv1d_bwd_data_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_bwd_data_instance)
target_link_libraries(ckProfiler PRIVATE device_conv3d_bwd_data_instance)
target_link_libraries(ckProfiler PRIVATE device_conv1d_bwd_weight_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_bwd_weight_instance)
target_link_libraries(ckProfiler PRIVATE device_conv3d_bwd_weight_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_bias_relu_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_bias_relu_add_instance)
target_link_libraries(ckProfiler PRIVATE device_normalization_instance)
target_link_libraries(ckProfiler PRIVATE device_reduce_instance)
