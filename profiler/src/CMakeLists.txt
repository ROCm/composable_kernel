# ck_profiler
set(PROFILER_SOURCE
    main.cpp
    profile_gemm.cpp
    profile_gemm_splitk.cpp
    profile_gemm_bilinear.cpp
    profile_gemm_bias_add_reduce.cpp
    profile_gemm_add_add_fastgelu.cpp
    profile_gemm_reduce.cpp
    profile_batched_gemm.cpp
    profile_batched_gemm_gemm.cpp
    profile_batched_gemm_add_relu_gemm_add.cpp
    profile_batched_gemm_reduce.cpp
    profile_grouped_gemm.cpp
    profile_conv_fwd.cpp
    profile_conv_fwd_bias_relu.cpp
    profile_conv_fwd_bias_relu_add.cpp
    profile_conv_bwd_data.cpp
    profile_grouped_conv_fwd.cpp
    profile_grouped_conv_bwd_weight.cpp
    profile_reduce.cpp
    profile_groupnorm.cpp
    profile_layernorm.cpp
    profile_softmax.cpp
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
target_link_libraries(ckProfiler PRIVATE device_batched_gemm_gemm_instance)
target_link_libraries(ckProfiler PRIVATE device_batched_gemm_add_relu_gemm_add_instance)
target_link_libraries(ckProfiler PRIVATE device_batched_gemm_reduce_instance)
target_link_libraries(ckProfiler PRIVATE device_grouped_gemm_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_instance)
target_link_libraries(ckProfiler PRIVATE device_grouped_conv1d_fwd_instance)
target_link_libraries(ckProfiler PRIVATE device_grouped_conv2d_fwd_instance)
target_link_libraries(ckProfiler PRIVATE device_grouped_conv3d_fwd_instance)
target_link_libraries(ckProfiler PRIVATE device_conv1d_bwd_data_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_bwd_data_instance)
target_link_libraries(ckProfiler PRIVATE device_conv3d_bwd_data_instance)
target_link_libraries(ckProfiler PRIVATE device_grouped_conv1d_bwd_weight_instance)
target_link_libraries(ckProfiler PRIVATE device_grouped_conv2d_bwd_weight_instance)
target_link_libraries(ckProfiler PRIVATE device_grouped_conv3d_bwd_weight_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_bias_relu_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_bias_relu_add_instance)
target_link_libraries(ckProfiler PRIVATE device_normalization_instance)
target_link_libraries(ckProfiler PRIVATE device_softmax_instance)
target_link_libraries(ckProfiler PRIVATE device_reduce_instance)

rocm_install(TARGETS ckProfiler COMPONENT profiler)
