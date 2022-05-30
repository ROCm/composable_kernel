include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/ck
    ${PROJECT_SOURCE_DIR}/include/ck/utility
    ${PROJECT_SOURCE_DIR}/include/ck/host_utility
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_description
    ${PROJECT_SOURCE_DIR}/include/ck/tensor
    ${PROJECT_SOURCE_DIR}/include/ck/problem_transform
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/device
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/grid
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/block
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/warp
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/thread
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/element
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/host_tensor
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/tensor_operation_instance
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/tensor_operation_instance/gpu/reduce
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/reference_tensor_operation/cpu
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/reference_tensor_operation/gpu
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/utility
    ${PROJECT_SOURCE_DIR}/profiler/include
    ${PROJECT_SOURCE_DIR}/external/include/half
)

# ck_profiler
set(PROFILER_SOURCE
    src/profiler.cpp
    src/profile_gemm.cpp
    src/profile_gemm_bias_2d.cpp
    src/profile_gemm_bias_relu.cpp
    src/profile_gemm_bias_relu_add.cpp
    src/profile_gemm_reduce.cpp
    src/profile_batched_gemm.cpp
    src/profile_conv_fwd_bias_relu.cpp
    src/profile_conv_fwd_bias_relu_add.cpp
    src/profile_conv_fwd_bias_relu_atomic_add.cpp
    src/profile_convnd_fwd.cpp
    src/profile_convnd_bwd_data.cpp
    src/profile_reduce.cpp
    src/profile_grouped_gemm.cpp
    src/profile_conv_bwd_weight.cpp
    src/profile_batched_gemm_reduce.cpp
)

add_executable(ckProfiler ${PROFILER_SOURCE})

target_link_libraries(ckProfiler PRIVATE host_tensor)
target_link_libraries(ckProfiler PRIVATE conv_util)
target_link_libraries(ckProfiler PRIVATE device_gemm_reduce_instance)
target_link_libraries(ckProfiler PRIVATE device_gemm_instance)
target_link_libraries(ckProfiler PRIVATE device_gemm_bias2d_instance)
target_link_libraries(ckProfiler PRIVATE device_gemm_bias_relu_instance)
target_link_libraries(ckProfiler PRIVATE device_gemm_bias_relu_add_instance)
target_link_libraries(ckProfiler PRIVATE device_batched_gemm_instance)
target_link_libraries(ckProfiler PRIVATE device_conv1d_fwd_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_instance)
target_link_libraries(ckProfiler PRIVATE device_conv3d_fwd_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_bias_relu_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_bias_relu_add_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_bias_relu_atomic_add_instance)
target_link_libraries(ckProfiler PRIVATE device_convnd_bwd_data_instance)
target_link_libraries(ckProfiler PRIVATE device_reduce_instance)
target_link_libraries(ckProfiler PRIVATE device_grouped_gemm_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_bwd_weight_instance)
target_link_libraries(ckProfiler PRIVATE device_batched_gemm_reduce_instance)
