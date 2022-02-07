include_directories(BEFORE
    include
    ${PROJECT_SOURCE_DIR}/host/host_tensor/include
    ${PROJECT_SOURCE_DIR}/device/include
    ${PROJECT_SOURCE_DIR}/device_operation/include
    ${PROJECT_SOURCE_DIR}/reference_operation/include
    ${PROJECT_SOURCE_DIR}/profiler/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/utility
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_description
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_operation
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/problem_transform
    ${PROJECT_SOURCE_DIR}/external/rocm/include
)

# ck_profiler
set(PROFILER_SOURCE 
    src/profiler.cpp
    src/profile_gemm.cpp
    src/profile_gemm_bias_relu.cpp
    src/profile_gemm_bias_relu_add.cpp
#   src/profile_conv_fwd.cpp
#   src/profile_conv_fwd_bias_relu.cpp
#   src/profile_conv_fwd_bias_relu_add.cpp
#   src/profile_conv_fwd_bias_relu_atomic_add.cpp
)

add_executable(ckProfiler ${PROFILER_SOURCE})

target_link_libraries(ckProfiler PRIVATE host_tensor)
target_link_libraries(ckProfiler PRIVATE device_gemm_instance)
target_link_libraries(ckProfiler PRIVATE device_gemm_bias_relu_instance)
target_link_libraries(ckProfiler PRIVATE device_gemm_bias_relu_add_instance)
#target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_instance)
#target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_bias_relu_instance)
#target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_bias_relu_add_instance)
#target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_bias_relu_atomic_add_instance)
