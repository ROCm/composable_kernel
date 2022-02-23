include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/host/host_tensor/include
    ${PROJECT_SOURCE_DIR}/host/device/include
    ${PROJECT_SOURCE_DIR}/device_operation/include
    ${PROJECT_SOURCE_DIR}/reference_operation/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/utility
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_description
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_operation
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/problem_transform
    ${PROJECT_SOURCE_DIR}/external/rocm/include
    ${PROJECT_SOURCE_DIR}/device_operation_reference/include
)

set(GEMM_XDL_SOURCE 1_gemm_xdl/gemm_xdl.cpp)
set(GEMM_XDL_BIAS_RELU_SOURCE 2_gemm_xdl_bias_relu/gemm_xdl_bias_relu.cpp)
set(GEMM_XDL_BIAS_RELU_ADD_SOURCE 3_gemm_xdl_bias_relu_add/gemm_xdl_bias_relu_add.cpp)
set(CONV2D_FWD_XDL_SOURCE 4_conv2d_fwd_xdl/conv2d_fwd_xdl.cpp)
set(CONV2D_FWD_XDL_BIAS_RELU_SOURCE 5_conv2d_fwd_xdl_bias_relu/conv2d_fwd_xdl_bias_relu.cpp)
set(CONV2D_FWD_XDL_BIAS_RELU_ADD_SOURCE 6_conv2d_fwd_xdl_bias_relu_add/conv2d_fwd_xdl_bias_relu_add.cpp)
set(CONV2D_FWD_XDL_BIAS_RELU_ATOMIC_ADD_SOURCE 7_conv2d_fwd_xdl_bias_relu_atomic_add/conv2d_fwd_xdl_bias_relu_atomic_add.cpp)
set(GEMM_XDL_ALPHA_BETA_SOURCE 8_gemm_xdl_alpha_beta/gemm_xdl_alpha_beta.cpp)
set(CONV2D_FWD_XDL_INT8_SOURCE 9_conv2d_fwd_xdl_int8/conv2d_fwd_xdl_int8.cpp)
set(CONV3D_FWD_XDL_SOURCE 10_conv3d_fwd_xdl/conv3d_fwd_xdl.cpp)

add_executable(gemm_xdl ${GEMM_XDL_SOURCE})
add_executable(gemm_xdl_bias_relu ${GEMM_XDL_BIAS_RELU_SOURCE})
add_executable(gemm_xdl_bias_relu_add ${GEMM_XDL_BIAS_RELU_ADD_SOURCE})
add_executable(conv2d_fwd_xdl ${CONV2D_FWD_XDL_SOURCE})
add_executable(conv2d_fwd_xdl_bias_relu ${CONV2D_FWD_XDL_BIAS_RELU_SOURCE})
add_executable(conv2d_fwd_xdl_bias_relu_add ${CONV2D_FWD_XDL_BIAS_RELU_ADD_SOURCE})
add_executable(conv2d_fwd_xdl_bias_relu_atomic_add ${CONV2D_FWD_XDL_BIAS_RELU_ATOMIC_ADD_SOURCE})
add_executable(gemm_xdl_alpha_beta ${GEMM_XDL_ALPHA_BETA_SOURCE})
add_executable(conv2d_fwd_xdl_int8 ${CONV2D_FWD_XDL_INT8_SOURCE})
add_executable(conv3d_fwd_xdl ${CONV3D_FWD_XDL_SOURCE})

target_link_libraries(gemm_xdl PRIVATE host_tensor)
target_link_libraries(gemm_xdl_bias_relu PRIVATE host_tensor)
target_link_libraries(gemm_xdl_bias_relu_add PRIVATE host_tensor)
target_link_libraries(conv2d_fwd_xdl PRIVATE host_tensor)
target_link_libraries(conv2d_fwd_xdl_bias_relu PRIVATE host_tensor)
target_link_libraries(conv2d_fwd_xdl_bias_relu_add PRIVATE host_tensor)
target_link_libraries(conv2d_fwd_xdl_bias_relu_atomic_add PRIVATE host_tensor)
target_link_libraries(gemm_xdl_alpha_beta PRIVATE host_tensor)
target_link_libraries(conv2d_fwd_xdl_int8 PRIVATE host_tensor)
target_link_libraries(conv3d_fwd_xdl PRIVATE host_tensor)

