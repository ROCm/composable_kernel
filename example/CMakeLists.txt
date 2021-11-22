include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/host/host_tensor/include
    ${PROJECT_SOURCE_DIR}/host/device/include
    ${PROJECT_SOURCE_DIR}/device_operation/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/utility
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_description
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_operation
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/problem_transform
    ${PROJECT_SOURCE_DIR}/external/rocm/include
)

set(GEMM_XDL_SOURCE 1_gemm_xdl/gemm_xdl.cpp)
set(GEMM_XDL_BIAS_ADD_SOURCE 2_gemm_xdl_bias_add/gemm_xdl_bias_add.cpp)

add_executable(gemm_xdl ${GEMM_XDL_SOURCE})
add_executable(gemm_xdl_bias_add ${GEMM_XDL_BIAS_ADD_SOURCE})

target_link_libraries(gemm_xdl PRIVATE host_tensor)
target_link_libraries(gemm_xdl_bias_add PRIVATE host_tensor)
