include_directories(BEFORE
    include
    ${PROJECT_SOURCE_DIR}/host/host_tensor/include
    ${PROJECT_SOURCE_DIR}/host/device/include
    ${PROJECT_SOURCE_DIR}/device_operation/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/utility
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_description
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_operation
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/problem_transform
    ${PROJECT_SOURCE_DIR}/external/rocm/include
    ${PROJECT_SOURCE_DIR}/reference_operation/include
    ${PROJECT_SOURCE_DIR}/test/include
)

# test_magic_number_division
set(MAGIC_NUMBER_DIVISISON_SOURCE magic_number_division/main.cpp)
add_executable(test_magic_number_division ${MAGIC_NUMBER_DIVISISON_SOURCE})
target_link_libraries(test_magic_number_division PRIVATE host_tensor)


set(CONV2D_FWD_SOURCE conv2d_fwd/main.cpp)

add_executable(test_conv2d_fwd ${CONV2D_FWD_SOURCE})
target_link_libraries(test_conv2d_fwd PRIVATE host_tensor)
target_link_libraries(test_conv2d_fwd PRIVATE device_conv2d_fwd_instance)

# test_split_k
set(SPLIT_K_SOURCE split_k/main.cpp)
add_executable(test_split_k ${SPLIT_K_SOURCE})
target_link_libraries(test_split_k PRIVATE host_tensor)
target_link_libraries(test_split_k PRIVATE device_gemm_instance)

# test_conv_util
set(CONV_UTIL_SOURCE conv_util/main.cpp)
add_executable(test_conv_util ${CONV_UTIL_SOURCE})
target_link_libraries(test_conv_util PRIVATE host_tensor)

# test_reference_conv_fwd
set(REFERENCE_CONV_FWD_SOURCE reference_conv_fwd/main.cpp)
add_executable(test_reference_conv_fwd ${REFERENCE_CONV_FWD_SOURCE})
target_link_libraries(test_reference_conv_fwd PRIVATE host_tensor)

# test_convnd_fwd_xdl
set(CONVND_FWD_XDL_SOURCE convnd_fwd_xdl/main.cpp)
add_executable(test_convnd_fwd_xdl ${CONVND_FWD_XDL_SOURCE})
target_link_libraries(test_convnd_fwd_xdl PRIVATE host_tensor)

# test_gemm_xdl_fp32
set(GEMM_XDL_FP32_SOURCE gemm_xdl/test_gemm_fp32.cpp)
add_executable(test_gemm_xdl_fp32 ${GEMM_XDL_FP32_SOURCE})
target_link_libraries(test_gemm_xdl_fp32 PRIVATE host_tensor)
target_link_libraries(test_gemm_xdl_fp32 PRIVATE device_gemm_instance)

# test_gemm_xdl_bf16
set(GEMM_XDL_BF16_SOURCE gemm_xdl/test_gemm_bf16.cpp)
add_executable(test_gemm_xdl_bf16 ${GEMM_XDL_BF16_SOURCE})
target_link_libraries(test_gemm_xdl_bf16 PRIVATE host_tensor)
target_link_libraries(test_gemm_xdl_bf16 PRIVATE device_gemm_instance)

# test_gemm_xdl_int8
set(GEMM_XDL_INT8_SOURCE gemm_xdl/test_gemm_int8.cpp)
add_executable(test_gemm_xdl_int8 ${GEMM_XDL_INT8_SOURCE})
target_link_libraries(test_gemm_xdl_int8 PRIVATE host_tensor)
target_link_libraries(test_gemm_xdl_int8 PRIVATE device_gemm_instance)

# test space_filling_curve_
set(SPACE_FILLING_CURVE_SOURCE space_filling_curve/space_filling_curve.cpp)
add_executable(space_filling_curve ${SPACE_FILLING_CURVE_SOURCE})
target_link_libraries(space_filling_curve PRIVATE host_tensor)
