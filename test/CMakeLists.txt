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
)

# test_magic_number_division
set(MAGIC_NUMBER_DIVISISON_SOURCE magic_number_division/main.cpp)
add_executable(test_magic_number_division ${MAGIC_NUMBER_DIVISISON_SOURCE})
target_link_libraries(test_magic_number_division PRIVATE host_tensor)

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
