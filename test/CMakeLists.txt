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

