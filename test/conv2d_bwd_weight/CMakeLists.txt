include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/profiler/include
    ${PROJECT_SOURCE_DIR}/external/include/half
)

add_test_executable(test_conv2d_bwd_weight conv2d_bwd_weight.cpp)
target_link_libraries(test_conv2d_bwd_weight PRIVATE host_tensor)
target_link_libraries(test_conv2d_bwd_weight PRIVATE device_conv2d_bwd_weight_instance)
