include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/profiler/include
    ${PROJECT_SOURCE_DIR}/external/include/half
)

add_test_executable(test_convnd_bwd_data convnd_bwd_data.cpp)
target_link_libraries(test_convnd_bwd_data PRIVATE host_tensor)
target_link_libraries(test_convnd_bwd_data PRIVATE device_convnd_bwd_data_instance)
