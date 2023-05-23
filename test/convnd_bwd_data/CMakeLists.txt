if(GPU_TARGETS MATCHES "gfx908" OR GPU_TARGETS MATCHES "gfx90a" OR GPU_TARGETS MATCHES "gfx940")
   add_gtest_executable(test_convnd_bwd_data convnd_bwd_data.cpp)
   target_link_libraries(test_convnd_bwd_data PRIVATE utility device_conv1d_bwd_data_instance device_conv2d_bwd_data_instance device_conv3d_bwd_data_instance)
endif()