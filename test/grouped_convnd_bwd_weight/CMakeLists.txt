if(GPU_TARGETS MATCHES "gfx908" OR GPU_TARGETS MATCHES "gfx90a" OR GPU_TARGETS MATCHES "gfx940")
   add_gtest_executable(test_grouped_convnd_bwd_weight grouped_convnd_bwd_weight.cpp)
    target_link_libraries(test_grouped_convnd_bwd_weight PRIVATE utility device_grouped_conv1d_bwd_weight_instance device_grouped_conv2d_bwd_weight_instance device_grouped_conv3d_bwd_weight_instance)
endif()