if(GPU_TARGETS MATCHES "gfx908" OR GPU_TARGETS MATCHES "gfx90a" OR GPU_TARGETS MATCHES "gfx940")
   add_gtest_executable(test_convnd_fwd convnd_fwd.cpp)
   target_link_libraries(test_convnd_fwd PRIVATE utility device_conv2d_fwd_instance)
endif()
