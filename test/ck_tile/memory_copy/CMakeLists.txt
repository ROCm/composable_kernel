if(GPU_TARGETS MATCHES "gfx950")
    add_gtest_executable(test_memory_copy test_copy.cpp)
endif()
