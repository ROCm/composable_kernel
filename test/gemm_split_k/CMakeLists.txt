if(GPU_TARGETS MATCHES "gfx908" OR GPU_TARGETS MATCHES "gfx90a" OR GPU_TARGETS MATCHES "gfx940")
   add_gtest_executable(test_gemm_splitk test_gemm_splitk.cpp)
   target_link_libraries(test_gemm_splitk PRIVATE utility device_gemm_splitk_instance)
endif()
