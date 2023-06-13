if(GPU_TARGETS MATCHES "gfx908" OR GPU_TARGETS MATCHES "gfx90a" OR GPU_TARGETS MATCHES "gfx940")
   add_custom_target(test_grouped_gemm)
   add_gtest_executable(test_grouped_gemm_splitk test_grouped_gemm_splitk.cpp)
   add_gtest_executable(test_grouped_gemm_interface test_grouped_gemm_interface.cpp)
   target_link_libraries(test_grouped_gemm_splitk PRIVATE utility device_grouped_gemm_instance)
   target_link_libraries(test_grouped_gemm_interface PRIVATE utility device_grouped_gemm_instance)
   
   add_dependencies(test_grouped_gemm test_grouped_gemm_splitk test_grouped_gemm_interface)
endif()
