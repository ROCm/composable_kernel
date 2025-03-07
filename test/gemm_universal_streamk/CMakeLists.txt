add_gtest_executable(test_gemm_universal_streamk test_gemm_universal_streamk_xdl.cpp)
if(result EQUAL 0)
   target_link_libraries(test_gemm_universal_streamk PRIVATE utility device_gemm_universal_streamk_instance)
 endif()
