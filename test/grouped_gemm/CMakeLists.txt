add_custom_target(test_grouped_gemm)

add_gtest_executable(test_grouped_gemm_splitk_xdl test_grouped_gemm_splitk_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_grouped_gemm_splitk_xdl PRIVATE utility device_grouped_gemm_instance)
    add_dependencies(test_grouped_gemm test_grouped_gemm_splitk_xdl)
endif()

add_gtest_executable(test_grouped_gemm_splitk_wmma test_grouped_gemm_splitk_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_grouped_gemm_splitk_wmma PRIVATE utility device_grouped_gemm_instance)
    add_dependencies(test_grouped_gemm test_grouped_gemm_splitk_wmma)
endif()

add_gtest_executable(test_grouped_gemm_interface test_grouped_gemm_interface_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_grouped_gemm_interface PRIVATE utility device_grouped_gemm_instance)
    add_dependencies(test_grouped_gemm test_grouped_gemm_interface)
endif()
