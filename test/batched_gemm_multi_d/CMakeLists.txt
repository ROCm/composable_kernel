# TODO: Enable for gfx90a after complier fix
if(NOT GPU_TARGETS MATCHES "gfx90a")
    add_gtest_executable(test_batched_gemm_multi_d test_batched_gemm_multi_d.cpp)
    target_link_libraries(test_batched_gemm_multi_d PRIVATE utility device_batched_gemm_multi_d_instance)
endif()
