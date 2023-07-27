# TODO: Enable for gfx90a after complier fix
if(DL_KERNELS)
  add_gtest_executable(test_batched_gemm_multi_d test_batched_gemm_multi_d.cpp)
  target_link_libraries(test_batched_gemm_multi_d PRIVATE utility device_batched_gemm_multi_d_instance)
endif()
