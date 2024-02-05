add_gtest_executable(test_wrapper_layout test_wrapper_layout.cpp)
target_link_libraries(test_wrapper_layout PRIVATE utility)
add_gtest_executable(test_wrapper_tensor test_wrapper_tensor.cpp)
target_link_libraries(test_wrapper_tensor PRIVATE utility)
add_gtest_executable(test_wrapper_copy test_wrapper_copy.cpp)
target_link_libraries(test_wrapper_copy PRIVATE utility)
add_gtest_executable(test_wrapper_partition test_wrapper_partition.cpp)
target_link_libraries(test_wrapper_partition PRIVATE utility)
if(GPU_TARGETS MATCHES "gfx908" OR GPU_TARGETS MATCHES "gfx90a" OR
   GPU_TARGETS MATCHES "gfx940" OR GPU_TARGETS MATCHES "gfx941" OR
   GPU_TARGETS MATCHES "gfx942")
    add_gtest_executable(test_wrapper_gemm test_wrapper_gemm.cpp)
    target_link_libraries(test_wrapper_gemm PRIVATE utility)
endif()
