add_gtest_executable(test_tensor_slice_transfer test_tensor_slice_transfer.cpp)
if(result EQUAL 0)
  target_link_libraries(test_tensor_slice_transfer PRIVATE utility)
endif()
