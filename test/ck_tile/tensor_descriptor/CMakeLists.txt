add_gtest_executable(test_tensor_descriptor test_tensor_descriptor.cpp)
if(result EQUAL 0)
    target_link_libraries(test_tensor_descriptor PRIVATE utility)
endif()