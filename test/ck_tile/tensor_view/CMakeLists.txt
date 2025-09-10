add_gtest_executable(test_tensor_view test_tensor_view.cpp)
if(result EQUAL 0)
    target_link_libraries(test_tensor_view PRIVATE utility)
endif()