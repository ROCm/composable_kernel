add_gtest_executable(test_coordinate_transform test_coordinate_transform.cpp)
if(result EQUAL 0)
    target_link_libraries(test_coordinate_transform PRIVATE utility)
endif()