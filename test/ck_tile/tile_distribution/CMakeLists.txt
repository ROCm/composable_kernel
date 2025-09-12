add_gtest_executable(test_tile_distribution test_tile_distribution.cpp)
if(result EQUAL 0)
    target_link_libraries(test_tile_distribution PRIVATE utility)
endif()