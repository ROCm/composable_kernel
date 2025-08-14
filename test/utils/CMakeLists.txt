add_gtest_executable(test_sequencies test_sequencies.cpp)
if(result EQUAL 0)
  target_link_libraries(test_sequencies PRIVATE utility)
endif()
