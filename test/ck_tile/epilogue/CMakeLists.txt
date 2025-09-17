add_gtest_executable(test_cshuffle_epilogue test_cshuffle_epilogue.cpp)
if(result EQUAL 0)
    target_link_libraries(test_cshuffle_epilogue PRIVATE utility)
endif()