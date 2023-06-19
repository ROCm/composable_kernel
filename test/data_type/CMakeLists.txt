if (USE_BITINT_EXTENSION_INT4)
  add_gtest_executable(test_int4 int4.cpp)
  target_link_libraries(test_int4 PRIVATE utility)
endif()

add_gtest_executable(test_fp8 fp8.cpp)
target_link_libraries(test_fp8 PRIVATE utility)
