# temporarily disable flaky test for all architectures
add_definitions(-DCK_SKIP_FLAKY_F8_TEST)
set(CK_SKIP_FLAKY_F8_TEST "ON")

if (USE_BITINT_EXTENSION_INT4)
  add_gtest_executable(test_int4 test_int4.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_int4 PRIVATE utility)
  endif()
endif()

add_gtest_executable(test_fp8 test_fp8.cpp)
if(result EQUAL 0)
  target_link_libraries(test_fp8 PRIVATE utility)
endif()
add_gtest_executable(test_bf8 test_bf8.cpp)
if(result EQUAL 0)
  target_link_libraries(test_bf8 PRIVATE utility)
endif()

add_gtest_executable(test_custom_type test_custom_type.cpp)
if(result EQUAL 0)
  target_link_libraries(test_custom_type PRIVATE utility)
endif()

add_gtest_executable(test_type_convert_const type_convert_const.cpp)
