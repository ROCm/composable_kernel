if (USE_BITINT_EXTENSION_INT4)
  add_gtest_executable(test_int4 int4.cpp)
  target_link_libraries(test_int4 PRIVATE utility)
endif()

if(DTYPES MATCHES "fp8" OR NOT DEFINED DTYPES)
  add_gtest_executable(test_f8 f8.cpp)
  target_link_libraries(test_f8 PRIVATE utility)
endif()

if(DTYPES MATCHES "bf8" OR NOT DEFINED DTYPES)
  add_gtest_executable(test_bf8 bf8.cpp)
  target_link_libraries(test_bf8 PRIVATE utility)
endif()
