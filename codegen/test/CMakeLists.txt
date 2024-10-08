list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
add_subdirectory(rtc)
file(GLOB TEST_SRCS CONFIGURE_DEPENDS *.cpp)

find_package(hip)

foreach(TEST_SRC ${TEST_SRCS})
  get_filename_component(BASE_NAME ${TEST_SRC} NAME_WE)
  rocm_add_test_executable(codegen_test_${BASE_NAME} ${TEST_SRC})
  target_link_libraries(codegen_test_${BASE_NAME} ck_rtc ck_host hip::device)
  target_include_directories(codegen_test_${BASE_NAME} PUBLIC include)
  target_include_directories(codegen_test_${BASE_NAME} PUBLIC ${CK_ROOT}/include)
  target_include_directories(codegen_test_${BASE_NAME} PUBLIC ${CK_ROOT}/library/include)

endforeach()
