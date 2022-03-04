include_directories(BEFORE
    include
    ${PROJECT_SOURCE_DIR}/host/host_tensor/include
    ${PROJECT_SOURCE_DIR}/host/device/include
    ${PROJECT_SOURCE_DIR}/device_operation/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/utility
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_description
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_operation
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/problem_transform
    ${PROJECT_SOURCE_DIR}/external/rocm/include
    ${PROJECT_SOURCE_DIR}/reference_operation/include
    ${PROJECT_SOURCE_DIR}/test/include
)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests)

function(add_test_executeable TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME} PRIVATE host_tensor)
    target_link_libraries(${TEST_NAME} PRIVATE device_gemm_instance)
    target_link_libraries(${TEST_NAME} PRIVATE device_conv2d_fwd_instance)
    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}> )
    add_dependencies(tests ${TEST_NAME})
    add_dependencies(check ${TEST_NAME})
endfunction(add_test_executeable TEST_NAME)


file(GLOB TESTS */*.cpp)

foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    message("adding test ${BASE_NAME}")
    add_test_executeable(test_${BASE_NAME} ${TEST})
endforeach(TEST ${TESTS})
