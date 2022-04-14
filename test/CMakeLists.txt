include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/ck
    ${PROJECT_SOURCE_DIR}/include/ck/utility
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_description
    ${PROJECT_SOURCE_DIR}/include/ck/tensor
    ${PROJECT_SOURCE_DIR}/include/ck/problem_transform
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/device
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/grid
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/block
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/warp
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/thread
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/element
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/host_tensor
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/tensor_operation_instance
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/tensor_operation_instance/gpu/reduce
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/reference_tensor_operation/cpu
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/reference_tensor_operation/gpu
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/utility
    ${PROJECT_SOURCE_DIR}/test/include
    ${PROJECT_SOURCE_DIR}/profiler/include
    ${PROJECT_SOURCE_DIR}/external/include/half
)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests)

function(add_test_executable TEST_NAME)
    message("adding test ${TEST_NAME}")
    add_executable(${TEST_NAME} ${ARGN})
    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}> )
    add_dependencies(tests ${TEST_NAME})
    add_dependencies(check ${TEST_NAME})
endfunction(add_test_executable TEST_NAME)

add_subdirectory(magic_number_division)
add_subdirectory(space_filling_curve)
add_subdirectory(conv_util)
add_subdirectory(reference_conv_fwd)
add_subdirectory(gemm)
add_subdirectory(gemm_dlops)
add_subdirectory(gemm_split_k)
add_subdirectory(gemm_reduce)
add_subdirectory(batched_gemm)
add_subdirectory(batched_gemm_reduce)
add_subdirectory(grouped_gemm)
add_subdirectory(convnd_fwd)
add_subdirectory(reduce)
add_subdirectory(conv2d_bwd_weight)
