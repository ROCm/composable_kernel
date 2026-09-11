# Only build this test for gfx125
if(GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_mx_coexecution mx_wmma_coexecution.cpp)
    add_gtest_executable(test_mx_wmma_op
        mx_wmma_op_fp8.cpp
        mx_wmma_op_fp6.cpp
        mx_wmma_op_fp4.cpp
        mx_wmma_op_unscaled.cpp
        mx_wmma_op_isoscale_fp8.cpp
        mx_wmma_op_isoscale_fp6.cpp
        mx_wmma_op_isoscale_fp4.cpp
    )
    if(result EQUAL 0)
        target_link_libraries(test_mx_coexecution PRIVATE utility)
        target_link_libraries(test_mx_wmma_op PRIVATE utility)
    endif()
else()
    message(STATUS "test_mx_coexecution and test_mx_wmma_op are only supported on gfx125x architectures")
endif()
