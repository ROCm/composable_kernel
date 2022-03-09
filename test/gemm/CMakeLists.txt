set(GEMM_SOURCE
    gemm_fp32.cpp
    gemm_bf16.cpp
    gemm_int8.cpp
)

foreach(TEST ${GEMM_SOURCE})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    message("adding test test_${BASE_NAME}")
    add_test_executable(test_${BASE_NAME} ${TEST})
    target_link_libraries(test_${BASE_NAME} PRIVATE host_tensor)
    target_link_libraries(test_${BASE_NAME} PRIVATE device_gemm_instance)
endforeach(TEST ${GEMM_SOURCE})
