set(GEMM_SPLIT_K_SOURCE
    gemm_split_k.cpp
)

get_filename_component(GEMM_SPLIT_K ${GEMM_SPLIT_K_SOURCE} NAME_WE)
add_test_executable(test_${GEMM_SPLIT_K} ${GEMM_SPLIT_K_SOURCE})
target_link_libraries(test_${GEMM_SPLIT_K} PRIVATE host_tensor)
target_link_libraries(test_${GEMM_SPLIT_K} PRIVATE device_gemm_instance)
