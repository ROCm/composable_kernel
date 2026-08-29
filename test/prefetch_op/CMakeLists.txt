if(SUPPORTED_GPU_TARGETS MATCHES "gfx125")
    add_test_executable(test_prefetch_op prefetch_op.cpp)
    target_link_libraries(test_prefetch_op PRIVATE utility)
endif()