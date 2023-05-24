if(GPU_TARGETS MATCHES "gfx908" OR GPU_TARGETS MATCHES "gfx90a" OR GPU_TARGETS MATCHES "gfx940")
    add_example_executable(example_gemm_bias_relu_xdl_fp16 gemm_bias_relu_xdl_fp16.cpp)
endif()