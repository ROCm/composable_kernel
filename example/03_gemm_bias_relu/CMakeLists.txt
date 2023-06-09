list(APPEND gpu_list gfx908 gfx90a gfx940 gfx941 gfx942)
if(GPU_TARGETS IN_LIST gpu_list)
    add_example_executable(example_gemm_bias_relu_xdl_fp16 gemm_bias_relu_xdl_fp16.cpp)
endif()
