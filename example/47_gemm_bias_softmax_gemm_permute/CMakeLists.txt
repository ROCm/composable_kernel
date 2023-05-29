if(GPU_TARGETS MATCHES "gfx908" OR GPU_TARGETS MATCHES "gfx90a" OR GPU_TARGETS MATCHES "gfx940")
   add_example_executable(example_gemm_bias_softmax_gemm_permute gemm_bias_softmax_gemm_permute.cpp)
endif()
