if(GPU_TARGETS MATCHES "gfx908" OR GPU_TARGETS MATCHES "gfx90a" OR GPU_TARGETS MATCHES "gfx940")
   add_example_executable(example_batched_gemm_reduce_xdl_fp16 batched_gemm_reduce_xdl_fp16.cpp)
endif()

