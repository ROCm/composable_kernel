add_example_executable(example_gemm_bilinear_xdl_fp16 gemm_bilinear_xdl_fp16.cpp)
if(GPU_TARGETS MATCHES "gfx1100")
    add_example_executable(example_gemm_bilinear_wmma_fp16 gemm_bilinear_wmma_fp16.cpp)
endif()
