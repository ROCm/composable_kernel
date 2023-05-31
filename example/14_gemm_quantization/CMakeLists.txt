# dlops
add_example_executable(example_gemm_dl_quantization_int8 gemm_dl_quantization_int8.cpp)

# xdlops
if(GPU_TARGETS MATCHES "gfx908" OR GPU_TARGETS MATCHES "gfx90a" OR GPU_TARGETS MATCHES "gfx940")
   add_example_executable(example_gemm_xdl_bias_relu_quantization_int8 gemm_xdl_bias_relu_quantization_int8.cpp)
   add_example_executable(example_gemm_xdl_quantization_int8 gemm_xdl_quantization_int8.cpp)
endif()