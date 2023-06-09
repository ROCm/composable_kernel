# dlops
add_example_executable(example_gemm_dl_quantization_int8 gemm_dl_quantization_int8.cpp)

# xdlops
list(APPEND gpu_list gfx908 gfx90a gfx940 gfx941 gfx942)
if(GPU_TARGETS IN_LIST gpu_list)
   add_example_executable(example_gemm_xdl_bias_relu_quantization_int8 gemm_xdl_bias_relu_quantization_int8.cpp)
   add_example_executable(example_gemm_xdl_quantization_int8 gemm_xdl_quantization_int8.cpp)
endif()