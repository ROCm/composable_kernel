if(DTYPES MATCHES "int8" OR NOT DEFINED DTYPES)
# dlops
add_example_executable(example_gemm_dl_quantization_int8 gemm_dl_quantization_int8.cpp)

# xdlops
list(APPEND gpu_list gfx908 gfx90a gfx940 gfx941 gfx942)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
 if(gpu IN_LIST gpu_list AND target EQUAL 0)
   add_example_executable(example_gemm_xdl_bias_relu_quantization_int8 gemm_xdl_bias_relu_quantization_int8.cpp)
   add_example_executable(example_gemm_xdl_quantization_int8 gemm_xdl_quantization_int8.cpp)
   set(target 1)
 endif()
endforeach()
endif()