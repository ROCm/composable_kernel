if((DTYPES MATCHES "fp8" AND DTYPES MATCHES "fp16") OR NOT DEFINED DTYPES)
  add_executable(client_splitK_gemm_fp16_f8 splitK_gemm_fp16_f8.cpp)
  target_link_libraries(client_splitK_gemm_fp16_f8 PRIVATE composable_kernel::device_gemm_operations)
endif()

if((DTYPES MATCHES "i8" AND DTYPES MATCHES "bf16") OR NOT DEFINED DTYPES)
  add_executable(client_splitK_gemm_bf16_i8 splitK_gemm_bf16_i8.cpp)
  target_link_libraries(client_splitK_gemm_bf16_i8 PRIVATE composable_kernel::device_gemm_operations)
endif()
