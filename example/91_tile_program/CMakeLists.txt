include_directories(AFTER
  ${CMAKE_CURRENT_LIST_DIR}
)

add_subdirectory(batched_gemm_softmax_gemm)
add_subdirectory(fmha)
add_subdirectory(gemm)
add_subdirectory(gemm_gemm)
add_subdirectory(gemm_softmax_gemm)
add_subdirectory(im2col)
add_subdirectory(reduce)
add_subdirectory(softmax)
add_subdirectory(variance2d)
