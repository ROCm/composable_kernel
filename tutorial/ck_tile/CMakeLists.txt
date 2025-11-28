include_directories(AFTER
  ${CMAKE_CURRENT_LIST_DIR}
)

add_subdirectory(00_copy_kernel)
add_subdirectory(01_naive_gemm)

