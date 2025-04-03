include_directories(AFTER
  ${CMAKE_CURRENT_LIST_DIR}
)

add_subdirectory(01_add)
add_subdirectory(02_gemm)
add_subdirectory(03_flash_attention_fwd)
