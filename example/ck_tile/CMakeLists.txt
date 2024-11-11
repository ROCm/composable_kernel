include_directories(AFTER
  ${CMAKE_CURRENT_LIST_DIR}
)

add_subdirectory(01_fmha)
add_subdirectory(02_layernorm2d)
add_subdirectory(03_gemm)
add_subdirectory(04_img2col)
add_subdirectory(05_reduce)
add_subdirectory(06_permute)
add_subdirectory(09_topk_softmax)
add_subdirectory(10_rmsnorm2d)
add_subdirectory(11_add_rmsnorm2d_rdquant)
add_subdirectory(12_smoothquant)
add_subdirectory(13_moe_sorting)
