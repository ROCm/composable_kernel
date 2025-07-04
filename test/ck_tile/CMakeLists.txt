add_subdirectory(image_to_column)
add_subdirectory(gemm)
add_subdirectory(batched_gemm)
add_subdirectory(grouped_gemm)
add_subdirectory(gemm_multi_d)
add_subdirectory(data_type)
add_subdirectory(flatmm)
# Not including these tests as there is a bug on gfx90a and gfx942
# resulting in "GPU core dump"
#add_subdirectory(moe_smoothquant)