set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx94|gfx95")
    # Split into three separate test executables for faster parallel compilation
    add_gtest_executable(test_ck_tile_grouped_gemm_quant_rowcol test_grouped_gemm_quant_rowcol.cpp)
    target_compile_options(test_ck_tile_grouped_gemm_quant_rowcol PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_ck_tile_grouped_gemm_quant_tensor test_grouped_gemm_quant_tensor.cpp)
    target_compile_options(test_ck_tile_grouped_gemm_quant_tensor PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_ck_tile_grouped_gemm_quant_bquant test_grouped_gemm_quant_bquant.cpp)
    target_compile_options(test_ck_tile_grouped_gemm_quant_bquant PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()

