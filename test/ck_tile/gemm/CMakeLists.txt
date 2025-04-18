# Currently ck_tile is only built on gfx94/gfx95
set(EXAMPLE_GEMM_COMPILE_OPTIONS "")
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
set(EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS "")
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS
    -mllvm
    -enable-noalias-to-md-conversion=0
)

if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
if(GPU_TARGETS MATCHES "gfx94" OR GPU_TARGETS MATCHES "gfx95")
    add_gtest_executable(test_ck_tile_gemm_pipeline_mem test_gemm_pipeline_mem.cpp)
    add_gtest_executable(test_ck_tile_gemm_pipeline_compv3 test_gemm_pipeline_compv3.cpp)
    add_gtest_executable(test_ck_tile_gemm_pipeline_compv4 test_gemm_pipeline_compv4.cpp)

    target_compile_options(test_ck_tile_gemm_pipeline_mem PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_options(test_ck_tile_gemm_pipeline_compv3 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_options(test_ck_tile_gemm_pipeline_compv4 PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS})
else()
    message("Skipping ck_tile_gemm tests for current target")
endif()
endif()
