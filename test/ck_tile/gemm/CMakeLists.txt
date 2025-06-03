# Currently ck_tile is only built on gfx94/gfx95
message("[DEBUG]: Start of CK_TILE_TEST_GEMM...")
set(EXAMPLE_GEMM_COMPILE_OPTIONS "")
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
message("[DEBUG]: First set of CK_TILE_TEST_GEMM...")
set(EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS "")
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
message("[DEBUG]: Second set of CK_TILE_TEST_GEMM...")
list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS
    -mllvm
    -enable-noalias-to-md-conversion=0
)

message("[DEBUG]: After list of CK_TILE_TEST_GEMM...")
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
#if(GPU_TARGETS MATCHES "gfx9")
    message("[DEBUG]: GPU TARGET MATCHED! in CK_TILE_TEST_GEMM...")
    add_gtest_executable(test_ck_tile_gemm_pipeline_mem test_gemm_pipeline_mem.cpp)
    add_gtest_executable(test_ck_tile_gemm_pipeline_compv3 test_gemm_pipeline_compv3.cpp)
    add_gtest_executable(test_ck_tile_gemm_pipeline_compv4 test_gemm_pipeline_compv4.cpp)

    target_compile_options(test_ck_tile_gemm_pipeline_mem PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_options(test_ck_tile_gemm_pipeline_compv3 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_options(test_ck_tile_gemm_pipeline_compv4 PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS})
#else()
#    message("Skipping ck_tile_gemm tests for current target")
#endif()

