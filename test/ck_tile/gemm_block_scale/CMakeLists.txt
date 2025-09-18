set(TEST_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
  list(APPEND TEST_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

list(APPEND TEST_GEMM_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)

if(GPU_TARGETS MATCHES "gfx94" OR GPU_TARGETS MATCHES "gfx95")
    # Comprehensive Unified GEMM Quantization Test Suite
    # Tests all quant types (AQuant, BQuant, RowColQuant, TensorQuant) with proper pipelines
    add_gtest_executable(test_tile_gemm_quant_unified test_gemm_quant_unified.cpp)
    target_compile_options(test_tile_gemm_quant_unified PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

else()
    message(DEBUG "Skipping ck_tile quant gemm tests for current target")
endif()
