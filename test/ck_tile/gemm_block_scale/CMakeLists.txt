set(TEST_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
  list(APPEND TEST_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

list(APPEND TEST_GEMM_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)

if(GPU_TARGETS MATCHES "gfx94" OR GPU_TARGETS MATCHES "gfx95")
    set(TEST_GEMM_NAME test_tile_gemm_aquant_basic)
    set(QUANT_TYPES fp8 bf8 i4fp8 i4bf8 i4f32fp8 i4f32bf8)

    foreach(QUANT_TYPE ${QUANT_TYPES})
        add_gtest_executable(${TEST_GEMM_NAME}_${QUANT_TYPE} test_gemm_aquant_basic_${QUANT_TYPE}.cpp)
        target_compile_options(${TEST_GEMM_NAME}_${QUANT_TYPE} PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})
    endforeach()

else()
    message(DEBUG "Skipping ck_tile quant gemm tests for current target")
endif()
