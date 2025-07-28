# Currently ck_tile_gemm is only built on gfx94/gfx95
set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
set(EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS
    -mllvm
    -enable-noalias-to-md-conversion=0
)

if(GPU_TARGETS MATCHES "gfx94" OR GPU_TARGETS MATCHES "gfx95")
    add_gtest_executable(test_ck_tile_gemm_pipeline_mem test_gemm_pipeline_mem.cpp)
    add_gtest_executable(test_ck_tile_gemm_pipeline_compv3 test_gemm_pipeline_compv3.cpp)
    add_gtest_executable(test_ck_tile_gemm_pipeline_compv4 test_gemm_pipeline_compv4.cpp)

    target_compile_options(test_ck_tile_gemm_pipeline_mem PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_options(test_ck_tile_gemm_pipeline_compv3 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_options(test_ck_tile_gemm_pipeline_compv4 PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS})


    add_test_executable(test_ck_tile_gemm_pipeline_universal_fp8 test_gemm_pipeline_universal_fp8.cpp)
    target_compile_options(test_ck_tile_gemm_pipeline_universal_fp8 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    add_test_executable(test_ck_tile_gemm_pipeline_universal_bf8 test_gemm_pipeline_universal_bf8.cpp)
    target_compile_options(test_ck_tile_gemm_pipeline_universal_bf8 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    add_test_executable(test_ck_tile_gemm_pipeline_basic_fp8 test_gemm_pipeline_basic_fp8.cpp)
    target_compile_options(test_ck_tile_gemm_pipeline_basic_fp8 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    add_test_executable(test_ck_tile_gemm_pipeline_basic_bf8 test_gemm_pipeline_basic_bf8.cpp)
    target_compile_options(test_ck_tile_gemm_pipeline_basic_bf8 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping ck_tile_gemm tests for current target")
endif()

if(GPU_TARGETS MATCHES "gfx94" OR GPU_TARGETS MATCHES "gfx95" OR GPU_TARGETS MATCHES "gfx90a")
    add_gtest_executable(test_ck_tile_gemm_pipeline_persistent test_gemm_pipeline_persistent.cpp)
    target_compile_options(test_ck_tile_gemm_pipeline_persistent PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

    add_test_executable(test_ck_tile_gemm_pipeline_universal_fp16 test_gemm_pipeline_universal_fp16.cpp)
    target_compile_options(test_ck_tile_gemm_pipeline_universal_fp16 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    add_test_executable(test_ck_tile_gemm_pipeline_universal_bf16 test_gemm_pipeline_universal_bf16.cpp)
    target_compile_options(test_ck_tile_gemm_pipeline_universal_bf16 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    add_test_executable(test_ck_tile_gemm_pipeline_basic_fp16 test_gemm_pipeline_basic_fp16.cpp)
    target_compile_options(test_ck_tile_gemm_pipeline_basic_fp16 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    add_test_executable(test_ck_tile_gemm_pipeline_basic_bf16 test_gemm_pipeline_basic_bf16.cpp)
    target_compile_options(test_ck_tile_gemm_pipeline_basic_bf16 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()
