# Currently ck_tile is only built on gfx9
set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx94" OR GPU_TARGETS MATCHES "gfx95")
    add_gtest_executable(test_ck_tile_gemm_multi_d test_gemm_multi_d.cpp)
    target_compile_definitions(test_ck_tile_gemm_multi_d PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()
