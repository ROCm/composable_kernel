# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_ck_tile_gemm_multi_d test_gemm_multi_d.cpp)
endif()
