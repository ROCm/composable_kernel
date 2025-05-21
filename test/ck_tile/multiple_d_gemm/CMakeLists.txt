# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_ck_tile_multiple_d_gemm test_multiple_d_gemm.cpp)
endif()
