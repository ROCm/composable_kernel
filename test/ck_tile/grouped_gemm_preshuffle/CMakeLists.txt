# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_ck_tile_grouped_gemm_preshuffle test_grouped_gemm_preshuffle.cpp)
endif()


