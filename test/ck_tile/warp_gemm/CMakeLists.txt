if(GPU_TARGETS MATCHES "gfx95")
    add_gtest_executable(test_ck_tile_wg_16x16x128_fp4 test_f32_16x16x128_fp4.cpp)
endif()
