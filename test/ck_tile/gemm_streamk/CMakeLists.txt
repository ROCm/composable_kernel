# Currently test_ck_tile_streamk is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")
    #TODO: support all arches
    add_gtest_executable(test_ck_tile_streamk test_gemm_streamk.cpp)
else()
    message(DEBUG "Skipping test_ck_tile_streamk tests for current target")
endif()
