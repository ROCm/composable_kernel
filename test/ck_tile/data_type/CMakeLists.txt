# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_ck_tile_pk_int4 test_pk_int4.cpp)
endif()
if(GPU_TARGETS MATCHES "gfx95")
    add_gtest_executable(test_ck_tile_pk_fp4 test_pk_fp4.cpp)
    add_gtest_executable(test_ck_tile_mx_scale test_mx_scale.cpp)
endif()