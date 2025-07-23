# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_ck_tile_pk_int4 test_pk_int4.cpp)
    add_gtest_executable(test_ck_tile_pk_fp4 test_pk_fp4.cpp)
endif()

add_gtest_executable(test_ck_tile_fp8 test_fp8.cpp)
# conditionally specify the use of OCP_FP8
if(CK_USE_OCP_FP8)
    target_compile_options(test_ck_tile_fp8 PRIVATE -DCK_TILE_USE_OCP_FP8)
endif()
