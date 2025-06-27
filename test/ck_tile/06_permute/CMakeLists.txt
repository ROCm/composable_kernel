# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")

add_test_executable(test_ck_tile_permute_fp16 permute_fp16.cpp)
add_test_executable(test_ck_tile_permute_fp8 permute_fp8.cpp)
add_test_executable(test_ck_tile_permute_fp32 permute_fp32.cpp)

if(NOT DEFINED PERMUTE_USE_ALTERNATIVE_IMPL)
set(PERMUTE_USE_ALTERNATIVE_IMPL true)
endif()

if(PERMUTE_USE_ALTERNATIVE_IMPL)
target_compile_options(test_ck_tile_permute_fp16 PRIVATE -DPERMUTE_USE_ALTERNATIVE_IMPL)
target_sources(test_ck_tile_permute_fp16 PRIVATE alternative_impl/matrix_core_swizzle.cpp)

target_compile_options(test_ck_tile_permute_fp8 PRIVATE -DPERMUTE_USE_ALTERNATIVE_IMPL)
target_sources(test_ck_tile_permute_fp8 PRIVATE alternative_impl/matrix_core_swizzle.cpp)

target_compile_options(test_ck_tile_permute_fp32 PRIVATE -DPERMUTE_USE_ALTERNATIVE_IMPL)
target_sources(test_ck_tile_permute_fp32 PRIVATE alternative_impl/matrix_core_swizzle.cpp)
endif()

else()
    message(DEBUG "Skipping ck_tile_permute tests for current target")
endif()
