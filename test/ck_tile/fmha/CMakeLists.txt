# Keep in sync with example/ck_tile/01_fmha/CMakeLists.txt
if(NOT SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx12")
    return()
endif()

set(FMHA_BWD_INSTANCES "tile_fmha_bwd_instances")
set(FMHA_FWD_INSTANCES "tile_fmha_fwd_instances")

add_custom_target(test_ck_tile_fmha)

add_gtest_executable(test_ck_tile_fmha_bwd_bf16 test_fmha_bwd_bf16.cpp)
target_link_libraries(test_ck_tile_fmha_bwd_bf16 PRIVATE ${FMHA_BWD_INSTANCES})

add_gtest_executable(test_ck_tile_fmha_bwd_fp16 test_fmha_bwd_fp16.cpp)
target_link_libraries(test_ck_tile_fmha_bwd_fp16 PRIVATE ${FMHA_BWD_INSTANCES})

add_gtest_executable(test_ck_tile_fmha_fwd_bf16 test_fmha_fwd_bf16.cpp)
target_link_libraries(test_ck_tile_fmha_fwd_bf16 PRIVATE ${FMHA_FWD_INSTANCES})

add_gtest_executable(test_ck_tile_fmha_fwd_fp16 test_fmha_fwd_fp16.cpp)
target_link_libraries(test_ck_tile_fmha_fwd_fp16 PRIVATE ${FMHA_FWD_INSTANCES})

add_gtest_executable(test_ck_tile_fmha_fwd_fp8 test_fmha_fwd_fp8.cpp)
target_link_libraries(test_ck_tile_fmha_fwd_fp8 PRIVATE ${FMHA_FWD_INSTANCES})

add_dependencies(test_ck_tile_fmha
    test_ck_tile_fmha_bwd_bf16
    test_ck_tile_fmha_bwd_fp16
    test_ck_tile_fmha_fwd_bf16
    test_ck_tile_fmha_fwd_fp16
    test_ck_tile_fmha_fwd_fp8
)

if(SUPPORTED_GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_ck_tile_fmha_bwd_fp32 test_fmha_bwd_fp32.cpp)
    target_link_libraries(test_ck_tile_fmha_bwd_fp32 PRIVATE ${FMHA_BWD_INSTANCES})

    add_gtest_executable(test_ck_tile_fmha_fwd_fp32 test_fmha_fwd_fp32.cpp)
    target_link_libraries(test_ck_tile_fmha_fwd_fp32 PRIVATE ${FMHA_FWD_INSTANCES})

    add_dependencies(test_ck_tile_fmha
        test_ck_tile_fmha_bwd_fp32
        test_ck_tile_fmha_fwd_fp32
    )
endif()
