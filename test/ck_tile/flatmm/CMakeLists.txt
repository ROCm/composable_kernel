# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")
    add_test_executable(test_ck_tile_flatmm_basic_fp16 flatmm_basic_fp16.cpp)
    add_test_executable(test_ck_tile_flatmm_basic_bf16 flatmm_basic_bf16.cpp)

    set(EXAMPLE_FLATMM_COMPILE_OPTIONS)
    # list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
    # list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -Wno-unused-variable -Wno-unused-parameter)
    target_compile_options(test_ck_tile_flatmm_basic_fp16 PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})
    target_compile_options(test_ck_tile_flatmm_basic_bf16 PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})

else()
    message(DEBUG "Skipping ck_tile flatmm tests for current target")
endif()