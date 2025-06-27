# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")
    add_test_executable(test_ck_tile_batched_transpose_fp16 batched_transpose_fp16.cpp batched_transpose_api.cpp)
    target_include_directories(test_ck_tile_batched_transpose_fp16 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

    add_test_executable(test_ck_tile_batched_transpose_fp8 batched_transpose_fp8.cpp batched_transpose_api.cpp)
    target_include_directories(test_ck_tile_batched_transpose_fp8 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

    add_test_executable(test_ck_tile_batched_transpose_bf16 batched_transpose_fp8.cpp batched_transpose_api.cpp)
    target_include_directories(test_ck_tile_batched_transpose_bf16 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

    # NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
    list(APPEND EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
    # list(APPEND EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
    target_compile_options(test_ck_tile_batched_transpose_fp16 PRIVATE ${EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS})
    target_compile_options(test_ck_tile_batched_transpose_fp8 PRIVATE ${EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS})
    target_compile_options(test_ck_tile_batched_transpose_bf16 PRIVATE ${EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS})

else()
    message(DEBUG "Skipping ck_tile_batched_transpose tests for current target")
endif()