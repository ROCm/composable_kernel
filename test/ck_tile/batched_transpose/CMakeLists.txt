# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")

    function (add_batched_transpose_test TARGET_NAME MAIN_SRC)
        message(DEBUG "adding ${TARGET_NAME}")

        add_test_executable(${TARGET_NAME} ${MAIN_SRC} batched_transpose_api.cpp)
        target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

        # NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
        list(APPEND EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
        # list(APPEND EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
        target_compile_options(${TARGET_NAME} PRIVATE ${EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS})

    endfunction(add_batched_transpose_test TARGET_NAME MAIN_SRC)

    set(CUSTOM_TARGET_NAME test_ck_tile_batched_transpose)

    add_custom_target(${CUSTOM_TARGET_NAME})

    add_batched_transpose_test(test_ck_tile_batched_transpose_fp16 batched_transpose_fp16.cpp)
    add_dependencies(${CUSTOM_TARGET_NAME} test_ck_tile_batched_transpose_fp16)

    add_batched_transpose_test(test_ck_tile_batched_transpose_fp8 batched_transpose_fp8.cpp)
    add_dependencies(${CUSTOM_TARGET_NAME} test_ck_tile_batched_transpose_fp8)

    add_batched_transpose_test(test_ck_tile_batched_transpose_bf16 batched_transpose_bf16.cpp)
    add_dependencies(${CUSTOM_TARGET_NAME} test_ck_tile_batched_transpose_bf16)
    add_gtest_executable(test_batched_transpose test_batched_transpose.cpp)
    set_property(TARGET test_batched_transpose PROPERTY CXX_STANDARD 20)
else()
    message(DEBUG "Skipping ck_tile batched_transpose tests for current target")
endif()
