# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")

    function(add_permute_test TARGET_NAME MAIN_SRC)
        add_test_executable(${TARGET_NAME} ${MAIN_SRC})

        if(NOT DEFINED PERMUTE_USE_ALTERNATIVE_IMPL)
        set(PERMUTE_USE_ALTERNATIVE_IMPL true)
        endif()

        if(PERMUTE_USE_ALTERNATIVE_IMPL)
        target_compile_options(${TARGET_NAME} PRIVATE -DPERMUTE_USE_ALTERNATIVE_IMPL)
        target_sources(${TARGET_NAME} PRIVATE alternative_impl/matrix_core_swizzle.cpp)
        endif()

    endfunction(add_permute_test TARGET_NAME MAIN_SRC)
    
    set(CUSTOM_TARGET_NAME test_ck_tile_permute)

    add_custom_target(${CUSTOM_TARGET_NAME})

    add_permute_test(test_ck_tile_permute_fp16 permute_fp16.cpp)
    add_dependencies(${CUSTOM_TARGET_NAME} test_ck_tile_permute_fp16)

    add_permute_test(test_ck_tile_permute_fp8 permute_fp8.cpp)
    add_dependencies(${CUSTOM_TARGET_NAME} test_ck_tile_permute_fp8)

    add_permute_test(test_ck_tile_permute_fp32 permute_fp32.cpp)
    add_dependencies(${CUSTOM_TARGET_NAME} test_ck_tile_permute_fp32)

else()
    message(DEBUG "Skipping ck_tile_permute tests for current target")
endif()
