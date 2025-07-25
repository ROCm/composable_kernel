# Currently ck_tile is only built on gfx90a, gfx942 and gfx950
if(GPU_TARGETS MATCHES "gfx942" OR GPU_TARGETS MATCHES "gfx950" OR GPU_TARGETS MATCHES "gfx90a")

    add_test_executable(test_ck_tile_moe_sorting_fp32 moe_sorting_fp32.cpp moe_sorting_api.cpp)
    target_include_directories(test_ck_tile_moe_sorting_fp32 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

    set(EXAMPLE_MOE_SORTING_COMPILE_OPTIONS)
    # NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
    list(APPEND EXAMPLE_MOE_SORTING_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
    # list(APPEND EXAMPLE_MOE_SORTING_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
    target_compile_options(test_ck_tile_moe_sorting_fp32 PRIVATE ${EXAMPLE_MOE_SORTING_COMPILE_OPTIONS})

else()
    message(DEBUG "Skipping ck_tile_moe_sorting tests for current target")
endif()
