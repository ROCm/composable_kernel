add_executable(tile_example_moe_flatmm EXCLUDE_FROM_ALL moe_flatmm.cpp)
add_executable(tile_example_a16w4_moe_flatmm EXCLUDE_FROM_ALL mixed_prec/a16w4_moe_flatmm.cpp)


set(EXAMPLE_MOE_FLATMM_COMPILE_OPTIONS)

list(APPEND EXAMPLE_MOE_FLATMM_COMPILE_OPTIONS -Wno-nrvo -Wno-unused-variable -Wno-unused-parameter -Wno-unused-local-typedef -Wno-float-equal)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_MOE_FLATMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

list(APPEND EXAMPLE_MOE_FLATMM_COMPILE_OPTIONS --save-temps)
target_compile_options(tile_example_moe_flatmm PRIVATE ${EXAMPLE_MOE_FLATMM_COMPILE_OPTIONS})
target_compile_options(tile_example_a16w4_moe_flatmm PRIVATE ${EXAMPLE_MOE_FLATMM_COMPILE_OPTIONS})
