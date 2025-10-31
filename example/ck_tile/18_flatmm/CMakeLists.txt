set(SUPPORTED_GPUS gfx908 gfx90a gfx942 gfx950)

set(has_supported_gpu FALSE)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST SUPPORTED_GPUS)
        set(has_supported_gpu TRUE)
        break()
    endif()
endforeach()

if(has_supported_gpu)
    add_executable(tile_example_flatmm_basic EXCLUDE_FROM_ALL flatmm_basic.cpp)
    add_executable(tile_example_mixed_prec_flatmm EXCLUDE_FROM_ALL mixed_prec/mixed_prec_flatmm.cpp)
    add_executable(tile_example_moe_flatmm EXCLUDE_FROM_ALL moe_flatmm.cpp)
    add_executable(tile_example_a16w4_moe_flatmm EXCLUDE_FROM_ALL mixed_prec/a16w4_moe_flatmm.cpp)
    add_executable(tile_example_grouped_flatmm EXCLUDE_FROM_ALL grouped_flatmm.cpp)
    add_executable(tile_example_mx_flatmm EXCLUDE_FROM_ALL mxgemm/mx_flatmm.cpp) # TODO: 950 only

    set(EXAMPLE_FLATMM_COMPILE_OPTIONS)
    set(EXAMPLE_MOE_FLATMM_COMPILE_OPTIONS)

    if(CK_USE_OCP_FP8)
        list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
    endif()

    target_compile_options(tile_example_flatmm_basic PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})
    target_compile_options(tile_example_mixed_prec_flatmm PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})
    target_compile_options(tile_example_moe_flatmm PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})
    target_compile_options(tile_example_a16w4_moe_flatmm PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})
    target_compile_options(tile_example_grouped_flatmm PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})
    target_compile_options(tile_example_mx_flatmm PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS}) # TODO: 950 only
endif()

