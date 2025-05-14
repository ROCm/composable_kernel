add_executable(tile_example_flatmm_basic EXCLUDE_FROM_ALL flatmm_basic.cpp)

set(EXAMPLE_FLATMM_COMPILE_OPTIONS)
# list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
# list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -Wno-unused-variable -Wno-unused-parameter)
list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -DUSING_MFMA_16x16x32=1 -DENABLE_FP8=1 -Wno-unused-local-typedef)
#list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -DUSING_MFMA_32x32x16=1 -DENABLE_FP8=1 -Wno-unused-local-typedef)
target_compile_options(tile_example_flatmm_basic PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})
