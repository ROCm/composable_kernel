add_executable(tile_example_moe_gemm EXCLUDE_FROM_ALL moe_gemm.cpp)

set(EXAMPLE_FLATMM_COMPILE_OPTIONS)
# list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
# list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -Wno-unused-variable -Wno-unused-parameter)
# list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -Wno-unused-local-typedef)
target_compile_options(tile_example_moe_gemm PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})
