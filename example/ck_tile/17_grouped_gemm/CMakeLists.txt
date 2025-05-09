set(EXAMPLE_COMPILE_OPTIONS)
list(APPEND EXAMPLE_COMPILE_OPTIONS -g -fverbose-asm --save-temps -Wno-gnu-line-marker -Rpass-analysis=kernel-resource-usage)

add_executable(tile_example_grouped_gemm EXCLUDE_FROM_ALL grouped_gemm.cpp)
target_compile_options(tile_example_grouped_gemm PRIVATE ${EXAMPLE_COMPILE_OPTIONS})

add_executable(tile_example_grouped_gemm_tileloop EXCLUDE_FROM_ALL grouped_gemm_tileloop.cpp)
target_compile_options(tile_example_grouped_gemm_tileloop PRIVATE ${EXAMPLE_COMPILE_OPTIONS})