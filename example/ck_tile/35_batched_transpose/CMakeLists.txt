add_executable(tile_example_batched_transpose EXCLUDE_FROM_ALL batched_transpose.cpp batched_transpose_api.cpp)
target_include_directories(tile_example_batched_transpose PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

set(EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS)
# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
# list(APPEND EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
target_compile_options(tile_example_topk_softmax PRIVATE ${EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS})

