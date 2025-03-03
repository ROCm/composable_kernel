set(TARGET_NAME tile_example_batched_transpose)
add_executable(${TARGET_NAME} EXCLUDE_FROM_ALL batched_transpose_example.cpp batched_transpose_api.cpp)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
# list(APPEND EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
target_compile_options(tile_example_batched_transpose PRIVATE ${EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS})

set(TARGET_2 tile_example_batched_transpose_2d)
add_executable(${TARGET_2} EXCLUDE_FROM_ALL batched_transpose_2d_example.cpp)
target_include_directories(${TARGET_2} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)
target_compile_options(${TARGET_2} PRIVATE ${EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS})
