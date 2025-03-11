set(TARGET_4d tile_example_batched_transpose_4d)
add_executable(${TARGET_4d} EXCLUDE_FROM_ALL batched_transpose_4d.cpp batched_transpose_4d_api.cpp)
target_include_directories(${TARGET_4d} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

set(TARGET_2d tile_example_batched_transpose_2d)
add_executable(${TARGET_2d} EXCLUDE_FROM_ALL batched_transpose_2d.cpp)
target_include_directories(${TARGET_2d} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)


# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
# list(APPEND EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
target_compile_options(${TARGET_4d} PRIVATE ${EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS})
target_compile_options(${TARGET_2d} PRIVATE ${EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS})
