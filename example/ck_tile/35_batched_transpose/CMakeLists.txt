add_library(batched_transpose_api OBJECT batched_transpose_api.cpp)

set(TARGET_4D tile_example_transpose_4d)
add_executable(${TARGET_4D} EXCLUDE_FROM_ALL transpose_4d.cpp $<TARGET_OBJECTS:batched_transpose_api>)
target_include_directories(${TARGET_4D} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

set(TARGET_2D tile_example_transpose_2d)
add_executable(${TARGET_2D} EXCLUDE_FROM_ALL transpose_2d.cpp $<TARGET_OBJECTS:batched_transpose_api>)
target_include_directories(${TARGET_2D} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)


# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
# list(APPEND EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
target_compile_options(${TARGET_4D} PRIVATE ${EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS})
target_compile_options(${TARGET_2D} PRIVATE ${EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS})
