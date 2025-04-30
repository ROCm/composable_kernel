set(TARGET_NAME tile_example_elementwise)
add_executable(${TARGET_NAME} elementwise_example.cpp)
rocm_install(TARGETS ${TARGET_NAME} COMPONENT examples)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_ELEMENTWISE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
# list(APPEND EXAMPLE_ELEMENTWISE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
target_compile_options(tile_example_elementwise PRIVATE ${EXAMPLE_ELEMENTWISE_COMPILE_OPTIONS})
