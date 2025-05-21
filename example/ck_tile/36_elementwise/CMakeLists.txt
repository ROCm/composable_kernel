# 1D elementwise target
# Elementwise target with 2D input
set(TARGET_NAME_2D_INPUT tile_example_elementwise)
add_executable(${TARGET_NAME_2D_INPUT} elementwise_example.cpp)
rocm_install(TARGETS ${TARGET_NAME_2D_INPUT} COMPONENT examples)
target_include_directories(${TARGET_NAME_2D_INPUT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

# Elementwise unary target with 2D input
set(TARGET_NAME_2D_INPUT_UNARY tile_example_elementwise_unary)
add_executable(${TARGET_NAME_2D_INPUT_UNARY} elementwise_example_unary.cpp)
rocm_install(TARGETS ${TARGET_NAME_2D_INPUT_UNARY} COMPONENT examples)
target_include_directories(${TARGET_NAME_2D_INPUT_UNARY} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_ELEMENTWISE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal -Wno-unused-variable -Rpass-analysis=kernel-resource-usage)
# list(APPEND EXAMPLE_ELEMENTWISE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
target_compile_options(${TARGET_NAME_2D_INPUT} PRIVATE ${EXAMPLE_ELEMENTWISE_COMPILE_OPTIONS})
target_compile_options(${TARGET_NAME_2D_INPUT_UNARY} PRIVATE ${EXAMPLE_ELEMENTWISE_COMPILE_OPTIONS})
