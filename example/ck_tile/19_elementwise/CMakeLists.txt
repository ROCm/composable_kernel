# Elementwise example targets 2D inputs
set(TARGET_NAME_2D_INPUT tile_example_elementwise)
add_executable(${TARGET_NAME_2D_INPUT} elementwise_example.cpp)

# Elementwise unary example targets 2D inputs
set(TARGET_NAME_2D_INPUT_UNARY tile_example_elementwise_unary)
add_executable(${TARGET_NAME_2D_INPUT_UNARY} elementwise_example_unary.cpp)

set(TARGET_NAME_2D_INPUT_TRANSPOSE tile_example_elementwise_transpose)
add_executable(${TARGET_NAME_2D_INPUT_TRANSPOSE} elementwise_example_transpose.cpp)
