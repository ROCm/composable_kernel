# Tutorial 02: Tensor Adaptors
# Advanced layout transformations using make_single_stage_tensor_adaptor,
# transform_tensor_adaptor, and chain_tensor_adaptors

# Create executable for tensor adaptors tutorial
add_executable(aa_tutorial_02_adaptors tensor_adaptors.cpp)

# Set properties
target_include_directories(aa_tutorial_02_adaptors PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Compile flags
target_compile_options(aa_tutorial_02_adaptors PRIVATE
    -Wall
    -O0
    -g
    --save-temps
)

# Message for build output
message(STATUS "Added Tutorial 02: Tensor Adaptors - Advanced layout transformations with adaptor methods")
