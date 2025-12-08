# Tutorial 04: Tensor Descriptor vs Tensor Adaptor
# Demonstrates the differences between tensor_adaptor and tensor_descriptor,
# including coordinate operations and when to use each

# Create executable for descriptor vs adaptor tutorial
add_executable(aa_tutorial_04_descriptor_vs_adaptor descriptor_vs_adaptor.cpp)

# Set properties
target_include_directories(aa_tutorial_04_descriptor_vs_adaptor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Compile flags
target_compile_options(aa_tutorial_04_descriptor_vs_adaptor PRIVATE
    -Wall
    -O0
    -g
    --save-temps
)

# Message for build output
message(STATUS "Added Tutorial 04: Descriptor vs Adaptor - Understanding tensor_adaptor and tensor_descriptor")
