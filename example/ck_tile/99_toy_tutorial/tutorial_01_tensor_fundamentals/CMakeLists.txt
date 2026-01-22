# Tutorial 01: Tensor Fundamentals
# Complete foundation covering descriptors, views, coordinates, and element access

# Create executable for tensor fundamentals tutorial
add_executable(aa_tutorial_01_fundamentals tensor_fundamentals.cpp)

# Set properties
target_include_directories(aa_tutorial_01_fundamentals PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Compile flags
target_compile_options(aa_tutorial_01_fundamentals PRIVATE
    -Wall
    -O0
    -g
    --save-temps
    -Wno-gnu-line-marker
)

# Message for build output
message(STATUS "Added Tutorial 01: Tensor Fundamentals - Complete foundation for ck_tile tensor system")