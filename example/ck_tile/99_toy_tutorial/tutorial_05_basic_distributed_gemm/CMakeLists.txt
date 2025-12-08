# Tutorial 05: Basic Distributed GEMM
# Demonstrates basic distributed GEMM with tile distributions
# Single warp per 16x16 output block

# Create executable for basic distributed GEMM tutorial
add_executable(aa_tutorial_05_basic_distributed_gemm basic_distributed_gemm.cpp)

# Set properties
target_include_directories(aa_tutorial_05_basic_distributed_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Compile flags
# target_compile_options(aa_tutorial_05_basic_distributed_gemm PRIVATE

#     -Wall
#     -O0
#     -g
#     --save-temps
# )

# Message for build output
message(STATUS "Added Tutorial 05: Basic Distributed GEMM - Understanding tile distributions for GEMM")
