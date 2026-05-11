# Tutorial 09: Optimized LDS Staging
# Demonstrates separate copy and GEMM distributions for production-ready kernels

# Create executable
add_executable(aa_tutorial_09_optimized_lds optimized_lds_gemm.cpp)

# Set properties
target_include_directories(aa_tutorial_09_optimized_lds PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

message(STATUS "Added Tutorial 09: Optimized LDS - Separate copy/GEMM distributions")
