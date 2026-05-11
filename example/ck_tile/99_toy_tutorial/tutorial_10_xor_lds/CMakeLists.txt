# Tutorial 10: Padded LDS for Bank Conflict Avoidance
# Demonstrates padding technique to reduce LDS bank conflicts

# Create executable
add_executable(aa_tutorial_10_xor_lds xor_lds_gemm.cpp)

# Set properties
target_include_directories(aa_tutorial_10_xor_lds PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Copy-only test to isolate XOR descriptor issues
add_executable(aa_tutorial_10_xor_copy_only_test xor_copy_only_test.cpp)

target_include_directories(aa_tutorial_10_xor_copy_only_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

message(STATUS "Added Tutorial 10: Padded LDS - Bank conflict avoidance via padding")
message(STATUS "Added Tutorial 10: XOR Copy-Only Test - Isolates XOR descriptor correctness")
