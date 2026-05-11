# Tutorial 12: XOR-Based Bank Conflict-Free LDS (Correct [N,K] Layout)
# Demonstrates proper XOR swizzling with correct dimension ordering

# Create executable
add_executable(aa_tutorial_12_xor_correct xor_correct_gemm.cpp)

# Set properties
target_include_directories(aa_tutorial_12_xor_correct PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

message(STATUS "Added Tutorial 12: XOR-Based Bank Conflict-Free LDS (Correct [N,K] layout)")
