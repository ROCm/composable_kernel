# Tutorial 11: XOR Descriptor Test
# Minimal test to understand XOR swizzling

add_executable(aa_tutorial_11_xor_test xor_test.cpp)

target_include_directories(aa_tutorial_11_xor_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Tutorial 11b: XOR Descriptor Test WITH Tile Window
add_executable(aa_tutorial_11_xor_test_tile_window xor_test_with_tile_window.cpp)

target_include_directories(aa_tutorial_11_xor_test_tile_window PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Tutorial 11c: XOR Descriptor Test WITH GEMM Distribution
add_executable(aa_tutorial_11_xor_test_gemm_dist xor_test_with_gemm_dist.cpp)

target_include_directories(aa_tutorial_11_xor_test_gemm_dist PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Tutorial 11d: XOR Test with SWAPPED Transposes
add_executable(aa_tutorial_11_xor_test_swapped xor_test_swapped_transpose.cpp)

target_include_directories(aa_tutorial_11_xor_test_swapped PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Tutorial 11e: XOR Toggle Test for Bank Conflict Profiling
add_executable(aa_tutorial_11_xor_toggle xor_test_toggle.cpp)

target_include_directories(aa_tutorial_11_xor_toggle PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Tutorial 11f: XOR Toggle Test with Transpose Access Pattern
add_executable(aa_tutorial_11_xor_toggle_transpose xor_test_toggle_transpose.cpp)

target_include_directories(aa_tutorial_11_xor_toggle_transpose PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Tutorial 11g: XOR Toggle Test - PROPER descriptor usage through tile_window
add_executable(aa_tutorial_11_xor_toggle_proper xor_test_toggle_proper.cpp)

target_include_directories(aa_tutorial_11_xor_toggle_proper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Tutorial 11h: XOR Test with Intentional Bank Conflict Pattern
add_executable(aa_tutorial_11_xor_conflict xor_test_conflict_pattern.cpp)

target_include_directories(aa_tutorial_11_xor_conflict PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Tutorial 11i: XOR Test with TRANSPOSE - Classic bank conflict pattern
add_executable(aa_tutorial_11_xor_transpose xor_test_transpose.cpp)

target_include_directories(aa_tutorial_11_xor_transpose PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Tutorial 11j: XOR Test with REAL TRANSPOSE - Actual column-major reads
add_executable(aa_tutorial_11_xor_real_transpose xor_test_real_transpose.cpp)

target_include_directories(aa_tutorial_11_xor_real_transpose PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Tutorial 11k: LDS Transpose with Manual XOR - Raw __shared__ with manual addressing
add_executable(aa_tutorial_11_xor_transpose_lds xor_test_transpose_lds.cpp)

target_include_directories(aa_tutorial_11_xor_transpose_lds PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Tutorial 11l: Plain Transpose ONLY - No XOR, for bank conflict profiling
add_executable(aa_tutorial_11_plain_transpose xor_test_plain_only.cpp)

target_include_directories(aa_tutorial_11_plain_transpose PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Tutorial 11m: Production Transpose - Single-pass transpose (no iteration amplification)
add_executable(aa_tutorial_11_production_transpose xor_test_production_transpose.cpp)

target_include_directories(aa_tutorial_11_production_transpose PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

message(STATUS "Added Tutorial 11: XOR Test - Minimal XOR swizzle experiment")
message(STATUS "Added Tutorial 11b: XOR Test with Tile Window - Tests XOR + tile_window compatibility")
message(STATUS "Added Tutorial 11c: XOR Test with GEMM Distribution - Tests XOR + GEMM distribution compatibility")
message(STATUS "Added Tutorial 11d: XOR Test SWAPPED - Tests if transpose order matters")
message(STATUS "Added Tutorial 11e: XOR Toggle Test - Compare bank conflicts with/without XOR")
message(STATUS "Added Tutorial 11f: XOR Toggle Transpose - Transpose access triggers bank conflicts")
message(STATUS "Added Tutorial 11g: XOR Toggle PROPER - Uses XOR descriptor through tile_window")
message(STATUS "Added Tutorial 11h: XOR Conflict Pattern - Intentional conflicts with K1=1 scalar loads")
message(STATUS "Added Tutorial 11i: XOR Transpose - THE classic bank conflict example")
message(STATUS "Added Tutorial 11j: XOR REAL Transpose - Actual stride-M column reads")
message(STATUS "Added Tutorial 11k: LDS Transpose - Raw __shared__ with manual XOR addressing")
message(STATUS "Added Tutorial 11l: Plain Transpose ONLY - For bank conflict profiling")
message(STATUS "Added Tutorial 11m: Production Transpose - Single-pass production transpose")
