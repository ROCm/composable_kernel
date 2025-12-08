# Tutorial 06: Tile Sweeping GEMM
# Demonstrates tile sweeping with multiple warps
# Multiple warps cooperate to compute larger output blocks

# Create executable for tile sweeping GEMM tutorial
add_executable(aa_tutorial_06_tile_sweeping_gemm tile_sweeping_gemm.cpp)

# Set properties
target_include_directories(aa_tutorial_06_tile_sweeping_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Compile flags
# target_compile_options(aa_tutorial_06_tile_sweeping_gemm PRIVATE
#     -Wall
#     -O0
#     -g
#     --save-temps
# )

# Message for build output
message(STATUS "Added Tutorial 06: Tile Sweeping GEMM - Multiple warps with tile sweeping pattern")

# Add test subdirectory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
endif()
