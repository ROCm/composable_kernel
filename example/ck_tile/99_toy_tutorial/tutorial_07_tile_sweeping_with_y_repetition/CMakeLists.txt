# Tutorial 07: Tile Sweeping with Y-Dimension Repetition
# Demonstrates true tile sweeping using Y-dimension repetition in distributions
# Follows the pattern from 02_gemm for production-ready code

# Create executable for tile sweeping with Y-repetition tutorial
add_executable(aa_tutorial_07_tile_sweeping_y_repetition tile_sweeping_with_y_repetition.cpp)

# Set properties
target_include_directories(aa_tutorial_07_tile_sweeping_y_repetition PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Compile flags
# target_compile_options(aa_tutorial_07_tile_sweeping_y_repetition PRIVATE
#     -Wall
#     -O0
#     -g
#     --save-temps
# )

# Message for build output
message(STATUS "Added Tutorial 07: Tile Sweeping with Y-Dimension Repetition - Multiple warps with Y-repetition for true tile sweeping")

# Add test subdirectory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
endif()
