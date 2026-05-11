# Tutorial 03: Padding with Tile Windows
# Demonstrates buffer_view + identity, tile_window, and load_tile with padding

# Create executable for padding with tiles tutorial
add_executable(aa_tutorial_03_padding padding_with_tiles.cpp)

# Set properties
target_include_directories(aa_tutorial_03_padding PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Compile flags
target_compile_options(aa_tutorial_03_padding PRIVATE
    -Wall
    -O0
    -g
    --save-temps
)

# Message for build output
message(STATUS "Added Tutorial 03: Padding with Tile Windows - buffer_view + identity + load_tile pattern")
