# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
add_executable(tile_example_permute EXCLUDE_FROM_ALL permute.cpp)
# target_compile_options(tile_example_permute PRIVATE -v --save-temps -Wno-gnu-line-marker)
