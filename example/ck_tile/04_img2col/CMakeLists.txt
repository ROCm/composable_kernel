# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
add_executable(tile_example_img2col EXCLUDE_FROM_ALL image_to_column.cpp)
target_compile_options(tile_example_img2col PRIVATE -save-temps=obj -Wno-gnu-line-marker)
