# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
add_executable(tile_example_layernorm2d_fwd EXCLUDE_FROM_ALL layernorm2d_fwd.cpp)
target_compile_options(tile_example_layernorm2d_fwd PRIVATE -DSAVE_MEAN_INV_STD)