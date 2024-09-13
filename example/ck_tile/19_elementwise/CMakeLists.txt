# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
add_executable(tile_example_elementwise EXCLUDE_FROM_ALL elementwise.cpp elementwise_api.cpp)
target_include_directories(tile_example_elementwise PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(tile_example_elementwise PRIVATE -v --save-temps -Wno-gnu-line-marker)
