add_test_executable(test_tile_reduce tile_reduce.cpp)
target_compile_options(test_tile_reduce PRIVATE -v --save-temps -Wno-gnu-line-marker)