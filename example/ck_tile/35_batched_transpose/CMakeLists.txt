add_executable(tile_example_batched_transpose EXCLUDE_FROM_ALL batched_transpose.cpp batched_transpose_api.cpp)
target_include_directories(tile_example_batched_transpose PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)
target_compile_options(tile_example_batched_transpose PRIVATE -v --save-temps -Wno-gnu-line-marker)
