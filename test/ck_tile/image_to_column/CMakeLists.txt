if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_tile_image_to_column test_tile_image_to_column.cpp)
endif()
