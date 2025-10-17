if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_reduce2d test_reduce2d.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_ck_tile_reduce2d PRIVATE utility)
    endif()
endif()

