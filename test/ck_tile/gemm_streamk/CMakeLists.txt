# Currently test_ck_tile_streamk is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")
    #TODO: support all arches
    #TODO: current stream-k c-shuffle only supports C layout as R
    add_gtest_executable(test_ck_tile_streamk ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_rrr_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                                              #${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_rrc_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                                              ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_rcr_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                                              #${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_rcc_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                                              ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_crr_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                                              #${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_crc_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                                              ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_ccr_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                                              #${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_ccc_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                                              ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_rcr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
        )
else()
    message(DEBUG "Skipping test_ck_tile_streamk tests for current target")
endif()
