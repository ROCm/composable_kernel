set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
set(EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS
    -mllvm
    -enable-noalias-to-md-conversion=0
)
set(EXAMPLE_GEMM_COMPILE_COMPUTE_ASYNC_OPTIONS ${EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS})

# Currently test_ck_tile_streamk is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")

    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
 
    #TODO: support all arches
    #TODO: current c-shuffle only supports C layout as R
    add_gtest_executable(test_ck_tile_streamk_smoke 
	                ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_rrr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
                        #${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_rrc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_rcr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
                        #${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_rcc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_crr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
                        #${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_crc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_ccr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
                        #${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f16_ccc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp

			${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/bf16_rrr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
                        #${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/bf16_rrc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/bf16_rcr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
                        #${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/bf16_rcc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/bf16_crr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
                        #${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/bf16_crc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/bf16_ccr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
                        #${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/bf16_ccc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp

			${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f8_rrr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f8_rcr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f8_crr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/f8_ccr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp

			${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/bf8_rrr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/bf8_rcr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/bf8_crr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/bf8_ccr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                        )
    # TODO: enable extended tests after tolerances for atomic reductions are addressed.
    # add_gtest_executable(test_ck_tile_streamk_extended 
    #                     # compv3 pipeline
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_rrr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_rrc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_rcr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_rcc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_crr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_crc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_ccr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_ccc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_rrr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_rrc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_rcr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_rcc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_crr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_crc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_ccr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/f16_ccc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                       
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_rrr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_rrc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_rcr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_rcc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_crr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_crc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_ccr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_ccc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_rrr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_rrc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_rcr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_rcc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_crr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_crc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_ccr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv3/bf16_ccc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
                        
    #                    # TODO: add compv4 pipeline
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_rrr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_rrc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_rcr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_rcc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_crr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_crc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_ccr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_ccc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_rrr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_rrc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_rcr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_rcc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_crr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_crc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_ccr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/f16_ccc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp

    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_rrr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_rrc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_rcr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_rcc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_crr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_crc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_ccr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_ccc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_rrr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_rrc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_rcr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_rcc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_crr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_crc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_ccr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                 #    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/compv4/bf16_ccc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent.cpp


    #                    # mem pipeline
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/f16_rrr_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/f16_rrc_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/f16_rcr_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/f16_rcc_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/f16_crr_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/f16_crc_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/f16_ccr_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/f16_ccc_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp

    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/bf16_rrr_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/bf16_rrc_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/bf16_rcr_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/bf16_rcc_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/bf16_crr_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/bf16_crc_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/bf16_ccr_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                    #${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/mem/bf16_ccc_mem_128x128x32_2x2x1_32x32x16_NonPersistent.cpp
    #                     )
    add_gtest_executable(test_ck_tile_streamk_tile_partitioner test_streamk_tile_partitioner.cpp)
    add_gtest_executable(test_ck_tile_streamk_reboot_smoke 
                         ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/test_gemm_streamk_reboot_fp16_persistent.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/test_gemm_streamk_reboot_bf16_persistent.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/test_gemm_streamk_reboot_fp16_nonpersistent.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/test_gemm_streamk_reboot_bf16_nonpersistent.cpp
                         test_gemm_streamk_reboot_util.cpp)
    add_gtest_executable(test_ck_tile_streamk_reboot_extended
                         ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_reboot_fp16_persistent.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_reboot_bf16_persistent.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_reboot_fp16_nonpersistent.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_reboot_bf16_nonpersistent.cpp
                         test_gemm_streamk_reboot_util.cpp)
    target_compile_options(test_ck_tile_streamk_smoke PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping test_ck_tile_streamk tests for current target")
endif()
