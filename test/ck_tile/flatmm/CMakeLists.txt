# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TEST_FLATMM_COMPILE_OPTIONS)
list(APPEND TEST_FLATMM_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)

if(CK_USE_OCP_FP8)
    list(APPEND TEST_FLATMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx95|gfx125")
    # Build the arch list additively. A multi-arch build (e.g.
    # GPU_TARGETS=gfx950;gfx1250) generates instances for every configured
    # arch's trait families simultaneously. The host side picks one arch at
    # compile time via GetCurrentTargetId() (driven by CK_USE_GFX1250) -- so
    # in a multi-arch build the gfx950 instances are dead code in the test
    # executable.
    set(MXFLATMM_ARCH)
    if(GPU_TARGETS MATCHES "gfx95")
        list(APPEND MXFLATMM_ARCH MXFlatmm_GFX950_)
    endif()
    if(GPU_TARGETS MATCHES "gfx125")
        list(APPEND MXFLATMM_ARCH MXFlatmm_GFX1250_ MXFlatmmTDM_GFX1250_)
    endif()

    set(C_DATA_TYPE FP16)
    set(A_LAYOUT ROW)
    set(B_LAYOUT COL)
    set(C_LAYOUT ROW)

    # Generate the kernel instance .cpp files into the build dir from the
    # mx_flatmm_instance.cpp.in template. SPLIT_K=true is omitted:
    # split-K is confirmed broken at the kernel level for all dtype
    # combinations and is not tested.
    #
    # Two combos are skipped from the cross product:
    #   1. FP6xFP6 + MXFlatmmTDM_*  -- FP6 is not supported by the GFX1250 TDM pipeline.
    #   2. FP4xFP4 + MXFlatmmTDM_*  -- FIXME: The kernel compiles and runs
    #      but produces numerically wrong results.
    #      Re-enable here AND in test_mx_flatmm_fp4fp4.cpp's typelist together
    #      once the kernel is fixed.
    set(FLATMM_INSTANCE_FILES)
    foreach(PERSISTENT false)
        foreach(DATA_TYPE FP4xFP4 FP8xFP8 FP6xFP6 FP8xFP4 FP4xFP8)
            string(REPLACE "x" ";" DATA_TYPE_AB ${DATA_TYPE})
            list(GET DATA_TYPE_AB 0 A_DATA_TYPE)
            list(GET DATA_TYPE_AB 1 B_DATA_TYPE)
            foreach(ARCH ${MXFLATMM_ARCH})
                # Skip 1: Not implemented on the TDM pipeline.
                if(DATA_TYPE MATCHES "FP6xFP6" AND ARCH MATCHES "MXFlatmmTDM_")
                    continue()
                endif()
                # Skip 2: Known-broken kernel.
                if(DATA_TYPE MATCHES "FP4xFP4" AND ARCH MATCHES "MXFlatmmTDM_")
                    continue()
                endif()
                set(MXFLATMM_ARCH_TRAITS "${ARCH}${A_DATA_TYPE}${B_DATA_TYPE}_Traits")
                foreach(SPLIT_K false)
                    foreach(HAS_HOT_LOOP false true)
                        foreach(TAIL_NUMBER ODD EVEN)
                            set(KERNEL_FILE instance_${ARCH}${DATA_TYPE}_${PERSISTENT}_${SPLIT_K}_${HAS_HOT_LOOP}_${TAIL_NUMBER}.cpp)
                            string(TOLOWER ${KERNEL_FILE} KERNEL_FILE)
                            configure_file(
                                ${CMAKE_CURRENT_SOURCE_DIR}/mx_flatmm_instance.cpp.in
                                ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_FILE}
                                @ONLY)
                            list(APPEND FLATMM_INSTANCE_FILES ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_FILE})
                        endforeach()
                    endforeach()
                endforeach()
            endforeach()
        endforeach()
    endforeach()

    # Compile the kernel instances once into an object library, shared
    # across all 5 test executables to avoid redundant GPU compilation.
    add_library(mx_flatmm_test_instances OBJECT ${FLATMM_INSTANCE_FILES})
    target_include_directories(mx_flatmm_test_instances PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_options(mx_flatmm_test_instances PRIVATE ${TEST_FLATMM_COMPILE_OPTIONS})

    foreach(DTYPE fp4fp4 fp8fp8 fp6fp6 fp8fp4 fp4fp8)
        add_gtest_executable(test_tile_mx_flatmm_${DTYPE}
            test_mx_flatmm_${DTYPE}.cpp
        )
        target_include_directories(test_tile_mx_flatmm_${DTYPE} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
        )
        target_compile_options(test_tile_mx_flatmm_${DTYPE} PRIVATE ${TEST_FLATMM_COMPILE_OPTIONS})
        target_link_libraries(test_tile_mx_flatmm_${DTYPE} PRIVATE mx_flatmm_test_instances)
    endforeach()

    # Standalone regression test for the multi-tile-per-workgroup LDS race in
    # MXFlatmmKernel's persistent path. Builds its kernel inline (does not link
    # mx_flatmm_test_instances). FP8xFP8 only -- the bug is at the kernel layer.
    add_gtest_executable(test_tile_mx_flatmm_persistent
        test_mx_flatmm_persistent.cpp
    )
    target_include_directories(test_tile_mx_flatmm_persistent PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_options(test_tile_mx_flatmm_persistent PRIVATE ${TEST_FLATMM_COMPILE_OPTIONS})

    # Umbrella target to build all flatmm tests at once
    add_custom_target(test_tile_mx_flatmm_all)
    add_dependencies(test_tile_mx_flatmm_all
        test_tile_mx_flatmm_fp4fp4
        test_tile_mx_flatmm_fp8fp8
        test_tile_mx_flatmm_fp6fp6
        test_tile_mx_flatmm_fp8fp4
        test_tile_mx_flatmm_fp4fp8
        test_tile_mx_flatmm_persistent
    )
else()
    message(DEBUG "Skipping ck_tile MX flatmm tests for current target")
endif()
