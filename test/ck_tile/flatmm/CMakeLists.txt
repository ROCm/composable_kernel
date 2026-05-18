# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TEST_FLATMM_COMPILE_OPTIONS)
list(APPEND TEST_FLATMM_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)

if(CK_USE_OCP_FP8)
    list(APPEND TEST_FLATMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx95")
    set(MXGEMM_EXAMPLE_DIR ${CMAKE_SOURCE_DIR}/example/ck_tile/18_flatmm/mxgemm)

    # Generate the 40 kernel instance .cpp files.
    # We inline the generation here (rather than calling mx_flatmm_instance_generate)
    # so that configure_file paths resolve correctly from this directory.
    set(C_DATA_TYPE FP16)
    set(A_LAYOUT ROW)
    set(B_LAYOUT COL)
    set(C_LAYOUT ROW)

    set(FLATMM_INSTANCE_FILES)
    foreach(PERSISTENT false)
        foreach(DATA_TYPE FP4xFP4 FP8xFP8 FP6xFP6 FP8xFP4 FP4xFP8)
            string(REPLACE "x" ";" DATA_TYPE_AB ${DATA_TYPE})
            list(GET DATA_TYPE_AB 0 A_DATA_TYPE)
            list(GET DATA_TYPE_AB 1 B_DATA_TYPE)
            set(ARCH MXFlatmm_GFX950_)
            set(MXFLATMM_ARCH_TRAITS "${ARCH}${A_DATA_TYPE}${B_DATA_TYPE}_Traits")
            foreach(SPLIT_K false)
                foreach(HAS_HOT_LOOP false true)
                    foreach(TAIL_NUMBER ODD EVEN)
                        set(KERNEL_FILE instance_${ARCH}${DATA_TYPE}_${PERSISTENT}_${SPLIT_K}_${HAS_HOT_LOOP}_${TAIL_NUMBER}.cpp)
                        string(TOLOWER ${KERNEL_FILE} KERNEL_FILE)
                        configure_file(
                            ${MXGEMM_EXAMPLE_DIR}/mx_flatmm_instance.cpp.in
                            ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_FILE}
                            @ONLY)
                        list(APPEND FLATMM_INSTANCE_FILES ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_FILE})
                    endforeach()
                endforeach()
            endforeach()
        endforeach()
    endforeach()

    # Compile the 20 kernel instances once into an object library,
    # shared across all 5 test executables to avoid redundant GPU compilation.
    # SPLIT_K=true instances are omitted: split-K is confirmed broken at the
    # kernel level for all dtype combinations and is not tested.
    add_library(mx_flatmm_test_instances OBJECT ${FLATMM_INSTANCE_FILES})
    target_include_directories(mx_flatmm_test_instances PRIVATE
        ${MXGEMM_EXAMPLE_DIR}
    )
    target_compile_options(mx_flatmm_test_instances PRIVATE ${TEST_FLATMM_COMPILE_OPTIONS})

    foreach(DTYPE fp4fp4 fp8fp8 fp6fp6 fp8fp4 fp4fp8)
        add_gtest_executable(test_tile_mx_flatmm_${DTYPE}
            test_mx_flatmm_${DTYPE}.cpp
        )
        target_include_directories(test_tile_mx_flatmm_${DTYPE} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${MXGEMM_EXAMPLE_DIR}
        )
        target_compile_options(test_tile_mx_flatmm_${DTYPE} PRIVATE ${TEST_FLATMM_COMPILE_OPTIONS})
        target_link_libraries(test_tile_mx_flatmm_${DTYPE} PRIVATE mx_flatmm_test_instances)
    endforeach()

    # Umbrella target to build all flatmm tests at once
    add_custom_target(test_tile_mx_flatmm_all)
    add_dependencies(test_tile_mx_flatmm_all
        test_tile_mx_flatmm_fp4fp4
        test_tile_mx_flatmm_fp8fp8
        test_tile_mx_flatmm_fp6fp6
        test_tile_mx_flatmm_fp8fp4
        test_tile_mx_flatmm_fp4fp8
    )
else()
    message(DEBUG "Skipping ck_tile flatmm tests for current target")
endif()
