# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(mx_flatmm_instance_generate FILE_LIST)
    set(C_DATA_TYPE FP16)
    set(A_LAYOUT ROW)
    set(B_LAYOUT COL)
    set(C_LAYOUT ROW)

    set(MXFLATMM_ARCH)

    if (GPU_TARGETS MATCHES "gfx95")
        list(APPEND MXFLATMM_ARCH MXFlatmm_GFX950_)
    endif()

    if (GPU_TARGETS MATCHES "gfx125")
        list(APPEND MXFLATMM_ARCH MXFlatmm_GFX1250_ MXFlatmmTDM_GFX1250_)
    endif()

    # foreach(PERSISTENT false true)
    # TODO: Persistent kernels are disabled due to compilation failures with some LLVM versions.
    foreach(PERSISTENT false)
        foreach(DATA_TYPE FP4xFP4 FP8xFP8 FP6xFP6 FP8xFP4 FP4xFP8)
            string(REPLACE "x" ";" DATA_TYPE_AB ${DATA_TYPE})
            list(GET DATA_TYPE_AB 0 A_DATA_TYPE)
            list(GET DATA_TYPE_AB 1 B_DATA_TYPE)
            foreach(ARCH ${MXFLATMM_ARCH})
                # FP6 is not supported on the GFX1250 TDM pipeline; only the non-TDM
                # MXFlatmm_GFX1250_ traits use the FP6-capable pipeline.
                if (DATA_TYPE MATCHES "FP6xFP6" AND ARCH MATCHES "MXFlatmmTDM_")
                    continue()
                endif()
                set(MXFLATMM_ARCH_TRAITS "${ARCH}${A_DATA_TYPE}${B_DATA_TYPE}_Traits")
                # SPLIT_K=true instances are omitted: split-K is confirmed broken at the
                # kernel level and the executable hardcodes split_k=false at every call site.
                foreach(SPLIT_K false)
                    foreach(HAS_HOT_LOOP false true)
                        foreach(TAIL_NUMBER ODD EVEN)
                            set(KERNEL_FILE mxgemm/instance_${ARCH}${DATA_TYPE}_${PERSISTENT}_${SPLIT_K}_${HAS_HOT_LOOP}_${TAIL_NUMBER}.cpp)
                            string(TOLOWER ${KERNEL_FILE} KERNEL_FILE)
                            configure_file(
                                ${CMAKE_CURRENT_SOURCE_DIR}/mxgemm/mx_flatmm_instance.cpp.in
                                ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_FILE}
                                @ONLY)
                            list(APPEND ${FILE_LIST} ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_FILE})
                        endforeach()
                    endforeach()
                endforeach()
            endforeach()
        endforeach()
    endforeach()
    set(${FILE_LIST} ${${FILE_LIST}} PARENT_SCOPE)
endfunction()
