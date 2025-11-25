function(mx_flatmm_instance_generate FILE_LIST)
    set(FLATMM_CONFIG MXfp4_FlatmmConfig16)
    set(A_DATA_TYPE FP4)
    set(B_DATA_TYPE FP4)
    set(C_DATA_TYPE FP16)
    set(A_LAYOUT ROW)
    set(B_LAYOUT COL)
    set(C_LAYOUT ROW)

    # foreach(PERSISTENT false true)
    # TODO: Persistent kernels are disabled due to compilation failures with some LLVM versions.  
    foreach(PERSISTENT false)
        foreach(SPLIT_K false true)
            foreach(HAS_HOT_LOOP false true)
                foreach(TAIL_NUMBER ODD EVEN)
                    set(KERNEL_FILE mxgemm/mx_flatmm_instance_${PERSISTENT}_${SPLIT_K}_${HAS_HOT_LOOP}_${TAIL_NUMBER}.cpp)
                    configure_file(
                        ${CMAKE_CURRENT_SOURCE_DIR}/mxgemm/mx_flatmm_instance.cpp.in
                        ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_FILE}
                        @ONLY)
                    list(APPEND ${FILE_LIST} ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_FILE})
                endforeach()
            endforeach()
        endforeach()
    endforeach()
    set(${FILE_LIST} ${${FILE_LIST}} PARENT_SCOPE)
endfunction()
