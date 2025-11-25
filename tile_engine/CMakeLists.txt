include_directories(BEFORE
        ${CMAKE_CURRENT_LIST_DIR}/include
    )

add_subdirectory(ops)
add_subdirectory(ops_new/gemm)
