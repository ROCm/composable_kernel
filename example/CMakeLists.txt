include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

add_custom_target(examples)

function(add_example_executable EXAMPLE_NAME FILE_NAME)
    message("adding example ${EXAMPLE_NAME}")
    set(result 1)
    if(DEFINED DTYPES)
    foreach(source IN LISTS FILE_NAME)
        if((NOT DTYPES MATCHES "fp8" AND source MATCHES "fp8") OR
                (NOT DTYPES MATCHES "fp8" AND source MATCHES "_f8") OR
                (NOT DTYPES MATCHES "fp16" AND source MATCHES "fp16") OR
                (NOT DTYPES MATCHES "fp32" AND source MATCHES "fp32") OR
                (NOT DTYPES MATCHES "fp64" AND source MATCHES "fp64") OR
                (NOT DTYPES MATCHES "bf16" AND source MATCHES "bf16") OR
                (NOT DTYPES MATCHES "int8" AND source MATCHES "int8") OR
                (NOT DTYPES MATCHES "int8" AND source MATCHES "_i8"))
                        message("removing source file ${source} ")
                        list(REMOVE_ITEM FILE_NAME "${source}")
        endif()
    endforeach()
    endif()
    foreach(source IN LISTS FILE_NAME)
        if(NOT DEFINED DL_KERNELS AND source MATCHES "_dl")
            message("removing dl example ${source} ")
            list(REMOVE_ITEM FILE_NAME "${source}")
        endif()
    endforeach()
    #only continue if there are some source files left on the list
    if(FILE_NAME)
        add_executable(${EXAMPLE_NAME} ${FILE_NAME})
        target_link_libraries(${EXAMPLE_NAME} PRIVATE utility)
        add_test(NAME ${EXAMPLE_NAME} COMMAND $<TARGET_FILE:${EXAMPLE_NAME}> ${ARGN})
        add_dependencies(examples ${EXAMPLE_NAME})
        add_dependencies(check ${EXAMPLE_NAME})
        rocm_install(TARGETS ${EXAMPLE_NAME} COMPONENT examples)
        set(result 0)
    endif()
    #message("add_example returns ${result}")
    return(PROPAGATE result)
endfunction(add_example_executable EXAMPLE_NAME)

function(add_example_executable_no_testing EXAMPLE_NAME FILE_NAME)
    message("adding example ${EXAMPLE_NAME}")
    set(result 1)
    if(DEFINED DTYPES)
    foreach(source IN LISTS FILE_NAME)
        if((NOT DTYPES MATCHES "fp8" AND source MATCHES "fp8") OR
                (NOT DTYPES MATCHES "fp8" AND source MATCHES "_f8") OR
                (NOT DTYPES MATCHES "fp16" AND source MATCHES "fp16") OR
                (NOT DTYPES MATCHES "fp32" AND source MATCHES "fp32") OR
                (NOT DTYPES MATCHES "fp64" AND source MATCHES "fp64") OR
                (NOT DTYPES MATCHES "bf16" AND source MATCHES "bf16") OR
                (NOT DTYPES MATCHES "int8" AND source MATCHES "int8") OR
                (NOT DTYPES MATCHES "int8" AND source MATCHES "_i8"))
                        message("removing source file ${source} ")
                        list(REMOVE_ITEM FILE_NAME "${source}")
        endif()
    endforeach()
    endif()
    foreach(source IN LISTS FILE_NAME)
        if(NOT DEFINED DL_KERNELS AND source MATCHES "_dl")
            message("removing dl example ${source} ")
            list(REMOVE_ITEM FILE_NAME "${source}")
        endif()
    endforeach()
    #only continue if there are some source files left on the list
    if(FILE_NAME)
        add_executable(${EXAMPLE_NAME} ${FILE_NAME})
        target_link_libraries(${EXAMPLE_NAME} PRIVATE utility)
        add_dependencies(examples ${EXAMPLE_NAME})
        rocm_install(TARGETS ${EXAMPLE_NAME} COMPONENT examples)
        set(result 0)
    endif()
    #message("add_example returns ${result}")
    return(PROPAGATE result)
endfunction(add_example_executable_no_testing EXAMPLE_NAME)

# add all example subdir
file(GLOB dir_list LIST_DIRECTORIES true *)
FOREACH(subdir ${dir_list})
    IF(IS_DIRECTORY "${subdir}")
        add_subdirectory(${subdir})
    ENDIF()
ENDFOREACH()
