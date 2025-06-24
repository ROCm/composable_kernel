
#set(GEMM_DATATYPE fp8 fp16)
set(GEMM_DATATYPE "fp8;fp16" CACHE STRING "List of datatypes for GEMM (semicolon-separated)")

function(build_gemm_for_datatype datatype)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/")
    #set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config_${datatype}.json")
    set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/user_provided_config.json")
    # Generate kernel list
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --config_json ${json_blob}
                --list_blobs
        RESULT_VARIABLE ret
    )
    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list kernels for ${datatype}: ${ret}")
    endif()
    file(STRINGS "${working_path}/gemm_instance_blobs.txt" codegen_blobs)
    set(blob_list ${codegen_blobs})
    file(STRINGS "${working_path}/gemm_instance_blobs_range.txt" codegen_blobs_range)
    set(blob_range ${codegen_blobs_range})

    add_custom_command(
        OUTPUT ${blob_list}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
                --working_path "${working_path}"
                --datatype ${datatype}
                --config_json "${json_blob}"
                --gen_blobs
    )

    set(cpp_files "")
    set(hpp_files "")
    set(intermediate_libs "")
    list(LENGTH codegen_blobs codegen_blobs_len)
    foreach(blob IN LISTS blob_range)
        string(STRIP "${blob}" stripped_blob)
        #message(STATUS "stripped_blob is: ${stripped_blob}")
        separate_arguments(spilit_blob UNIX_COMMAND "${stripped_blob}")
        
        list(GET spilit_blob 0 name)
        list(GET spilit_blob 1 first)
        list(GET spilit_blob 2 last)
        math(EXPR total_files "${last} - ${first}")
        #message(STATUS "total files are: ${total_files}")
        #Object libraries (chunked)
        set(sub_intermediate_libs "")
        set(chunk_size 3)
        
        math(EXPR num_chunks "( ${total_files} + ${chunk_size} - 1 ) / ${chunk_size}")
        math(EXPR num_chunks_minus_1 "${num_chunks} - 1")
        #message(STATUS "total num_chunks are: ${num_chunks_minus_1}")
        foreach(i RANGE 0 ${num_chunks_minus_1})
            math(EXPR start "${i} * ${chunk_size} + ${first}")
            math(EXPR end "${start} + ${chunk_size} - 1")
            set(chunk_files "")
            #message(STATUS "start: ${start}, end: ${end}")
            foreach(j RANGE ${start} ${end})
                if(j LESS ${last} AND j LESS ${codegen_blobs_len})
                    list(GET codegen_blobs ${j} f)
                    list(APPEND chunk_files "${f}")
                    #message(STATUS "file name: ${f}")
                endif()
            endforeach()
            if(chunk_files)
                set(sub_intermediate_lib_name "gemm_objlib_${name}_${i}_${datatype}")
                #message(STATUS "sub lib name: ${sub_intermediate_lib_name}")
                add_library(${sub_intermediate_lib_name} OBJECT ${chunk_files})
                list(APPEND sub_intermediate_libs ${sub_intermediate_lib_name})
            endif()
        endforeach()
        set(intermediate_lib_name "gemm_staticlib_${name}_${datatype}")
        #message(STATUS "lib name: ${intermediate_lib_name}, linking ${sub_intermediate_libs}")
        add_library(${intermediate_lib_name} STATIC)
        foreach(objlib IN LISTS sub_intermediate_libs)
            target_sources(${intermediate_lib_name} PRIVATE $<TARGET_OBJECTS:${objlib}>)
        endforeach()
        #target_sources(${intermediate_lib_name} PRIVATE ${sub_intermediate_libs})
        list(APPEND intermediate_libs ${intermediate_lib_name})
        #message(STATUS "name: ${name}, first: ${first}, last: ${last}")   
    endforeach()

    #message(STATUS "intermediate libs are: ${intermediate_libs}")
    # Static library for instances
    add_library(gemm_template_instances_${datatype} INTERFACE)
    target_link_libraries(gemm_template_instances_${datatype} INTERFACE ${intermediate_libs})
    target_include_directories(gemm_template_instances_${datatype} INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}
        "${working_path}" 
    )
    set_target_properties(gemm_template_instances_${datatype} PROPERTIES LINKER_LANGUAGE CXX)
    
    
    # Host API interface library
    add_library(gemm_host_api_${datatype} INTERFACE)
    target_include_directories(gemm_host_api_${datatype} INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}
        "${working_path}"  
    )
    target_link_libraries(gemm_host_api_${datatype} INTERFACE gemm_template_instances_${datatype})

    # Executable per datatype
    set(exec_name "benchmark_gemm_${datatype}")
    add_executable(${exec_name} benchmark_gemm.cpp)
    target_link_libraries(${exec_name} PRIVATE gemm_host_api_${datatype})
    target_compile_options(${exec_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
    )
endfunction()

# Process each datatype in isolation
foreach(dt IN LISTS GEMM_DATATYPE)
    build_gemm_for_datatype(${dt})
endforeach()
