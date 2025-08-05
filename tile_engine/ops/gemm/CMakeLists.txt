
set(GEMM_DATATYPE "fp8;fp16" CACHE STRING "List of datatypes for GEMM (semicolon-separated)")
set(GEMM_LAYOUT "rcr" CACHE STRING "List of layout for GEMM (semicolon-separated)")

function(build_gemm_for_datatype datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Comment this if-else block when using user_provided_config
    if(layout STREQUAL "rcr")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/custom_ci_config.json")
    endif()

    # uncomment this if you want to use user_provided_config.json
    # set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/user_provided_config.json")
    
    # Generate kernel list
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${json_blob}
                --list_blobs
        RESULT_VARIABLE ret
    )
    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list kernels for ${datatype} ${layout}: ${ret}")
    endif()

    file(STRINGS "${working_path}/gemm_instance_blobs.txt" codegen_blobs)
    file(STRINGS "${working_path}/gemm_instance_blobs_range.txt" codegen_blobs_range)
    
    # Generate the blobs
    add_custom_command(
        OUTPUT ${codegen_blobs}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
                --working_path "${working_path}"
                --datatype ${datatype}
                --layout ${layout}
                --config_json "${json_blob}"
                --gen_blobs
        COMMENT "Generating GEMM instance sources for ${datatype} ${layout}"
    )
    add_custom_target(gemm_gen_${datatype}_${layout} DEPENDS ${codegen_blobs})

    set(intermediate_libs)
    list(LENGTH codegen_blobs codegen_blobs_len)

    foreach(blob IN LISTS codegen_blobs_range)
        string(STRIP "${blob}" stripped_blob)
        separate_arguments(spilit_blob UNIX_COMMAND "${stripped_blob}")
        # Each line is: <trait_name> <first_index_inclusive> <last_index_exclusive>   
        list(GET spilit_blob 0 name)
        list(GET spilit_blob 1 first)
        list(GET spilit_blob 2 last)
        math(EXPR total_files "${last} - ${first}")
        if(total_files EQUAL 0)
            continue()        # nothing for this trait
        endif()

        # Object libraries (chunked) per trait
        set(sub_intermediate_libs)
        set(chunk_size 3)
        math(EXPR num_chunks "( ${total_files} + ${chunk_size} - 1 ) / ${chunk_size}")
        math(EXPR num_chunks_minus_1 "${num_chunks} - 1")
        
        foreach(i RANGE 0 ${num_chunks_minus_1})
            math(EXPR start "${first} + ${i} * ${chunk_size} ")
            math(EXPR end "${start} + ${chunk_size} - 1")

            set(chunk_files)
            foreach(j RANGE ${start} ${end})
                if(j LESS ${last} AND j LESS ${codegen_blobs_len})
                    list(GET codegen_blobs ${j} f)
                    list(APPEND chunk_files "${f}")
                endif()
            endforeach()

            #list(LENGTH chunk_files chunk_files_len)
            #if(chunk_files_len AND chunk_files_len GREATER 1)
            if(chunk_files)
                set(sub_intermediate_lib_name "gemm_objlib_${name}_${i}_${datatype}_${layout}")
                add_library(${sub_intermediate_lib_name} OBJECT ${chunk_files})
                list(APPEND sub_intermediate_libs ${sub_intermediate_lib_name})
            endif()

        endforeach()

        # ------------------ Bundle the object libs into one static lib ---------
        #list(LENGTH sub_intermediate_libs sub_intermediate_libs_len)
        #if(sub_intermediate_libs AND sub_intermediate_libs_len GREATER 1)
        if(sub_intermediate_libs)
            set(intermediate_lib_name "gemm_staticlib_${name}_${datatype}_${layout}")
            # Collect the $<TARGET_OBJECTS:...> expressions
            
            set(obj_exprs)
            foreach(objlib IN LISTS sub_intermediate_libs)
                list(APPEND obj_exprs $<TARGET_OBJECTS:${objlib}>)
            endforeach()
            
            add_library(${intermediate_lib_name} STATIC ${obj_exprs})
            add_dependencies(${intermediate_lib_name} gemm_gen_${datatype}_${layout})
            #foreach(objlib IN LISTS sub_intermediate_libs)
            #    target_sources(${intermediate_lib_name} PRIVATE $<TARGET_OBJECTS:${objlib}>)
            #endforeach()
            list(APPEND intermediate_libs ${intermediate_lib_name})
        endif()

    endforeach()
    
    # Interface library for instances
    add_library(gemm_template_instances_${datatype}_${layout} INTERFACE)
    add_dependencies(gemm_template_instances_${datatype}_${layout} gemm_gen_${datatype}_${layout})
    target_link_libraries(gemm_template_instances_${datatype}_${layout} INTERFACE ${intermediate_libs})
    target_include_directories(gemm_template_instances_${datatype}_${layout} INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}
        "${working_path}"
    )
    set_target_properties(gemm_template_instances_${datatype}_${layout} PROPERTIES LINKER_LANGUAGE CXX)
    
    # Host API interface library
    add_library(gemm_host_api_${datatype}_${layout} INTERFACE)
    target_link_libraries(gemm_host_api_${datatype}_${layout} INTERFACE gemm_template_instances_${datatype}_${layout})
    target_include_directories(gemm_host_api_${datatype}_${layout} INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}
        "${working_path}"
    )
    

    # Executable per datatype
    set(exec_name "benchmark_gemm_${datatype}_${layout}")
    add_executable(${exec_name} benchmark_gemm.cpp)
    target_link_libraries(${exec_name} PRIVATE gemm_host_api_${datatype}_${layout})
    target_compile_options(${exec_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
    )
endfunction()

# Process each datatype in isolation
foreach(dt IN LISTS GEMM_DATATYPE)
    foreach(l IN LISTS GEMM_LAYOUT)
        build_gemm_for_datatype(${dt} ${l})
    endforeach()
endforeach()
