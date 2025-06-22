
set(GEMM_DATATYPE fp16 fp8)

function(build_gemm_for_datatype datatype)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/")
    set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/user_provided_config_${datatype}.json")
    # Generate kernel list
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
                --working_path ${working_path}
                --config_json ${json_blob}
                --list_blobs
        RESULT_VARIABLE ret
    )
    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list kernels for ${datatype}: ${ret}")
    endif()
    file(STRINGS "${working_path}/gemm_instance_blobs.txt" codegen_blobs)
    set(blob_list ${codegen_blobs})

    add_custom_command(
        OUTPUT ${blob_list}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
                --working_path "${working_path}"
                --config_json "${json_blob}"
                --gen_blobs
    )

    set(cpp_files "")
    set(hpp_files "")
    foreach(blob IN LISTS blob_list)
        string(STRIP "${blob}" stripped_blob)
        if(stripped_blob MATCHES "\\.cpp$")
            list(APPEND cpp_files "${stripped_blob}")
        elseif(stripped_blob MATCHES "\\.hpp$")
            list(APPEND hpp_files "${stripped_blob}")
        endif()
    endforeach()
    
    #Object libraries (chunked)
    set(intermediate_libs "")
    set(chunk_size 15)
    list(LENGTH cpp_files total_files)
    math(EXPR num_chunks "( ${total_files} + ${chunk_size} - 1 ) / ${chunk_size}")

    foreach(i RANGE 0 ${num_chunks})
        math(EXPR start "${i} * ${chunk_size}")
        math(EXPR end "${start} + ${chunk_size} - 1")
        set(chunk_files "")
        foreach(j RANGE ${start} ${end})
            if(j LESS total_files)
                list(GET cpp_files ${j} f)
                list(APPEND chunk_files "${f}")
            endif()
        endforeach()
        if(chunk_files)
            add_library(gemm_objlib_${i}_${datatype} OBJECT ${chunk_files})
            target_include_directories(gemm_objlib_${i}_${datatype} PRIVATE "${working_path}")
            list(APPEND intermediate_libs $<TARGET_OBJECTS:gemm_objlib_${i}_${datatype}>)
        endif()
    endforeach()

    # Static library for instances
    add_library(gemm_template_instances_${datatype} STATIC ${intermediate_libs})
    target_include_directories(gemm_template_instances_${datatype} PRIVATE
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
