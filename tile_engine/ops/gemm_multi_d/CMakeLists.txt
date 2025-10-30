
set(GEMM_MULTI_D_DATATYPE "fp16" CACHE STRING "List of datatypes for GEMM Multi D (semicolon-separated)")
set(GEMM_MULTI_D_LAYOUT "rcrr" CACHE STRING "List of layout for GEMM Multi D(semicolon-separated)")
set(GEMM_MULTI_D_ELEMENTWISE_FUNCTION "mul"  CACHE STRING "Elementwise function")

function(build_gemm_multi_d_for_datatype_layout datatype layout)
    # Filter GPU targets to only gfx90a, gfx942, and gfx950
    set(GEMM_GPU_TARGETS "")
    set(DESIRED_TARGETS "gfx90a;gfx942;gfx950")
    
    foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
        if(target IN_LIST DESIRED_TARGETS)
            list(APPEND GEMM_GPU_TARGETS ${target})
        endif()
    endforeach()
    
    # Skip compilation if no matching targets found
    if(NOT GEMM_GPU_TARGETS)
        message(WARNING "Skipping Tile Engine GEMM Multi D compilation: No supported GPU targets (gfx90a, gfx942, gfx950) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
        return()
    endif()
    
    message(STATUS "Building GEMM Multi D for GPU targets: ${GEMM_GPU_TARGETS}")
    
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Comment this if-else block when using user_provided_config
    if(layout STREQUAL "rcrr")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/custom_ci_config.json")
    endif()

    # uncomment this if you want to use user_provided_config.json
    # set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/user_provided_config.json")
    
    # Generate kernel list
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_multi_d_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --elementwise_function ${GEMM_MULTI_D_ELEMENTWISE_FUNCTION}
                --config_json ${json_blob}
                --list_blobs
                --gpu_target ${GEMM_GPU_TARGETS}
        RESULT_VARIABLE ret
    )
    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list kernels for ${datatype} ${layout}: ${ret}")
    endif()

    file(STRINGS "${working_path}/gemm_multi_d_instance_blobs.txt" codegen_blobs)
    file(STRINGS "${working_path}/gemm_multi_d_instance_blobs_range.txt" codegen_blobs_range)
    
    # Generate the blobs
    add_custom_command(
        OUTPUT ${codegen_blobs}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_multi_d_instance_builder.py
                --working_path "${working_path}"
                --datatype ${datatype}
                --layout ${layout}
                --elementwise_function ${GEMM_MULTI_D_ELEMENTWISE_FUNCTION}
                --config_json "${json_blob}"
                --gen_blobs
                --gpu_target ${GEMM_GPU_TARGETS}
        COMMENT "Generating GEMM Multi D instance sources for ${datatype} ${layout}"
    )
    add_custom_target(gemm_multi_d_gen_${datatype}_${layout} DEPENDS ${codegen_blobs})

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
                set(sub_intermediate_lib_name "gemm_multi_d_objlib_${name}_${i}_${datatype}_${layout}")
                add_library(${sub_intermediate_lib_name} OBJECT ${chunk_files})
                set_property(TARGET ${sub_intermediate_lib_name} PROPERTY HIP_ARCHITECTURES ${GEMM_GPU_TARGETS})
                list(APPEND sub_intermediate_libs ${sub_intermediate_lib_name})
            endif()

        endforeach()

        # ------------------ Bundle the object libs into one static lib ---------
        #list(LENGTH sub_intermediate_libs sub_intermediate_libs_len)
        #if(sub_intermediate_libs AND sub_intermediate_libs_len GREATER 1)
        if(sub_intermediate_libs)
            set(intermediate_lib_name "gemm_multi_d_staticlib_${name}_${datatype}_${layout}")
            # Collect the $<TARGET_OBJECTS:...> expressions
            
            set(obj_exprs)
            foreach(objlib IN LISTS sub_intermediate_libs)
                list(APPEND obj_exprs $<TARGET_OBJECTS:${objlib}>)
            endforeach()
            
            add_library(${intermediate_lib_name} STATIC ${obj_exprs})
            add_dependencies(${intermediate_lib_name} gemm_multi_d_gen_${datatype}_${layout})
            set_property(TARGET ${intermediate_lib_name} PROPERTY HIP_ARCHITECTURES ${GEMM_GPU_TARGETS})
            #foreach(objlib IN LISTS sub_intermediate_libs)
            #    target_sources(${intermediate_lib_name} PRIVATE $<TARGET_OBJECTS:${objlib}>)
            #endforeach()
            list(APPEND intermediate_libs ${intermediate_lib_name})
        endif()

    endforeach()
    
    # Interface library for instances
    add_library(gemm_multi_d_template_instances_${datatype}_${layout} INTERFACE)
    add_dependencies(gemm_multi_d_template_instances_${datatype}_${layout} gemm_multi_d_gen_${datatype}_${layout})
    target_link_libraries(gemm_multi_d_template_instances_${datatype}_${layout} INTERFACE ${intermediate_libs})
    target_include_directories(gemm_multi_d_template_instances_${datatype}_${layout} INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}
        "${working_path}"
    )
    set_target_properties(gemm_multi_d_template_instances_${datatype}_${layout} PROPERTIES LINKER_LANGUAGE CXX)
    
    # Host API interface library
    add_library(gemm_multi_d_host_api_${datatype}_${layout} INTERFACE)
    target_link_libraries(gemm_multi_d_host_api_${datatype}_${layout} INTERFACE gemm_multi_d_template_instances_${datatype}_${layout})
    target_include_directories(gemm_multi_d_host_api_${datatype}_${layout} INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}
        "${working_path}"
    )

    

    # Executable per datatype
    set(exec_name "benchmark_gemm_multi_d_${datatype}_${layout}")
    add_executable(${exec_name} benchmark_gemm_multi_d.cpp)
    set_property(TARGET ${exec_name} PROPERTY HIP_ARCHITECTURES ${GEMM_GPU_TARGETS})
    target_link_libraries(${exec_name} PRIVATE gemm_multi_d_host_api_${datatype}_${layout})
    target_compile_options(${exec_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
    )
endfunction()

# Process each datatype in isolation
foreach(dt IN LISTS GEMM_MULTI_D_DATATYPE)
    foreach(l IN LISTS GEMM_MULTI_D_LAYOUT)
        build_gemm_multi_d_for_datatype_layout(${dt} ${l})
    endforeach()
endforeach()
