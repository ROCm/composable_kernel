set(GEMM_DATATYPE "fp8;fp16" CACHE STRING "List of datatypes for GEMM (semicolon-separated)")
set(GEMM_LAYOUT "rcr" CACHE STRING "List of layout for GEMM (semicolon-separated)")

# Option to enable parallel trait compilation
option(GEMM_PARALLEL_TRAITS "Build separate executables per trait for parallel compilation" ON)

# Pre-generate all kernel lists to avoid blocking during parallel builds
foreach(dt IN LISTS GEMM_DATATYPE)
    foreach(l IN LISTS GEMM_LAYOUT)
        set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${dt}/${l}")
        file(MAKE_DIRECTORY ${working_path})
        
        if(l STREQUAL "rcr")
            set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
        else()
            set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/custom_ci_config.json")
        endif()
        
        # Only run if files don't exist
        if(NOT EXISTS "${working_path}/gemm_instance_blobs.txt")
            execute_process(
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
                        --working_path ${working_path}
                        --datatype ${dt}
                        --layout ${l}
                        --config_json ${json_blob}
                        --list_blobs
                RESULT_VARIABLE ret
            )
            if(NOT ret EQUAL 0)
                message(FATAL_ERROR "Failed to pre-generate kernel list for ${dt} ${l}")
            endif()
        endif()
    endforeach()
endforeach()

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
    
    # Read pre-generated kernel lists (already created above)
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

    if(GEMM_PARALLEL_TRAITS)
        # Parse ranges to identify unique trait names
        set(unique_traits)
        foreach(range_line IN LISTS codegen_blobs_range)
            string(STRIP "${range_line}" stripped_line)
            separate_arguments(split_line UNIX_COMMAND "${stripped_line}")
            list(GET split_line 0 trait_name)
            list(APPEND unique_traits ${trait_name})
        endforeach()
        list(REMOVE_DUPLICATES unique_traits)
        
        # Build each trait separately
        foreach(trait IN LISTS unique_traits)
            set(trait_files)
            
            # Collect files for this trait
            foreach(range_line IN LISTS codegen_blobs_range)
                string(STRIP "${range_line}" stripped_line)
                separate_arguments(split_line UNIX_COMMAND "${stripped_line}")
                list(GET split_line 0 name)
                
                if(name STREQUAL trait)
                    list(GET split_line 1 first)
                    list(GET split_line 2 last)
                    math(EXPR total_files "${last} - ${first}")
                    
                    if(total_files GREATER 0)
                        foreach(j RANGE ${first} ${last}-1)
                            list(LENGTH codegen_blobs blobs_len)
                            if(j LESS blobs_len)
                                list(GET codegen_blobs ${j} f)
                                list(APPEND trait_files "${f}")
                            endif()
                        endforeach()
                    endif()
                endif()
            endforeach()
            
            if(trait_files)
                # Create object libraries with improved chunking
                set(chunk_size 12)  # Increased for better compilation performance
                list(LENGTH trait_files num_files)
                math(EXPR num_chunks "( ${num_files} + ${chunk_size} - 1 ) / ${chunk_size}")
                
                set(trait_obj_libs)
                foreach(i RANGE 0 ${num_chunks}-1)
                    math(EXPR start "${i} * ${chunk_size}")
                    math(EXPR end "${start} + ${chunk_size} - 1")
                    
                    set(chunk_files)
                    foreach(j RANGE ${start} ${end})
                        if(j LESS ${num_files})
                            list(GET trait_files ${j} f)
                            list(APPEND chunk_files "${f}")
                        endif()
                    endforeach()
                    
                    if(chunk_files)
                        set(obj_lib_name "gemm_obj_${trait}_${i}_${datatype}_${layout}")
                        add_library(${obj_lib_name} OBJECT ${chunk_files})
                        add_dependencies(${obj_lib_name} gemm_gen_${datatype}_${layout})
                        
                        # Faster compilation flags
                        target_compile_options(${obj_lib_name} PRIVATE
                            -Wno-undefined-func-template
                            -Wno-float-equal
                            --offload-compress
                            -O3
                            -fno-exceptions  # If exceptions not needed
                        )
                        
                        # Enable unity build for this object library
                        set_target_properties(${obj_lib_name} PROPERTIES
                            UNITY_BUILD ON
                            UNITY_BUILD_BATCH_SIZE 6
                        )
                        
                        list(APPEND trait_obj_libs ${obj_lib_name})
                    endif()
                endforeach()
                
                # Create static library for this trait
                if(trait_obj_libs)
                    set(trait_lib_name "gemm_lib_${trait}_${datatype}_${layout}")
                    
                    set(obj_exprs)
                    foreach(objlib IN LISTS trait_obj_libs)
                        list(APPEND obj_exprs $<TARGET_OBJECTS:${objlib}>)
                    endforeach()
                    
                    add_library(${trait_lib_name} STATIC ${obj_exprs})
                    add_dependencies(${trait_lib_name} gemm_gen_${datatype}_${layout})
                    
                    # Create trait-specific executable
                    set(exec_name "benchmark_gemm_${datatype}_${layout}_${trait}")
                    add_executable(${exec_name} benchmark_gemm.cpp)
                    target_link_libraries(${exec_name} PRIVATE ${trait_lib_name})
                    target_include_directories(${exec_name} PRIVATE
                        ${CMAKE_CURRENT_LIST_DIR}
                        "${working_path}"
                    )
                    target_compile_definitions(${exec_name} PRIVATE
                        GEMM_TRAIT_FILTER="${trait}"
                    )
                    target_compile_options(${exec_name} PRIVATE
                        -Wno-undefined-func-template
                        -Wno-float-equal
                        --offload-compress
                    )
                endif()
            endif()
        endforeach()
        
        # Create master executable that includes all traits
        set(all_trait_libs)
        foreach(trait IN LISTS unique_traits)
            if(TARGET gemm_lib_${trait}_${datatype}_${layout})
                list(APPEND all_trait_libs gemm_lib_${trait}_${datatype}_${layout})
            endif()
        endforeach()
        
        if(all_trait_libs)
            add_executable(benchmark_gemm_${datatype}_${layout} benchmark_gemm.cpp)
            target_link_libraries(benchmark_gemm_${datatype}_${layout} PRIVATE ${all_trait_libs})
            target_include_directories(benchmark_gemm_${datatype}_${layout} PRIVATE
                ${CMAKE_CURRENT_LIST_DIR}
                "${working_path}"
            )
            target_compile_options(benchmark_gemm_${datatype}_${layout} PRIVATE
                -Wno-undefined-func-template
                -Wno-float-equal
                --offload-compress
            )
        endif()
        
    else()
        # Original monolithic build (fallback)
        set(intermediate_libs)
        list(LENGTH codegen_blobs codegen_blobs_len)

        foreach(blob IN LISTS codegen_blobs_range)
            string(STRIP "${blob}" stripped_blob)
            separate_arguments(spilit_blob UNIX_COMMAND "${stripped_blob}")
            list(GET spilit_blob 0 name)
            list(GET spilit_blob 1 first)
            list(GET spilit_blob 2 last)
            math(EXPR total_files "${last} - ${first}")
            if(total_files EQUAL 0)
                continue()
            endif()

            set(sub_intermediate_libs)
            set(chunk_size 12)
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

                if(chunk_files)
                    set(sub_intermediate_lib_name "gemm_objlib_${name}_${i}_${datatype}_${layout}")
                    add_library(${sub_intermediate_lib_name} OBJECT ${chunk_files})
                    
                    # Enable unity build and faster compilation
                    set_target_properties(${sub_intermediate_lib_name} PROPERTIES
                        UNITY_BUILD ON
                        UNITY_BUILD_BATCH_SIZE 6
                    )
                    
                    target_compile_options(${sub_intermediate_lib_name} PRIVATE
                        -O3
                        -fvisibility=hidden
                        -fvisibility-inlines-hidden
                    )
                    
                    list(APPEND sub_intermediate_libs ${sub_intermediate_lib_name})
                endif()
            endforeach()

            if(sub_intermediate_libs)
                set(intermediate_lib_name "gemm_staticlib_${name}_${datatype}_${layout}")
                set(obj_exprs)
                foreach(objlib IN LISTS sub_intermediate_libs)
                    list(APPEND obj_exprs $<TARGET_OBJECTS:${objlib}>)
                endforeach()
                
                add_library(${intermediate_lib_name} STATIC ${obj_exprs})
                add_dependencies(${intermediate_lib_name} gemm_gen_${datatype}_${layout})
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
    endif()
endfunction()

# Process each datatype in isolation
foreach(dt IN LISTS GEMM_DATATYPE)
    foreach(l IN LISTS GEMM_LAYOUT)
        build_gemm_for_datatype(${dt} ${l})
    endforeach()
endforeach()

# Create master targets for parallel builds
set(ALL_GEMM_TARGETS)
foreach(dt IN LISTS GEMM_DATATYPE)
    foreach(l IN LISTS GEMM_LAYOUT)
        list(APPEND ALL_GEMM_TARGETS "benchmark_gemm_${dt}_${l}")
    endforeach()
endforeach()

add_custom_target(benchmark_gemm_all DEPENDS ${ALL_GEMM_TARGETS})

# Use faster linker if available
find_program(LLD_LINKER "ld.lld")
find_program(MOLD_LINKER "mold")
if(MOLD_LINKER)
    message(STATUS "Using mold linker for faster linking")
    add_link_options(-fuse-ld=mold)
elseif(LLD_LINKER)
    message(STATUS "Using lld linker for faster linking")
    add_link_options(-fuse-ld=lld)
endif()