# Currently ck_tile_gemm is only built on gfx94/gfx95
set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
set(EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS
    -mllvm
    -enable-noalias-to-md-conversion=0
)

if(GPU_TARGETS MATCHES "gfx94" OR GPU_TARGETS MATCHES "gfx95")
    add_gtest_executable(test_ck_tile_gemm_pipeline_mem test_gemm_pipeline_mem.cpp)
    add_gtest_executable(test_ck_tile_gemm_pipeline_compv3 test_gemm_pipeline_compv3.cpp)
    add_gtest_executable(test_ck_tile_gemm_pipeline_compv4 test_gemm_pipeline_compv4.cpp)

    target_compile_options(test_ck_tile_gemm_pipeline_mem PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_options(test_ck_tile_gemm_pipeline_compv3 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_options(test_ck_tile_gemm_pipeline_compv4 PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS})
else()
    message(DEBUG "Skipping ck_tile_gemm tests for current target")
endif()

if(GPU_TARGETS MATCHES "gfx94" OR GPU_TARGETS MATCHES "gfx95" OR GPU_TARGETS MATCHES "gfx90a")
    add_gtest_executable(test_ck_tile_gemm_pipeline_persistent test_gemm_pipeline_persistent.cpp)
    target_compile_options(test_ck_tile_gemm_pipeline_persistent PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

    function(generate_ck_tile_test_instances datatype)
        set(working_path "${CMAKE_CURRENT_BINARY_DIR}/test_instances_${datatype}/")
        set(json_config "${CMAKE_CURRENT_LIST_DIR}/configs/default_config_sample.json")
        
        # List blobs
        execute_process(
            COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm/gemm_instance_builder.py
                    --working_path ${working_path}
                    --datatype ${datatype}
                    --config_json ${json_config}
                    --list_blobs
            RESULT_VARIABLE ret
        )
        if(NOT ret EQUAL 0)
            message(FATAL_ERROR "Failed to list test kernels for ${datatype}: ${ret}")
        endif()
        
        file(STRINGS "${working_path}/gemm_instance_blobs.txt" test_codegen_blobs)
        file(STRINGS "${working_path}/gemm_instance_blobs_range.txt" test_codegen_blobs_range)
        
        # Generate the kernel blobs
        add_custom_command(
            OUTPUT ${test_codegen_blobs}
            COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm/gemm_instance_builder.py
                    --working_path "${working_path}"
                    --datatype ${datatype}
                    --config_json "${json_config}"
                    --gen_blobs
            DEPENDS ${json_config}
                    ${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm/gemm_instance_builder.py
            COMMENT "Generating CK Tile GEMM test instance sources for ${datatype}"
        )

        # Create custom target for generated blobs
        add_custom_target(ck_tile_test_gen_${datatype} DEPENDS ${test_codegen_blobs})

        # Build object libraries with chunking 
        set(test_intermediate_libs)
        list(LENGTH test_codegen_blobs test_codegen_blobs_len)

        foreach(blob IN LISTS test_codegen_blobs_range)
            string(STRIP "${blob}" stripped_blob)
            separate_arguments(split_blob UNIX_COMMAND "${stripped_blob}")
            # Each line is: <trait_name> <first_index_inclusive> <last_index_exclusive>   
            list(GET split_blob 0 name)
            list(GET split_blob 1 first)
            list(GET split_blob 2 last)
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
                    if(j LESS ${last} AND j LESS ${test_codegen_blobs_len})
                        list(GET test_codegen_blobs ${j} f)
                        list(APPEND chunk_files "${f}")
                    endif()
                endforeach()

                if(chunk_files)
                    set(sub_intermediate_lib_name "test_gemm_objlib_${name}_${i}_${datatype}")
                    add_library(${sub_intermediate_lib_name} OBJECT ${chunk_files})
                    add_dependencies(${sub_intermediate_lib_name} ck_tile_test_gen_${datatype})
                    list(APPEND sub_intermediate_libs ${sub_intermediate_lib_name})
                endif()
            endforeach()

            # Bundle the object libs into one static lib
            if(sub_intermediate_libs)
                set(intermediate_lib_name "test_gemm_staticlib_${name}_${datatype}")
                # Collect the $<TARGET_OBJECTS:...> expressions
                set(obj_exprs)
                foreach(objlib IN LISTS sub_intermediate_libs)
                    list(APPEND obj_exprs $<TARGET_OBJECTS:${objlib}>)
                endforeach()
                
                add_library(${intermediate_lib_name} STATIC ${obj_exprs})
                add_dependencies(${intermediate_lib_name} ck_tile_test_gen_${datatype})
                list(APPEND test_intermediate_libs ${intermediate_lib_name})
            endif()
        endforeach()
        
        # Create interface library for test instances
        add_library(test_gemm_instances_${datatype} INTERFACE)
        add_dependencies(test_gemm_instances_${datatype} ck_tile_test_gen_${datatype})
        target_include_directories(test_gemm_instances_${datatype} INTERFACE
            ${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm
            "${working_path}"
        )
        set_target_properties(test_gemm_instances_${datatype} PROPERTIES LINKER_LANGUAGE CXX)
    endfunction()
    
    # Generate test instances
    generate_ck_tile_test_instances(fp16)
    generate_ck_tile_test_instances(bf16)


    # Add test executables for each datatype
    add_gtest_executable(test_ck_tile_gemm_instance_validation_fp16 test_gemm_instance_validation_fp16.cpp)
    target_link_libraries(test_ck_tile_gemm_instance_validation_fp16 PRIVATE test_gemm_instances_fp16)
    target_compile_options(test_ck_tile_gemm_instance_validation_fp16 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_include_directories(test_ck_tile_gemm_instance_validation_fp16 PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/test_instances_fp16
        ${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm
    )

    add_gtest_executable(test_ck_tile_gemm_instance_validation_bf16 test_gemm_instance_validation_bf16.cpp)
    target_link_libraries(test_ck_tile_gemm_instance_validation_bf16 PRIVATE test_gemm_instances_bf16)
    target_compile_options(test_ck_tile_gemm_instance_validation_bf16 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_include_directories(test_ck_tile_gemm_instance_validation_bf16 PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/test_instances_bf16
        ${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm
)

endif()
