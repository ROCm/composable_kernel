set(GEMM_DATATYPE "fp8;fp16" CACHE STRING "List of datatypes for GEMM (semicolon-separated)")
set(GEMM_LAYOUT "rcr" CACHE STRING "List of layout for GEMM (semicolon-separated)")

# Pre-generate all kernel lists to avoid blocking during parallel builds
foreach(dt IN LISTS GEMM_DATATYPE)
    foreach(l IN LISTS GEMM_LAYOUT)
        set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${dt}/${l}")
        file(MAKE_DIRECTORY "${working_path}")

        if (l STREQUAL "rcr")
            set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
        else()
            set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/custom_ci_config.json")
        endif()

        # Only run if files don't exist
        if (NOT EXISTS "${working_path}/gemm_instance_blobs.txt")
            execute_process(
                COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py"
                        --working_path "${working_path}"
                        --datatype "${dt}"
                        --layout "${l}"
                        --config_json "${json_blob}"
                        --list_blobs
                RESULT_VARIABLE ret
            )
            if (NOT ret EQUAL 0)
                message(FATAL_ERROR "Failed to pre-generate kernel list for ${dt} ${l}")
            endif()
        endif()
    endforeach()
endforeach()

function(build_gemm_for_datatype datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    if (layout STREQUAL "rcr")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/custom_ci_config.json")
    endif()
    # Uncomment to override:
    # set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/user_provided_config.json")

    # Read pre-generated kernel lists
    file(STRINGS "${working_path}/gemm_instance_blobs.txt" codegen_blobs)
    file(STRINGS "${working_path}/gemm_instance_blobs_range.txt" codegen_blobs_range)

    # Generate the blobs
    add_custom_command(
        OUTPUT ${codegen_blobs}
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py"
                --working_path "${working_path}"
                --datatype "${datatype}"
                --layout "${layout}"
                --config_json "${json_blob}"
                --gen_blobs
        COMMENT "Generating GEMM instance sources for ${datatype} ${layout}"
    )
    add_custom_target(gemm_gen_${datatype}_${layout} DEPENDS ${codegen_blobs})

    # Parse ranges to identify unique trait names
    set(unique_traits)
    foreach(range_line IN LISTS codegen_blobs_range)
        string(STRIP "${range_line}" stripped_line)
        separate_arguments(split_line UNIX_COMMAND "${stripped_line}")
        list(GET split_line 0 trait_name)
        list(APPEND unique_traits "${trait_name}")
    endforeach()
    list(REMOVE_DUPLICATES unique_traits)

    # Build each trait separately
    foreach(trait IN LISTS unique_traits)
        set(trait_files)
        foreach(range_line IN LISTS codegen_blobs_range)
            string(STRIP "${range_line}" stripped_line)
            separate_arguments(split_line UNIX_COMMAND "${stripped_line}")
            list(GET split_line 0 name)
            if (name STREQUAL trait)
                list(GET split_line 1 first)
                list(GET split_line 2 last)
                math(EXPR total_files "${last} - ${first}")
                if (total_files GREATER 0)
                    foreach(j RANGE ${first} ${last}-1)
                        list(LENGTH codegen_blobs blobs_len)
                        if (j LESS blobs_len)
                            list(GET codegen_blobs ${j} f)
                            list(APPEND trait_files "${f}")
                        endif()
                    endforeach()
                endif()
            endif()
        endforeach()

        if (trait_files)
            # Create object libraries with chunking
            set(chunk_size 3)  # adjust as needed for memory vs parallelism
            list(LENGTH trait_files num_files)
            math(EXPR num_chunks "( ${num_files} + ${chunk_size} - 1 ) / ${chunk_size}")

            set(trait_obj_libs)
            foreach(i RANGE 0 ${num_chunks}-1)
                math(EXPR start "${i} * ${chunk_size}")
                math(EXPR end "${start} + ${chunk_size} - 1")

                set(chunk_files)
                foreach(j RANGE ${start} ${end})
                    if (j LESS ${num_files})
                        list(GET trait_files ${j} f)
                        list(APPEND chunk_files "${f}")
                    endif()
                endforeach()

                if (chunk_files)
                    set(obj_lib_name "gemm_obj_${trait}_${i}_${datatype}_${layout}")
                    add_library(${obj_lib_name} OBJECT ${chunk_files})
                    add_dependencies(${obj_lib_name} gemm_gen_${datatype}_${layout})

                    target_compile_options(${obj_lib_name} PRIVATE
                        -Wno-undefined-func-template
                        -Wno-float-equal
                        --offload-compress
                        -O3
                        -fno-exceptions
                    )

                    set_target_properties(${obj_lib_name} PROPERTIES
                        UNITY_BUILD ON
                        UNITY_BUILD_BATCH_SIZE 2
                    )

                    list(APPEND trait_obj_libs "${obj_lib_name}")
                endif()
            endforeach()

            # Static library for this trait
            if (trait_obj_libs)
                set(trait_lib_name "gemm_lib_${trait}_${datatype}_${layout}")
                set(obj_exprs)
                foreach(objlib IN LISTS trait_obj_libs)
                    list(APPEND obj_exprs "$<TARGET_OBJECTS:${objlib}>")
                endforeach()

                add_library(${trait_lib_name} STATIC ${obj_exprs})
                add_dependencies(${trait_lib_name} gemm_gen_${datatype}_${layout})

                # Trait-specific executable
                set(exec_name "benchmark_gemm_${datatype}_${layout}_${trait}")
                add_executable(${exec_name} benchmark_gemm.cpp)
                target_link_libraries(${exec_name} PRIVATE ${trait_lib_name})
                target_include_directories(${exec_name} PRIVATE
                    "${CMAKE_CURRENT_LIST_DIR}"
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

    # Master executable including all traits
    set(all_trait_libs)
    foreach(trait IN LISTS unique_traits)
        if (TARGET gemm_lib_${trait}_${datatype}_${layout})
            list(APPEND all_trait_libs "gemm_lib_${trait}_${datatype}_${layout}")
        endif()
    endforeach()

    if (all_trait_libs)
        add_executable(benchmark_gemm_${datatype}_${layout} benchmark_gemm.cpp)
        target_link_libraries(benchmark_gemm_${datatype}_${layout} PRIVATE ${all_trait_libs})
        target_include_directories(benchmark_gemm_${datatype}_${layout} PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}"
            "${working_path}"
        )
        target_compile_options(benchmark_gemm_${datatype}_${layout} PRIVATE
            -Wno-undefined-func-template
            -Wno-float-equal
            --offload-compress
        )
    endif()
endfunction()

# Process each datatype/layout
foreach(dt IN LISTS GEMM_DATATYPE)
    foreach(l IN LISTS GEMM_LAYOUT)
        build_gemm_for_datatype("${dt}" "${l}")
    endforeach()
endforeach()

# Master target for parallel builds
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
if (MOLD_LINKER)
    message(STATUS "Using mold linker for faster linking")
    add_link_options(-fuse-ld=mold)
elseif (LLD_LINKER)
    message(STATUS "Using lld linker for faster linking")
    add_link_options(-fuse-ld=lld)
endif()