cmake_minimum_required(VERSION 4.2)

enable_testing()

set(MULTI_REDUCE_DATATYPE "fp16" CACHE STRING "List of datatypes Multi Reduce (semicolon-separated)")
set(MULTI_REDUCE_VARIANTS "multiops_multiblock;multiops_threadwise" CACHE STRING "List of variants for Multi Reduce (semicolon-separated)")
set(MULTI_REDUCE_FUNCTIONS "add"  CACHE STRING "Reduction functions (elementwise ops, intrablock reduce, accumulator ops, interblock reduce)")

function(build_multi_reduce_for_datatype datatype variant)
    # Filter GPU targets to only gfx942, and gfx950
    set(GPU_TARGETS "")
    set(DESIRED_TARGETS "gfx942;gfx950")
    set(VALID_VARIANTS "multiops_multiblock;multiops_threadwise")
    
    foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
        if(target IN_LIST DESIRED_TARGETS)
            list(APPEND GPU_TARGETS ${target})
        endif()
    endforeach()
    
    # Skip compilation if no matching targets found
    if(NOT GPU_TARGETS)
        message(WARNING "Skipping Tile Engine for Multi Reduction Kernel: No supported GPU targets (gfx942, gfx950) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
        return()
    endif()
    
    message(STATUS "Building Reduction for GPU targets: ${GPU_TARGETS}")
    
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${variant}")
    file(MAKE_DIRECTORY "${working_path}")

    # Comment this if-else block when using user_provided_config
    if(variant IN_LIST VALID_VARIANTS)
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_multi_reduce_multiblock_config.json")
    else()
        # set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/custom_ci_config.json")
        message(WARNING "Unknown Multi Reduce variant: ${variant}.")
        return()
    endif()

    # uncomment this if you want to use user_provided_config.json
    # set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/user_provided_config.json")
    
    # Generate kernel list
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/reduce_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --variant ${variant}
                --config_json ${json_blob}
                --list_blobs
                --gpu_target ${GPU_TARGETS}
        RESULT_VARIABLE ret
    )
    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list kernels for ${datatype} ${variant}: ${ret}")
    endif()

    file(STRINGS "${working_path}/reduce_${variant}_blobs_list.txt" codegen_blobs)
    # file(STRINGS "${working_path}/gemm_multi_d_instance_blobs_range.txt" codegen_blobs_range)
    
    # Generate the blobs
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/reduce_instance_builder.py
                --working_path "${working_path}"
                --datatype ${datatype}
                --config_json "${json_blob}"
                --variant "${variant}"
                --gen_blobs
                --gpu_target ${GPU_TARGETS}
        RESULT_VARIABLE ret
    )
    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to generate kernels for ${datatype} ${variant}: ${ret}")
    endif()

    message(STATUS "Generated ${datatype} ${variant} reduction kernel blobs at: ${working_path}")

    # # Add test executables for each generated test
    file(STRINGS "${working_path}/reduce_${variant}_blobs_list.txt" test_basenames)

    foreach(test_base IN LISTS test_basenames)
        string(PREPEND test_base "test_")
        set(test_src "${working_path}/${test_base}.cpp")
        set(test_target "${test_base}")

        add_executable(${test_target} ${test_src})
        target_include_directories(${test_target} PRIVATE
            "${CMAKE_SOURCE_DIR}/test/ck_tile/reduce/"
            ${working_path}
        )

        target_compile_options(${test_target} PRIVATE -Wno-global-constructors -Wno-dev)
        target_link_libraries(${test_target} PRIVATE gtest gtest_main)

        add_test(NAME ${test_target} COMMAND ${test_target})
        set_tests_properties(${test_target} PROPERTIES LABELS "multi_reduce")
    endforeach()
    # add_custom_target(test_reduce_${variant}_${datatype} DEPENDS ${codegen_blobs})

    # # Generating a single binary from all the tests (debug-only)
    # set(test_srcs)
    # foreach(test_base IN LISTS test_basenames)
    #     list(APPEND test_srcs "${working_path}/test_${test_base}.cpp")
    # endforeach()

    # if(test_srcs)
    #     set(test_target "test_reduce_${variant}_${datatype}")
    #     add_executable(${test_target} ${test_srcs})
    #     target_include_directories(${test_target} PRIVATE
    #         ${working_path}
    #         "${CMAKE_SOURCE_DIR}/test/ck_tile/reduce/"
    #     )
    #     target_compile_options(${test_target} PRIVATE -Wno-global-constructors -Wno-dev)
    #     target_link_libraries(${test_target} PRIVATE gtest gtest_main)
    # endif()

    # set(intermediate_libs)
    # list(LENGTH codegen_blobs codegen_blobs_len)

    # foreach(blob IN LISTS codegen_blobs_range)
    #     string(STRIP "${blob}" stripped_blob)
    #     separate_arguments(spilit_blob UNIX_COMMAND "${stripped_blob}")
    #     # Each line is: <trait_name> <first_index_inclusive> <last_index_exclusive>   
    #     list(GET spilit_blob 0 name)
    #     list(GET spilit_blob 1 first)
    #     list(GET spilit_blob 2 last)
    #     math(EXPR total_files "${last} - ${first}")
    #     if(total_files EQUAL 0)
    #         continue()        # nothing for this trait
    #     endif()

    #     # Object libraries (chunked) per trait
    #     set(sub_intermediate_libs)
    #     set(chunk_size 3)
    #     math(EXPR num_chunks "( ${total_files} + ${chunk_size} - 1 ) / ${chunk_size}")
    #     math(EXPR num_chunks_minus_1 "${num_chunks} - 1")
        
    #     foreach(i RANGE 0 ${num_chunks_minus_1})
    #         math(EXPR start "${first} + ${i} * ${chunk_size} ")
    #         math(EXPR end "${start} + ${chunk_size} - 1")

    #         set(chunk_files)
    #         foreach(j RANGE ${start} ${end})
    #             if(j LESS ${last} AND j LESS ${codegen_blobs_len})
    #                 list(GET codegen_blobs ${j} f)
    #                 list(APPEND chunk_files "${f}")
    #             endif()
    #         endforeach()

    #         #list(LENGTH chunk_files chunk_files_len)
    #         #if(chunk_files_len AND chunk_files_len GREATER 1)
    #         if(chunk_files)
    #             set(sub_intermediate_lib_name "gemm_multi_d_objlib_${name}_${i}_${datatype}_${layout}")
    #             add_library(${sub_intermediate_lib_name} OBJECT ${chunk_files})
    #             set_property(TARGET ${sub_intermediate_lib_name} PROPERTY HIP_ARCHITECTURES ${GEMM_GPU_TARGETS})
    #             list(APPEND sub_intermediate_libs ${sub_intermediate_lib_name})
    #         endif()

    #     endforeach()

    #     # ------------------ Bundle the object libs into one static lib ---------
    #     #list(LENGTH sub_intermediate_libs sub_intermediate_libs_len)
    #     #if(sub_intermediate_libs AND sub_intermediate_libs_len GREATER 1)
    #     if(sub_intermediate_libs)
    #         set(intermediate_lib_name "gemm_multi_d_staticlib_${name}_${datatype}_${layout}")
    #         # Collect the $<TARGET_OBJECTS:...> expressions
            
    #         set(obj_exprs)
    #         foreach(objlib IN LISTS sub_intermediate_libs)
    #             list(APPEND obj_exprs $<TARGET_OBJECTS:${objlib}>)
    #         endforeach()
            
    #         add_library(${intermediate_lib_name} STATIC ${obj_exprs})
    #         add_dependencies(${intermediate_lib_name} gemm_multi_d_gen_${datatype}_${layout})
    #         set_property(TARGET ${intermediate_lib_name} PROPERTY HIP_ARCHITECTURES ${GEMM_GPU_TARGETS})
    #         #foreach(objlib IN LISTS sub_intermediate_libs)
    #         #    target_sources(${intermediate_lib_name} PRIVATE $<TARGET_OBJECTS:${objlib}>)
    #         #endforeach()
    #         list(APPEND intermediate_libs ${intermediate_lib_name})
    #     endif()

    # endforeach()
    
    # # Interface library for instances
    # add_library(gemm_multi_d_template_instances_${datatype}_${layout} INTERFACE)
    # add_dependencies(gemm_multi_d_template_instances_${datatype}_${layout} gemm_multi_d_gen_${datatype}_${layout})
    # target_link_libraries(gemm_multi_d_template_instances_${datatype}_${layout} INTERFACE ${intermediate_libs})
    # target_include_directories(gemm_multi_d_template_instances_${datatype}_${layout} INTERFACE
    #     ${CMAKE_CURRENT_LIST_DIR}
    #     "${working_path}"
    # )
    # set_target_properties(gemm_multi_d_template_instances_${datatype}_${layout} PROPERTIES LINKER_LANGUAGE CXX)
    
    # # Host API interface library
    # add_library(gemm_multi_d_host_api_${datatype}_${layout} INTERFACE)
    # target_link_libraries(gemm_multi_d_host_api_${datatype}_${layout} INTERFACE gemm_multi_d_template_instances_${datatype}_${layout})
    # target_include_directories(gemm_multi_d_host_api_${datatype}_${layout} INTERFACE
    #     ${CMAKE_CURRENT_LIST_DIR}
    #     "${working_path}"
    # )

    

    # # Executable per datatype
    # set(exec_name "benchmark_gemm_multi_d_${datatype}_${layout}")
    # add_executable(${exec_name} benchmark_gemm_multi_d.cpp)
    # set_property(TARGET ${exec_name} PROPERTY HIP_ARCHITECTURES ${GEMM_GPU_TARGETS})
    # target_link_libraries(${exec_name} PRIVATE gemm_multi_d_host_api_${datatype}_${layout})
    # target_compile_options(${exec_name} PRIVATE
    #     -Wno-undefined-func-template
    #     -Wno-float-equal
    #     --offload-compress
    # )
endfunction()

# Process each datatype in isolation
foreach(dt IN LISTS MULTI_REDUCE_DATATYPE)
    foreach(l IN LISTS MULTI_REDUCE_VARIANTS)
        build_multi_reduce_for_datatype(${dt} ${l})
    endforeach()
endforeach()