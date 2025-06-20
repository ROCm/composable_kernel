
# generate a list of kernels, but not actually emit files at config stage
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
            --working_path ${CMAKE_CURRENT_BINARY_DIR}
            # --config_json ${CMAKE_CURRENT_LIST_DIR}/configs/user_provided_config.json
            --list_blobs
    RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "Fail to list kernels via Python. ${ret}")
endif()

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/gemm_instance_blobs.txt GEMM_CODEGEN_BLOBS)

set(GEMM_CODEGEN_CPP_FILES "")
set(GEMM_CODEGEN_HPP_FILES "")


add_custom_command(
    OUTPUT  ${GEMM_CODEGEN_BLOBS}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
            --working_path ${CMAKE_CURRENT_BINARY_DIR}
            # --config_json ${CMAKE_CURRENT_LIST_DIR}/configs/user_provided_config.json
            --gen_blobs
)

foreach(blob ${GEMM_CODEGEN_BLOBS})
    string(STRIP "${blob}" stripped_blob)
    
    if(stripped_blob MATCHES "\\.cpp$")
        message(STATUS "Adding gemm codegen file: ${stripped_blob}")
        list(APPEND GEMM_CODEGEN_CPP_FILES "${stripped_blob}")
    elseif(stripped_blob MATCHES "\\.hpp$")
        list(APPEND GEMM_CODEGEN_HPP_FILES "${stripped_blob}")
    endif()
endforeach()


set(chunk_size 15)
set(chunk_index 0)
set(intermediate_libs "")

list(LENGTH GEMM_CODEGEN_CPP_FILES total_files)
message(STATUS "Total gemm codegen files: ${total_files}")
math(EXPR num_chunks "( ${total_files} + ${chunk_size} - 1 ) / ${chunk_size}")

foreach(i RANGE 0 ${num_chunks})
    set(chunk_files "")
    math(EXPR start "${i} * ${chunk_size}")
    math(EXPR end "${start} + ${chunk_size} - 1")
    foreach(j RANGE ${start} ${end})
        if(j LESS total_files)
            list(GET GEMM_CODEGEN_CPP_FILES ${j} file)
            list(APPEND chunk_files ${file})
        endif()
    endforeach()
    if(chunk_files)
        set(lib_name "gemm_objlib_${i}")
        add_library(${lib_name} OBJECT ${chunk_files})
        list(APPEND intermediate_libs $<TARGET_OBJECTS:${lib_name}>)
    endif()
endforeach()
add_library(gemm_template_instances STATIC EXCLUDE_FROM_ALL ${intermediate_libs})

#add_library(gemm_template_instances STATIC EXCLUDE_FROM_ALL ${GEMM_CODEGEN_CPP_FILES})
# Explicitly set LINKER_LANGUAGE to avoid build config failures with Ninja.
set_target_properties(gemm_template_instances PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(gemm_template_instances PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(gemm_template_instances PRIVATE ${GEMM_CODEGEN_HPP_FILES})

set(BENCHMARK_GEMM_EXECUTABLE "benchmark_gemm")
message(DEBUG "adding example ${BENCHMARK_GEMM_EXECUTABLE}")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(gemm_host_api INTERFACE EXCLUDE_FROM_ALL)
target_include_directories(gemm_host_api INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_sources(gemm_host_api INTERFACE ${GEMM_CODEGEN_HPP_FILES} gemm_host_api.hpp)
target_link_libraries(gemm_host_api INTERFACE gemm_template_instances)

add_executable(${BENCHMARK_GEMM_EXECUTABLE} EXCLUDE_FROM_ALL benchmark_gemm.cpp)
target_include_directories(${BENCHMARK_GEMM_EXECUTABLE} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${BENCHMARK_GEMM_EXECUTABLE} PRIVATE benchmark_gemm.hpp gemm_profiler.hpp)
target_link_libraries(${BENCHMARK_GEMM_EXECUTABLE} PRIVATE gemm_host_api)

set(EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS)

list(APPEND EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS
     -Wno-undefined-func-template
     -Wno-float-equal
     --offload-compress)

target_compile_options(${BENCHMARK_GEMM_EXECUTABLE} PRIVATE ${EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS})

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)