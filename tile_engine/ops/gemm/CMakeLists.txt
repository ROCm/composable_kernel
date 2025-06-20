
set(GEMM_DATATYPE fp16;fp8)

list(GET GEMM_DATATYPE 0 datatype)
#list(GET GEMM_DATATYPE ${idx} datatype)
set(GEMM_CODEGEN_CPP_FILES_FP16 "")
set(GEMM_CODEGEN_HPP_FILES_FP16 "")

set(json_blob_FP16 "${CMAKE_CURRENT_LIST_DIR}/configs/user_provided_config_${datatype}.json")
set(working_path_FP16 "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/")

# generate a list of kernels, but not actually emit files at config stage
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
            --working_path ${working_path_FP16}
            --config_json ${json_blob_FP16}
            --list_blobs
    RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
message( FATAL_ERROR "Fail to list kernels via Python. ${ret}")
endif()

file(STRINGS ${working_path_FP16}/gemm_instance_blobs.txt GEMM_CODEGEN_BLOBS_FP16)

add_custom_command(
    OUTPUT  ${GEMM_CODEGEN_BLOBS_FP16}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
            --working_path ${working_path_FP16}
            --config_json ${json_blob_FP16}
            --gen_blobs
)

foreach(blob ${GEMM_CODEGEN_BLOBS_FP16})
    string(STRIP "${blob}" stripped_blob)
    message(STATUS "file is: ${stripped_blob}")
    if(stripped_blob MATCHES "\\.cpp$")
        list(APPEND GEMM_CODEGEN_CPP_FILES_FP16 "${stripped_blob}")
    elseif(stripped_blob MATCHES "\\.hpp$")
        list(APPEND GEMM_CODEGEN_HPP_FILES_FP16 "${stripped_blob}")
    endif()
endforeach()


set(chunk_size_FP16 15)
set(intermediate_libs_FP16 "")

list(LENGTH GEMM_CODEGEN_CPP_FILES_FP16 total_files_FP16)
message(STATUS "Total gemm codegen files: ${total_files_FP16}")
math(EXPR num_chunks_FP16 "( ${total_files_FP16} + ${chunk_size_FP16} - 1 ) / ${chunk_size_FP16}")

foreach(i RANGE 0 ${num_chunks_FP16})
    set(chunk_files_FP16 "")
    math(EXPR start "${i} * ${chunk_size_FP16}")
    math(EXPR end "${start} + ${chunk_size_FP16} - 1")
    foreach(j RANGE ${start} ${end})
        if(j LESS total_files_FP16)
            list(GET GEMM_CODEGEN_CPP_FILES_FP16 ${j} file)
            list(APPEND chunk_files_FP16 ${file})
        endif()
    endforeach()
    if(chunk_files_FP16)
        set(lib_name_FP16 "gemm_objlib_${i}_${datatype}")
        add_library(${lib_name_FP16} OBJECT ${chunk_files_FP16})
        list(APPEND intermediate_libs_FP16 $<TARGET_OBJECTS:${lib_name_FP16}>)
    endif()
endforeach()
set(lib_name_FP16 "gemm_template_instances_${datatype}")
add_library(${lib_name_FP16} STATIC EXCLUDE_FROM_ALL ${intermediate_libs_FP16})

# Explicitly set LINKER_LANGUAGE to avoid build config failures with Ninja.
set_target_properties(${lib_name_FP16} PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(${lib_name_FP16} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${lib_name_FP16} PRIVATE ${GEMM_CODEGEN_HPP_FILES_FP16})

set(BENCHMARK_GEMM_EXECUTABLE_FP16 "benchmark_gemm_${datatype}")
message(DEBUG "adding example ${BENCHMARK_GEMM_EXECUTABLE}")

include_directories(${working_path})

set(lib_name_host_FP16 "gemm_host_api_${datatype}")
add_library(${lib_name_host_FP16} INTERFACE EXCLUDE_FROM_ALL)
target_include_directories(${lib_name_host_FP16} INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${lib_name_host_FP16} INTERFACE ${GEMM_CODEGEN_HPP_FILES_FP16} gemm_host_api.hpp)
target_link_libraries(${lib_name_host_FP16} INTERFACE ${lib_name_FP16})

add_executable(${BENCHMARK_GEMM_EXECUTABLE_FP16} EXCLUDE_FROM_ALL benchmark_gemm.cpp)
target_include_directories(${BENCHMARK_GEMM_EXECUTABLE_FP16} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${BENCHMARK_GEMM_EXECUTABLE_FP16} PRIVATE benchmark_gemm.hpp gemm_profiler.hpp)
target_link_libraries(${BENCHMARK_GEMM_EXECUTABLE_FP16} PRIVATE ${lib_name_host_FP16})

set(EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS)

list(APPEND EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS
    -Wno-undefined-func-template
    -Wno-float-equal
    --offload-compress)

target_compile_options(${BENCHMARK_GEMM_EXECUTABLE_FP16} PRIVATE ${EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS})

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)




list(GET GEMM_DATATYPE 1 datatype)
#list(GET GEMM_DATATYPE ${idx} datatype)
set(GEMM_CODEGEN_CPP_FILES "")
set(GEMM_CODEGEN_HPP_FILES "")

set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/user_provided_config_${datatype}.json")
set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/")

# generate a list of kernels, but not actually emit files at config stage
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
            --working_path ${working_path}
            --config_json ${json_blob}
            --list_blobs
    RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
message( FATAL_ERROR "Fail to list kernels via Python. ${ret}")
endif()

file(STRINGS ${working_path}/gemm_instance_blobs.txt GEMM_CODEGEN_BLOBS)

add_custom_command(
    OUTPUT  ${GEMM_CODEGEN_BLOBS}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
            --working_path ${working_path}
            --config_json ${json_blob}
            --gen_blobs
)

foreach(blob ${GEMM_CODEGEN_BLOBS})
    string(STRIP "${blob}" stripped_blob)
    message(STATUS "file is: ${stripped_blob}")
    if(stripped_blob MATCHES "\\.cpp$")
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
        set(lib_name "gemm_objlib_${i}_${datatype}")
        add_library(${lib_name} OBJECT ${chunk_files})
        list(APPEND intermediate_libs $<TARGET_OBJECTS:${lib_name}>)
    endif()
endforeach()
set(lib_name "gemm_template_instances_${datatype}")
add_library(${lib_name} STATIC EXCLUDE_FROM_ALL ${intermediate_libs})

# Explicitly set LINKER_LANGUAGE to avoid build config failures with Ninja.
set_target_properties(${lib_name} PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(${lib_name} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${lib_name} PRIVATE ${GEMM_CODEGEN_HPP_FILES})

set(BENCHMARK_GEMM_EXECUTABLE "benchmark_gemm_${datatype}")
message(DEBUG "adding example ${BENCHMARK_GEMM_EXECUTABLE}")

include_directories(${working_path})

set(lib_name_host "gemm_host_api_${datatype}")
add_library(${lib_name_host} INTERFACE EXCLUDE_FROM_ALL)
target_include_directories(${lib_name_host} INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${lib_name_host} INTERFACE ${GEMM_CODEGEN_HPP_FILES} gemm_host_api.hpp)
target_link_libraries(${lib_name_host} INTERFACE ${lib_name})

add_executable(${BENCHMARK_GEMM_EXECUTABLE} EXCLUDE_FROM_ALL benchmark_gemm.cpp)
target_include_directories(${BENCHMARK_GEMM_EXECUTABLE} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${BENCHMARK_GEMM_EXECUTABLE} PRIVATE benchmark_gemm.hpp gemm_profiler.hpp)
target_link_libraries(${BENCHMARK_GEMM_EXECUTABLE} PRIVATE ${lib_name_host})

set(EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS)

list(APPEND EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS
    -Wno-undefined-func-template
    -Wno-float-equal
    --offload-compress)

target_compile_options(${BENCHMARK_GEMM_EXECUTABLE} PRIVATE ${EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS})

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
