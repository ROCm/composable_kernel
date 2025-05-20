
# generate a list of kernels, but not actually emit files at config stage
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
            --working_path ${CMAKE_CURRENT_BINARY_DIR}
            --list_blobs
    RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "Fail to list kernels via Python. ${ret}")
endif()

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/gemm_instance_blobs.txt GEMM_CODEGEN_BLOBS)

set(GEMM_CODEGEN_CPP_FILES "")
set(GEMM_CODEGEN_HPP_FILES "")

foreach(blob ${GEMM_CODEGEN_BLOBS})
    string(STRIP "${blob}" stripped_blob)
    
    if(stripped_blob MATCHES "\\.cpp$")
        list(APPEND GEMM_CODEGEN_CPP_FILES "${stripped_blob}")
    elseif(stripped_blob MATCHES "\\.hpp$")
        list(APPEND GEMM_CODEGEN_HPP_FILES "${stripped_blob}")
    endif()
endforeach()

add_custom_command(
    OUTPUT  ${GEMM_CODEGEN_BLOBS}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
            --working_path ${CMAKE_CURRENT_BINARY_DIR}
            --gen_blobs
)

add_library(gemm_template_instances OBJECT ${GEMM_CODEGEN_CPP_FILES})
target_include_directories(gemm_template_instances PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(gemm_template_instances PRIVATE ${GEMM_CODEGEN_HPP_FILES})

set(EXECUTABLE_GEMM_INSTANCE "tile_engine_gemm")
message("adding example ${EXECUTABLE_GEMM_INSTANCE}")

include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_executable(${EXECUTABLE_GEMM_INSTANCE} EXCLUDE_FROM_ALL gemm_host_api.cpp)
target_include_directories(${EXECUTABLE_GEMM_INSTANCE} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${EXECUTABLE_GEMM_INSTANCE} PRIVATE ${GEMM_CODEGEN_HPP_FILES})
target_link_libraries(${EXECUTABLE_GEMM_INSTANCE} PRIVATE gemm_template_instances)

set(EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS)

list(APPEND EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS
     -Wno-undefined-func-template
     -Wno-float-equal
     --offload-compress)

target_compile_options(${EXECUTABLE_GEMM_INSTANCE} PRIVATE ${EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS})

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)