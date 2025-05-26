

# generate a list of kernels, but not actually emit files at config stage
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
            --working_path ${CMAKE_CURRENT_BINARY_DIR}
            --json ${CMAKE_CURRENT_LIST_DIR}/configs/instance_combination.json
            --list_blobs
            RESULT_VARIABLE ret
)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
  ${CMAKE_CURRENT_LIST_DIR}/configs/instance_combination.json
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "Fail to generate kernels via Python. ${ret}")
endif()

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/gemm_instance_blobs.txt GEMM_CODEGEN_BLOBS)

add_custom_command(
    OUTPUT  ${GEMM_CODEGEN_BLOBS}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
            --working_path ${CMAKE_CURRENT_BINARY_DIR}
            --json ${CMAKE_CURRENT_LIST_DIR}/configs/instance_combination.json
            --gen_blobs
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/gemm_instance_builder.py
            ${CMAKE_CURRENT_BINARY_DIR}/gemm_instance_blobs.txt
            ${CMAKE_CURRENT_LIST_DIR}/configs/instance_combination.json
)

set(EXECUTABLE_GEMM_INSTANCE "tile_engine_gemm")
message("adding example ${EXECUTABLE_GEMM_INSTANCE}")

# use build as include directory
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_executable(${EXECUTABLE_GEMM_INSTANCE} EXCLUDE_FROM_ALL gemm_host_api.cpp)
target_include_directories(${EXECUTABLE_GEMM_INSTANCE} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${EXECUTABLE_GEMM_INSTANCE} PRIVATE ${GEMM_CODEGEN_BLOBS})

set(EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS)

list(APPEND EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS
     -Wno-undefined-func-template
     -Wno-float-equal
     --offload-compress)

target_compile_options(${EXECUTABLE_GEMM_INSTANCE} PRIVATE ${EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS})

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)