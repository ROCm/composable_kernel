set(GEMM_CODEGEN_KNOWN_APIS "single;benchmark")
set(GEMM_CODEGEN_KNOWN_APIS "single" CACHE STRING
    "semicolon-separated list of APIs to generate (${GEMM_CODEGEN_KNOWN_APIS}) & link, or \"all\".")
if(GEMM_CODEGEN_ENABLE_APIS  STREQUAL "all")
    set(GEMM_CODEGEN_ENABLE_APIS  ${GEMM_CODEGEN_KNOWN_APIS})
endif()

foreach(api ${GEMM_CODEGEN_ENABLE_APIS})
    if(NOT "${api}" IN_LIST GEMM_CODEGEN_KNOWN_APIS)
        message(FATAL_ERROR "${api} isn't a known api: ${GEMM_CODEGEN_KNOWN_APIS}.")
    endif()
endforeach()

# "single" is a must-have api for the fmha_fwd example, add it if not specified
if(NOT "single" IN_LIST GEMM_CODEGEN_ENABLE_APIS)
  list(APPEND GEMM_CODEGEN_ENABLE_APIS "single")
endif()

# generate a list of kernels, but not actually emit files at config sta
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/code_generate.py
            --api ${GEMM_CODEGEN_ENABLE_APIS}
            --working_path ${CMAKE_CURRENT_BINARY_DIR}
            -j ${CMAKE_CURRENT_LIST_DIR}/config.json
            --list_blobs
            RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "Fail to generate kernels via Python. ${ret}")
endif()

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/gemm_instance_blobs.txt GEMM_CODEGEN_BLOBS)

# add_custom_command(
#   OUTPUT
#   COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/code_generate.py
#   --api ${GEMM_CODEGEN_ENABLE_APIS} --working_path ${CMAKE_CURRENT_BINARY_DIR} -j ${CMAKE_CURRENT_LIST_DIR}/config.json
#   --gen_blobs
# )

# To ensure dependency for generating tensor_configuration.hpp before generating executable
add_custom_command(
    OUTPUT  ${GEMM_CODEGEN_BLOBS}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/code_generate.py
            --api ${GEMM_CODEGEN_ENABLE_APIS}
            --working_path ${CMAKE_CURRENT_BINARY_DIR}
            -j ${CMAKE_CURRENT_LIST_DIR}/config.json
            --gen_blobs
    DEPENDS ${GEMM_CODEGEN_BLOBS}
)
add_custom_target(TENSOR_CONFIG DEPENDS ${GEMM_CODEGEN_BLOBS})


set(EXECUTABLE_GEMM_INSTANCE "tile_gemm_codegen")
message("adding example ${EXECUTABLE_GEMM_INSTANCE}")

# use build as include directory
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_executable(${EXECUTABLE_GEMM_INSTANCE} EXCLUDE_FROM_ALL gemm_calc.cpp)
# Ensure EXECUTABLE_GEMM_INSTANCE depends on TENSOR_CONFIG
add_dependencies(${EXECUTABLE_GEMM_INSTANCE}  TENSOR_CONFIG)
target_include_directories(${EXECUTABLE_GEMM_INSTANCE} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${EXECUTABLE_GEMM_INSTANCE} PRIVATE ${GEMM_CODEGEN_BLOBS})

set(EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS)

list(APPEND EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS
     -Wno-undefined-func-template
     -Wno-float-equal
     --offload-compress)

target_compile_options(${EXECUTABLE_GEMM_INSTANCE} PRIVATE ${EXECUTABLE_GEMM_INSTANCE_COMPILE_OPTIONS})

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)