# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(LAYERNORM2D_FWD_KNOWN_APIS "fwd;bwd")
set(LAYERNORM2D_FWD_ENABLE_APIS  "fwd" CACHE STRING
    "semicolon-separated list of APIs to generate (${LAYERNORM2D_FWD_KNOWN_APIS}) & link, or \"all\".")
if(LAYERNORM2D_FWD_ENABLE_APIS  STREQUAL "all")
  set(LAYERNORM2D_FWD_ENABLE_APIS  ${LAYERNORM2D_FWD_KNOWN_APIS})
endif()

set(LAYERNORM2D_FWD_INSTANCE_FILTER "" CACHE STRING
    "fnmatch pattern selecting LayerNorm forward instance names; empty generates all")
set(_layernorm_filter_args)
if(NOT LAYERNORM2D_FWD_INSTANCE_FILTER STREQUAL "")
    list(APPEND _layernorm_filter_args --filter "${LAYERNORM2D_FWD_INSTANCE_FILTER}")
endif()

# generate a list of kernels, but not actually emit files at config sta
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api ${LAYERNORM2D_FWD_ENABLE_APIS} --working_path ${CMAKE_CURRENT_BINARY_DIR} --list_blobs ${_layernorm_filter_args}
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "Fail to generate kernels via Python. ${ret}")
endif()

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/layernorm2d_fwd_blobs.txt LAYERNORM2D_FWD_GEN_BLOBS)

add_custom_command(
  OUTPUT ${LAYERNORM2D_FWD_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api ${LAYERNORM2D_FWD_ENABLE_APIS} --working_path ${CMAKE_CURRENT_BINARY_DIR} --gen_blobs ${_layernorm_filter_args}
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/generate.py
  VERBATIM
)

set(EXAMPLE_LAYERNORM2D_FWD "tile_example_layernorm2d_fwd")

message(DEBUG "adding example ${EXAMPLE_LAYERNORM2D_FWD}")
add_executable(${EXAMPLE_LAYERNORM2D_FWD} layernorm2d_fwd.cpp)
target_include_directories(${EXAMPLE_LAYERNORM2D_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${EXAMPLE_LAYERNORM2D_FWD} PRIVATE ${LAYERNORM2D_FWD_GEN_BLOBS})

set(EXAMPLE_LAYERNORM2D_FWD_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_LAYERNORM2D_FWD_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal --offload-compress)

target_compile_options(${EXAMPLE_LAYERNORM2D_FWD} PRIVATE ${EXAMPLE_LAYERNORM2D_FWD_COMPILE_OPTIONS})

if(BUILD_TESTING)
    add_test(
        NAME test_ck_tile_layernorm2d_fwd_codegen
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_generate.py -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(test_ck_tile_layernorm2d_fwd_codegen PROPERTIES
        LABELS "CK_TILE_LAYERNORM2D_TESTS;CK_TILE_LAYERNORM2D_CODEGEN_TESTS")
endif()

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
