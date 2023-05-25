#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
find_program(EMBED_LD ld)
find_program(EMBED_OBJCOPY objcopy)

function(generate_embed_source EMBED_NAME)
    set(options)
    set(oneValueArgs SRC HEADER RELATIVE)
    set(multiValueArgs OBJECTS SYMBOLS FILES)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(EXTERNS)
    set(INIT_KERNELS)

    list(LENGTH PARSE_SYMBOLS SYMBOLS_LEN)
    list(LENGTH PARSE_OBJECTS OBJECTS_LEN)
    if(NOT ${SYMBOLS_LEN} EQUAL ${OBJECTS_LEN})
        message(FATAL_ERROR "Symbols and objects dont match: ${SYMBOLS_LEN} != ${OBJECTS_LEN}")
    endif()
    math(EXPR LEN "${SYMBOLS_LEN} - 1")

    foreach(idx RANGE ${LEN})
        list(GET PARSE_SYMBOLS ${idx} SYMBOL)
        list(GET PARSE_OBJECTS ${idx} OBJECT)
        list(GET PARSE_FILES ${idx} FILE)
        set(START_SYMBOL "_binary_${SYMBOL}_start")
        set(END_SYMBOL "_binary_${SYMBOL}_end")
        string(APPEND EXTERNS "
            extern const char ${START_SYMBOL}[];
            extern const char ${END_SYMBOL}[];
        ")

        
        file(RELATIVE_PATH BASE_NAME ${PARSE_RELATIVE} "${FILE}")

        string(APPEND INIT_KERNELS "
            { \"${BASE_NAME}\", { ${START_SYMBOL}, ${END_SYMBOL}} },
        ")
    endforeach()

    file(WRITE "${PARSE_HEADER}" "
#include <unordered_map>
#include <string>
#include <utility>
const std::unordered_map<std::string, std::pair<const char*,const char*>>& ${EMBED_NAME}();
")

    file(WRITE "${PARSE_SRC}" "
#include <${EMBED_NAME}.hpp>
${EXTERNS}
const std::unordered_map<std::string, std::pair<const char*,const char*>>& ${EMBED_NAME}()
{
    static const std::unordered_map<std::string, std::pair<const char*,const char*>> result = {${INIT_KERNELS}};
    return result;
}
")
endfunction()

function(embed_file OUTPUT_FILE OUTPUT_SYMBOL FILE)
    set(WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    # Glob is used to compute the relative path
    file(GLOB FILES RELATIVE ${WORKING_DIRECTORY} ${FILE})
    foreach(REL_FILE ${FILES})
        string(MAKE_C_IDENTIFIER "${REL_FILE}" SYMBOL)
        get_filename_component(OUTPUT_FILE_DIR "${REL_FILE}" DIRECTORY)
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE_DIR}")
        set(OUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${REL_FILE}.o")
        set(${OUTPUT_SYMBOL} ${SYMBOL} PARENT_SCOPE)
        set(${OUTPUT_FILE} "${OUT_FILE}" PARENT_SCOPE)
        add_custom_command(
            OUTPUT "${OUT_FILE}"
            COMMAND ${EMBED_LD} -r -o "${OUT_FILE}" -z noexecstack --format=binary "${REL_FILE}" 
            COMMAND ${EMBED_OBJCOPY} --rename-section .data=.rodata,alloc,load,readonly,data,contents "${OUT_FILE}"
            WORKING_DIRECTORY ${WORKING_DIRECTORY}
            DEPENDS ${FILE}
            VERBATIM
        )
    endforeach()
endfunction()

function(add_embed_library EMBED_NAME)
    set(options)
    set(oneValueArgs RELATIVE)
    set(multiValueArgs)
    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/embed)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/embed/${EMBED_NAME})
    set(EMBED_DIR ${CMAKE_CURRENT_BINARY_DIR}/embed/${EMBED_NAME})
    set(SRC_FILE "${EMBED_DIR}/${EMBED_NAME}.cpp")
    set(HEADER_FILE "${EMBED_DIR}/include/${EMBED_NAME}.hpp")
    set(WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set(OUTPUT_FILES)
    set(SYMBOLS)
    message(STATUS "Embedding files")
    foreach(FILE ${PARSE_UNPARSED_ARGUMENTS})
        embed_file(OUTPUT_FILE OUTPUT_SYMBOL ${FILE})
        list(APPEND OUTPUT_FILES ${OUTPUT_FILE})
        list(APPEND SYMBOLS ${OUTPUT_SYMBOL})
    endforeach()
    message(STATUS "Generating embedding library ${EMBED_NAME}")
    generate_embed_source(${EMBED_NAME} SRC ${SRC_FILE} HEADER ${HEADER_FILE} OBJECTS ${OUTPUT_FILES} SYMBOLS ${SYMBOLS} RELATIVE ${PARSE_RELATIVE} FILES ${PARSE_UNPARSED_ARGUMENTS})
    add_library(${EMBED_NAME} STATIC ${OUTPUT_FILES} "${SRC_FILE}")
    target_include_directories(${EMBED_NAME} PUBLIC "$<BUILD_INTERFACE:${EMBED_DIR}/include>")
    target_compile_options(${EMBED_NAME} PRIVATE -Wno-reserved-identifier)
    set_target_properties(${EMBED_NAME} PROPERTIES POSITION_INDEPENDENT_CODE On)
endfunction()
