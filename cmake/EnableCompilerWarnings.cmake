################################################################################
#
# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
# - Enable warning all for gcc/clang or use /W4 for visual studio

## Strict compile options for Visual C++ compiler
set(__default_msvc_compile_options /w)

## Strict compile options for GNU/Clang compilers
set(__default_compile_options
        -Wall -Wextra
        -Wcomment
        -Wendif-labels
        -Wformat
        -Winit-self
        -Wreturn-type
        -Wsequence-point
        -Wswitch
        -Wtrigraphs
        -Wundef
        -Wuninitialized
        -Wunreachable-code
        -Wunused
        -Wno-reserved-identifier
        -Werror
        -Wno-option-ignored
        -Wsign-compare
        -Wno-extra-semi-stmt
    )

## Strict compile options for Clang compilers
set(__default_clang_compile_options
        -Weverything
        -Wshadow
        -Wno-c++98-compat
        -Wno-c++98-compat-pedantic
        -Wno-conversion
        -Wno-double-promotion
        -Wno-exit-time-destructors
        -Wno-extra-semi
        -Wno-float-conversion
        -Wno-gnu-anonymous-struct
        -Wno-gnu-zero-variadic-macro-arguments
        -Wno-missing-prototypes
        -Wno-nested-anon-types
        -Wno-padded
        -Wno-return-std-move-in-c++11
        -Wno-shorten-64-to-32
        -Wno-sign-conversion
        -Wno-unknown-warning-option
        -Wno-unused-command-line-argument
        -Wno-weak-vtables
        -Wno-covered-switch-default
        -Wno-unsafe-buffer-usage)

if(WIN32)
    list(APPEND __default_clang_compile_options
        -fms-extensions
        -fms-compatibility
        -fdelayed-template-parsing)
endif()

set(__default_gnu_compile_options
        -Wduplicated-branches
        -Wduplicated-cond
        -Wno-noexcept-type
        -Wno-ignored-attributes
        -Wodr
        -Wshift-negative-value
        -Wshift-overflow=2
        -Wno-missing-field-initializers
        -Wno-maybe-uninitialized
        -Wno-deprecated-declarations)

add_compile_options(
        "$<$<OR:$<CXX_COMPILER_ID:MSVC>,$<C_COMPILER_ID:MSVC>>:${__default_msvc_compile_options}>"
        "$<$<OR:$<CXX_COMPILER_ID:GNU,Clang>,$<C_COMPILER_ID:GNU,Clang>>:${__default_compile_options}>"
        "$<$<OR:$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,7>>,$<AND:$<C_COMPILER_ID:GNU>,$<VERSION_GREATER_EQUAL:$<C_COMPILER_VERSION>,7>>>:${__default_gnu_compile_options}>"
        "$<$<OR:$<CXX_COMPILER_ID:Clang>,$<C_COMPILER_ID:Clang>>:${__default_clang_compile_options}>")

unset(__default_msvc_compile_options)
unset(__default_compile_options)
unset(__default_gnu_compile_options)
unset(__default_clang_compile_options)

