set(EXAMPLE_BASIC_GEMM "basic_gemm")

message("adding example ${EXAMPLE_BASIC_GEMM}")

add_executable(${EXAMPLE_BASIC_GEMM} EXCLUDE_FROM_ALL gemm.cpp)
target_include_directories(${EXAMPLE_BASIC_GEMM} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set(EXAMPLE_BASIC_GEMM_COMPILE_OPTIONS)

# generate assembly
# list(APPEND EXAMPLE_BASIC_GEMM_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_BASIC_GEMM_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)

if(DEFINED kernel)
    message("Compiling with Kernel: ${kernel}")
    target_compile_definitions(${EXAMPLE_BASIC_GEMM} PRIVATE KERNEL_${kernel}=1)
endif()

target_compile_options(${EXAMPLE_BASIC_GEMM} PRIVATE ${EXAMPLE_BASIC_GEMM_COMPILE_OPTIONS})

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
