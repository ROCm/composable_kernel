set(EXAMPLE_NAIVE_GEMM "naive_gemm")

message(DEBUG "adding example ${EXAMPLE_NAIVE_GEMM}")

add_executable(${EXAMPLE_NAIVE_GEMM} EXCLUDE_FROM_ALL gemm.cpp)
target_include_directories(${EXAMPLE_NAIVE_GEMM} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
target_compile_options(${EXAMPLE_NAIVE_GEMM} PRIVATE
    -Wno-undefined-func-template
    -Wno-float-equal)

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
