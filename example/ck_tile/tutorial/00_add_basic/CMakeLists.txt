set(EXAMPLE_ADD_BASIC "add_basic")

message(DEBUG "adding example ${EXAMPLE_ADD_BASIC}")

add_executable(${EXAMPLE_ADD_BASIC} EXCLUDE_FROM_ALL add_basic.cpp)
target_include_directories(${EXAMPLE_ADD_BASIC} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set(EXAMPLE_ADD_BASIC_COMPILE_OPTIONS)

# generate assembly
# list(APPEND EXAMPLE_ADD_BASIC_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_ADD_BASIC_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)

target_compile_options(${EXAMPLE_ADD_BASIC} PRIVATE ${EXAMPLE_ADD_BASIC_COMPILE_OPTIONS})

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
