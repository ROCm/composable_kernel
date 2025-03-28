set(EXAMPLE_HSTU_ATTENTION "tile_example_hstu_attention")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message("adding example ${EXAMPLE_HSTU_ATTENTION}")
##file(GLOB INSTANCE_SRCS instances/*.cpp)
add_executable(${EXAMPLE_HSTU_ATTENTION} EXCLUDE_FROM_ALL example_hstu_attention.cpp)
target_include_directories(${EXAMPLE_HSTU_ATTENTION} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
##target_sources(${EXAMPLE_HSTU_ATTENTION} PRIVATE hstu_attention_bf16.cpp  hstu_attention_fp16.cpp ${INSTANCE_SRCS})

set(EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS)

list(APPEND EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)

target_compile_options(${EXAMPLE_HSTU_ATTENTION} PRIVATE ${EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS})

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)

