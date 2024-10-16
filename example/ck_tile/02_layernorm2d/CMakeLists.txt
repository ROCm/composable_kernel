set(EXAMPLE_LAYERNORM2D_FWD "tile_example_layernorm2d_fwd")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message("adding example ${EXAMPLE_LAYERNORM2D_FWD}")
file(GLOB INSTANCE_SRCS instances/*.cpp)
add_executable(${EXAMPLE_LAYERNORM2D_FWD} EXCLUDE_FROM_ALL example_layernorm2d_fwd.cpp)
target_include_directories(${EXAMPLE_LAYERNORM2D_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${EXAMPLE_LAYERNORM2D_FWD} PRIVATE layernorm2d_fwd_api.cpp ${INSTANCE_SRCS})

set(EXAMPLE_LAYERNORM2D_FWD_COMPILE_OPTIONS)

# list(APPEND EXAMPLE_LAYERNORM2D_FWD_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)

target_compile_options(${EXAMPLE_LAYERNORM2D_FWD} PRIVATE ${EXAMPLE_LAYERNORM2D_FWD_COMPILE_OPTIONS})

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
