set(TILE_SMOOTHQUANT "tile_smoothquant")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message("adding ${TILE_SMOOTHQUANT}")
file(GLOB INSTANCE_SRCS instances/*.cpp)
add_executable(${TILE_SMOOTHQUANT} EXCLUDE_FROM_ALL smoothquant.cpp)
target_include_directories(${TILE_SMOOTHQUANT} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${TILE_SMOOTHQUANT} PRIVATE ${INSTANCE_SRCS})

set(TILE_SMOOTHQUANT_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND TILE_SMOOTHQUANT_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)

target_compile_options(${TILE_SMOOTHQUANT} PRIVATE ${TILE_SMOOTHQUANT_COMPILE_OPTIONS})

set(EXAMPLE_SMOOTHQUANT "tile_example_smoothquant")
add_executable(${EXAMPLE_SMOOTHQUANT} EXCLUDE_FROM_ALL example_smoothquant.cpp)
target_compile_options(${EXAMPLE_SMOOTHQUANT} PRIVATE ${TILE_SMOOTHQUANT_COMPILE_OPTIONS})

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
