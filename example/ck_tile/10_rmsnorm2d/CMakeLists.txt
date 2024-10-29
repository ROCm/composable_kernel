set(TILE_RMSNORM2D_FWD "tile_rmsnorm2d_fwd")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message("adding ${TILE_RMSNORM2D_FWD}")
file(GLOB INSTANCE_SRCS instances/*.cpp)
add_executable(${TILE_RMSNORM2D_FWD} EXCLUDE_FROM_ALL rmsnorm2d_fwd.cpp)
target_include_directories(${TILE_RMSNORM2D_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${TILE_RMSNORM2D_FWD} PRIVATE ${INSTANCE_SRCS})

set(TILE_RMSNORM2D_FWD_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND TILE_RMSNORM2D_FWD_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)

target_compile_options(${TILE_RMSNORM2D_FWD} PRIVATE ${TILE_RMSNORM2D_FWD_COMPILE_OPTIONS})

set(EXAMPLE_RMSNORM2D_FWD "tile_example_rmsnorm2d_fwd")
add_executable(${EXAMPLE_RMSNORM2D_FWD} EXCLUDE_FROM_ALL example_rmsnorm2d_fwd.cpp)
target_compile_options(${EXAMPLE_RMSNORM2D_FWD} PRIVATE ${TILE_RMSNORM2D_FWD_COMPILE_OPTIONS})

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
