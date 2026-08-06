# Common compile options shared by all targets (executables and libraries) below.
set(EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS)

list(APPEND EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3)

if(GPU_TARGETS MATCHES "gfx95" AND NOT GPU_TARGETS MATCHES "gfx94" AND NOT GPU_TARGETS MATCHES "gfx90")
    ## disable slp-vectorize improve pipelines performance on gfx950
    list(APPEND EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS -DBUILD_HSTU_FOR_GFX95 -fno-slp-vectorize)
endif()

# Source lists for the forward path
file(GLOB FWD_INSTANCE_SRCS instances/*_forward_*.cpp)
set(FWD_INTERFACE_SRCS  hstu_attention_no_group_forward_bf16.cpp  hstu_attention_no_group_forward_fp16.cpp hstu_attention_group_forward_bf16.cpp hstu_attention_group_forward_fp16.cpp)

# Source lists for the backward path
file(GLOB BWD_INSTANCE_SRCS instances/*_backward_*.cpp)
set(BWD_INTERFACE_SRCS  hstu_attention_no_group_backward_bf16.cpp  hstu_attention_no_group_backward_fp16.cpp hstu_attention_group_backward_bf16.cpp hstu_attention_group_backward_fp16.cpp)

# Optional single-kernel HSTU backward path. When enabled, the four backward
# interface TUs dispatch to run_{batched,group}_backward_single_dispatch and the
# extra single-kernel instances under instances_single/ are compiled into hstubwd.
# CONFIGURE_DEPENDS: newly generated instance files must re-trigger cmake, otherwise
# they would be silently dropped from the build.
option(HSTU_BWD_SINGLE_KERNEL "Build the single-kernel HSTU backward path" OFF)
if(HSTU_BWD_SINGLE_KERNEL)
    file(GLOB BWD_SINGLE_INSTANCE_SRCS CONFIGURE_DEPENDS instances_single/*_backward_single_*.cpp)
    list(APPEND BWD_INSTANCE_SRCS ${BWD_SINGLE_INSTANCE_SRCS})
endif()

file(GLOB OTHER_INTERFACE_SRCS "hstu_generate_*_random_number_*.cpp")

# Forward library: libhstufwd.so
add_library(hstufwd SHARED EXCLUDE_FROM_ALL ${FWD_INTERFACE_SRCS} ${FWD_INSTANCE_SRCS})
target_include_directories(hstufwd PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(hstufwd PRIVATE ${EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS})

# Backward library: libhstubwd.so
add_library(hstubwd SHARED EXCLUDE_FROM_ALL ${BWD_INTERFACE_SRCS} ${BWD_INSTANCE_SRCS})
target_include_directories(hstubwd PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(hstubwd PRIVATE ${EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS})
if(HSTU_BWD_SINGLE_KERNEL)
    # The four backward interface TUs (BWD_INTERFACE_SRCS) belong to hstubwd, so a
    # PRIVATE definition here reaches exactly the code that needs the macro. The bwd
    # executable only calls the extern interface functions declared in
    # hstu_attention_api.hpp and never sees this macro, which is correct.
    target_compile_definitions(hstubwd PRIVATE HSTU_BWD_SINGLE_KERNEL)
endif()

set(EXAMPLE_HSTU_ATTENTION_FWD "tile_example_hstu_attention_fwd")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message("adding example ${EXAMPLE_HSTU_ATTENTION_FWD}")
add_executable(${EXAMPLE_HSTU_ATTENTION_FWD} EXCLUDE_FROM_ALL example_hstu_attention_fwd.cpp)
target_include_directories(${EXAMPLE_HSTU_ATTENTION_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${EXAMPLE_HSTU_ATTENTION_FWD} PRIVATE ${OTHER_INTERFACE_SRCS})
target_compile_options(${EXAMPLE_HSTU_ATTENTION_FWD} PRIVATE ${EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS})
target_link_libraries(${EXAMPLE_HSTU_ATTENTION_FWD} PRIVATE hstufwd)

set(EXAMPLE_HSTU_ATTENTION_BWD "tile_example_hstu_attention_bwd")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message("adding example ${EXAMPLE_HSTU_ATTENTION_BWD}")
add_executable(${EXAMPLE_HSTU_ATTENTION_BWD} EXCLUDE_FROM_ALL example_hstu_attention_bwd.cpp)
target_include_directories(${EXAMPLE_HSTU_ATTENTION_BWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${EXAMPLE_HSTU_ATTENTION_BWD} PRIVATE ${OTHER_INTERFACE_SRCS})
target_compile_options(${EXAMPLE_HSTU_ATTENTION_BWD} PRIVATE ${EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS})
target_link_libraries(${EXAMPLE_HSTU_ATTENTION_BWD} PRIVATE hstufwd hstubwd)

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
