set(EXAMPLE_HSTU_ATTENTION_FWD "tile_example_hstu_attention_fwd")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message("adding example ${EXAMPLE_HSTU_ATTENTION_FWD}")
file(GLOB FWD_INSTANCE_SRCS instances/*_forward_*.cpp)
set(FWD_INTERFACE_SRCS  hstu_attention_no_group_forward_bf16.cpp  hstu_attention_no_group_forward_fp16.cpp hstu_attention_group_forward_bf16.cpp hstu_attention_group_forward_fp16.cpp)
file(GLOB OTHER_INTERFACE_SRCS "hstu_generate_*_random_number_*.cpp")
add_executable(${EXAMPLE_HSTU_ATTENTION_FWD} EXCLUDE_FROM_ALL example_hstu_attention_fwd.cpp)
target_include_directories(${EXAMPLE_HSTU_ATTENTION_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${EXAMPLE_HSTU_ATTENTION_FWD} PRIVATE ${FWD_INTERFACE_SRCS}  ${FWD_INSTANCE_SRCS} ${OTHER_INTERFACE_SRCS})

set(EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS)

list(APPEND EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3)

if(GPU_TARGETS MATCHES "gfx95" AND NOT GPU_TARGETS MATCHES "gfx94" AND NOT GPU_TARGETS MATCHES "gfx90")
    ## disable slp-vectorize improve pipelines performance on gfx950
    list(APPEND EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS -DBUILD_HSTU_FOR_GFX95 -fno-slp-vectorize)
endif()

target_compile_options(${EXAMPLE_HSTU_ATTENTION_FWD} PRIVATE ${EXAMPLE_HSTU_ATTENTION_COMPILE_OPTIONS})

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)

