set(EXAMPLE_REDUCE "basic_flash_attention_fwd")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message("adding example ${EXAMPLE_REDUCE}")

add_executable(${EXAMPLE_REDUCE} EXCLUDE_FROM_ALL flash_attention_fwd.cpp)
target_include_directories(${EXAMPLE_REDUCE} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set(EXAMPLE_REDUCE_COMPILE_OPTIONS)

# list(APPEND EXAMPLE_REDUCE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_REDUCE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)

option(ENABLE_TOY_FA_FWD_OPT "Enable toy FA fwd optimization" OFF)
if(ENABLE_TOY_FA_FWD_OPT)
    message("Compiling with toy FA fwd optimization")
    target_compile_definitions(${EXAMPLE_REDUCE} PRIVATE TOY_FA_FWD_OPT)
endif()

option(ENABLE_TOY_FA_FWD_QK_SWIZZLE "Enable toy FA fwd QK swizzle" OFF)
if(ENABLE_TOY_FA_FWD_QK_SWIZZLE)
    message("Compiling with toy FA fwd QK swizzle")
    target_compile_definitions(${EXAMPLE_REDUCE} PRIVATE TOY_FA_FWD_QK_SWIZZLE)
endif()

option(ENABLE_TOY_FA_FWD_CACHE_AWARE "Enable toy FA fwd cache aware" OFF)
if(ENABLE_TOY_FA_FWD_CACHE_AWARE)
    message("Compiling with toy FA fwd cache aware")
    target_compile_definitions(${EXAMPLE_REDUCE} PRIVATE TOY_FA_FWD_CACHE_AWARE)
endif()

target_compile_options(${EXAMPLE_REDUCE} PRIVATE ${EXAMPLE_REDUCE_COMPILE_OPTIONS})

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
