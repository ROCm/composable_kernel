function (add_smoothquant_test TARGET_NAME MAIN_SRC)
message(DEBUG "adding ${TARGET_NAME}")

add_test_executable(${TARGET_NAME} ${MAIN_SRC})
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

foreach(source IN LISTS ARGN)
list(APPEND INSTANCE_SRCS ${source})
endforeach()

target_sources(${TARGET_NAME} PRIVATE ${INSTANCE_SRCS})

set(COMPILE_OPTIONS)
# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)

target_compile_options(${TARGET_NAME} PRIVATE ${COMPILE_OPTIONS})
endfunction(add_smoothquant_test TARGET_NAME MAIN_SRC)

# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")
    add_smoothquant_test(test_ck_tile_smoothquant_fp16 smoothquant_fp16.cpp)
    add_smoothquant_test(test_ck_tile_smoothquant_bf16 smoothquant_bf16.cpp)
else()
    message(DEBUG "Skipping ck_tile_permute tests for current target")
endif()