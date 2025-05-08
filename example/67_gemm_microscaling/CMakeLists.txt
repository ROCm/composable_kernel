add_custom_target(example_gemm_mx)

#add_example_executable(example_gemm_mx_fp8 gemm_mx_fp8.cpp)
#add_example_dependencies(example_gemm_mx example_gemm_mx_fp8)
#
#add_example_executable(example_gemm_mx_bf8 gemm_mx_bf8.cpp)
#add_example_dependencies(example_gemm_mx example_gemm_mx_bf8)
#
#add_example_executable(example_gemm_mx_fp8_bf8 gemm_mx_fp8_bf8.cpp)
#add_example_dependencies(example_gemm_mx example_gemm_mx_fp8_bf8)

add_executable(ffmx EXCLUDE_FROM_ALL gemm_mx_bpreshuffle.cpp)
set(EXAMPLE_FF_COMPILE_OPTIONS)
list(APPEND EXAMPLE_FF_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
list(APPEND EXAMPLE_FF_COMPILE_OPTIONS -Wno-unused-variable -Wno-unused-parameter)
list(APPEND EXAMPLE_FF_COMPILE_OPTIONS -Wno-unused-local-typedef)
#list(APPEND EXAMPLE_FF_COMPILE_OPTIONS -O0)
list(APPEND EXAMPLE_FF_COMPILE_OPTIONS -v)
#list(APPEND EXAMPLE_FF_COMPILE_OPTIONS -ggdb -g)
list(APPEND EXAMPLE_FF_COMPILE_OPTIONS -save-temps)
# list(APPEND EXAMPLE_FF_COMPILE_OPTIONS  -mllvm -greedy-reverse-local-assignment=1)
list(APPEND EXAMPLE_FF_COMPILE_OPTIONS -Wno-gnu-line-marker)
list(APPEND EXAMPLE_FF_COMPILE_OPTIONS -DFF_DBG=1 -DFF_DBG_CNT=64)

target_link_libraries(ffmx PRIVATE utility)
#add_test(NAME ${EXAMPLE_NAME} COMMAND $<TARGET_FILE:${EXAMPLE_NAME}> ${ARGN})
#set_property(TARGET ${EXAMPLE_NAME} PROPERTY HIP_ARCHITECTURES ${EX_TARGETS} )
add_dependencies(examples ffmx)
add_dependencies(check ffmx)

target_compile_options(ffmx PRIVATE ${EXAMPLE_FF_COMPILE_OPTIONS})

