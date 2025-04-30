add_example_executable(example_gemm_multiply_multiply_xdl_fp8 gemm_multiply_multiply_xdl_fp8.cpp)
add_example_executable(example_gemm_multiply_multiply_xdl_fp8_ab_scale gemm_multiply_multiply_xdl_fp8_ab_scale.cpp)
add_example_executable(example_gemm_multiply_multiply_xdl_fp8_bpreshuffle gemm_multiply_multiply_xdl_fp8_bpreshuffle.cpp)
add_example_executable(example_gemm_add_add_xdl_fp16 gemm_add_add_xdl_fp16.cpp)
add_example_executable(example_gemm_multiply_multiply_xdl_int8 gemm_multiply_multiply_xdl_int8.cpp)
add_example_executable(example_moe_gemm1_xdl_fp8 moe_gemm1_xdl_fp8.cpp)
add_example_executable(example_moe_gemm2_xdl_fp8 moe_gemm2_xdl_fp8.cpp)
add_example_executable(example_moe_gemm2_xdl_fp8_blockscale moe_gemm2_xdl_fp8_blockscale.cpp)

list(APPEND gpu_list gfx942)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list AND target EQUAL 0)
        add_example_executable(example_moe_gemm1_xdl_pk_i4 moe_gemm1_xdl_pk_i4.cpp)
        add_example_executable(example_moe_gemm2_xdl_pk_i4 moe_gemm2_xdl_pk_i4.cpp)
        set(target 1)
    endif()
endforeach()
set(EXAMPLE_COMPILE_OPTIONS)
list(APPEND EXAMPLE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
list(APPEND EXAMPLE_COMPILE_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1 -mllvm --schedmodel=0 -mllvm -misched=gcn-iterative-max-occupancy-experimental")
target_compile_options(example_moe_gemm2_xdl_fp8_blockscale PRIVATE ${EXAMPLE_COMPILE_OPTIONS})

#hacky fix for bs_moe_stage2 with rocm < 6.4
add_custom_command(
    TARGET example_moe_gemm2_xdl_fp8_blockscale
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/hsa/
    ${CMAKE_CURRENT_BINARY_DIR}/hsa/
)

target_compile_definitions(example_moe_gemm2_xdl_fp8_blockscale PRIVATE MOE_STAGE2_ASM_DIR="${CMAKE_CURRENT_BINARY_DIR}/hsa/")