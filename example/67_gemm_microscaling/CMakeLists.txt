add_custom_target(example_gemm_mx)

add_example_executable(example_gemm_mx_fp8 gemm_mx_f8.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_fp8)
if (TARGET example_gemm_mx_fp8)
target_compile_definitions(example_gemm_mx_fp8 PRIVATE 
    EXAMPLE_A_DATA_TYPE=f8_t
    EXAMPLE_B_DATA_TYPE=f8_t
    EXAMPLE_C_DATA_TYPE=half_t
    EXAMPLE_A_LAYOUT=Row
    EXAMPLE_B_LAYOUT=Col
    EXAMPLE_C_LAYOUT=Row
)
endif()


add_example_executable(example_gemm_mx_bf8 gemm_mx_f8.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_bf8)
if (TARGET example_gemm_mx_bf8)
# TODO: Fix B_LAYOUT=ROW
target_compile_definitions(example_gemm_mx_bf8 PRIVATE 
    EXAMPLE_A_DATA_TYPE=bf8_t
    EXAMPLE_B_DATA_TYPE=bf8_t
    EXAMPLE_C_DATA_TYPE=bhalf_t
    EXAMPLE_A_LAYOUT=Row
    EXAMPLE_B_LAYOUT=Col
    EXAMPLE_C_LAYOUT=Row
)
endif()


add_example_executable(example_gemm_mx_fp8_bf8 gemm_mx_f8.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_fp8_bf8)
if (TARGET example_gemm_mx_fp8_bf8)
# TODO: Fix B_LAYOUT=ROW
target_compile_definitions(example_gemm_mx_fp8_bf8 PRIVATE 
    EXAMPLE_A_DATA_TYPE=f8_t
    EXAMPLE_B_DATA_TYPE=bf8_t
    EXAMPLE_C_DATA_TYPE=bhalf_t
    EXAMPLE_A_LAYOUT=Row
    EXAMPLE_B_LAYOUT=Col
    EXAMPLE_C_LAYOUT=Row
)
endif()

add_example_executable(example_gemm_mx_fp4 gemm_mx_fp4.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_fp4)

set(FP4_MXGEMM_OPTIONS)
# list(APPEND FP4_MXGEMM_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1 -mllvm --slp-threshold=-32")
list(APPEND FP4_MXGEMM_OPTIONS -v --save-temps -Wno-gnu-line-marker)
target_compile_options(example_gemm_mx_fp4 PRIVATE ${FP4_MXGEMM_OPTIONS})