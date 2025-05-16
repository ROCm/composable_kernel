add_custom_target(example_gemm_mx)

add_example_executable(example_gemm_mx_fp8 gemm_mx_fp8.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_fp8)

add_example_executable(example_gemm_mx_bf8 gemm_mx_bf8.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_bf8)

add_example_executable(example_gemm_mx_fp8_bf8 gemm_mx_fp8_bf8.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_fp8_bf8)

add_example_executable(example_gemm_mx_fp4 gemm_mx_fp4.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_fp4)

set(FP4_MXGEMM_OPTIONS)
list(APPEND FP4_MXGEMM_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1")
list(APPEND FP4_MXGEMM_OPTIONS -v --save-temps -Wno-gnu-line-marker -ftemplate-backtrace-limit=0)
set(FP8_MXGEMM_OPTIONS)
list(APPEND FP8_MXGEMM_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1 -mllvm --slp-threshold=-32")
list(APPEND FP8_MXGEMM_OPTIONS -v --save-temps -Wno-gnu-line-marker -ftemplate-backtrace-limit=0)
target_compile_options(example_gemm_mx_fp4 PRIVATE ${FP4_MXGEMM_OPTIONS})
target_compile_options(example_gemm_mx_fp8 PRIVATE ${FP8_MXGEMM_OPTIONS})