add_custom_target(example_gemm_mx)

add_example_executable(example_gemm_mx_fp8 gemm_mx_fp8.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_fp8)

add_example_executable(example_gemm_mx_bf8 gemm_mx_bf8.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_bf8)

#add_example_executable(example_gemm_mx_fp8_bf8 gemm_mx_fp8_bf8.cpp)
# add_example_dependencies(example_gemm_mx example_gemm_mx_fp8_bf8) TOFO: Fix RRR

add_example_executable(example_gemm_mx_fp4 gemm_mx_fp4.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_fp4)

add_example_executable(example_gemm_mx_fp4_bpreshuffle gemm_mx_fp4_bpreshuffle.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_fp4_bpreshuffle)

#add_example_executable(example_moe_gemm1_xdl_mx_fp4 moe_gemm1_xdl_mx_fp4.cpp)
# add_example_dependencies(example_gemm_mx example_moe_gemm1_xdl_mx_fp4) TODO: Fix

#add_example_executable(example_moe_gemm1_xdl_mx_fp4_bns moe_gemm1_xdl_mx_fp4_bns.cpp)
#add_example_dependencies(example_gemm_mx example_moe_gemm1_xdl_mx_fp4_bns)

#add_example_executable(example_moe_gemm2_xdl_mx_fp4 moe_gemm2_xdl_mx_fp4.cpp)
# add_example_dependencies(example_gemm_mx example_moe_gemm2_xdl_mx_fp4) TODO: Fix

#add_example_executable(example_moe_gemm2_xdl_mx_fp4_bns moe_gemm2_xdl_mx_fp4_bns.cpp)
#add_example_dependencies(example_gemm_mx example_moe_gemm2_xdl_mx_fp4_bns)

set(FP4_MXGEMM_OPTIONS)
list(APPEND FP4_MXGEMM_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1 -mllvm --amdgpu-use-amdgpu-trackers=1")
#list(APPEND FP4_MXGEMM_OPTIONS -v --save-temps -Wno-gnu-line-marker -ftemplate-backtrace-limit=0)
example_compile_options(example_gemm_mx_fp4 PRIVATE ${FP4_MXGEMM_OPTIONS})
example_compile_options(example_gemm_mx_fp4_bpreshuffle PRIVATE ${FP4_MXGEMM_OPTIONS})
# example_compile_options(example_moe_gemm1_xdl_mx_fp4 PRIVATE ${FP4_MXGEMM_OPTIONS})
# example_compile_options(example_moe_gemm2_xdl_mx_fp4 PRIVATE ${FP4_MXGEMM_OPTIONS})
# example_compile_options(example_moe_gemm1_xdl_mx_fp4_bns PRIVATE ${FP4_MXGEMM_OPTIONS})
# example_compile_options(example_moe_gemm2_xdl_mx_fp4_bns PRIVATE ${FP4_MXGEMM_OPTIONS})

set(FP8_MXGEMM_OPTIONS)
list(APPEND FP8_MXGEMM_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1 -mllvm --slp-threshold=-32")
#list(APPEND FP8_MXGEMM_OPTIONS -v --save-temps -Wno-gnu-line-marker -ftemplate-backtrace-limit=0)
example_compile_options(example_gemm_mx_fp8 PRIVATE ${FP8_MXGEMM_OPTIONS})
example_compile_options(example_gemm_mx_bf8 PRIVATE ${FP8_MXGEMM_OPTIONS})
