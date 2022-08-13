add_example_executable(example_gemm_add_add_mean_meansquare_xdl_fp16 gemm_add_add_mean_meansquare_xdl_fp16.cpp)
add_example_executable(example_gemm_mean_meansquare_xdl_fp16 gemm_mean_meansquare_xdl_fp16.cpp)

#exclude GEMM+max exampe from testing, since there is random failure on gfx908
#https://github.com/ROCmSoftwarePlatform/composable_kernel/issues/358
#TODO: fix the failure and re-enable this test
add_example_executable_no_testing(example_gemm_max_xdl_fp16 gemm_max_xdl_fp16.cpp)
