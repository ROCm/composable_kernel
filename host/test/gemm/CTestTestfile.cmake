# CMake generated Testfile for 
# Source directory: /root/workspace/composable_kernel/test/gemm
# Build directory: /root/workspace/composable_kernel/host/test/gemm
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_gemm_fp32 "/root/workspace/composable_kernel/host/bin/test_gemm_fp32")
set_tests_properties(test_gemm_fp32 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;57;add_test;/root/workspace/composable_kernel/test/gemm/CMakeLists.txt;1;add_test_executable;/root/workspace/composable_kernel/test/gemm/CMakeLists.txt;0;")
add_test(test_gemm_fp16 "/root/workspace/composable_kernel/host/bin/test_gemm_fp16")
set_tests_properties(test_gemm_fp16 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;57;add_test;/root/workspace/composable_kernel/test/gemm/CMakeLists.txt;5;add_test_executable;/root/workspace/composable_kernel/test/gemm/CMakeLists.txt;0;")
add_test(test_gemm_standalone_xdl_fp16 "/root/workspace/composable_kernel/host/bin/test_gemm_standalone_xdl_fp16")
set_tests_properties(test_gemm_standalone_xdl_fp16 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;57;add_test;/root/workspace/composable_kernel/test/gemm/CMakeLists.txt;16;add_test_executable;/root/workspace/composable_kernel/test/gemm/CMakeLists.txt;0;")
add_test(test_gemm_bf16 "/root/workspace/composable_kernel/host/bin/test_gemm_bf16")
set_tests_properties(test_gemm_bf16 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;57;add_test;/root/workspace/composable_kernel/test/gemm/CMakeLists.txt;21;add_test_executable;/root/workspace/composable_kernel/test/gemm/CMakeLists.txt;0;")
add_test(test_gemm_int8 "/root/workspace/composable_kernel/host/bin/test_gemm_int8")
set_tests_properties(test_gemm_int8 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;57;add_test;/root/workspace/composable_kernel/test/gemm/CMakeLists.txt;25;add_test_executable;/root/workspace/composable_kernel/test/gemm/CMakeLists.txt;0;")
