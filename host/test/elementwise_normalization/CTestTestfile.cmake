# CMake generated Testfile for 
# Source directory: /root/workspace/composable_kernel/test/elementwise_normalization
# Build directory: /root/workspace/composable_kernel/host/test/elementwise_normalization
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_elementwise_layernorm_fp16 "/root/workspace/composable_kernel/host/bin/test_elementwise_layernorm_fp16")
set_tests_properties(test_elementwise_layernorm_fp16 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/elementwise_normalization/CMakeLists.txt;2;add_gtest_executable;/root/workspace/composable_kernel/test/elementwise_normalization/CMakeLists.txt;0;")
