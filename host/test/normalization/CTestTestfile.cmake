# CMake generated Testfile for 
# Source directory: /root/workspace/composable_kernel/test/normalization
# Build directory: /root/workspace/composable_kernel/host/test/normalization
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_layernorm2d_fp32 "/root/workspace/composable_kernel/host/bin/test_layernorm2d_fp32")
set_tests_properties(test_layernorm2d_fp32 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/normalization/CMakeLists.txt;2;add_gtest_executable;/root/workspace/composable_kernel/test/normalization/CMakeLists.txt;0;")
add_test(test_groupnorm_fp32 "/root/workspace/composable_kernel/host/bin/test_groupnorm_fp32")
set_tests_properties(test_groupnorm_fp32 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/normalization/CMakeLists.txt;7;add_gtest_executable;/root/workspace/composable_kernel/test/normalization/CMakeLists.txt;0;")
add_test(test_layernorm2d_fp16 "/root/workspace/composable_kernel/host/bin/test_layernorm2d_fp16")
set_tests_properties(test_layernorm2d_fp16 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/normalization/CMakeLists.txt;12;add_gtest_executable;/root/workspace/composable_kernel/test/normalization/CMakeLists.txt;0;")
add_test(test_groupnorm_fp16 "/root/workspace/composable_kernel/host/bin/test_groupnorm_fp16")
set_tests_properties(test_groupnorm_fp16 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/normalization/CMakeLists.txt;17;add_gtest_executable;/root/workspace/composable_kernel/test/normalization/CMakeLists.txt;0;")
