# CMake generated Testfile for 
# Source directory: /root/workspace/composable_kernel/test/data_type
# Build directory: /root/workspace/composable_kernel/host/test/data_type
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_fp8 "/root/workspace/composable_kernel/host/bin/test_fp8")
set_tests_properties(test_fp8 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/data_type/CMakeLists.txt;8;add_gtest_executable;/root/workspace/composable_kernel/test/data_type/CMakeLists.txt;0;")
add_test(test_bf8 "/root/workspace/composable_kernel/host/bin/test_bf8")
set_tests_properties(test_bf8 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/data_type/CMakeLists.txt;12;add_gtest_executable;/root/workspace/composable_kernel/test/data_type/CMakeLists.txt;0;")
