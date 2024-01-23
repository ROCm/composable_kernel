# CMake generated Testfile for 
# Source directory: /root/workspace/composable_kernel/test/softmax
# Build directory: /root/workspace/composable_kernel/host/test/softmax
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_softmax_rank3 "/root/workspace/composable_kernel/host/bin/test_softmax_rank3")
set_tests_properties(test_softmax_rank3 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/softmax/CMakeLists.txt;3;add_gtest_executable;/root/workspace/composable_kernel/test/softmax/CMakeLists.txt;0;")
add_test(test_softmax_rank4 "/root/workspace/composable_kernel/host/bin/test_softmax_rank4")
set_tests_properties(test_softmax_rank4 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/softmax/CMakeLists.txt;4;add_gtest_executable;/root/workspace/composable_kernel/test/softmax/CMakeLists.txt;0;")
add_test(test_softmax_interface "/root/workspace/composable_kernel/host/bin/test_softmax_interface")
set_tests_properties(test_softmax_interface PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/softmax/CMakeLists.txt;5;add_gtest_executable;/root/workspace/composable_kernel/test/softmax/CMakeLists.txt;0;")
