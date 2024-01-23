# CMake generated Testfile for 
# Source directory: /root/workspace/composable_kernel/test/reduce
# Build directory: /root/workspace/composable_kernel/host/test/reduce
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_reduce_no_index "/root/workspace/composable_kernel/host/bin/test_reduce_no_index")
set_tests_properties(test_reduce_no_index PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;57;add_test;/root/workspace/composable_kernel/test/reduce/CMakeLists.txt;1;add_test_executable;/root/workspace/composable_kernel/test/reduce/CMakeLists.txt;0;")
add_test(test_reduce_with_index "/root/workspace/composable_kernel/host/bin/test_reduce_with_index")
set_tests_properties(test_reduce_with_index PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;57;add_test;/root/workspace/composable_kernel/test/reduce/CMakeLists.txt;2;add_test_executable;/root/workspace/composable_kernel/test/reduce/CMakeLists.txt;0;")
