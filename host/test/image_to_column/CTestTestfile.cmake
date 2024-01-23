# CMake generated Testfile for 
# Source directory: /root/workspace/composable_kernel/test/image_to_column
# Build directory: /root/workspace/composable_kernel/host/test/image_to_column
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_image_to_column "/root/workspace/composable_kernel/host/bin/test_image_to_column")
set_tests_properties(test_image_to_column PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/image_to_column/CMakeLists.txt;1;add_gtest_executable;/root/workspace/composable_kernel/test/image_to_column/CMakeLists.txt;0;")
add_test(test_image_to_column_interface "/root/workspace/composable_kernel/host/bin/test_image_to_column_interface")
set_tests_properties(test_image_to_column_interface PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/image_to_column/CMakeLists.txt;3;add_gtest_executable;/root/workspace/composable_kernel/test/image_to_column/CMakeLists.txt;0;")
