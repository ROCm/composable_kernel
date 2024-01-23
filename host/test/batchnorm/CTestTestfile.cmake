# CMake generated Testfile for 
# Source directory: /root/workspace/composable_kernel/test/batchnorm
# Build directory: /root/workspace/composable_kernel/host/test/batchnorm
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_batchnorm_fwd_rank_4 "/root/workspace/composable_kernel/host/bin/test_batchnorm_fwd_rank_4")
set_tests_properties(test_batchnorm_fwd_rank_4 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/batchnorm/CMakeLists.txt;1;add_gtest_executable;/root/workspace/composable_kernel/test/batchnorm/CMakeLists.txt;0;")
add_test(test_batchnorm_bwd_rank_4 "/root/workspace/composable_kernel/host/bin/test_batchnorm_bwd_rank_4")
set_tests_properties(test_batchnorm_bwd_rank_4 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/batchnorm/CMakeLists.txt;2;add_gtest_executable;/root/workspace/composable_kernel/test/batchnorm/CMakeLists.txt;0;")
add_test(test_batchnorm_infer_rank_4 "/root/workspace/composable_kernel/host/bin/test_batchnorm_infer_rank_4")
set_tests_properties(test_batchnorm_infer_rank_4 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;121;add_test;/root/workspace/composable_kernel/test/batchnorm/CMakeLists.txt;3;add_gtest_executable;/root/workspace/composable_kernel/test/batchnorm/CMakeLists.txt;0;")
