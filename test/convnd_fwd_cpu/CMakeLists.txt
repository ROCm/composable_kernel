add_test_executable(test_conv2d_fwd_cpu conv2d_fwd_cpu.cpp)
target_link_libraries(test_conv2d_fwd_cpu PRIVATE host_tensor)
target_link_libraries(test_conv2d_fwd_cpu PRIVATE device_conv2d_fwd_cpu_instance)
# 3.13 introduce target_link_directories, which is better
set_target_properties(test_conv2d_fwd_cpu PROPERTIES LINK_FLAGS "${OMP_LINK_FLAG}")
target_link_libraries(test_conv2d_fwd_cpu PRIVATE "${OMP_LIBRARY}")
target_compile_options(test_conv2d_fwd_cpu PRIVATE "${OMP_CXX_FLAG}")
