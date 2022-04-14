add_test_executable(test_conv2d_fwd_cpu conv2d_fwd_cpu.cpp)
target_link_libraries(test_conv2d_fwd_cpu PRIVATE host_tensor)
target_link_libraries(test_conv2d_fwd_cpu PRIVATE device_conv2d_fwd_cpu_instance)
# 3.13 introduce target_link_directories, which is better
set_target_properties(test_conv2d_fwd_cpu PROPERTIES LINK_FLAGS -Wl,-rpath,/opt/rocm/llvm/lib )
target_link_libraries(test_conv2d_fwd_cpu PRIVATE /opt/rocm/llvm/lib/libomp.so)
target_compile_options(test_conv2d_fwd_cpu PRIVATE -fopenmp=libomp -Wno-unused-command-line-argument)
