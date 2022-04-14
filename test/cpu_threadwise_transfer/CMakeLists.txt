add_test_executable(test_cpu_threadwise_transfer cpu_threadwise_transfer.cpp)
target_link_libraries(test_cpu_threadwise_transfer PRIVATE host_tensor)
# 3.13 introduce target_link_directories, which is better
set_target_properties(test_cpu_threadwise_transfer PROPERTIES LINK_FLAGS -Wl,-rpath,/opt/rocm/llvm/lib )
target_link_libraries(test_cpu_threadwise_transfer PRIVATE /opt/rocm/llvm/lib/libomp.so)
target_compile_options(test_cpu_threadwise_transfer PRIVATE -fopenmp=libomp -Wno-unused-command-line-argument)
