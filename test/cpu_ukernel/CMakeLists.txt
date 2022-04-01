add_test_executable(test_cpu_gemm_uk cpu_gemm_uk.cpp)
target_link_libraries(test_cpu_gemm_uk PRIVATE host_tensor)
# 3.13 introduce target_link_directories, which is better
set_target_properties(test_cpu_gemm_uk PROPERTIES LINK_FLAGS -Wl,-rpath,/opt/rocm/llvm/lib )
target_link_libraries(test_cpu_gemm_uk PRIVATE /opt/rocm/llvm/lib/libomp.so)
target_compile_options(test_cpu_gemm_uk PRIVATE -fopenmp=libomp -Wno-unused-command-line-argument)
