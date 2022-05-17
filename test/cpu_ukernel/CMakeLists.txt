add_test_executable(test_cpu_gemm_uk cpu_gemm_uk.cpp)
target_link_libraries(test_cpu_gemm_uk PRIVATE host_tensor)
# 3.13 introduce target_link_directories, which is better
set_target_properties(test_cpu_gemm_uk PROPERTIES LINK_FLAGS "${OMP_LINK_FLAG}")
target_link_libraries(test_cpu_gemm_uk PRIVATE "${OMP_LIBRARY}")
target_compile_options(test_cpu_gemm_uk PRIVATE "${OMP_CXX_FLAG}")
