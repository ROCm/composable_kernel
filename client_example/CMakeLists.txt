cmake_minimum_required(VERSION 3.15)
project(ck_app)
add_compile_options(-std=c++17)

find_package(composable_kernel 1.0.0 COMPONENTS device_operations)
find_package(hip REQUIRED PATHS /opt/rocm)
message(STATUS "Build with HIP ${hip_VERSION}")

add_subdirectory(01_gemm)
add_subdirectory(02_gemm_add_add_fastgelu)
add_subdirectory(03_gemm_layernorm)
add_subdirectory(04_contraction)
