cmake_minimum_required(VERSION 3.15)
project(ck_app)
add_compile_options(-std=c++14)

find_package(composable_kernel 1.0.0 COMPONENTS device_operations host_tensor)
find_package(hip REQUIRED PATHS /opt/rocm)
message(STATUS "Build with HIP ${hip_VERSION}")

add_executable(test_client_app client_app.cpp)

target_link_libraries(test_client_app PRIVATE composable_kernel::device_operations composable_kernel::host_tensor hip::host)
