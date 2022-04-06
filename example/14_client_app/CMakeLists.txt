cmake_minimum_required(VERSION 3.15)
project(ck_app)
add_compile_options(-std=c++14)
# add_link_options(--offload-arch=gfx908)

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(composable_kernel 1.0.0 COMPONENTS device_operations host_tensor)
find_package(hip REQUIRED PATHS /opt/rocm)
message(STATUS "Build with HIP ${hip_VERSION}")
# target_flags(HIP_COMPILER_FLAGS hip::device)
# Remove cuda arch flags
#string(REGEX REPLACE --cuda-gpu-arch=[a-z0-9]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
#string(REGEX REPLACE --offload-arch=[a-z0-9]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
#string(REPLACE "$<LINK_LANGUAGE:CXX>" "1" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
#string(REPLACE "SHELL:" "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")


add_executable(myexe client_app.cpp)

target_link_libraries(myexe PRIVATE composable_kernel::device_operations composable_kernel::host_tensor hip::host)
#
