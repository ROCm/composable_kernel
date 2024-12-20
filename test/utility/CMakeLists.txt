cmake_minimum_required(VERSION 3.27)

project(utilities-tests LANGUAGES CXX HIP)

enable_testing()
include(../../cmake/gtest.cmake)

add_definitions(-D__HIP_PLATFORM_AMD__)
set(TEST_SOURCE_FILES test_utility.cpp)

set(AMDDeviceLibs_DIR /opt/rocm/lib/cmake/AMDDeviceLibs)
set(amd_comgr_DIR /opt/rocm/lib/cmake/amd_comgr)
set(hsa-runtime64_DIR /opt/rocm/lib/cmake/hsa-runtime64)
find_package(ROCM REQUIRED PATHS /opt/rocm)
find_package(hip REQUIRED PATHS /opt/rocm)

include_directories(
    ${PROJECT_SOURCE_DIR}/../include/ck
    ${PROJECT_SOURCE_DIR}/../include/ck/utility
    ${PROJECT_SOURCE_DIR}/../include/ck/host_utility
    /opt/rocm/include
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")

add_executable(utility_tests ${CSRC_FILES} ${TEST_SOURCE_FILES})
target_link_libraries(utility_tests ${GTEST_LIBRARIES} GTest::gtest GTest::gtest_main hip::device pthread stdc++)

set_target_properties(utility_tests PROPERTIES LINK_FLAGS "-Wl,-allow-shlib-undefined")

add_test(NAME utility_tests COMMAND utility_tests)
