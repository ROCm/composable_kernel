cmake_minimum_required(VERSION 2.8.3)
project(modular_convolution)

#c++
enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")

#boost
find_package(Boost REQUIRED)

message("Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message("Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")

include_directories(BEFORE ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

#OpenMP
if( NOT( ${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang") )
    find_package(OpenMP REQUIRED)

    message("OpenMP_CXX_LIB_NAMES: ${OpenMP_CXX_LIB_NAMES}")
    message("OpenMP_gomp_LIBRARY: ${OpenMP_gomp_LIBRARY}")
    message("OpenMP_pthread_LIBRARY: ${OpenMP_pthread_LIBRARY}")
    message("OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    link_libraries(${OpenMP_gomp_LIBRARY})
    link_libraries(${OpenMP_pthread_LIBRARY})
endif( NOT( ${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang") )

#GPU backend
if(DEVICE_BACKEND STREQUAL "AMD")
    set(CMAKE_MODULE_PATH "/opt/rocm/hip/cmake" ${CMAKE_MODULE_PATH})
    find_package(HIP REQUIRED)
elseif(DEVICE_BACKEND STREQUAL "NVIDIA")
	enable_language(CUDA)
	include_directories(BEFORE ${CUDA_COMMON_INCLUDE_DIR})
endif()

#
include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/composable_kernel/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/utility
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_description
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_operation
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/kernel_algorithm
    ${PROJECT_SOURCE_DIR}/external/include
    ${PROJECT_SOURCE_DIR}/driver/include
    ${PROJECT_BINARY_DIR}/composable_kernel/include/utility
)

if(DEVICE_BACKEND STREQUAL "AMD")
    configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/config.amd.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/config.hpp")
    configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/float_type.amd.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/float_type.hpp")
    configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/in_memory_operation.amd.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/in_memory_operation.hpp")
elseif(DEVICE_BACKEND STREQUAL "NVIDIA")
    configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/config.nvidia.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/config.hpp")
    configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/float_type.nvidia.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/float_type.hpp")
    configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/in_memory_operation.nvidia.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/in_memory_operation.hpp")
endif()

add_subdirectory(driver)
