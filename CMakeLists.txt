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
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	# workaround issue hipcc in rocm3.5 cannot find openmp
	set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
	set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
	set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
	set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
else()
	find_package(OpenMP REQUIRED)
endif()

message("OpenMP_CXX_LIB_NAMES: ${OpenMP_CXX_LIB_NAMES}")
message("OpenMP_gomp_LIBRARY: ${OpenMP_gomp_LIBRARY}")
message("OpenMP_pthread_LIBRARY: ${OpenMP_pthread_LIBRARY}")
message("OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
link_libraries(${OpenMP_gomp_LIBRARY})
link_libraries(${OpenMP_pthread_LIBRARY})

#GPU backend
if(DEVICE_BACKEND STREQUAL "AMD")
    find_package(HIP REQUIRED)
elseif(DEVICE_BACKEND STREQUAL "NVIDIA")
    enable_language(CUDA)
endif()

#
include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/composable_kernel/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/utility
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_description
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_operation
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/kernel_algorithm
    ${PROJECT_SOURCE_DIR}/external/half/include
    ${PROJECT_SOURCE_DIR}/driver/include
    ${PROJECT_BINARY_DIR}/composable_kernel/include/utility
)

if(DEVICE_BACKEND STREQUAL "AMD")
    include_directories(BEFORE
        ${PROJECT_SOURCE_DIR}/external/rocm/include
    )
endif()

if(DEVICE_BACKEND STREQUAL "AMD")
    configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/config.amd.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/config.hpp")
    configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/float_type.amd.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/float_type.hpp")
    configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/in_memory_operation.amd.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/in_memory_operation.hpp")
    configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/synchronization.amd.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/synchronization.hpp")
elseif(DEVICE_BACKEND STREQUAL "NVIDIA")
    configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/config.nvidia.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/config.hpp")
    configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/float_type.nvidia.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/float_type.hpp")
    configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/in_memory_operation.nvidia.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/in_memory_operation.hpp")
    configure_file("${PROJECT_SOURCE_DIR}/composable_kernel/include/utility/synchronization.nvidia.hpp.in" "${PROJECT_BINARY_DIR}/composable_kernel/include/utility/synchronization.hpp")
endif()

add_subdirectory(driver)
