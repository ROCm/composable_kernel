cmake_minimum_required(VERSION 2.8.3)
project(modular_convolution)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(TargetFlags)
include(AddKernels)

#c++
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")

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
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/driver
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

message("Compiling options for drivers: ${CMAKE_CXX_FLAGS}")

if(DEVICE_BACKEND STREQUAL "AMD")
    set(CONV_SOURCE driver/conv_driver.cpp)
    set(CONV_BWD_DATA_SOURCE driver/conv_bwd_data_driver.cpp)
    set(CONV_V2_SOURCE driver/conv_driver_v2.cpp)
    set(CONV_BWD_DATA_V2_SOURCE driver/conv_bwd_data_driver_v2.cpp)
    set(CONV_V2_OLC_SOURCE driver/conv_driver_v2_olc.cpp)
elseif(DEVICE_BACKEND STREQUAL "NVIDIA")
    set(CONV_SOURCE driver/conv_driver.cu)
    set(CONV_BWD_DATA_SOURCE driver/conv_bwd_data_driver.cu)
endif()

add_executable(conv_driver ${CONV_SOURCE})
add_executable(conv_bwd_data_driver ${CONV_BWD_DATA_SOURCE})
add_executable(conv_driver_v2 ${CONV_V2_SOURCE})
add_executable(conv_bwd_data_driver_v2 ${CONV_BWD_DATA_V2_SOURCE})
add_executable(conv_driver_v2_olc ${CONV_V2_OLC_SOURCE})

target_include_directories(conv_driver_v2_olc PRIVATE driver/olCompiling/include/)

target_link_libraries(conv_driver PRIVATE modConv)
target_link_libraries(conv_bwd_data_driver PRIVATE modConv)
target_link_libraries(conv_driver_v2 PRIVATE modConv)
target_link_libraries(conv_bwd_data_driver_v2 PRIVATE modConv)
target_link_libraries(conv_driver_v2_olc PRIVATE modConv)


