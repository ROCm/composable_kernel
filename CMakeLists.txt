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
    set(CK_DEVICE_BACKEND_AMD 1)

    set(CMAKE_MODULE_PATH "/opt/rocm/hip/cmake" ${CMAKE_MODULE_PATH})
    find_package(HIP REQUIRED)
elseif(DEVICE_BACKEND STREQUAL "NVIDIA")
    set(CK_DEVICE_BACKEND_NVIDIA 1)

	enable_language(CUDA)
	include_directories(BEFORE ${CUDA_COMMON_INCLUDE_DIR})
endif()

#
include_directories(BEFORE
    include
    ${PROJECT_BINARY_DIR}/include
)
add_subdirectory(src)
add_subdirectory(driver)
