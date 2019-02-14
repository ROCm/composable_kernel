cmake_minimum_required(VERSION 2.8.3)
project(modular_convolution)

#c++
enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")

#boost
find_package(Boost REQUIRED)

message("Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message("Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")

include_directories(BEFORE ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

#openMP
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

#cuda
enable_language(CUDA)
include_directories(BEFORE ${CUDA_COMMON_INCLUDE_DIR})

#
include_directories(BEFORE src/include)
add_subdirectory(src)
add_subdirectory(driver)
