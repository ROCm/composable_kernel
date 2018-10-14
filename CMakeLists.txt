cmake_minimum_required(VERSION 3.9)
project(convolution LANGUAGES CXX CUDA)

#c++
message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
add_compile_options(-std=c++14)

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

#python
find_package(PythonLibs 3 REQUIRED)

message("PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
message("PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")

include_directories(BEFORE ${PYTHON_INCLUDE_DIRS})
link_libraries(${PYTHON_LIBRARIES})

#cuda
include_directories(BEFORE ${CUDA_COMMON_INCLUDE_DIR})

#
add_subdirectory(src)
add_subdirectory(driver)
