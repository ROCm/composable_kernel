set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_compile_options(-std=c++17)
find_package(hip)
add_custom_target(codegen)

# add include directories
include_directories(BEFORE
            ${PROJECT_BINARY_DIR}/include
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/library/include
            ${HIP_INCLUDE_DIRS}
            )

list(APPEND CMAKE_MODULE_PATH ${CK_ROOT}/cmake)
include(Embed)
file(GLOB_RECURSE KERNEL_FILES CONFIGURE_DEPENDS
	${CK_ROOT}/include/ck/*.hpp)
# printouts for debug purposes
#message(STATUS "KERNEL_FILES: ${KERNEL_FILES}")
#message(STATUS "RELATIVE: ${CK_ROOT}/include")
add_embed_library(ck_headers ${KERNEL_FILES} RELATIVE ${CK_ROOT}/include)

file(GLOB SOURCES CONFIGURE_DEPENDS src/*.cpp)

#message(STATUS "SOURCE_FILES: ${SOURCES}")
# TODO: Use object library
add_library(ck_host STATIC ${SOURCES})
add_library(composable_kernel::ck_host ALIAS ck_host)

set_target_properties(ck_host PROPERTIES
	LINKER_LANGUAGE CXX
        POSITION_INDEPENDENT_CODE ON)

target_include_directories(ck_host SYSTEM PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/solution_instances>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/embed/ck_headers/include>
)

target_link_libraries(ck_host PRIVATE $<BUILD_INTERFACE:ck_headers>)

target_include_directories(ck_host PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(ck-template-driver driver/main.cpp)
target_link_libraries(ck-template-driver ck_host)

rocm_install(
    TARGETS ck_host
    EXPORT ck_hostTargets
)
rocm_install(EXPORT ck_hostTargets 
	FILE composable_kernelck_hostTargets.cmake 
	NAMESPACE composable_kernel:: 
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel)
rocm_install(DIRECTORY include/ck DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

rocm_install(
    EXPORT ck_hostTargets
    FILE composable_kernelck_hostTargets.cmake
    NAMESPACE composable_kernel::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)

if (NOT CK_BUILD_HOST_LIB AND BUILD_TESTING)
    add_subdirectory(test)
endif()
