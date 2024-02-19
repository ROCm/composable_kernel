include(Embed)
file(GLOB_RECURSE KERNEL_FILES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/include/ck/*.hpp)
message(STATUS "KERNEL_FILES: ${KERNEL_FILES}")
message(STATUS "RELATIVE: ${PROJECT_SOURCE_DIR}/include")
add_embed_library(ck_headers ${KERNEL_FILES} RELATIVE ${PROJECT_SOURCE_DIR}/include)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter)

execute_process(
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/util/make_instance_strings.py
        ${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance/gpu
        ${CMAKE_CURRENT_BINARY_DIR}/solution_instances
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../tensor_operation_instance/gpu/
)

add_library(jit_library STATIC 
    src/device_batched_gemm_softmax_gemm.cpp
    src/device_gemm_multiple_d.cpp
    src/common.cpp
)
add_library(composable_kernel::jit_library ALIAS jit_library)

set_target_properties(jit_library PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(jit_library SYSTEM PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/jit_library/solution_instances>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/solution_instances>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/embed/ck_headers/include>
)

target_link_libraries(jit_library PRIVATE $<BUILD_INTERFACE:ck_headers>)

rocm_install(
    TARGETS jit_library 
    EXPORT jit_libraryTargets
)

rocm_install(DIRECTORY include/ck DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
rocm_install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ck DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

rocm_install(
    EXPORT jit_libraryTargets
    FILE composable_kerneljit_libraryTargets.cmake
    NAMESPACE composable_kernel::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)
