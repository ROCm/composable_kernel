include(Embed)
file(GLOB_RECURSE KERNEL_FILES ${CONFIGURE_DEPENDS}
    ${PROJECT_SOURCE_DIR}/include/ck/*.hpp)
message(STATUS "KERNEL_FILES: ${KERNEL_FILES}")
add_embed_library(ck_headers ${KERNEL_FILES} RELATIVE ${PROJECT_SOURCE_DIR}/build/include)

execute_process(
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/util/make_instance_strings.py 
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../tensor_operation_instance/gpu/
)


set(JIT_LIB_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/device_gemm_multiple_d_xdlop_cshuffle.hpp
)

add_library(jit_library STATIC ${JIT_LIB_SOURCE})
add_library(composable_kernel::jit_library ALIAS jit_library)

set_target_properties(jit_library PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(jit_library PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(jit_library PRIVATE ck_headers)

rocm_install(
    TARGETS jit_library ck_headers
    EXPORT jit_libraryTargets
)

set(INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/include/ck/
    ${PROJECT_SOURCE_DIR}/library/src/jit_library/include
    ${PROJECT_SOURCE_DIR}/library/src/jit_library/solution_instances
    ${CMAKE_CURRENT_BINARY_DIR}/embed/ck_headers/include
)

rocm_install(DIRECTORY ${INCLUDE_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ck)

rocm_install(
    EXPORT jit_libraryTargets
    FILE composable_kerneljit_libraryTargets.cmake
    NAMESPACE composable_kernel::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)


