## host_tensor
include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/ck
    ${PROJECT_SOURCE_DIR}/include/ck/utility
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/host_tensor
)

set(HOST_TENSOR_SOURCE
    device.cpp
    host_tensor.cpp
)

add_library(host_tensor STATIC ${HOST_TENSOR_SOURCE})
add_library(composable_kernel::host_tensor ALIAS host_tensor)

target_compile_features(host_tensor PUBLIC)
set_target_properties(host_tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(host_tensor SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)

target_include_directories(host_tensor PUBLIC 
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/utility>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/host_tensor>"
)

install(TARGETS host_tensor 
        EXPORT host_tensorTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT host_tensorTargets
    FILE composable_kernelhost_tensorTargets.cmake 
    NAMESPACE composable_kernel::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)

clang_tidy_check(host_tensor)
