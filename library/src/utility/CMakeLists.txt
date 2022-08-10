## utility
set(UTILITY_SOURCE
    device_memory.cpp
    host_tensor.cpp
    convolution_parameter.cpp
)

add_library(utility STATIC ${UTILITY_SOURCE})
add_library(composable_kernel::utility ALIAS utility)

set(UTILITY_CMAKE_CXX_FLAGS)

if(USE_BITINT_EXTENSION_INT4)
    list(APPEND UTILITY_CMAKE_CXX_FLAGS
         -Wno-bit-int-extension
    )
endif()

target_compile_options(utility PRIVATE ${UTILITY_CMAKE_CXX_FLAGS})

target_include_directories(utility PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/utility>"
)

rocm_install(
    TARGETS utility
    EXPORT utilityTargets
)

rocm_install(
    EXPORT utilityTargets
    FILE composable_kernelutilityTargets.cmake
    NAMESPACE composable_kernel::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)

clang_tidy_check(utility)
