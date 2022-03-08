#file(GLOB_RECURSE COMPOSABLE_KERNEL_HEADERS "composable_kernel/include/*/*.hpp")
#file(GLOB_RECURSE DEVICE_OPS_HEADERS "device_operation/include/*.hpp")
#
#file(GLOB_RECURSE DEVICE_OPS_SOURCE "device_operation/*.cpp")
#
#set(CK_HEADERS ${COMPOSABLE_KERNEL_HEADERS} ${DEVICE_OPS_HEADERS})
#set(CK_SOURCE ${DEVICE_OPS_SOURCE})
#add_library(composable_kernel ${CK_SOURCE})

add_subdirectory(src/host_tensor)
add_subdirectory(src/tensor_operation_instance/gpu)

## composable_kernel
file(GLOB_RECURSE CK_LIBRARY_SOURCE "src/*/*.cpp")
set(CK_SOURCE ${CK_LIBRARY_SOURCE})
add_library(composable_kernel ${CK_SOURCE})

target_include_directories(composable_kernel PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_include_directories(composable_kernel PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)

target_include_directories(composable_kernel PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
)

include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/composable_kernel/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/utility
    include
)

set(HOST_TENSOR_SOURCE
    src/host_tensor.cpp;
    src/device.cpp;
)

clang_tidy_check(composable_kernel)
