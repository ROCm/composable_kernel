if(GPU_TARGETS MATCHES "gfx9")
    add_executable(client_grouped_conv2d_fwd grouped_conv2d_fwd.cpp)
    target_link_libraries(client_grouped_conv2d_fwd PRIVATE composable_kernel::device_conv_operations)

    add_executable(client_grouped_conv1d_fwd grouped_conv1d_fwd.cpp)
    target_link_libraries(client_grouped_conv1d_fwd PRIVATE composable_kernel::device_conv_operations)
endif()