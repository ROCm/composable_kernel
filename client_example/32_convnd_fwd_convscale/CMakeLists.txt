if((DTYPES MATCHES "fp8") OR NOT DEFINED DTYPES)
    add_executable(client_conv3d_fwd_convscale_fp8 conv3d_fwd_convscale_fp8.cpp)
    target_link_libraries(client_conv3d_fwd_convscale_fp8 PRIVATE composable_kernel::device_conv_operations)
endif()
