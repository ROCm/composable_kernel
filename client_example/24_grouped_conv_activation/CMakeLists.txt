# Fwd scaleadd scaleadd relu
add_executable(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_fp32 
               grouped_convnd_fwd_scaleadd_scaleadd_relu/grouped_conv_fwd_scaleadd_scaleadd_relu_fp32.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_fp32 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_fp16
               grouped_convnd_fwd_scaleadd_scaleadd_relu/grouped_conv_fwd_scaleadd_scaleadd_relu_fp16.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_fp16 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_bf16
               grouped_convnd_fwd_scaleadd_scaleadd_relu/grouped_conv_fwd_scaleadd_scaleadd_relu_bf16.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_bf16 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_int8
               grouped_convnd_fwd_scaleadd_scaleadd_relu/grouped_conv_fwd_scaleadd_scaleadd_relu_int8.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_int8 PRIVATE composable_kernel::device_conv_operations)
# Fwd scaleadd AB
add_executable(client_grouped_convnd_fwd_scaleadd_ab_fp32
               grouped_convnd_fwd_scaleadd_ab/grouped_conv_fwd_scaleadd_ab_fp32.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_ab_fp32 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_grouped_convnd_fwd_scaleadd_ab_fp16
               grouped_convnd_fwd_scaleadd_ab/grouped_conv_fwd_scaleadd_ab_fp16.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_ab_fp16 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_grouped_convnd_fwd_scaleadd_ab_bf16
               grouped_convnd_fwd_scaleadd_ab/grouped_conv_fwd_scaleadd_ab_bf16.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_ab_bf16 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_grouped_convnd_fwd_scaleadd_ab_int8
               grouped_convnd_fwd_scaleadd_ab/grouped_conv_fwd_scaleadd_ab_int8.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_ab_int8 PRIVATE composable_kernel::device_conv_operations)
# Fwd bilinear
add_executable(client_grouped_convnd_fwd_bilinear_residual_fp16
               grouped_convnd_fwd_bilinear/grouped_conv_fwd_bilinear_residual_fp16.cpp)
target_link_libraries(client_grouped_convnd_fwd_bilinear_residual_fp16 PRIVATE composable_kernel::device_conv_operations)
# Bwd data bilinear
add_executable(client_grouped_convnd_bwd_data_bilinear_residual_fp16
               grouped_convnd_bwd_data_bilinear/grouped_conv_bwd_data_bilinear_residual_fp16.cpp)
target_link_libraries(client_grouped_convnd_bwd_data_bilinear_residual_fp16 PRIVATE composable_kernel::device_conv_operations)
