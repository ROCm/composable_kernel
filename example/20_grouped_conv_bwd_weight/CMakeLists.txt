add_custom_target(example_grouped_conv_bwd_weight)

add_example_executable(example_grouped_conv_bwd_weight_xdl_fp16 grouped_conv_bwd_weight_xdl_fp16.cpp)
add_example_executable(example_grouped_conv_bwd_weight_xdl_bf16 grouped_conv_bwd_weight_xdl_bf16.cpp)


add_dependencies(example_grouped_conv_bwd_weight example_grouped_conv_bwd_weight_xdl_fp16
                                                 example_grouped_conv_bwd_weight_xdl_bf16)

add_custom_target(example_conv_bwd_weight)

add_example_executable(example_convnd_bwd_weight_dl_fp16 convnd_bwd_weight_dl_fp16.cpp)

add_dependencies(example_conv_bwd_weight example_convnd_bwd_weight_dl_fp16)
