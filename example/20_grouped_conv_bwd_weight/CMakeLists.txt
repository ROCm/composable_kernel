if(GPU_TARGETS MATCHES "gfx908" OR GPU_TARGETS MATCHES "gfx90a" OR GPU_TARGETS MATCHES "gfx940")
   add_custom_target(example_grouped_conv_bwd_weight)

   add_example_executable(example_grouped_conv_bwd_weight_xdl_fp16 grouped_conv_bwd_weight_xdl_fp16.cpp)
   add_example_executable(example_grouped_conv_bwd_weight_xdl_bf16 grouped_conv_bwd_weight_xdl_bf16.cpp)

   add_dependencies(example_grouped_conv_bwd_weight example_grouped_conv_bwd_weight_xdl_fp16
                                                 example_grouped_conv_bwd_weight_xdl_bf16)
endif()

add_custom_target(example_grouped_conv_bwd_weight_dl)

add_example_executable(example_grouped_conv_bwd_weight_dl_fp16 grouped_conv_bwd_weight_dl_fp16.cpp)

add_dependencies(example_grouped_conv_bwd_weight_dl example_grouped_conv_bwd_weight_dl_fp16)
