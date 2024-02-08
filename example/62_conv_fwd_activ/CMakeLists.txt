list(APPEND gpu_list gfx908 gfx90a gfx940 gfx941 gfx942)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
 if(gpu IN_LIST gpu_list AND target EQUAL 0)
      add_custom_target(example_convnd_fwd_activ_xdl)
      # Sigmoid
      add_example_executable(example_convnd_fwd_xdl_sigmoid_fp16 convnd_fwd_xdl_sigmoid_fp16.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_convnd_fwd_xdl_sigmoid_fp16)
      # Tanh
      add_example_executable(example_convnd_fwd_xdl_tanh_fp16 convnd_fwd_xdl_tanh_fp16.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_convnd_fwd_xdl_tanh_fp16)
      # Relu
      add_example_executable(example_convnd_fwd_xdl_relu_fp16 convnd_fwd_xdl_relu_fp16.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_convnd_fwd_xdl_relu_fp16)
      # SoftRelu
      add_example_executable(example_convnd_fwd_xdl_softrelu_fp16 convnd_fwd_xdl_softrelu_fp16.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_convnd_fwd_xdl_softrelu_fp16)
      # Abs
      add_example_executable(example_convnd_fwd_xdl_abs_fp16 convnd_fwd_xdl_abs_fp16.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_convnd_fwd_xdl_abs_fp16)
      # Pow
      add_example_executable(example_convnd_fwd_xdl_pow_fp16 convnd_fwd_xdl_pow_fp16.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_convnd_fwd_xdl_pow_fp16)
      # Clipped Relu
      add_example_executable(example_convnd_fwd_xdl_clippedrelu_fp16 convnd_fwd_xdl_clippedrelu_fp16.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_convnd_fwd_xdl_clippedrelu_fp16)
      # Leaky Relu
      add_example_executable(example_convnd_fwd_xdl_leakyrelu_fp16 convnd_fwd_xdl_leakyrelu_fp16.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_convnd_fwd_xdl_leakyrelu_fp16)
      # Elu
      add_example_executable(example_convnd_fwd_xdl_elu_fp16 convnd_fwd_xdl_elu_fp16.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_convnd_fwd_xdl_elu_fp16)
      # ScaleAdd on A and B
      add_example_executable(example_conv_fwd_xdl_scaleadd_ab_fp16 multi_AB/conv_fwd_xdl_scaleadd_ab_fp16.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_conv_fwd_xdl_scaleadd_ab_fp16)
      add_example_executable(example_conv_fwd_xdl_scaleadd_ab_fp32 multi_AB/conv_fwd_xdl_scaleadd_ab_fp32.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_conv_fwd_xdl_scaleadd_ab_fp32)
      add_example_executable(example_conv_fwd_xdl_scaleadd_ab_bf16 multi_AB/conv_fwd_xdl_scaleadd_ab_bf16.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_conv_fwd_xdl_scaleadd_ab_bf16)
      add_example_executable(example_conv_fwd_xdl_scaleadd_ab_int8 multi_AB/conv_fwd_xdl_scaleadd_ab_int8.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_conv_fwd_xdl_scaleadd_ab_int8)
      # ScaleAdd ScaleAdd Relu
      add_example_executable(example_convnd_fwd_xdl_scaleadd_scaleadd_relu_fp16 convnd_fwd_xdl_scaleadd_scaleadd_relu_fp16.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_convnd_fwd_xdl_scaleadd_scaleadd_relu_fp16)
      add_example_executable(example_convnd_fwd_xdl_scaleadd_scaleadd_relu_bcasted_bias_fp16 convnd_fwd_xdl_scaleadd_scaleadd_relu_bcasted_bias_fp16.cpp)
      add_example_dependencies(example_convnd_fwd_activ_xdl example_convnd_fwd_xdl_scaleadd_scaleadd_relu_bcasted_bias_fp16)
   set(target 1)
 endif()
endforeach()
