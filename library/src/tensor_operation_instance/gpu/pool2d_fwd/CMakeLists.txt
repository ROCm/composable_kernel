set(DEVICE_POOL2D_FWD_INSTANCES)
list(APPEND DEVICE_POOL2D_FWD_INSTANCES device_pool2d_avg_fwd_nhwc_f16_instance.cpp
                                        device_pool2d_max_fwd_nhwc_f16_instance.cpp
                                        device_pool2d_avg_fwd_nhwc_f32_instance.cpp
                                        device_pool2d_max_fwd_nhwc_f32_instance.cpp
                                        device_pool2d_avg_fwd_nhwc_bf16_instance.cpp
                                        device_pool2d_max_fwd_nhwc_bf16_instance.cpp)
add_instance_library(device_pool2d_fwd_instance ${DEVICE_POOL2D_FWD_INSTANCES})
