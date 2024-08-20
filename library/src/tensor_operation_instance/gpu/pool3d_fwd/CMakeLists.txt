set(DEVICE_POOL3D_FWD_INSTANCES)
list(APPEND DEVICE_POOL3D_FWD_INSTANCES device_pool3d_avg_fwd_ndhwc_f16_instance.cpp
                                        device_pool3d_max_fwd_ndhwc_f16_instance.cpp
                                        device_pool3d_avg_fwd_ndhwc_f32_instance.cpp
                                        device_pool3d_max_fwd_ndhwc_f32_instance.cpp
                                        device_pool3d_avg_fwd_ndhwc_bf16_instance.cpp
                                        device_pool3d_max_fwd_ndhwc_bf16_instance.cpp)
add_instance_library(device_pool3d_fwd_instance ${DEVICE_POOL3D_FWD_INSTANCES})
