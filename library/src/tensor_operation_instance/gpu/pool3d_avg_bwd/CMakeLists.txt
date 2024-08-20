set(DEVICE_AVGPOOL_BWD_INSTANCES)
list(APPEND DEVICE_AVGPOOL_BWD_INSTANCES device_pool3d_avg_bwd_ndhwc_f16_instance.cpp
                                         device_pool3d_avg_bwd_ndhwc_bf16_instance.cpp
                                         device_pool3d_avg_bwd_ndhwc_f32_instance.cpp)
add_instance_library(device_pool3d_avg_bwd_instance ${DEVICE_AVGPOOL_BWD_INSTANCES})
