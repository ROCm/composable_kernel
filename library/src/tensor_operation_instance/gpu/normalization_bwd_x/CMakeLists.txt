set(DEVICE_NORMALIZATION_BWD_X_INSTANCES)

list(APPEND DEVICE_NORMALIZATION_BWD_X_INSTANCES
    device_groupnorm_bwd_x_f32_instance.cpp
    device_layernorm2d_bwd_x_f16_instance.cpp
    device_layernorm2d_bwd_x_f32_instance.cpp)

add_instance_library(device_normalization_bwd_x_instance ${DEVICE_NORMALIZATION_BWD_X_INSTANCES})
