set(DEVICE_NORMALIZATION_INSTANCES)

list(APPEND DEVICE_NORMALIZATION_INSTANCES 
    device_layernorm2d_f16_instance.cpp
    device_layernorm4d_f16_instance.cpp
    device_groupnorm_f16_instance.cpp
    device_groupnorm_swish_f16_instance.cpp
    device_groupnorm_swish_f16_f32_f32_f16_instance.cpp
    device_layernorm2d_f32_instance.cpp
    device_layernorm4d_f32_instance.cpp
    device_groupnorm_f32_instance.cpp
    device_groupnorm_swish_f32_instance.cpp)

add_instance_library(device_normalization_instance ${DEVICE_NORMALIZATION_INSTANCES})
