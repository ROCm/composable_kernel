# device_normalization_instance
set(DEVICE_NORMALIZATION_INSTANCE_SOURCE
    device_softmax_f32_f32_instance.cpp
    device_softmax_f16_f16_instance.cpp
)

add_instance_library(device_normalization_instance OBJECT ${DEVICE_NORMALIZATION_INSTANCE_SOURCE})
