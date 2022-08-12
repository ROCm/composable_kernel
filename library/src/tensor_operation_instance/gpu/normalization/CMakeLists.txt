# device_normalization_instance
set(DEVICE_NORMALIZATION_INSTANCE_SOURCE
    device_layernorm_f16_instance.cpp
    device_layernorm_f32_instance.cpp
    device_softmax_f32_f32_instance.cpp
    device_softmax_f16_f16_instance.cpp
)

add_library(device_normalization_instance OBJECT ${DEVICE_NORMALIZATION_INSTANCE_SOURCE})
set_target_properties(device_normalization_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)

clang_tidy_check(device_normalization_instance)
