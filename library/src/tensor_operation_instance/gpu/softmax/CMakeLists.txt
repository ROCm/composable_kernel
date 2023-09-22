set(DEVICE_SOFTMAX_INSTANCES)
if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    list(APPEND DEVICE_SOFTMAX_INSTANCES device_softmax_f16_f16_instance_rank3_reduce1.cpp
    device_softmax_f16_f16_instance_rank3_reduce2.cpp
    device_softmax_f16_f16_instance_rank3_reduce3.cpp
    device_softmax_f16_f16_instance_rank4_reduce1.cpp
    device_softmax_f16_f16_instance_rank4_reduce2.cpp
    device_softmax_f16_f16_instance_rank4_reduce3.cpp
    device_softmax_f16_f16_instance_rank4_reduce4.cpp)
endif()
if(DTYPES MATCHES "fp32" OR NOT DEFINED DTYPES)
    list(APPEND DEVICE_SOFTMAX_INSTANCES device_softmax_f32_f32_instance_rank3_reduce1.cpp
    device_softmax_f32_f32_instance_rank3_reduce2.cpp
    device_softmax_f32_f32_instance_rank3_reduce3.cpp
    device_softmax_f32_f32_instance_rank4_reduce1.cpp
    device_softmax_f32_f32_instance_rank4_reduce2.cpp
    device_softmax_f32_f32_instance_rank4_reduce3.cpp
    device_softmax_f32_f32_instance_rank4_reduce4.cpp)
endif()
add_instance_library(device_softmax_instance ${DEVICE_SOFTMAX_INSTANCES})
