if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
add_instance_library(device_elementwise_normalization_instance
    device_elementwise_normalization_f16_instance.cpp
)
endif()
