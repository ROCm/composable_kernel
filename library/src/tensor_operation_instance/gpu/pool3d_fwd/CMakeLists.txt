set(DEVICE_POOL3D_FWD_INSTANCES)
if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    list(APPEND DEVICE_POOL3D_FWD_INSTANCES device_avg_pool3d_fwd_ndhwc_f16_instance.cpp
                                          device_max_pool3d_fwd_ndhwc_f16_instance.cpp)
endif()
if(DTYPES MATCHES "bf16" OR NOT DEFINED DTYPES)
    list(APPEND DEVICE_POOL3D_FWD_INSTANCES device_avg_pool3d_fwd_ndhwc_bf16_instance.cpp
                                          device_max_pool3d_fwd_ndhwc_bf16_instance.cpp)
endif()
if(DTYPES MATCHES "fp32" OR NOT DEFINED DTYPES)
    list(APPEND DEVICE_POOL3D_FWD_INSTANCES device_avg_pool3d_fwd_ndhwc_f32_instance.cpp
                                          device_max_pool3d_fwd_ndhwc_f32_instance.cpp)
endif()
add_instance_library(device_pool3d_fwd_instance ${DEVICE_POOL3D_FWD_INSTANCES})
