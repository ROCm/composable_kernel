set(GROUPED_CONV1D_BWD_WEIGHT
    device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_f16_instance.cpp
    device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_f32_instance.cpp
    device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_bf16_instance.cpp)

if(DL_KERNELS)
    list(APPEND GROUPED_CONV1D_BWD_WEIGHT
        device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_f16_instance.cpp
        device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_f32_instance.cpp
        device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_bf16_instance.cpp
        device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f16_instance.cpp
        device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f32_instance.cpp
        device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_bf16_instance.cpp)
endif()

add_instance_library(device_grouped_conv1d_bwd_weight_instance ${GROUPED_CONV1D_BWD_WEIGHT})
