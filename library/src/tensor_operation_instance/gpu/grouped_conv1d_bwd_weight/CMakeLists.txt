# ONLY XDL_AND_DL_KERNELS
set(GROUPED_CONV1D_BWD_WEIGHT_GNWC
    xdl/device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_f16_instance.cpp
    xdl/device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_f32_instance.cpp
    xdl/device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_bf16_f32_bf16_instance.cpp)

if(DL_KERNELS)
    list(APPEND GROUPED_CONV1D_BWD_WEIGHT_GNWC
        dl/device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_f16_instance.cpp
        dl/device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_f32_instance.cpp
        dl/device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_bf16_f32_bf16_instance.cpp)
    set(GROUPED_CONV1D_BWD_WEIGHT_NWGC
        dl/device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f16_instance.cpp
        dl/device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f32_instance.cpp
        dl/device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_bf16_f32_bf16_instance.cpp)
    add_instance_library(device_grouped_conv1d_nwgc_bwd_weight_instance ${GROUPED_CONV1D_BWD_WEIGHT_NWGC})
endif()

add_instance_library(device_grouped_conv1d_gnwc_bwd_weight_instance ${GROUPED_CONV1D_BWD_WEIGHT_GNWC})
