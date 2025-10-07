# ONLY DL_KERNELS
if(DL_KERNELS)
    set(GROUPED_CONV1D_BWD_WEIGHT_NWGC
        dl/device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f16_instance.cpp
        dl/device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f32_instance.cpp
        dl/device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_bf16_f32_bf16_instance.cpp)
    add_instance_library(device_grouped_conv1d_nwgc_bwd_weight_instance ${GROUPED_CONV1D_BWD_WEIGHT_NWGC})
endif()
