set(GROUPED_CONV3D_BWD_WEIGHT
    device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
    device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f32_instance.cpp
    device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_bf16_instance.cpp
    device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instance.cpp
    device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instance.cpp
    device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instance.cpp)

if(DL_KERNELS)
    list(APPEND GROUPED_CONV3D_BWD_WEIGHT
        device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
        device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f32_instance.cpp
        device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_bf16_instance.cpp
        device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_f16_instance.cpp
        device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_f32_instance.cpp
        device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_bf16_instance.cpp)
endif()

add_instance_library(device_grouped_conv3d_bwd_weight_instance ${GROUPED_CONV3D_BWD_WEIGHT})
