 # XDL_DL_WMMA_KERNELS
set(GROUPED_CONV3D_BWD_WEIGHT_GNDHWC
     xdl/gndhwc_gkzyxc_gndhwk/device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
     xdl/gndhwc_gkzyxc_gndhwk/device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f32_instance.cpp
     xdl/gndhwc_gkzyxc_gndhwk/device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_bf16_f32_bf16_instance.cpp
     wmma/device_grouped_conv3d_bwd_weight_wmma_gndhwc_gkzyxc_gndhwk_f16_1x1s1p0_instance.cpp
     wmma/device_grouped_conv3d_bwd_weight_wmma_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
     wmma/device_grouped_conv3d_bwd_weight_wmma_gndhwc_gkzyxc_gndhwk_i8_1x1s1p0_instance.cpp
     wmma/device_grouped_conv3d_bwd_weight_wmma_gndhwc_gkzyxc_gndhwk_i8_instance.cpp
)

if(DL_KERNELS)
    list(APPEND GROUPED_CONV3D_BWD_WEIGHT_GNDHWC
        dl/device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
        dl/device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f32_instance.cpp
        dl/device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_bf16_f32_bf16_instance.cpp)
endif()

add_instance_library(device_grouped_conv3d_gndhwc_bwd_weight_instance ${GROUPED_CONV3D_BWD_WEIGHT_GNDHWC})
