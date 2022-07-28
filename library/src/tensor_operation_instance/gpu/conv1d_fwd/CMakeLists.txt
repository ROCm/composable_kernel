# device_conv1d_fwd_instance
set(DEVICE_CONV1D_FWD_INSTANCE_SOURCE
   device_conv1d_fwd_xdl_nwc_kxc_nwk_bf16_instance.cpp;
   device_conv1d_fwd_xdl_nwc_kxc_nwk_f16_instance.cpp;
   device_conv1d_fwd_xdl_nwc_kxc_nwk_f32_instance.cpp;
   device_conv1d_fwd_xdl_nwc_kxc_nwk_int8_instance.cpp;
)

add_instance_library(device_conv1d_fwd_instance OBJECT ${DEVICE_CONV1D_FWD_INSTANCE_SOURCE})
