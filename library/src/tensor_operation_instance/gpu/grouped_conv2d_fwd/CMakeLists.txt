# device_grouped_conv2d_fwd_instance
set(DEVICE_GROUPED_CONV2D_FWD_INSTANCE_SOURCE
   # GNHWC, GKYXC, GNHWK
   device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_bf16_instance.cpp;
   device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f16_instance.cpp;
   device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f32_instance.cpp;
   device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_int8_instance.cpp;
   # NHWGC, KYXGC, NHWGK
   device_grouped_conv2d_fwd_xdl_nhwgc_kyxgc_nhwgk_f16_instance.cpp;
)

add_library(device_grouped_conv2d_fwd_instance OBJECT ${DEVICE_GROUPED_CONV2D_FWD_INSTANCE_SOURCE}) 
set_target_properties(device_grouped_conv2d_fwd_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)

clang_tidy_check(device_grouped_conv2d_fwd_instance)
