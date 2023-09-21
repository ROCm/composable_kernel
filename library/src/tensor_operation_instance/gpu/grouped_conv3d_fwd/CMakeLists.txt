add_instance_library(device_grouped_conv3d_fwd_instance
if((DTYPES MATCHES "fp16") OR NOT DEFINED DTYPES)
   device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
   device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instance.cpp
   device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_f8_instance.cpp
endif()


if((DTYPES MATCHES "fp32") OR NOT DEFINED DTYPES)
   device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_f32_instance.cpp
   device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instance.cpp
endif()

if((DTYPES MATCHES "bf16") OR NOT DEFINED DTYPES)
   device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_bf16_instance.cpp
   device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instance.cpp
endif()

if((DTYPES MATCHES "int8") OR NOT DEFINED DTYPES)
   device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_int8_instance.cpp
   device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_int8_instance.cpp
endif()

)
